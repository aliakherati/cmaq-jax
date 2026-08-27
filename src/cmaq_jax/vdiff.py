"""Vertical diffusion — the ACM2 solvers.

Ports `vdiff/acm2_m3dry/`. CMAQ's science driver runs vertical diffusion
**first**, before coupling and transport (`sciproc.F`), on uncoupled
concentrations — so it is not part of the coupled transport block that
:func:`cmaq_jax.api.transport_step` implements.

ACM2 (Pleim 2007) splits each sub-step in two:

* a **non-local convective stage**, active only where the column is convective,
  in which mass leaves the surface layer and arrives directly in every layer of
  the convective boundary layer, returning by layer-to-layer subsidence. Because
  every row couples to column 1, the matrix is *tridiagonal plus a full first
  column*, and needs :func:`solve_acm1` rather than :func:`solve_tridiagonal`;
* a **local diffusion stage**, always active, which is ordinary vertical
  diffusion and uses the Thomas algorithm.

Both stages are Crank–Nicolson (`THETA = 0.5`, `vdiffacmx.F:94`), so the
sub-step limit is an *accuracy* constraint rather than a stability one — a real
difference from horizontal diffusion, where the analogous constant sits past the
stability boundary.

Both solvers take one matrix and many right-hand sides: the column's
factorisation is shared across every species, and doing it per species would be
both slower and a different computation to read.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cmaq_jax.config import DEFAULT_ACM2, ACM2Constants

#: Gravitational acceleration, m/s^2. ``CONST.EXT:69``.
GRAV = 9.80622

#: Dry-air gas constant, J/(kg K). ``CONST.EXT:116``.
RDGAS = 287.07548994

#: Water-vapour gas constant, J/(kg K). ``CONST.EXT``.
RWVAP = 461.52492604

__all__ = [
    "ACM2Column",
    "ColumnGeometry",
    "ColumnState",
    "SurfaceExchange",
    "VerticalMeteorology",
    "acm2_column_step",
    "acm2_setup",
    "column_geometry",
    "eddy_diffusivity",
    "solve_acm1",
    "solve_tridiagonal",
    "substep_counts",
    "vdiff_step",
]


def solve_tridiagonal(sub: Array, diag: Array, sup: Array, rhs: Array) -> Array:
    """Solve a tridiagonal system by the Thomas algorithm. Ports ``tri.F``.

    The matrix follows CMAQ's storage (``tri.F:40-46``): row ``k`` holds
    ``sub[k]`` at column ``k-1``, ``diag[k]`` at ``k`` and ``sup[k]`` at ``k+1``.
    So ``sub[0]`` and ``sup[-1]`` are never referenced — the golden generator
    fills them with a poison value so that a port which touches one disagrees
    loudly rather than subtly.

    ``rhs`` is ``(nspcs, nlays)``; the result has the same shape. The layer axis
    is last because that is the axis the recurrence runs along, and CMAQ stores
    it that way too.

    The recurrence is sequential by nature. At ``nlays ~ 35`` a scan is
    inexpensive; on a GPU a short sequential scan can dominate, which is a
    question for the benchmark rather than an assumption to make here.
    """
    sub, diag, sup, rhs = (jnp.asarray(a) for a in (sub, diag, sup, rhs))
    nlays = diag.shape[0]

    def forward(
        carry: tuple[Array, Array], k: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        bet, prev = carry
        gam = bet * sup[k - 1]
        bet = 1.0 / (diag[k] - sub[k] * gam)
        x = bet * (rhs[:, k] - sub[k] * prev)
        return (bet, x), (x, gam)

    bet0 = 1.0 / diag[0]
    x0 = bet0 * rhs[:, 0]
    (_, _), (xs, gams) = jax.lax.scan(forward, (bet0, x0), jnp.arange(1, nlays))

    # xs/gams cover layers 1..nlays-1; prepend layer 0 (whose gamma is unused).
    x_all = jnp.concatenate([x0[:, None], xs.T], axis=1)
    gam_all = jnp.concatenate([jnp.zeros_like(gams[:1]), gams])

    def backward(nxt: Array, k: Array) -> tuple[Array, Array]:
        # GAM(K+1), not GAM(K) -- tri.F:113 indexes the gamma of the layer
        # *above* the one being corrected.
        x = x_all[:, k] - gam_all[k + 1] * nxt
        return x, x

    _, back = jax.lax.scan(backward, x_all[:, -1], jnp.arange(nlays - 2, -1, -1))

    # `back` is layers nlays-2 .. 0, so reverse it and re-attach the top layer.
    return jnp.concatenate([back[::-1].T, x_all[:, -1:]], axis=1)


def solve_acm1(col: Array, diag: Array, sup: Array, rhs: Array, kl: Array | int) -> Array:
    """Solve the ACM1 system — tridiagonal plus a full first column.

    Ports ``matrix1.F``. Over rows ``1..kl`` (1-based, as CMAQ counts):

    * row 1: ``diag[0]·x[0] + sup[1]·x[1]``
    * rows 2..kl-1: ``col[L]·x[0] + diag[L]·x[L] + sup[L+1]·x[L+1]``
    * row kl: ``col[kl]·x[0] + diag[kl]·x[kl]``

    ``col`` is a *column*, not a subdiagonal, and ``sup[L]`` sits in row ``L-1``.
    ``col[0]`` and ``sup[0]`` are unused. This layout was confirmed by residual
    against the compiled Fortran, not read off the comment block.

    ``kl`` is the top of the convective boundary layer and varies per column in
    a real run, so it is handled by masking rather than by slicing: rows above it
    are returned as zero, matching a caller that zeroes ``X`` before the call.

    The elimination uses a running product ``alpha[L] = Π(-sup/diag)``, which
    over a deep CBL gets very small — measured at 1.9e-38 on the
    ``alpha_underflow`` golden, right at float32's smallest normal. That is
    harmless: ``alpha`` weights contributions to ``gama`` that are already
    negligible beside ``diag[0]``, so losing them to underflow costs nothing.
    Guarding it would be guarding against the arithmetic working correctly.
    """
    col, diag, sup, rhs = (jnp.asarray(a) for a in (col, diag, sup, rhs))
    nlays = diag.shape[0]
    layer = jnp.arange(nlays)
    kl = jnp.asarray(kl)

    # Rows 1..kl-1 (0-based) take part in the elimination; row 0 seeds it.
    inner = (layer >= 1) & (layer < kl)

    # alpha[L] = prod_{k=1..L} (-sup[k]/diag[k]); 1 outside the active range so
    # the cumulative product is unaffected, then zeroed so it contributes
    # nothing to the sums below.
    ratio = jnp.where(inner, -sup / diag, 1.0)
    alpha = jnp.where(inner, jnp.cumprod(ratio), 0.0)

    beta = rhs[:, 0] + (alpha * rhs).sum(axis=1)
    gama = diag[0] + (alpha * col).sum()
    x0 = beta / gama

    # Row kl-1 has no super-diagonal term inside the system, so zero sup there
    # and one downward recurrence covers every active row including the last.
    sup_eff = jnp.where(layer < kl, sup, 0.0)

    def backward(nxt: Array, k: Array) -> tuple[Array, Array]:
        upper = jnp.where(k + 1 < nlays, sup_eff[jnp.minimum(k + 1, nlays - 1)], 0.0)
        x = (rhs[:, k] - col[k] * x0 - upper * nxt) / diag[k]
        return x, x

    _, back = jax.lax.scan(backward, jnp.zeros_like(x0), jnp.arange(nlays - 1, 0, -1))

    solution = jnp.concatenate([x0[:, None], back[::-1].T], axis=1)
    return jnp.where(layer < kl, solution, 0.0)


def _wind_shear_squared(uwind: Array, vwind: Array, c_staggered: bool) -> Array:
    """Component-wise squared wind difference across each layer interface.

    Ports ``eddyx.F:138-152``. ``uwind``/``vwind`` are dot-dimensioned
    ``(ncols+1, nrows+1, nlays)``. The two branches average a different number
    of points: C-staggered winds need only the two faces bounding a cell
    (factor 1/4), while B-staggered winds are at cell corners and take all four
    (factor 1/16).
    """
    du = uwind[..., 1:] - uwind[..., :-1]
    dv = vwind[..., 1:] - vwind[..., :-1]
    if c_staggered:
        u_sum = du[1:, :-1] + du[:-1, :-1]
        v_sum = dv[:-1, 1:] + dv[:-1, :-1]
        return 0.25 * (u_sum**2 + v_sum**2)
    u_sum = du[:-1, :-1] + du[1:, :-1] + du[:-1, 1:] + du[1:, 1:]
    v_sum = dv[:-1, :-1] + dv[1:, :-1] + dv[:-1, 1:] + dv[1:, 1:]
    return (1.0 / 16.0) * (u_sum**2 + v_sum**2)


class VerticalMeteorology(NamedTuple):
    """The met fields ``eddyx.F`` reads, named as ``Met_Data`` names them.

    Grouped rather than passed separately because there are twelve of them and
    their order carries no meaning — the same reason
    :class:`cmaq_jax.api.Meteorology` exists.

    Surface fields are ``(ncols, nrows)``; layer fields ``(ncols, nrows, nlays)``;
    the winds are dot-dimensioned ``(ncols+1, nrows+1, nlays)``.

    ``moli`` is the *inverse* Monin–Obukhov length, so its sign selects the
    stability regime — negative unstable, zero neutral, positive stable. Storing
    the inverse avoids the singularity at neutral, where ``L`` itself is
    infinite.
    """

    pbl: Array
    ustar: Array
    moli: Array
    zf: Array
    zh: Array
    kzmin: Array
    thetav: Array
    ta: Array
    qv: Array
    qc: Array
    uwind: Array
    vwind: Array


def eddy_diffusivity(
    met: VerticalMeteorology,
    *,
    c_staggered: bool = True,
    constants: ACM2Constants = DEFAULT_ACM2,
) -> Array:
    """Vertical eddy diffusivity ``Kz``, m^2/s. Ports ``eddyx.F:104-215``.

    Returns ``(ncols, nrows, nlays)`` with the **top layer zero**: the
    diffusivity lives on layer interfaces, of which there are ``nlays - 1``.

    Three regimes, combined by taking the larger of two estimates below the PBL:

    * **surface layer** (``z < PBL``) — Monin–Obukhov similarity,
      ``Kz = κ·(u*/φ_h)·z·(1 − z/h)²``. In neutral conditions ``φ_h = 1`` and
      this is exact in closed form, which is what the ``neutral`` golden checks.
    * **free atmosphere** — a Richardson-number-damped mixing length, with
      different formulae either side of ``Ri = 0``.
    * **moist correction** — where cloud water exceeds ``0.01 g/kg``, ``Ri`` is
      rescaled for latent heating (HIRPBL). Omitting it would pass every dry
      case, so one golden is cloudy.

    Every Fortran branch becomes a ``jnp.where``; the guards on ``sqrt`` and
    division follow the same double-``where`` discipline as the advection
    kernels, so gradients stay finite where the Fortran merely stays defined.
    """
    pbl, ustar, moli, zf, zh, kzmin, thetav, ta, qv, qc, uwind, vwind = met
    pbl, ustar, moli, zf, zh, kzmin, thetav, ta, qv, qc, uwind, vwind = (
        jnp.asarray(a) for a in (pbl, ustar, moli, zf, zh, kzmin, thetav, ta, qv, qc, uwind, vwind)
    )

    hpbl = jnp.maximum(pbl, 20.0)[..., None]
    zfl = zf[..., :-1]
    zol = zfl * moli[..., None]

    # --- surface-layer similarity (eddyx.F:112-136) ------------------------
    # Unstable: the stability function is frozen above 0.1*PBL, so the
    # diffusivity does not keep growing through the mixed layer.
    zsol = 0.1 * hpbl * moli[..., None]
    unstable_arg = jnp.where(zfl < 0.1 * hpbl, zol, zsol)
    # Guard the sqrt: 1 - GAMAH*z/L is positive whenever z/L < 0, which is the
    # only branch that reaches it, but the *other* branch still traces through.
    radicand = 1.0 - constants.gamah * unstable_arg
    safe_radicand = jnp.where(radicand > 0.0, radicand, 1.0)
    phih_unstable = 1.0 / jnp.sqrt(safe_radicand)

    phih_stable = jnp.where(zol < 1.0, 1.0 + constants.betah * zol, constants.betah + zol)
    phih = jnp.where(zol < 0.0, phih_unstable, phih_stable)

    zfunc = 1.0 - zfl / hpbl
    edyz = constants.karman * (ustar[..., None] / phih) * zfl * zfunc * zfunc
    edyz = jnp.maximum(edyz, kzmin[..., :-1])
    below_pbl = zfl < hpbl
    edyz = jnp.where(below_pbl, edyz, 0.0)

    # --- free atmosphere (eddyx.F:155-190) ---------------------------------
    dzl = zh[..., 1:] - zh[..., :-1]
    ww2 = _wind_shear_squared(uwind, vwind, c_staggered)
    ws2 = ww2 / (dzl * dzl) + 1.0e-9

    theta_lo, theta_hi = thetav[..., :-1], thetav[..., 1:]
    rib = 2.0 * GRAV * (theta_hi - theta_lo) / (dzl * ws2 * (theta_hi + theta_lo))

    # Moist correction, applied where either bounding layer holds cloud water.
    qmean = 0.5 * (qv[..., :-1] + qv[..., 1:])
    tmean = 0.5 * (ta[..., :-1] + ta[..., 1:])
    xlv = (2.501 - 0.00237 * (tmean - 273.15)) * 1.0e6
    alph = xlv * qmean / RDGAS / tmean
    cpair = 1004.67 * (1.0 + 0.84 * qv[..., :-1])
    chi = xlv * xlv * qmean / (cpair * RWVAP * tmean * tmean)
    rib_moist = (1.0 + alph) * (
        rib - GRAV * GRAV / (ws2 * tmean * cpair) * ((chi - alph) / (1.0 + chi))
    )
    cloudy = (qc[..., :-1] > constants.qc_threshold) | (qc[..., 1:] > constants.qc_threshold)
    rib = jnp.where(cloudy, rib_moist, rib)

    zk = constants.karman * zfl
    sql = zk * constants.rlam / (constants.rlam + zk)
    sql = sql * sql

    fh_denominator = 1.0 + rib * (10.0 + rib * (50.0 + 5000.0 * rib * rib))
    fh = 0.0012 + 1.0 / fh_denominator
    stable_eddv = kzmin[..., :-1] + jnp.sqrt(ws2) * fh * sql
    unstable_eddv = kzmin[..., :-1] + jnp.sqrt(ws2 * (1.0 - 25.0 * jnp.minimum(rib, 0.0))) * sql
    eddv = jnp.where(rib >= 0.0, stable_eddv, unstable_eddv)

    # Below the PBL the similarity estimate wins if it is larger.
    eddv = jnp.where(below_pbl & (edyz > eddv), edyz, eddv)
    eddv = jnp.minimum(eddv, constants.eddy_max)

    # The top layer has no interface above it.
    return jnp.concatenate([eddv, jnp.zeros_like(eddv[..., :1])], axis=-1)


class ColumnGeometry(NamedTuple):
    """Layer thicknesses and interface spacings for one column.

    ``dzh[0] = zf[0]`` — the first layer is measured from the ground, not from
    a face below it (``vdiffacmx.F:445``).
    """

    dzh: Array  # layer thickness, m
    dzhi: Array  # 1/dzh
    dzfi: Array  # 1/(zh[L+1] - zh[L]); the top entry repeats the one below


def column_geometry(zf: Array, zh: Array) -> ColumnGeometry:
    """Ports ``vdiffacmx.F:445-454``."""
    zf, zh = jnp.asarray(zf), jnp.asarray(zh)
    dzh = jnp.concatenate([zf[:1], zf[1:] - zf[:-1]])
    spacing = zh[1:] - zh[:-1]
    # The top layer has no interface above it, so CMAQ repeats the one below.
    dzfi = jnp.concatenate([1.0 / spacing, (1.0 / spacing)[-1:]])
    return ColumnGeometry(dzh=dzh, dzhi=1.0 / dzh, dzfi=dzfi)


class ACM2Column(NamedTuple):
    """The per-column ACM2 setup: mixing rates, the split diffusivity, the step.

    ``seddy`` is *not* the input diffusivity. Inside the convective boundary
    layer the convective stage takes a fraction ``fnl`` of it and carries that
    non-locally instead, leaving ``(1 - fnl)`` for the local stage
    (``vdiffacmx.F:493-499``). That reallocation is what makes ACM2 asymmetric,
    and it is the single easiest thing to miss when reading the routine.
    """

    seddy: Array  # (nlays,) diffusivity after the split
    mbarks: Array  # (nlays,) upward non-local rate
    mdwn: Array  # (nlays,) downward subsidence rate
    lcbl: Array  # top of the convective boundary layer, 1-based count
    nlp: Array  # sub-steps this column needs


def acm2_setup(
    seddy: Array,
    geometry: ColumnGeometry,
    *,
    pbl: Array,
    zf: Array,
    lpbl: Array,
    hol: Array,
    convective: Array,
    dtsec: float,
    constants: ACM2Constants = DEFAULT_ACM2,
) -> ACM2Column:
    """The ACM2 mixing rates and sub-step count for one column.

    Ports ``vdiffacmx.F:455-516``. ``hol`` is the PBL depth over the
    Monin–Obukhov length, negative when convective; ``lpbl`` is the layer index
    of the PBL top, 1-based as CMAQ counts.

    The sub-step limit takes the smaller of a diffusion constraint,
    ``0.75/(K·dzh⁻¹·dzf⁻¹)``, and — where convective — an ACM constraint
    ``0.75/(mbar·rz)``. Both are *accuracy* limits: Crank–Nicolson is
    unconditionally stable, so unlike horizontal diffusion there is no
    stability boundary to sit the wrong side of.
    """
    nlays = seddy.shape[0]
    layer = jnp.arange(nlays)

    # Diffusion limit, over interfaces only (the top layer has none).
    interior = layer < nlays - 1
    limit = jnp.where(
        interior,
        constants.substep_factor / (seddy * geometry.dzhi * geometry.dzfi),
        jnp.inf,
    )
    dtlim = jnp.minimum(dtsec, limit.min())

    # Non-local mixing rate. FNL is the fraction of the total mixing the plume
    # carries; the local stage keeps the rest.
    meddy = seddy[0] * geometry.dzfi[0] / (pbl - zf[0])
    fnl = 1.0 / (
        1.0
        + ((constants.karman / jnp.maximum(-hol, 1.0e-12)) ** 0.3333) / (0.72 * constants.karman)
    )
    mbar = meddy * fnl

    lcbl = jnp.where(convective, lpbl, 1)
    in_cbl = layer < lcbl

    mbarks = jnp.where(in_cbl, mbar, 0.0)
    mdwn = jnp.where(
        in_cbl & (layer >= 1),
        mbar * (pbl - jnp.concatenate([zf[:1], zf[:-1]])) * geometry.dzhi,
        0.0,
    )
    # The top CBL layer mixes up at the rate it subsides -- the closure that
    # keeps the column conservative (vdiffacmx.F:501).
    top = lcbl - 1
    mbarks = mbarks.at[top].set(jnp.where(convective, mdwn[top], mbarks[top]))

    split_seddy = jnp.where(in_cbl, (1.0 - fnl) * seddy, seddy)

    rz = (zf[top] - zf[0]) * geometry.dzhi[0]
    dtacm = 1.0 / (mbar * rz)
    dtlim = jnp.where(convective, jnp.minimum(constants.substep_factor * dtacm, dtlim), dtlim)

    seddy_out = jnp.where(convective, split_seddy, seddy)
    mbarks = jnp.where(convective, mbarks, 0.0)
    mdwn = jnp.where(convective, mdwn, 0.0)

    # NLP = int(DTSEC/DTLIM + 0.99): a ceiling, so the actual step is at or
    # below the limit rather than straddling it.
    nlp = jnp.floor(dtsec / dtlim + 0.99).astype(jnp.int32)
    return ACM2Column(seddy=seddy_out, mbarks=mbarks, mdwn=mdwn, lcbl=lcbl, nlp=jnp.maximum(nlp, 1))


class SurfaceExchange(NamedTuple):
    """Deposition and emission at the surface, per species.

    Inputs to this operator, not outputs of it — the modules that compute them
    (``depv/m3dry``, the DESID emission machinery) are a separate concern, in
    the same way meteorology is an input to advection.

    ``depv`` must be **strictly positive**. The surface layer relaxes toward the
    equilibrium ``pldv/depv`` (``vdiffacmx.F:625``), so a zero deposition
    velocity is 0/0 and turns the column into NaN. A run with negligible
    deposition uses a small value, not zero.
    """

    depv: Array  # (nspc,) deposition velocity, m/s
    pldv: Array  # (nspc,) surface emission flux
    emis: Array  # (nspc, nlays) layered emission, already scaled by the sub-step


def _shift_up(a: Array) -> Array:
    """``a`` shifted toward the surface, repeating the top entry."""
    return jnp.concatenate([a[1:], a[-1:]])


def _shift_down(a: Array) -> Array:
    """``a`` shifted away from the surface, repeating the bottom entry."""
    return jnp.concatenate([a[:1], a[:-1]])


class _Coefficients(NamedTuple):
    """Matrix entries for both stages. Assembled once per sync step, since none
    of them depends on the concentration."""

    aa: Array
    bb1: Array
    ee1: Array
    mfac: Array
    lfac1: Array
    lfac2: Array
    cc: Array
    bb2: Array
    ee2: Array
    lfac3: Array
    lfac4: Array


def _acm2_matrices(
    setup: ACM2Column,
    geometry: ColumnGeometry,
    *,
    dfacp: Array,
    dfacq: Array,
    dfsp: Array,
    dfsq: Array,
    eddy: Array,
    pbl: Array,
    zf: Array,
) -> _Coefficients:
    """Both stages' matrices. Ports ``vdiffacmx.F:656-679``.

    The ``lfac*`` terms belong to the explicit half of the Crank-Nicolson step
    and so appear on the right-hand side; the rest form the two matrices.
    """
    nlays = geometry.dzh.shape[0]
    layer = jnp.arange(nlays)
    in_cbl = layer < setup.lcbl
    delp = pbl - zf[0]

    # --- convective stage (vdiffacmx.F:656-667) --------------------------
    in_cbl = layer < setup.lcbl
    delp = pbl - zf[0]

    aa = jnp.where(in_cbl & (layer >= 1), -dfacp * setup.mbarks, 0.0)
    bb1 = jnp.where(
        layer == 0,
        1.0 + delp * dfsp[0] * setup.mbarks[0],
        jnp.where(in_cbl, 1.0 + dfacp * setup.mdwn, 1.0),
    )
    ee1 = jnp.where(
        in_cbl & (layer >= 1),
        -_shift_down(dfsp) * geometry.dzh * setup.mdwn,
        0.0,
    )
    mfac = jnp.where(
        in_cbl & (layer >= 1),
        _shift_up(geometry.dzh) * geometry.dzhi * _shift_up(setup.mdwn),
        0.0,
    )
    lfac1 = dfsq[0] * delp * setup.mbarks[0]
    lfac2 = dfsq[0] * _shift_up(setup.mdwn)[0] * geometry.dzh[1]

    # --- local stage (vdiffacmx.F:669-679) -------------------------------
    ee2 = -dfsp * eddy
    lfac3 = dfsq * eddy
    cc = jnp.where(layer >= 1, -dfsp * _shift_down(eddy), 0.0)
    bb2 = jnp.where(layer == 0, 1.0 - ee2, 1.0 - cc - ee2)
    lfac4 = jnp.where(layer >= 1, dfsq * _shift_down(eddy), 0.0)
    # The top layer has no interface above it, so no upward flux term.
    ee2 = jnp.where(layer < nlays - 1, ee2, 0.0)
    lfac3 = jnp.where(layer < nlays - 1, lfac3, 0.0)

    return _Coefficients(
        aa=aa,
        bb1=bb1,
        ee1=ee1,
        mfac=mfac,
        lfac1=lfac1,
        lfac2=lfac2,
        cc=cc,
        bb2=bb2,
        ee2=ee2,
        lfac3=lfac3,
        lfac4=lfac4,
    )


def acm2_column_step(
    conc: Array,
    setup: ACM2Column,
    geometry: ColumnGeometry,
    surface: SurfaceExchange,
    *,
    pbl: Array,
    zf: Array,
    dens1: Array,
    rdepvht: Array,
    dtsec: float,
    max_substeps: int,
    constants: ACM2Constants = DEFAULT_ACM2,
) -> tuple[Array, Array]:
    """One sync step of ACM2 for a single column. Ports ``vdiffacmx.F:640-1130``.

    ``conc`` is ``(nspc, nlays)``. Returns the diffused column and the
    accumulated dry deposition, ``(nspc,)``.

    Each sub-step is four moves:

    1. the surface layer relaxes toward ``pldv/depv`` over the implicit half
       step, and the deposited mass is accumulated;
    2. **the convective stage**, where the column is convective — the non-local
       plume, solved with :func:`solve_acm1`;
    3. **the local stage**, always — Crank–Nicolson diffusion plus the layered
       emission source, solved with :func:`solve_tridiagonal`;
    4. the surface layer relaxes again over the explicit half step.

    ``max_substeps`` is a static bound. Columns needing fewer stop early by
    masking, so each column takes exactly its own ``nlp`` steps of its own
    ``dtsec/nlp`` — the ragged loop CMAQ writes, made rectangular without
    changing the arithmetic.
    """
    nspc, nlays = conc.shape
    layer = jnp.arange(nlays)

    dts = dtsec / setup.nlp
    dfacp = constants.theta * dts
    dfacq = constants.theta_bar * dts

    dfsp = dfacp * geometry.dzhi
    dfsq = dfacq * geometry.dzhi
    eddy = setup.seddy * geometry.dzfi

    # --- surface exchange coefficients (vdiffacmx.F:618-626) ---------------
    rp = dfacp * rdepvht
    rq = dfacq * rdepvht
    efac1 = jnp.exp(-surface.depv * rp)
    efac2 = jnp.exp(-surface.depv * rq)
    pol = surface.pldv / surface.depv
    dd_fac = dts * dens1 * surface.depv
    evasion = dts * dens1 * surface.pldv

    coeff = _acm2_matrices(
        setup,
        geometry,
        dfacp=dfacp,
        dfacq=dfacq,
        dfsp=dfsp,
        dfsq=dfsq,
        eddy=eddy,
        pbl=pbl,
        zf=zf,
    )
    aa, bb1, ee1, mfac, lfac1, lfac2, cc, bb2, ee2, lfac3, lfac4 = coeff
    in_cbl = layer < setup.lcbl

    convective = setup.lcbl > 1

    def substep(carry: tuple[Array, Array], step: Array) -> tuple[tuple[Array, Array], None]:
        current, deposited = carry
        active = step < setup.nlp

        # 1. surface relaxation, implicit half.
        surface_new = pol + (current[:, 0] - pol) * efac1
        updated = current.at[:, 0].set(jnp.where(active, surface_new, current[:, 0]))
        deposited = deposited + jnp.where(
            active, constants.theta * (dd_fac * updated[:, 0] - evasion), 0.0
        )

        # 2. convective stage.
        upper = jnp.concatenate([updated[:, 1:], updated[:, -1:]], axis=1)
        rhs_conv = jnp.where(
            layer == 0,
            updated - lfac1 * updated + lfac2 * upper,
            updated + dfacq * (setup.mbarks * updated[:, :1] - setup.mdwn * updated + mfac * upper),
        )
        solved = solve_acm1(aa, bb1, ee1, rhs_conv, setup.lcbl)
        after_convective = jnp.where(in_cbl, solved, updated)
        updated = jnp.where(convective & active, after_convective, updated)

        # 3. local stage.
        lower = jnp.concatenate([updated[:, :1], updated[:, :-1]], axis=1)
        upper = jnp.concatenate([updated[:, 1:], updated[:, -1:]], axis=1)
        rhs_local = (
            updated + lfac3 * (upper - updated) - lfac4 * (updated - lower) + surface.emis * dts
        )
        diffused = solve_tridiagonal(cc, bb2, ee2, rhs_local)
        updated = jnp.where(active, diffused, updated)

        # 4. surface relaxation, explicit half.
        surface_new = pol + (updated[:, 0] - pol) * efac2
        updated = updated.at[:, 0].set(jnp.where(active, surface_new, updated[:, 0]))
        # The evasion term appears in *both* halves for a plain species
        # (vdiffacmx.F:696 and 1104-1106). Only the heterogeneous-HONO branches
        # omit it from the second, and copying their form here costs exactly
        # THBAR * DTS * DENS1 * PLDV -- invisible unless emissions are on.
        deposited = deposited + jnp.where(
            active, constants.theta_bar * (dd_fac * updated[:, 0] - evasion), 0.0
        )

        return (updated, deposited), None

    (final, ddep), _ = jax.lax.scan(
        substep, (conc, jnp.zeros((nspc,), dtype=conc.dtype)), jnp.arange(max_substeps)
    )
    return final, ddep


class ColumnState(NamedTuple):
    """Everything one column of :func:`vdiff_step` needs, per column.

    Arrays are ``(ncols, nrows, ...)`` with the layer axis last, matching the
    rest of the package. CMAQ passes ``SEDDY`` layer-first
    (``vdiffproc.F:160``); the transpose belongs at the boundary, not in the
    kernel.
    """

    seddy: Array  # (ncols, nrows, nlays)
    zf: Array
    zh: Array
    pbl: Array  # (ncols, nrows)
    lpbl: Array
    hol: Array
    dens1: Array
    rdepvht: Array
    convective: Array


def vdiff_step(
    conc: Array,
    state: ColumnState,
    surface: SurfaceExchange,
    *,
    dtsec: float,
    max_substeps: int,
    constants: ACM2Constants = DEFAULT_ACM2,
) -> tuple[Array, Array]:
    """ACM2 vertical diffusion over a domain. Ports ``vdiffacmx.F``.

    ``conc`` is ``(ncols, nrows, nlays, nspc)``, matching the rest of the
    package; each column is transposed internally to ``(nspc, nlays)`` because
    that is the axis order both solvers recurse along. ``surface`` carries
    ``(ncols, nrows, nspc)`` fields and ``(ncols, nrows, nlays, nspc)``
    emissions. Returns the diffused concentrations and the accumulated dry
    deposition, ``(ncols, nrows, nspc)``.

    **Not part of the coupled transport block.** ``sciproc.F`` runs vertical
    diffusion *first*, on uncoupled concentrations, before ``COUPLE``. It is
    applied outside :func:`cmaq_jax.api.transport_step`, not appended to it.

    ``max_substeps`` bounds the sub-step loop. Every column runs the same number
    of scan iterations and masks those past its own ``nlp``, so the arithmetic
    matches CMAQ's ragged loop exactly while the shape stays rectangular. Take
    the bound from :func:`substep_counts`; a column needing more is silently
    under-stepped, which is why that helper exists rather than leaving the
    caller to guess.
    """
    conc = jnp.asarray(conc)

    def one_column(column: Array, col: ColumnState, surf: SurfaceExchange) -> tuple[Array, Array]:
        geometry = column_geometry(col.zf, col.zh)
        setup = acm2_setup(
            col.seddy,
            geometry,
            pbl=col.pbl,
            zf=col.zf,
            lpbl=col.lpbl,
            hol=col.hol,
            convective=col.convective,
            dtsec=dtsec,
            constants=constants,
        )
        return acm2_column_step(
            column.T,
            setup,
            geometry,
            SurfaceExchange(depv=surf.depv, pldv=surf.pldv, emis=surf.emis.T),
            pbl=col.pbl,
            zf=col.zf,
            dens1=col.dens1,
            rdepvht=col.rdepvht,
            dtsec=dtsec,
            max_substeps=max_substeps,
            constants=constants,
        )

    over_domain = jax.vmap(jax.vmap(one_column, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
    diffused, ddep = over_domain(conc, state, surface)
    # Back to (ncols, nrows, nlays, nspc).
    return jnp.swapaxes(diffused, -1, -2), ddep


def substep_counts(
    state: ColumnState, dtsec: float, *, constants: ACM2Constants = DEFAULT_ACM2
) -> Array:
    """``NLP`` for every column — the quantity :func:`vdiff_step` needs bounded.

    Kept separate because the bound has to be a Python integer for the scan
    length: take ``int(substep_counts(...).max())``. Hiding that reduction
    inside a jitted step would force a host sync on every call.
    """

    def one(col: ColumnState) -> Array:
        setup = acm2_setup(
            col.seddy,
            column_geometry(col.zf, col.zh),
            pbl=col.pbl,
            zf=col.zf,
            lpbl=col.lpbl,
            hol=col.hol,
            convective=col.convective,
            dtsec=dtsec,
            constants=constants,
        )
        return setup.nlp

    return jax.vmap(jax.vmap(one))(state)

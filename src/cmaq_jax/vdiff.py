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
    "VerticalMeteorology",
    "eddy_diffusivity",
    "solve_acm1",
    "solve_tridiagonal",
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

"""Horizontal diffusion.

Ports ``hdiff/multiscale/`` — `hdiff.F`, `hcdiff3d.F` and `deform.F`. CMAQ's
science driver runs this immediately after vertical advection
(``sciproc.F:250-278``).

The scheme is an **explicit five-point Laplacian in mixing-ratio space**,
sub-cycled for stability, with an eddy diffusivity that grows with the local
wind deformation. There is no limiter, no reconstruction and no iteration, which
makes it far simpler than the PPM advection in :mod:`cmaq_jax.ppm`.

The one structural point worth stating up front, because intuition from
advection is actively misleading here: **rho*J is not carried as an extra
species slot**. Advection transports it alongside everything else, and that is
its mass-conservation mechanism. Diffusion instead acts on the mixing ratio
``q = c / rhoJ`` and puts ``rhoJ`` into the flux coefficient, so ``rhoJ`` is an
input that comes back unchanged. Treating it as a species here would diffuse the
density field, which nothing in CMAQ does.

Three boundary conventions are in play, on three *different* edges, and they are
easy to conflate:

* ``deform.F:337-343`` zeroes the deformation over its full ``(ncols+1, nrows+1)``
  extent. Where deformation is zero the diffusivity is **not** zero — it sits on
  the ``KHMIN`` floor.
* ``deform.F:420-421`` zeroes the cross-gradients ``du/dy`` at the first and last
  *row* and ``dv/dx`` at the first and last *column*.
* ``hcdiff3d.F:216,226`` zeroes ``K11BAR`` on the last row and ``K22BAR`` on the
  last column, so no flux crosses the domain edge.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from cmaq_jax.config import DEFAULT_HDIFF, GridConfig, HDiffConstants

__all__ = [
    "contravariant_winds",
    "deformation",
    "diffusion_coefficients",
    "eddy_diffusivity",
    "face_coefficients",
    "halo_density",
    "hdiff_step",
    "stable_timestep",
    "substep_count",
]


def halo_density(rhoj: Array, ring: Array) -> Array:
    """``DENSJ`` — rho*J extended by one cell all round. Ports ``deform.F:254-292``.

    ``rhoj`` is ``(ncols, nrows, nlays)``; ``ring`` is the perimeter,
    ``(nbndy, nlays)`` with ``nbndy = 2*(ncols + nrows + 2)``, in CMAQ's
    South/East/North/West order. Returns ``(ncols+2, nrows+2, nlays)``, indexed
    so that ``[i, j]`` is Fortran's ``DENSJ(i, j)`` over ``0:ncols+1``.

    This exists because the halo density is a real input, not a convenience: it
    divides the contravariant wind at the domain-edge faces, and substituting a
    zero-gradient extrapolation there changes the answer wherever the density
    varies along the boundary.
    """
    rhoj = jnp.asarray(rhoj)
    ring = jnp.asarray(ring)
    ncols, nrows, nlays = rhoj.shape
    expected = 2 * (ncols + nrows + 2)
    if ring.shape != (expected, nlays):
        raise ValueError(
            f"ring must be ({expected}, {nlays}) for a {ncols}x{nrows} domain, got {ring.shape}"
        )

    out = jnp.zeros((ncols + 2, nrows + 2, nlays), dtype=rhoj.dtype)
    out = out.at[1 : ncols + 1, 1 : nrows + 1].set(rhoj)

    take = 0

    def segment(length: int) -> Array:
        nonlocal take
        piece = ring[take : take + length]
        take += length
        return piece.astype(rhoj.dtype)

    out = out.at[1 : ncols + 2, 0].set(segment(ncols + 1))  # South
    out = out.at[ncols + 1, 1 : nrows + 2].set(segment(nrows + 1))  # East
    out = out.at[0 : ncols + 1, nrows + 1].set(segment(ncols + 1))  # North
    out = out.at[0, 0 : nrows + 1].set(segment(nrows + 1))  # West
    return out


def contravariant_winds(uhat_jd: Array, vhat_jd: Array, densj: Array) -> tuple[Array, Array]:
    """``LOC_UWIND``/``LOC_VWIND`` — winds recovered from the flux form.

    Ports ``deform.F:310-332``. ``uhat_jd``/``vhat_jd`` are ``UHAT_JD``/``VHAT_JD``
    on the dot grid, ``(ncols+1, nrows+1, nlays)``; ``densj`` is the haloed
    density from :func:`halo_density`. Each face divides by the mean of the two
    cells it separates.

    The unfilled parts are returned as zeros, matching the Fortran's
    ``LOC_UWIND = 0.0`` before the loop: ``u`` is filled on rows ``1..nrows``
    only, ``v`` on columns ``1..ncols`` only. Those zeros are never read by
    :func:`deformation` — its stencils stop one short — but they are part of the
    array's definition, so they are reproduced rather than left to chance.
    """
    uhat_jd = jnp.asarray(uhat_jd)
    vhat_jd = jnp.asarray(vhat_jd)
    densj = jnp.asarray(densj)
    ncols_p1, nrows_p1, _ = uhat_jd.shape
    ncols, nrows = ncols_p1 - 1, nrows_p1 - 1

    # u face (c, r) separates DENSJ(c, r+1) and DENSJ(c+1, r+1).
    dj_u = 0.5 * (densj[1 : ncols + 2, 1 : nrows + 1] + densj[0 : ncols + 1, 1 : nrows + 1])
    u = jnp.zeros_like(uhat_jd).at[:, :nrows].set(uhat_jd[:, :nrows] / dj_u)

    # v face (c, r) separates DENSJ(c+1, r) and DENSJ(c+1, r+1).
    dj_v = 0.5 * (densj[1 : ncols + 1, 1 : nrows + 2] + densj[1 : ncols + 1, 0 : nrows + 1])
    v = jnp.zeros_like(vhat_jd).at[:ncols].set(vhat_jd[:ncols] / dj_v)

    return u, v


def _cross_gradient_rows(field: Array, spacing: float) -> Array:
    """``du/dy`` by the gradient of column-pair averages (``deform.F:396-405``).

    ``UBAR1 - UBAR2`` over ``4*dy``, where each ``UBAR`` sums the value at this
    column and the next. That is a gradient *of averages*, not an average of
    gradients — a distinction that changes the stencil and so the answer.

    Zero on the first and last row, matching ``deform.F:420``: the stencil
    reaches one row either side and there is nothing outside the domain to
    reach.
    """
    # field is (ncols+1, nrows+1, nlays) on dot points; pairs adjacent columns.
    pair = field[:-1] + field[1:]  # (ncols, nrows+1, nlays)
    interior = (pair[:, 2:] - pair[:, :-2]) / (4.0 * spacing)  # (ncols, nrows-1, nlays)
    ncols, _, nlays = pair.shape
    edge = jnp.zeros((ncols, 1, nlays), dtype=interior.dtype)
    # Rows 1 and nrows carry no cross-gradient; interior covers rows 2..nrows-1.
    return jnp.concatenate([edge, interior[:, :-1], edge], axis=1)


def _cross_gradient_cols(field: Array, spacing: float) -> Array:
    """``dv/dx`` by the gradient of row-pair averages (``deform.F:408-418``).

    The mirror of :func:`_cross_gradient_rows`, zero on the first and last
    *column* (``deform.F:421``).
    """
    pair = field[:, :-1] + field[:, 1:]  # (ncols+1, nrows, nlays)
    interior = (pair[2:] - pair[:-2]) / (4.0 * spacing)  # (ncols-1, nrows, nlays)
    _, nrows, nlays = pair.shape
    edge = jnp.zeros((1, nrows, nlays), dtype=interior.dtype)
    return jnp.concatenate([edge, interior[:-1], edge], axis=0)


def deformation(u: Array, v: Array, *, dx1: float, dx2: float) -> Array:
    """Total wind deformation at cell centres. Ports ``deform.F:352-432``.

    ``u`` and ``v`` are contravariant velocities on the dot grid, shaped
    ``(ncols+1, nrows+1, nlays)`` — that is, ``UHAT_JD``/``VHAT_JD`` already
    divided by the face-interpolated density (see
    :func:`cmaq_jax.velocity.face_velocity_from_flux`).

    Returns ``(ncols+1, nrows+1, nlays)``, with values only on
    ``(1:ncols, 1:nrows)`` and zeros elsewhere — the padded row and column are
    zero by design (``deform.F:337-343``), not by omission.

    The quantity is the second invariant of the strain-rate tensor::

        DF1 = du/dx - dv/dy       stretching
        DF2 = dv/dx + du/dy       shearing
        deformation = sqrt(DF1^2 + DF2^2)

    so it is invariant under a rotation of the frame — exactly so, and a useful
    test, but only away from the edges where the cross-gradients are zeroed.
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")
    if u.ndim != 3:
        raise ValueError(f"expected (ncols+1, nrows+1, nlays) winds, got {u.shape}")

    ncols_p1, nrows_p1, nlays = u.shape
    ncols, nrows = ncols_p1 - 1, nrows_p1 - 1

    # Divergence terms: plain one-sided differences across the cell
    # (deform.F:385-386). Note the sign -- the Fortran differences the *next*
    # face minus this one, over a cell width.
    dudx = (u[1:, :-1] - u[:-1, :-1]) / dx1  # (ncols, nrows, nlays)
    dvdy = (v[:-1, 1:] - v[:-1, :-1]) / dx2  # (ncols, nrows, nlays)

    dudy = _cross_gradient_rows(u, dx2)  # (ncols, nrows, nlays)
    dvdx = _cross_gradient_cols(v, dx1)  # (ncols, nrows, nlays)

    df1 = dudx - dvdy
    df2 = dvdx + dudy
    interior = jnp.sqrt(df1 * df1 + df2 * df2)

    # Pad back to dot-point extent. Zero, deliberately: deform.F:337-343.
    out = jnp.zeros((ncols_p1, nrows_p1, nlays), dtype=interior.dtype)
    return out.at[:ncols, :nrows].set(interior)


def eddy_diffusivity(
    deform: Array,
    msfd2: Array,
    *,
    dx1: float,
    dx2: float,
    constants: HDiffConstants = DEFAULT_HDIFF,
) -> Array:
    """Contravariant horizontal eddy diffusivity. Ports ``hcdiff3d.F:180-200``.

    ``deform`` is the field from :func:`deformation`; ``msfd2`` is the squared
    map scale factor at dot points, ``(ncols+1, nrows+1)``.

    ::

        KHD   = max(KHMIN, ACOEF * deformation)
        EDDYH = MSFD2 * KHA * KHD / (KHA + KHD)

    The second line is a parallel-resistor blend. It rises with deformation but
    saturates at ``KHA``, so a strongly sheared cell cannot run away — and note
    that it is *not* zero where deformation is zero, but ``KHA*KHMIN/(KHA+KHMIN)``.
    Writing the blend upside-down gives a plausible-looking field on mild winds
    and the wrong asymptote on sharp ones.
    """
    deform = jnp.asarray(deform)
    msfd2 = jnp.asarray(msfd2)
    kha = constants.base_diffusivity(dx1, dx2)
    acoef = constants.deformation_coefficient(dx1, dx2)

    khd = jnp.maximum(constants.khmin, acoef * deform)
    return msfd2[..., None] * kha * khd / (kha + khd)


def face_coefficients(eddyh: Array) -> tuple[Array, Array]:
    """Flux-point diffusivities ``(K11BAR, K22BAR)``. Ports ``hcdiff3d.F:210-230``.

    Each averages *across* its own direction — ``K11`` lives on x faces and
    averages over rows, ``K22`` on y faces averaging over columns::

        K11BAR(c, r) = 0.5 * (EDDYH(c, r+1) + EDDYH(c, r))
        K22BAR(c, r) = 0.5 * (EDDYH(c, r)   + EDDYH(c+1, r))

    Both are returned at ``(ncols+1, nrows+1, nlays)``, with ``K11``'s last row
    and ``K22``'s last column zeroed (``hcdiff3d.F:216,226``) so that no flux
    crosses the domain edge. Those are different edges from the ones
    :func:`deformation` zeroes, and conflating them is the easy mistake.
    """
    eddyh = jnp.asarray(eddyh)

    k11 = jnp.zeros_like(eddyh)
    k11 = k11.at[:, :-1].set(0.5 * (eddyh[:, 1:] + eddyh[:, :-1]))

    k22 = jnp.zeros_like(eddyh)
    k22 = k22.at[:-1].set(0.5 * (eddyh[:-1] + eddyh[1:]))

    return k11, k22


def stable_timestep(
    k11: Array, k22: Array, *, dx1: float, dx2: float, constants: HDiffConstants = DEFAULT_HDIFF
) -> Array:
    """Largest stable diffusion step, seconds. Ports ``hcdiff3d.F:253``.

    ``CFC * dx1 * dx2 / max(K)``, the maximum taken over the interior only
    (``hcdiff3d.F:238-244`` reduces over ``1..NCOLS, 1..NROWS``) and, in CMAQ,
    over MPI ranks as well — every rank has to agree on the step.
    """
    ncols_p1, nrows_p1, _ = k11.shape
    interior = slice(0, ncols_p1 - 1), slice(0, nrows_p1 - 1)
    effkb = jnp.maximum(k11[interior].max(), k22[interior].max())
    return constants.cfc * dx1 * dx2 / effkb


def substep_count(sync_seconds: float, dt_stable: float) -> int:
    """Diffusion sub-steps in one sync step. Ports ``hdiff.F:336-338``.

    ``int(DTSEC/DT) + 1``. The ``+ 1`` means at least one step always runs, and
    that the actual step is *shorter* than the stable one rather than equal to
    it — CMAQ then divides the sync step evenly, ``DT = DTSEC / NSTEPS``.
    """
    return int(sync_seconds / dt_stable) + 1


def diffusion_coefficients(
    u: Array,
    v: Array,
    msfd2: Array,
    *,
    cfg: GridConfig,
    constants: HDiffConstants = DEFAULT_HDIFF,
) -> tuple[Array, Array, Array]:
    """``(K11BAR, K22BAR, dt_stable)`` from the winds — the whole ``hcdiff3d`` chain.

    Convenience over :func:`deformation`, :func:`eddy_diffusivity`,
    :func:`face_coefficients` and :func:`stable_timestep`, which are exposed
    separately because each is pinned to its own golden.
    """
    deform = deformation(u, v, dx1=cfg.dx1, dx2=cfg.dx2)
    eddyh = eddy_diffusivity(deform, msfd2, dx1=cfg.dx1, dx2=cfg.dx2, constants=constants)
    k11, k22 = face_coefficients(eddyh)
    dt = stable_timestep(k11, k22, dx1=cfg.dx1, dx2=cfg.dx2, constants=constants)
    return k11, k22, dt


def _face_densities(densj: Array) -> tuple[Array, Array]:
    """``0.5*(rhoJ(c) + rhoJ(c-1))`` on x faces, and the row analogue on y.

    Ports ``hdiff.F:317-324``. ``densj`` is the haloed density from
    :func:`halo_density`; both results are ``(ncols+1, nrows+1, nlays)``,
    matching ``RK11``/``RK22`` before they are multiplied by the diffusivities.
    """
    ncols_p2, nrows_p2, _ = densj.shape
    ncols, nrows = ncols_p2 - 2, nrows_p2 - 2
    rk11 = 0.5 * (densj[1 : ncols + 2, 1 : nrows + 2] + densj[0 : ncols + 1, 1 : nrows + 2])
    rk22 = 0.5 * (densj[1 : ncols + 2, 1 : nrows + 2] + densj[1 : ncols + 2, 0 : nrows + 1])
    return rk11, rk22


def _seed_halo(q: Array) -> Array:
    """Extend the mixing ratio by one cell, copying the edge value outward.

    Ports ``hdiff.F:355-400``. Each halo cell takes its neighbouring edge
    cell's value, so the gradient across the domain boundary is zero and no
    flux crosses it — the Dirichlet/no-flux condition named at ``hdiff.F:25``.

    ``q`` is ``(ncols, nrows, nlays, nspc)``; the result is
    ``(ncols+2, nrows+2, nlays, nspc)``.

    Corners are never read: the five-point stencil only ever reaches a halo
    cell directly north, south, east or west of an interior cell. They are
    filled anyway, from the adjacent edge, rather than left undefined.
    """
    padded = jnp.pad(q, ((1, 1), (1, 1), (0, 0), (0, 0)), mode="edge")
    return padded


def hdiff_step(
    state: Array,
    densj: Array,
    k11: Array,
    k22: Array,
    *,
    cfg: GridConfig,
    sync_seconds: float,
    nsteps: int,
) -> Array:
    """One sync step of horizontal diffusion. Ports ``hdiff.F:455-530``.

    ``state`` is ``(ncols, nrows, nlays, nspc)`` in coupled units, with rho*J in
    the last slot; ``densj`` is the haloed density from :func:`halo_density`;
    ``k11``/``k22`` are the face diffusivities from :func:`face_coefficients`.
    ``nsteps`` is the sub-step count from :func:`substep_count` — host-side, so
    the loop has a static trip count, exactly as ``astep_seconds`` is for
    advection.

    **The rho*J slot is not diffused.** ``DIFF_MAP`` (``hdiff.F:276-292``) covers
    the transported species only, with no ``+ 1`` for density — unlike
    ``ADV_MAP``, which does include it. Density is a coefficient here, not a
    tracer, and diffusing it would smooth the meteorology.

    **The halo is frozen across sub-steps**, reproducing ``hdiff.F``: the halo
    arrays are filled once, before the ``DO 344`` loop, from the *initial*
    mixing ratio, while the interior is reloaded from ``CGRID`` every sub-step.
    The zero-gradient boundary is therefore exact only on the first sub-step and
    drifts slightly afterwards. That is a behavioural detail a tidier rewrite
    would silently change, so it is kept and pinned by a test.
    """
    state = jnp.asarray(state, dtype=cfg.numpy_dtype)
    densj = jnp.asarray(densj, dtype=cfg.numpy_dtype)
    k11 = jnp.asarray(k11, dtype=cfg.numpy_dtype)
    k22 = jnp.asarray(k22, dtype=cfg.numpy_dtype)
    if nsteps < 1:
        raise ValueError(f"nsteps must be >= 1, got {nsteps}")

    ncols, nrows = state.shape[0], state.shape[1]
    rhoj = densj[1 : ncols + 1, 1 : nrows + 1]  # interior, (ncols, nrows, nlays)

    rk11, rk22 = _face_densities(densj)
    rk11 = (rk11 * k11)[..., None]  # (ncols+1, nrows+1, nlays, 1)
    rk22 = (rk22 * k22)[..., None]

    dt = sync_seconds / nsteps
    dtdx1s = dt / (cfg.dx1 * cfg.dx1)
    dtdx2s = dt / (cfg.dx2 * cfg.dx2)

    species, density = state[..., :-1], state[..., -1:]
    rhoj_s = rhoj[..., None]

    # Filled once, from the initial mixing ratio, and held fixed -- see above.
    halo = _seed_halo(species / rhoj_s)

    def substep(_: int, current: Array) -> Array:
        q = current / rhoj_s
        # Interior refreshed each pass; the halo ring stays as first seeded.
        padded = halo.at[1 : ncols + 1, 1 : nrows + 1].set(q)

        east = padded[2 : ncols + 2, 1 : nrows + 1] - q
        west = q - padded[0:ncols, 1 : nrows + 1]
        north = padded[1 : ncols + 1, 2 : nrows + 2] - q
        south = q - padded[1 : ncols + 1, 0:nrows]

        return (
            current
            + dtdx1s * (rk11[1 : ncols + 1, 0:nrows] * east - rk11[0:ncols, 0:nrows] * west)
            + dtdx2s * (rk22[0:ncols, 1 : nrows + 1] * north - rk22[0:ncols, 0:nrows] * south)
        )

    diffused = jax.lax.fori_loop(0, nsteps, substep, species)
    return jnp.concatenate([diffused, density], axis=-1)

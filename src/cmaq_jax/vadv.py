"""Vertical advection: the PPM column solve and its flux-matching adjustment.

Ports ``vppm.F``. The vertical differs from the horizontal in two ways that
matter:

* **layer thickness varies**, so the reconstruction uses the non-uniform form
  (:func:`cmaq_jax.ppm.ppm_parabola_nonuniform`);
* **the face velocity is corrected before use.** ``zadvppmwrf.F`` diagnoses a
  face mass flux from the rho*J budget, but the PPM flux computed from that
  velocity does not generally equal it. ``vppm.F:200-246`` rescales each face
  velocity until the two agree, and only then advects the species.

That second step is what keeps advected density tracking the meteorology. It is
also the one place in the operator that is genuinely iterative.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from cmaq_jax.config import DEFAULT_PPM, PPMConstants
from cmaq_jax.ppm import NonUniformMesh, Parabola, ppm_parabola_nonuniform

__all__ = ["AdjustedVelocity", "vppm", "vppm_adjust_velocity"]


class AdjustedVelocity(NamedTuple):
    """Face velocities rescaled to match the diagnosed mass flux.

    ``residual`` is the relative flux mismatch left after the fixed iteration
    count, per face. CMAQ loops to convergence and calls ``M3EXIT`` on failure;
    a fixed count is what makes this jittable, so the failure has to be
    reported rather than raised. Anything above
    ``PPMConstants.velocity_flux_tolerance`` did not converge.
    """

    vel: Array
    residual: Array


def _face_flux(
    vel: Array,
    parabola: Parabola,
    ds: Array,
    *,
    dt: float | Array,
    ppm: PPMConstants,
) -> Array:
    """PPM mass flux through each face, given the face velocities.

    Ports the flux expressions at ``vppm.F:205-208`` and ``vppm.F:229-231``.

    A face draws on the cell upwind of it, and which cell that is depends on the
    sign of the velocity: a downward face (``vel < 0``) takes the left edge of
    the cell *above* it, an upward face the right edge of the cell *below*. The
    two branches are mutually exclusive, so every face is handled exactly once
    and the whole thing vectorises -- there is no sequential dependence along
    the column, despite the Fortran being written as a loop that mutates ``VEL``
    in place.

    ``parabola`` covers the ``ni`` cells; ``vel`` and the result cover the
    ``ni + 1`` faces.
    """
    edge = jnp.zeros((1, *vel.shape[1:]), dtype=vel.dtype)

    # Downward: face k uses cell k (the cell above the face).
    y_down = -vel[:-1] * dt
    x_down = y_down / ds
    flux_down = y_down * (
        parabola.cl + 0.5 * x_down * (parabola.dc + parabola.c6 * (1.0 - ppm.two_thirds * x_down))
    )

    # Upward: face k uses cell k-1 (the cell below the face).
    y_up = vel[1:] * dt
    x_up = y_up / ds
    flux_up = y_up * (
        parabola.cr - 0.5 * x_up * (parabola.dc - parabola.c6 * (1.0 - ppm.two_thirds * x_up))
    )

    return jnp.where(
        vel < 0.0,
        jnp.concatenate([flux_down, edge]),
        jnp.where(vel > 0.0, jnp.concatenate([edge, flux_up]), 0.0),
    )


def _target_flux(flx: Array, vel: Array, dt: float | Array) -> Array:
    """The mass flux the velocity is being matched to.

    ``vppm.F:202`` uses ``FDN = -FLX(I)*DT`` on a downward face and
    ``vppm.F:225`` ``FUP = FLX(I+1)*DT`` on an upward one, so the sign follows
    the face direction and the target is positive in both cases.
    """
    return jnp.where(vel < 0.0, -flx * dt, flx * dt)


def _adjustable(vel: Array) -> Array:
    """Which faces CMAQ actually adjusts.

    ``vppm.F:199-246`` loops ``DO I = 1, NI``, handling the downward branch for
    face ``I`` and the upward branch for face ``I+1``. So a face is only
    adjusted if the cell it draws from exists:

    * the **bottom** face has no cell below it, so it is adjusted only when the
      flow is downward (there, it draws on cell 1);
    * the **top** face has no cell above it, so it is adjusted only when the
      flow is upward.

    The excluded cases are not failures. A downward-flowing top face is simply
    left at its diagnosed velocity, and reporting a residual for it would flag a
    convergence problem that does not exist -- which is exactly what an earlier
    version of this did on ``vppm_downward_flux``.
    """
    interior = jnp.ones_like(vel, dtype=bool).at[0].set(False).at[-1].set(False)
    return jnp.where(
        vel < 0.0,
        interior | (jnp.arange(vel.shape[0]).reshape((-1,) + (1,) * (vel.ndim - 1)) == 0),
        jnp.where(
            vel > 0.0,
            interior
            | (jnp.arange(vel.shape[0]).reshape((-1,) + (1,) * (vel.ndim - 1)) == vel.shape[0] - 1),
            False,
        ),
    )


def vppm_adjust_velocity(
    vel: Array,
    flx: Array,
    parabola: Parabola,
    ds: Array,
    *,
    dt: float | Array,
    ppm: PPMConstants = DEFAULT_PPM,
) -> AdjustedVelocity:
    """Rescale face velocities so the PPM flux matches the diagnosed flux.

    Ports ``vppm.F:200-246``. The update is ``vel <- vel * sqrt(target / flux)``,
    which converges quadratically because the flux is close to quadratic in the
    velocity -- the comment at ``vppm.F:103-108`` records that the square root
    replaced a linear correction precisely because a linear one misbehaved at
    Courant numbers near one.

    CMAQ iterates to a 1e-3 relative tolerance with a cap of 50. Here the count
    is fixed (``PPMConstants.velocity_adjust_iterations``) so the loop jits, and
    the leftover mismatch is returned instead of raising.

    Faces where the flux and the target disagree in sign are left alone. Their
    ratio is negative, and the Fortran would take the square root of it and
    produce a NaN; reporting a large residual is more useful than poisoning the
    column.
    """
    vel = jnp.asarray(vel)
    target = _target_flux(jnp.asarray(flx), vel, dt)
    active = _adjustable(vel)

    def refine(_: int, current: Array) -> Array:
        flux = _face_flux(current, parabola, ds, dt=dt, ppm=ppm)
        ratio = jnp.where(flux != 0.0, target / flux, 1.0)
        converged = jnp.abs(flux - target) <= ppm.velocity_flux_tolerance * jnp.abs(target)
        step = jnp.where(ratio > 0.0, jnp.sqrt(jnp.abs(ratio)), 1.0)
        return jnp.where(active & ~converged, current * step, current)

    adjusted = jax.lax.fori_loop(0, ppm.velocity_adjust_iterations, refine, vel)

    final = _face_flux(adjusted, parabola, ds, dt=dt, ppm=ppm)
    scale = jnp.where(jnp.abs(target) > 0.0, jnp.abs(target), 1.0)
    residual = jnp.where(active, jnp.abs(final - target) / scale, 0.0)
    return AdjustedVelocity(vel=adjusted, residual=residual)


def _advect_column(
    con: Array,
    vel: Array,
    parabola: Parabola,
    ds: Array,
    *,
    dt: float | Array,
    ppm: PPMConstants,
) -> Array:
    """One species advected through a column, given its parabola.

    Ports ``vppm.F:259-288``. Two asymmetries against the horizontal case:

    * the bottom face carries no flux at all -- ``FP(0)`` is initialised to zero
      and never written, because ``zadvppmwrf.F`` sets ``VEL(1) = 0`` at the
      impermeable ground;
    * the top face uses a plain donor-cell flux from the topmost cell
      (``vppm.F:280-283``) rather than the parabola.
    """
    zero = jnp.zeros((1, *con.shape[1:]), dtype=con.dtype)

    y_down = -vel[:-1] * dt
    x_down = y_down / ds
    fm_interior = jnp.where(
        vel[:-1] < 0.0,
        y_down
        * (
            parabola.cl
            + 0.5 * x_down * (parabola.dc + parabola.c6 * (1.0 - ppm.two_thirds * x_down))
        ),
        0.0,
    )
    fm_top = jnp.where(vel[-1:] < 0.0, -vel[-1:] * dt * con[-1:], 0.0)
    fm = jnp.concatenate([fm_interior, fm_top])

    y_up = vel[1:] * dt
    x_up = y_up / ds
    fp_interior = jnp.where(
        vel[1:] > 0.0,
        y_up
        * (parabola.cr - 0.5 * x_up * (parabola.dc - parabola.c6 * (1.0 - ppm.two_thirds * x_up))),
        0.0,
    )
    fp = jnp.concatenate([zero, fp_interior])

    return con + (fp[:-1] - fp[1:] + fm[1:] - fm[:-1]) / ds


def vppm(
    con: Array,
    vel: Array,
    flx: Array,
    ds: Array,
    mesh: NonUniformMesh,
    *,
    dt: float | Array,
    ppm: PPMConstants = DEFAULT_PPM,
) -> tuple[Array, AdjustedVelocity]:
    """Advect a column of species vertically.

    Ports ``VPPM`` (``vppm.F``). ``con`` is ``(nlays, nspc)`` with the last
    species slot holding rho*J; ``vel`` and ``flx`` are the ``nlays + 1`` face
    velocities and diagnosed mass fluxes.

    The order matters and is not incidental: the velocity is first matched
    against the **rho*J column alone**, and the corrected velocity is then used
    for every species. That is what makes advected density reproduce the
    meteorology, and why rho*J has to ride along as a species rather than being
    reconstructed afterwards.

    Returns the advected column and the velocity adjustment, whose ``residual``
    reports any face that did not converge.
    """
    con = jnp.asarray(con)
    ds = jnp.asarray(ds)

    density = ppm_parabola_nonuniform(con[:, -1], mesh)
    adjusted = vppm_adjust_velocity(vel, flx, density, ds, dt=dt, ppm=ppm)

    parabola = ppm_parabola_nonuniform(con, mesh)
    advected = _advect_column(con, adjusted.vel[:, None], parabola, ds[:, None], dt=dt, ppm=ppm)
    return advected, adjusted

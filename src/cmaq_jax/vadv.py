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

__all__ = [
    "AdjustedVelocity",
    "ZadvDiagnostics",
    "diagnose_flux",
    "face_velocity_from_flux",
    "vppm",
    "vppm_adjust_velocity",
    "zadv",
]


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


def _with_layer_axis(dt: float | Array, extra: int = 0) -> Array:
    """Give a per-column time step the axes it needs to broadcast.

    Under CFL sub-stepping every column carries its own ``dt``, shaped like the
    batch. The arrays it multiplies are layer-first, and sometimes
    species-last, so the scalar case passes through and the per-column case
    gains a leading layer axis plus ``extra`` trailing ones.
    """
    step = jnp.asarray(dt)
    if step.ndim == 0:
        return step
    return step.reshape((1, *step.shape, *([1] * extra)))


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
    vel = jnp.asarray(vel)
    # ds is per-layer; give it trailing singletons so it broadcasts against
    # whatever batch dimensions the caller brought.
    ds_faces = jnp.asarray(ds).reshape((-1,) + (1,) * (vel.ndim - 1))
    ds_cells = jnp.asarray(ds).reshape((-1,) + (1,) * (con.ndim - 1))

    density = ppm_parabola_nonuniform(con[..., -1], mesh)
    adjusted = vppm_adjust_velocity(vel, flx, density, ds_faces, dt=_with_layer_axis(dt), ppm=ppm)

    parabola = ppm_parabola_nonuniform(con, mesh)
    advected = _advect_column(
        con,
        adjusted.vel[..., None],
        parabola,
        ds_cells,
        dt=_with_layer_axis(dt, extra=1),
        ppm=ppm,
    )
    return advected, adjusted


class ZadvDiagnostics(NamedTuple):
    """What the vertical solve had to do, per column.

    ``substeps`` counts the CFL sub-steps taken; ``max_courant`` is the largest
    Courant number seen on the first pass; ``residual`` is the worst leftover
    flux mismatch from the velocity adjustment. All three are the signals CMAQ
    would otherwise have turned into an ``M3EXIT``.
    """

    substeps: Array
    max_courant: Array
    residual: Array


def diagnose_flux(
    rhoj_met: Array,
    rhoj_transported: Array,
    ds: Array,
    dt: float | Array,
) -> Array:
    """Face mass flux implied by the density budget.

    Ports ``zadvppmwrf.F:343-372``. Horizontal advection leaves the transported
    rho*J disagreeing with the meteorology; the vertical flux is chosen to
    close that gap, which is what makes advected density track the met fields.

    ``DIVV(l) = (rhoj_met(l) - rhoj_transported(l)) * ds(l) / dt`` is the
    per-layer mismatch, ``DRJ = -sum(DIVV)`` the column total, and

        FLX(1) = 0,   FLX(l+1) = FLX(l) - ds(l)*DRJ - DIVV(l)

    a running sum from the impermeable ground upward.

    Two things drop out of the upstream expression, both because ``FBLN`` is
    hard-set to 1.0 (``zadvppmwrf.F:249``, the sigmoid commented out):

    * the blend against the alternative ``FLUX`` accumulator is a no-op;
    * ``RHOJM2``, the end-of-step met density, feeds only that dead accumulator
      and is therefore **not needed at all**. Only the start-of-step density
      appears here.

    The layer axis comes first; the result has one more entry along it.
    """
    rhoj_met = jnp.asarray(rhoj_met)
    rhoj_transported = jnp.asarray(rhoj_transported)
    thickness = jnp.asarray(ds).reshape((-1,) + (1,) * (rhoj_met.ndim - 1))

    divv = (rhoj_met - rhoj_transported) * thickness / dt
    column_total = -jnp.sum(divv, axis=0, keepdims=True)

    increments = -thickness * column_total - divv
    surface = jnp.zeros((1, *increments.shape[1:]), dtype=increments.dtype)
    return jnp.concatenate([surface, jnp.cumsum(increments, axis=0)])


def face_velocity_from_flux(flx: Array, rhoj_transported: Array) -> Array:
    """Face velocity from face flux, upwinded on the sign of the flux.

    Ports ``zadvppmwrf.F:374-381``: ``VEL(l) = FLX(l)/RJT(l-1)`` when the flux
    is non-negative, else ``FLX(l)/RJT(l)``. The bottom face is pinned to zero
    — the ground is impermeable — and the top face always draws on the layer
    below it, there being nothing above.
    """
    flx = jnp.asarray(flx)
    rhoj = jnp.asarray(rhoj_transported)

    below = jnp.concatenate([rhoj[:1], rhoj])  # donor when flux is upward
    above = jnp.concatenate([rhoj, rhoj[-1:]])  # donor when flux is downward
    velocity = jnp.where(flx >= 0.0, flx / below, flx / above)
    return velocity.at[0].set(0.0)


def _max_courant(vel: Array, ds: Array, dt: float | Array) -> Array:
    """Largest Courant number over a column's faces.

    Ports ``zadvppmwrf.F:383-410``. Each face is measured against the layer the
    flow is coming *from*: upward against the layer below, downward against the
    layer above. The top face is the exception — the Fortran uses the layer
    below it for both signs, since there is no layer above.

    The bottom face is skipped; its velocity is pinned to zero.
    """
    step = _with_layer_axis(dt)
    thickness = jnp.asarray(ds).reshape((-1,) + (1,) * (vel.ndim - 1))
    # Interior faces are 1 .. nlays-1. Face f draws on layer f-1 when the flow
    # is upward and layer f when it is downward, so the two donor arrays are
    # the thicknesses shifted against each other -- both nlays-1 long, matching
    # vel[1:-1].
    below = thickness[:-1]
    above = thickness[1:]

    interior = vel[1:-1]
    courant_interior = jnp.where(interior > 0.0, interior * step / below, -interior * step / above)
    # Top face: `DS(LVL-1)` on both branches (zadvppmwrf.F:404, :407).
    top = jnp.abs(vel[-1:]) * step / thickness[-1:]

    return jnp.max(jnp.concatenate([courant_interior, top]), axis=0)


def zadv(
    con: Array,
    rhoj_met: Array,
    ds: Array,
    mesh: NonUniformMesh,
    *,
    dt: float,
    ppm: PPMConstants = DEFAULT_PPM,
) -> tuple[Array, ZadvDiagnostics]:
    """Vertical advection over one sync step.

    Ports ``ZADV`` (``zadvppmwrf.F``). ``con`` is ``(nlays, ..., nspc)`` in
    coupled transport units with rho*J last; ``rhoj_met`` is the meteorological
    density at the **start** of the sync step, ``(nlays, ...)``.

    The Courant number is checked against the diagnosed velocity, and a column
    that exceeds one is advanced in sub-steps of ``0.9*dt/CC`` until the
    remaining time is safe (``zadvppmwrf.F:412-459``). CMAQ writes that as a
    ``GO TO`` loop with a cap of 30; here it is a fixed-count ``fori_loop``
    carrying the remaining time, with finished columns masked off. Columns
    advance independently, which is what the mask buys — a whole grid runs at
    once even though each column needs a different number of sub-steps.

    ``FLX`` is computed once, outside the loop. In the Fortran it sits *inside*
    the retry block and looks as though it were being refreshed, but with
    ``FBLN`` at 1.0 it depends only on quantities fixed before the loop starts,
    so recomputing it would be wasted work. The velocity *is* refreshed each
    sub-step, since it divides by the transported density that has just changed.
    """
    con = jnp.asarray(con)
    flx = diagnose_flux(rhoj_met, con[..., -1], ds, dt)

    batch = con.shape[1:-1]
    remaining = jnp.full(batch, float(dt), dtype=con.dtype)
    taken = jnp.zeros(batch, dtype=jnp.int32)

    def sub_step(
        _: int, carry: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        state, left, count, worst_residual = carry
        vel = face_velocity_from_flux(flx, state[..., -1])
        courant = _max_courant(vel, ds, left)

        # Safe column: finish the remaining time in one go. Otherwise take the
        # CFL-limited slice, floored at a second (zadvppmwrf.F:426).
        limited = jnp.maximum(
            ppm.cfl_safety * left / jnp.maximum(courant, 1e-30), ppm.min_substep_seconds
        )
        step = jnp.where(courant > 1.0, jnp.minimum(limited, left), left)
        active = left > 0.0

        advected, adjusted = vppm(state, vel, flx, ds, mesh, dt=step, ppm=ppm)
        keep = active[None, ..., None]
        return (
            jnp.where(keep, advected, state),
            jnp.where(active, left - step, left),
            count + active.astype(count.dtype),
            jnp.maximum(worst_residual, jnp.where(active, jnp.max(adjusted.residual, axis=0), 0.0)),
        )

    first_courant = _max_courant(face_velocity_from_flux(flx, con[..., -1]), ds, dt)
    state, left, taken, residual = jax.lax.fori_loop(
        0, ppm.max_substeps, sub_step, (con, remaining, taken, jnp.zeros(batch, dtype=con.dtype))
    )

    return state, ZadvDiagnostics(
        substeps=taken,
        max_courant=first_courant,
        # A column with time left over never finished; surface that as a failed
        # adjustment rather than letting it pass silently.
        residual=jnp.where(left > 0.0, jnp.inf, residual),
    )

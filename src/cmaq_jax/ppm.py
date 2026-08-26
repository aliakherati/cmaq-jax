"""The piecewise parabolic method, as CMAQ implements it.

Ports the numerics of ``hppm.F`` (uniform cell spacing, both horizontal sweeps)
following Colella & Woodward (1984). Equation numbers in the comments are
theirs, and match the numbering in the Fortran.

Everything here operates on the **sweep axis first**: ``con`` has shape
``(ncells_padded, ...)`` and any trailing axes ride along untouched. Stages are
written as whole-array slices rather than a ``vmap`` over 1-D columns, so a
sweep compiles to one fused kernel over every row, layer and species at once.

Every Fortran ``IF`` becomes a :func:`jax.numpy.where`; there are no data
dependent branches.

Index convention
----------------
The Fortran array is ``CON(1-SWP : NI+SWP, NSPCS)``, so Fortran cell ``i`` sits
at Python index ``p = i + SWP - 1``. The interior is ``con[SWP : SWP + ni]``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from cmaq_jax.config import DEFAULT_PPM, PPMConstants

__all__ = [
    "Parabola",
    "ppm_advect_uniform",
    "ppm_flux_uniform",
    "ppm_parabola_uniform",
]


class Parabola(NamedTuple):
    """A monotonised parabola per cell, Colella & Woodward eq. (1.4).

    ``cl``/``cr`` are the left and right edge values, ``dc = cr - cl`` is the
    eq. (1.5) slope and ``c6`` the curvature term. All four are aligned on the
    same cell window as the ``con`` slice they were built from.
    """

    cl: Array
    cr: Array
    dc: Array
    c6: Array


def _fortran_sign(magnitude: Array, sign_source: Array) -> Array:
    """Fortran's ``SIGN(magnitude, sign_source)``.

    Not ``jnp.sign``: Fortran treats ``+0.0`` as positive and returns
    ``+magnitude``, whereas ``jnp.sign(0.0)`` is ``0.0``.
    """
    return jnp.where(sign_source >= 0.0, magnitude, -magnitude)


def _van_leer_limit(dc: Array, back: Array, fwd: Array) -> Array:
    """Monotonicity constraint on a cell slope, eq. (1.8).

    ``back = c_i - c_{i-1}`` and ``fwd = c_{i+1} - c_i``. Where those disagree
    in sign the cell is a local extremum and the slope is zeroed; otherwise the
    slope is clipped to twice the smaller neighbouring difference.

    Ports ``hppm.F:296-303`` and the identical block at ``vppm.F:493-500``.
    """
    limited = jnp.minimum(jnp.abs(dc), jnp.minimum(2.0 * jnp.abs(back), 2.0 * jnp.abs(fwd)))
    return jnp.where(back * fwd > 0.0, _fortran_sign(limited, dc), 0.0)


def _monotonise(
    con: Array,
    cl: Array,
    cr: Array,
    pad: list[tuple[int, int]] | None = None,
) -> Parabola:
    """Force the parabola to be monotone inside its cell, eqs. (1.10) and (1.5).

    Three cases, exactly as ``hppm.F:327-351`` / ``vppm.F:518-540``:

    * the cell value lies outside ``[cl, cr]`` -- a local extremum -- so the
      interpolant collapses to a constant;
    * the parabola overshoots past the left edge, so ``cl`` is reset to put the
      extremum on the boundary;
    * likewise on the right.

    The two overshoot cases are mutually exclusive (both would require
    ``0 > 2*dc**2``), and the Fortran's ``ELSE IF`` means each uses the
    *original* edge values -- which is what the simultaneous ``where`` below
    reproduces.

    ``pad``, when given, is applied to every output, letting the caller shift
    the result into a wider index space.
    """
    interior = (cr - con) * (con - cl) > 0.0

    dc_trial = cr - cl
    c6_trial = 6.0 * (con - 0.5 * (cl + cr))
    overshoot_left = dc_trial * c6_trial > dc_trial * dc_trial
    overshoot_right = -dc_trial * dc_trial > dc_trial * c6_trial

    cl_fixed = jnp.where(overshoot_left, 3.0 * con - 2.0 * cr, cl)
    cr_fixed = jnp.where(overshoot_right & ~overshoot_left, 3.0 * con - 2.0 * cl, cr)

    cl_out = jnp.where(interior, cl_fixed, con)
    cr_out = jnp.where(interior, cr_fixed, con)

    parabola = Parabola(
        cl=cl_out,
        cr=cr_out,
        dc=cr_out - cl_out,
        c6=6.0 * (con - 0.5 * (cl_out + cr_out)),
    )
    if pad is None:
        return parabola
    return Parabola(*(jnp.pad(a, pad) for a in parabola))


def ppm_parabola_uniform(con: Array, ppm: PPMConstants = DEFAULT_PPM) -> Parabola:
    """Build the monotonised parabola on a uniformly spaced grid.

    Ports ``hppm.F:283-353``. ``con`` is the padded array with the sweep axis
    first; the returned arrays are aligned on Python index ``p`` and are valid
    for ``p`` in ``[2, con.shape[0] - 4]``, which covers Fortran cells
    ``0 .. NI+1`` -- exactly the window the flux stage needs.
    """
    con = jnp.asarray(con)
    # Limited slope per cell, eqs. (1.7)-(1.8). hppm.F:289-303.
    diff = jnp.diff(con, axis=0)
    slope = 0.5 * (diff[:-1] + diff[1:])
    dc = _van_leer_limit(slope, back=diff[:-1], fwd=diff[1:])
    # Pad back to the con index space; the two edge cells are never read.
    pad = [(1, 1)] + [(0, 0)] * (con.ndim - 1)
    dc = jnp.pad(dc, pad)

    # Trial edge value, eq. (1.6). hppm.F:309-310.
    #   cm_p = 0.5*(con[p] + con[p-1]) - (1/6)*(dc[p] - dc[p-1])
    # Stored offset by one: cm[j] holds cm_{j+1}.
    cm = 0.5 * (con[1:] + con[:-1]) - ppm.sixth * (dc[1:] - dc[:-1])

    # eq. (1.15): a cell's right edge is the next cell's left edge, so
    #   cl_p = cm_p     = cm[p-1]
    #   cr_p = cm_{p+1} = cm[p]
    # Both are available for p in [1, M-2]; pad by one at each end so the
    # returned arrays are indexed by p directly, which keeps every downstream
    # slice in one index space. The padded edges are never read -- the flux
    # stage only touches p in [2, ni+3].
    return _monotonise(con[1:-1], cl=cm[:-1], cr=cm[1:], pad=pad)


def ppm_flux_uniform(
    parabola: Parabola,
    con_window: Array,
    vel: Array,
    *,
    dt: float | Array,
    ds: float | Array,
    ppm: PPMConstants = DEFAULT_PPM,
) -> tuple[Array, Array]:
    """Upwind fluxes across every cell face, eq. (1.12).

    Ports ``hppm.F:368-439``. ``parabola`` and ``con_window`` cover Fortran
    cells ``0 .. NI+1`` (length ``ni + 2``); ``vel`` holds the ``ni + 1`` face
    velocities ``VEL(1..NI+1)``, broadcastable against the trailing axes.

    Returns ``(fp, fm)``, each of length ``ni + 1``:

    * ``fp[i]`` -- mass leaving cell ``i`` through its upper face, non-zero
      only where the face velocity is positive;
    * ``fm[i+1]`` -- mass leaving cell ``i+1`` through its lower face,
      non-zero only where the face velocity is negative.

    The outermost faces use a plain donor-cell flux rather than the parabola
    (``hppm.F:422-439``). In CMAQ that is conditional on this process owning a
    domain boundary; single-device, it always owns both.
    """
    con_window = jnp.asarray(con_window)
    vel = jnp.asarray(vel)
    zero = jnp.zeros((), dtype=con_window.dtype)

    # Outflux through upper faces: cells 0..NI paired with VEL(1..NI+1).
    y_up = vel * dt
    x_up = y_up / ds
    fp = jnp.where(
        vel > 0.0,
        y_up
        * (
            parabola.cr[:-1]
            - 0.5 * x_up * (parabola.dc[:-1] - parabola.c6[:-1] * (1.0 - ppm.two_thirds * x_up))
        ),
        zero,
    )

    # Outflux through lower faces: cells 1..NI+1 paired with the same faces.
    y_dn = -vel * dt
    x_dn = y_dn / ds
    fm = jnp.where(
        vel < 0.0,
        y_dn
        * (
            parabola.cl[1:]
            + 0.5 * x_dn * (parabola.dc[1:] + parabola.c6[1:] * (1.0 - ppm.two_thirds * x_dn))
        ),
        zero,
    )

    # Donor-cell flux at the two domain edges, overriding the parabolic value.
    fp = fp.at[0].set(jnp.where(vel[0] > 0.0, y_up[0] * con_window[0], zero))
    fm = fm.at[-1].set(jnp.where(vel[-1] < 0.0, y_dn[-1] * con_window[-1], zero))

    return fp, fm


def ppm_advect_uniform(
    con: Array,
    vel: Array,
    dt: float | Array,
    ds: float | Array,
    ppm: PPMConstants = DEFAULT_PPM,
) -> Array:
    """One PPM sweep along a uniformly spaced axis.

    The JAX equivalent of a single ``HPPM`` call. ``con`` is padded with
    ``ppm.halo_width`` ghost cells at each end of axis 0; ``vel`` holds the
    ``ni + 1`` face velocities. Only the interior is updated -- the halo comes
    back untouched, matching ``hppm.F:443`` (``DO I = 1, NI``), because
    refilling it is the caller's job.

    ``vel`` must broadcast against ``con``'s trailing axes; add singleton axes
    (e.g. ``vel[:, None]`` for a ``(cells, species)`` array) before calling.
    """
    con = jnp.asarray(con)
    vel = jnp.asarray(vel)
    swp = ppm.halo_width
    ni = con.shape[0] - 2 * swp
    if ni < 1:
        raise ValueError(f"con has {con.shape[0]} cells, too few for a halo of {swp}")

    # Fortran cells 0..NI+1 start at Python index swp - 1.
    lo = swp - 1
    window = slice(lo, lo + ni + 2)

    parabola = ppm_parabola_uniform(con, ppm)
    parabola = Parabola(*(a[window] for a in parabola))
    fp, fm = ppm_flux_uniform(parabola, con[window], vel, dt=dt, ds=ds, ppm=ppm)

    # Conservative update, eq. (1.13). hppm.F:444-445.
    delta = (fp[:-1] - fp[1:] + fm[1:] - fm[:-1]) / ds
    return con.at[swp : swp + ni].add(delta)

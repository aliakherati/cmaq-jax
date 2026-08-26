"""Horizontal advection: the sweeps and the driver that sequences them.

Ports ``x_ppm.F``, ``y_ppm.F`` and ``hadvppm.F``. The two sweeps differ only in
which axis they run along and which pair of boundaries they consult, so they are
one function here rather than the two near-duplicate 660-line files upstream.

The driver reproduces three behaviours that the sweep alone does not have, and
that a property test cannot see:

* **per-layer sub-stepping** -- each layer advances on its own ``ASTEP``,
  repeating until it reaches the sync step (``hadvppm.F:199-257``);
* **X-Y / Y-X alternation** -- the sweep order flips on *every sub-step*, per
  layer, carried across calls in a saved flag (``hadvppm.F:215-251``);
* **layer independence** -- layers never exchange information horizontally.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from cmaq_jax.bc import fill_halo
from cmaq_jax.config import DEFAULT_PPM, GridConfig, PPMConstants
from cmaq_jax.ppm import ppm_advect_uniform

__all__ = ["BoundaryConditions", "hadv", "sweep"]

COLUMN_AXIS = 0
ROW_AXIS = 1


class BoundaryConditions(NamedTuple):
    """Inflow concentrations on each domain edge, in coupled transport units.

    CMAQ stores these in one flat ring, ``BCON(NBNDY, N_SPC_ADV)``, indexed
    through per-edge offsets: ``SFX = 0`` and ``NFX = NCOLS+NROWS+3``
    (``y_ppm.F:203``), ``EFX = NCOLS+1`` and ``WFX = 2*NCOLS+NROWS+4``
    (``x_ppm.F:208``). Splitting the ring into named edges here keeps that
    arithmetic out of the advection code; ``scripts/generate_goldens.py``
    holds the conversion.

    ``west``/``east`` are ``(nrows, nlays, nspc)``; ``south``/``north`` are
    ``(ncols, nlays, nspc)``.
    """

    west: Array
    east: Array
    south: Array
    north: Array


def sweep(
    cgrid: Array,
    vel: Array,
    bcon_lo: Array,
    bcon_hi: Array,
    *,
    dt: float | Array,
    ds: float | Array,
    axis: int,
    ppm: PPMConstants = DEFAULT_PPM,
) -> Array:
    """One PPM sweep along ``axis`` of the whole grid.

    The JAX equivalent of an ``X_PPM`` or ``Y_PPM`` call, minus the row loop:
    where CMAQ gathers one row at a time into a 1-D buffer, this sweeps every
    row, layer and species at once.

    ``cgrid`` is the interior ``(ncols, nrows, nlays, nspc)``. ``vel`` holds the
    face velocities of ``axis`` -- one longer than the cell count along it, and
    without a species axis, since the wind is the same for every species.
    ``bcon_lo``/``bcon_hi`` supply the inflow values for the low and high edges.
    """
    cgrid = jnp.asarray(cgrid)
    swp = ppm.halo_width

    moved = jnp.moveaxis(cgrid, axis, 0)
    # The wind does not vary by species; give it a trailing axis to broadcast.
    moved_vel = jnp.moveaxis(jnp.asarray(vel), axis, 0)[..., None]

    if moved_vel.shape[0] != moved.shape[0] + 1:
        raise ValueError(
            f"vel has {moved_vel.shape[0]} entries along axis {axis} but the grid has "
            f"{moved.shape[0]} cells; face velocities need one more"
        )

    # The pad values are placeholders -- fill_halo overwrites the whole region.
    pad = [(swp, swp)] + [(0, 0)] * (moved.ndim - 1)
    padded = jnp.pad(moved, pad)

    filled = fill_halo(padded, moved_vel, bcon_lo, bcon_hi, ppm)
    advected = ppm_advect_uniform(filled, moved_vel, dt, ds, ppm)

    return jnp.moveaxis(advected[swp:-swp], 0, axis)


def _substep_count(sync_seconds: int, astep_seconds: int) -> int:
    """How many advection steps a layer takes per sync step.

    ``hadvppm.F:199-256`` primes ``DSTEP = STEP``, does the work, adds ``STEP``
    and repeats while ``DSTEP <= SYNCSTEP`` -- which runs the body
    ``floor(SYNCSTEP / STEP)`` times. CMAQ's ``ADVSTEP`` chooses ``ASTEP`` to
    divide the sync step exactly, so the floor is normally not doing anything.
    """
    if astep_seconds <= 0:
        raise ValueError(f"astep must be positive, got {astep_seconds}")
    count = sync_seconds // astep_seconds
    if count < 1:
        raise ValueError(
            f"astep of {astep_seconds}s exceeds the {sync_seconds}s sync step, "
            "so the layer would never advance"
        )
    return int(count)


def _sweep_pair(
    cgrid: Array,
    uhat: Array,
    vhat: Array,
    bcon: BoundaryConditions,
    *,
    cfg: GridConfig,
    dt: float,
    x_first: bool,
) -> Array:
    """Both sweeps, in the requested order."""
    do_x = (
        uhat,
        bcon.west,
        bcon.east,
        cfg.dx1,
        COLUMN_AXIS,
    )
    do_y = (
        vhat,
        bcon.south,
        bcon.north,
        cfg.dx2,
        ROW_AXIS,
    )
    for vel, lo, hi, ds, axis in (do_x, do_y) if x_first else (do_y, do_x):
        cgrid = sweep(cgrid, vel, lo, hi, dt=dt, ds=ds, axis=axis, ppm=cfg.ppm)
    return cgrid


def hadv(
    cgrid: Array,
    uhat: Array,
    vhat: Array,
    bcon: BoundaryConditions,
    *,
    cfg: GridConfig,
    astep_seconds: NDArray[np.integer],
    sync_seconds: int,
    xyfirst: NDArray[np.bool_],
) -> tuple[Array, NDArray[np.bool_]]:
    """Advance one sync step of horizontal advection.

    Ports ``HADV`` (``hadvppm.F``). ``cgrid`` is ``(ncols, nrows, nlays, nspc)``
    in coupled transport units, with the last species slot holding rho*J.

    ``astep_seconds`` gives each layer its own advection step, and
    ``sync_seconds`` the step they all have to reach. ``xyfirst`` is the saved
    alternation flag, one per layer; the updated copy comes back with the
    result, since CMAQ keeps it in a ``SAVE``d array across calls and the
    sequence of orders is part of the answer.

    Layers are grouped by ``(sub-step count, starting sweep order)`` -- both
    host-side, both static -- so each group runs a plain Python loop of sweeps
    with no masking and no ``lax.while_loop``. The alternative, carrying a
    per-layer flag into the kernel, would mean computing both sweep orders
    everywhere and selecting, which doubles the work to no purpose.

    The wind is taken as fixed across the sync step. CMAQ re-reads it at each
    sub-step's midpoint, but that is time interpolation inside the met reader
    rather than anything advection does; it belongs with ``io_mcip`` (A3.5).
    """
    cgrid = jnp.asarray(cgrid)
    nlays = cgrid.shape[2]

    astep = np.asarray(astep_seconds, dtype=np.int64)
    order = np.asarray(xyfirst, dtype=bool).copy()
    if astep.shape != (nlays,):
        raise ValueError(f"astep_seconds must have one entry per layer, got {astep.shape}")
    if order.shape != (nlays,):
        raise ValueError(f"xyfirst must have one entry per layer, got {order.shape}")

    groups: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for layer in range(nlays):
        groups[(_substep_count(sync_seconds, int(astep[layer])), bool(order[layer]))].append(layer)

    result = cgrid
    for (nsub, starts_with_x), layers in sorted(groups.items()):
        index = np.asarray(layers)
        block = result[:, :, index, :]
        block_bcon = BoundaryConditions(
            west=bcon.west[:, index, :],
            east=bcon.east[:, index, :],
            south=bcon.south[:, index, :],
            north=bcon.north[:, index, :],
        )
        dt = float(astep[index[0]])

        x_first = starts_with_x
        for _ in range(nsub):
            block = _sweep_pair(
                block,
                uhat[:, :, index],
                vhat[:, :, index],
                block_bcon,
                cfg=cfg,
                dt=dt,
                x_first=x_first,
            )
            # hadvppm.F flips the flag inside the sub-step loop, not once per
            # call, so a layer taking two sub-steps sweeps X-Y then Y-X and
            # comes back to where it started.
            x_first = not x_first

        result = result.at[:, :, index, :].set(block)
        order[index] = x_first

    return result, order

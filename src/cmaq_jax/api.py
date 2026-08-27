"""The advection operator as CMAQ's science driver invokes it.

``sciproc.F:250-278`` runs, per sync step::

    COUPLE -> HADV -> ZADV -> HDIFF -> DECOUPLE

This module supplies the ``HADV -> ZADV`` pair. Coupling is deliberately left
out: it is a unit conversion between the model's storage units and transport
units, not transport, and it belongs with the meteorology reader.

The state arrives in **coupled transport units** with the layer axis third and
the species axis last, matching Fortran's ``CGRID(COL, ROW, LAY, SPC)``. The
last species slot holds rho*J, which is advected alongside everything else —
that is CMAQ's mass-conservation mechanism, not an optimisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from cmaq_jax.config import GridConfig
from cmaq_jax.hadv import BoundaryConditions, advance_xyfirst, hadv_step
from cmaq_jax.ppm import NonUniformMesh
from cmaq_jax.vadv import ZadvDiagnostics, zadv

__all__ = ["Meteorology", "advance_xyfirst", "advect_step"]

LAYER_AXIS = 2


class Meteorology(NamedTuple):
    """Everything advection needs from outside itself, for one sync step.

    ``uhat``/``vhat`` are contravariant face velocities -- see
    :mod:`cmaq_jax.velocity`, and note that on the modern C-staggered path they
    are ``UWINDC``/``VWINDC`` unchanged. ``rhoj_met`` is the meteorological
    density at the **start** of the step; the end-of-step field CMAQ also reads
    is unused, since ``FBLN`` is fixed at 1.0.
    """

    uhat: Array  # (ncols+1, nrows, nlays)
    vhat: Array  # (ncols, nrows+1, nlays)
    rhoj_met: Array  # (ncols, nrows, nlays)
    bcon: BoundaryConditions


def advect_step(
    state: Array,
    met: Meteorology,
    mesh: NonUniformMesh,
    *,
    cfg: GridConfig,
    astep_seconds: NDArray[np.integer],
    sync_seconds: int,
    xyfirst: Sequence[bool],
) -> tuple[Array, ZadvDiagnostics]:
    """One sync step of three-dimensional advection.

    ``state`` is ``(ncols, nrows, nlays, nspc)`` in coupled units. Returns the
    advected state and the vertical diagnostics -- where the failure modes are,
    since a column that exhausts its sub-steps reports an infinite residual.

    The sweep-order flags are **not** returned. They are host-side control
    state, and putting them in a jitted function's output turns them into
    traced arrays, which is exactly what stops them being usable as flags on
    the next call. Advance them alongside instead::

        step = jax.jit(functools.partial(
            advect_step, mesh=mesh, cfg=cfg, astep_seconds=astep,
            sync_seconds=sync, xyfirst=xyfirst))
        state, diagnostics = step(state, met)
        xyfirst = advance_xyfirst(xyfirst, astep, sync)

    Horizontal runs first, as in ``sciproc.F``. The order is not arbitrary --
    the vertical flux is diagnosed from the gap between the transported density
    and the meteorology, and it is horizontal advection that opens that gap.
    Running the vertical first would leave it correcting a gap that has not been
    created yet.

    The two operators want different memory layouts: horizontal sweeps read
    ``CGRID`` order directly, while the vertical works layer-first so a column
    is contiguous. The transpose between them is a relabelling that XLA folds
    into the neighbouring kernels.
    """
    horizontal = hadv_step(
        state,
        met.uhat,
        met.vhat,
        met.bcon,
        cfg=cfg,
        astep_seconds=astep_seconds,
        sync_seconds=sync_seconds,
        xyfirst=xyfirst,
    )

    column_state = jnp.moveaxis(horizontal, LAYER_AXIS, 0)
    column_met = jnp.moveaxis(jnp.asarray(met.rhoj_met, dtype=horizontal.dtype), LAYER_AXIS, 0)
    advected, vertical = zadv(
        column_state,
        column_met,
        jnp.asarray(cfg.ds, dtype=horizontal.dtype),
        mesh,
        dt=float(sync_seconds),
        ppm=cfg.ppm,
    )

    return jnp.moveaxis(advected, 0, LAYER_AXIS), vertical

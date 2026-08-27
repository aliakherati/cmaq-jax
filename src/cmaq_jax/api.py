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
from cmaq_jax.hdiff import hdiff_step
from cmaq_jax.ppm import NonUniformMesh
from cmaq_jax.vadv import ZadvDiagnostics, zadv
from cmaq_jax.vdiff import ColumnState, SurfaceExchange, vdiff_step

__all__ = [
    "Diffusivity",
    "Meteorology",
    "advance_xyfirst",
    "advect_step",
    "science_step",
    "transport_step",
]

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


class Diffusivity(NamedTuple):
    """What horizontal diffusion needs, for one sync step.

    Separate from :class:`Meteorology` because it is *derived* from it rather
    than read: build it with :func:`cmaq_jax.hdiff.diffusion_coefficients` and
    :func:`cmaq_jax.hdiff.halo_density`. Keeping the two apart makes it explicit
    that a run reusing meteorology across steps can reuse these too.
    """

    densj: Array  # (ncols+2, nrows+2, nlays), rho*J with its halo ring
    k11: Array  # (ncols+1, nrows+1, nlays), x-face diffusivity
    k22: Array  # (ncols+1, nrows+1, nlays), y-face diffusivity


def transport_step(
    state: Array,
    met: Meteorology,
    diffusivity: Diffusivity,
    mesh: NonUniformMesh,
    *,
    cfg: GridConfig,
    astep_seconds: NDArray[np.integer],
    sync_seconds: int,
    xyfirst: Sequence[bool],
    diffusion_substeps: int,
) -> tuple[Array, ZadvDiagnostics]:
    """One sync step of transport: HADV, then ZADV, then HDIFF.

    The order is ``sciproc.F:250-278``'s, minus the coupling either side, which
    is a unit conversion rather than transport. It is not a free choice: the
    vertical flux is diagnosed from the gap horizontal advection opens, and
    diffusion runs on the result of both.

    ``diffusion_substeps`` is host-side, from
    :func:`cmaq_jax.hdiff.substep_count`, so the diffusion loop has a static
    trip count — the same arrangement as ``astep_seconds`` for advection.

    Returns the transported state and the vertical diagnostics. Diffusion has no
    diagnostics of its own: it cannot fail to converge, having nothing to
    converge.
    """
    advected, vertical = advect_step(
        state,
        met,
        mesh,
        cfg=cfg,
        astep_seconds=astep_seconds,
        sync_seconds=sync_seconds,
        xyfirst=xyfirst,
    )
    diffused = hdiff_step(
        advected,
        diffusivity.densj,
        diffusivity.k11,
        diffusivity.k22,
        cfg=cfg,
        sync_seconds=float(sync_seconds),
        nsteps=diffusion_substeps,
    )
    return diffused, vertical


def science_step(
    state: Array,
    met: Meteorology,
    diffusivity: Diffusivity,
    *,
    vertical: ColumnState,
    surface: SurfaceExchange,
    mesh: NonUniformMesh,
    cfg: GridConfig,
    astep_seconds: NDArray[np.integer],
    sync_seconds: int,
    xyfirst: Sequence[bool],
    diffusion_substeps: int,
    vdiff_substeps: int,
) -> tuple[Array, Array, ZadvDiagnostics]:
    """One sync step of the transport block: VDIFF, then HADV, ZADV, HDIFF.

    ``sciproc.F:231-278``'s order, and the ordering point that is easy to get
    wrong: **vertical diffusion runs first, and it runs on uncoupled
    concentrations**, before ``COUPLE``. It is not part of the coupled block
    that :func:`transport_step` implements, so it is applied here rather than
    appended to that chain.

    This port does not implement ``COUPLE``/``DECOUPLE`` — they are a unit
    conversion, not transport — so ``state`` arrives already in coupled units
    and the vertical-diffusion stage is applied to it directly. On a real run
    that conversion would sit between the two calls below. The distinction is
    recorded rather than silently elided because it changes what the
    concentrations mean.

    Returns the advanced state, the accumulated dry deposition and the vertical
    advection diagnostics.
    """
    diffused, ddep = vdiff_step(
        state,
        vertical,
        surface,
        dtsec=float(sync_seconds),
        max_substeps=vdiff_substeps,
    )
    transported, diagnostics = transport_step(
        diffused,
        met,
        diffusivity,
        mesh,
        cfg=cfg,
        astep_seconds=astep_seconds,
        sync_seconds=sync_seconds,
        xyfirst=xyfirst,
        diffusion_substeps=diffusion_substeps,
    )
    return transported, ddep, diagnostics

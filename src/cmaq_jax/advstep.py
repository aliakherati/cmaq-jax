"""Choosing the sync step and the per-layer advection step.

Ports ``advstep.F``. This is the piece that decides *how often* advection runs,
by asking how fast the wind is relative to the grid and how strongly it
diverges. Nothing here touches concentrations.

Two constraints, both per layer:

* **Courant** -- ``max(|u|/dx, |v|/dy) * dt < CFL``, so a parcel cannot cross a
  cell in one step;
* **horizontal divergence** -- ``max(HDIV) * dt < HDIV_LIM``, added in 2009
  because a strongly divergent wind can empty a cell even at a safe Courant
  number.

All of it is host-side integer arithmetic over a handful of layers, so it stays
in plain Python rather than JAX: the answer is static configuration for the step
that follows, and making it a traced computation would force the sub-step
counts to become dynamic for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_LIMITS",
    "AdvectionSchedule",
    "StepLimits",
    "advstep",
    "sync_top_layer",
    "wind_index",
]


@dataclass(frozen=True)
class StepLimits:
    """Bounds and safety factors for the step-size search.

    Defaults are CMAQ's, from ``RUNTIME_VARS.F:132-214``. All are environment
    variables upstream, so a run can differ; they are arguments here for the
    same reason.
    """

    cfl: float = 0.75
    """Largest Courant number allowed. ``RUNTIME_VARS.F:214``."""

    hdiv_lim: float = 0.9
    """Largest ``divergence * dt`` allowed. ``RUNTIME_VARS.F:213``.

    Added in 2009 (``advstep.F:84``): the Courant condition alone lets a
    strongly divergent wind drain a cell within a single step.
    """

    min_sync: int = 60
    """Floor on the sync step, seconds. ``RUNTIME_VARS.F:133``."""

    max_sync: int = 720
    """Ceiling on the sync step, seconds. ``RUNTIME_VARS.F:132``."""

    sigma_sync_top: float = 0.7
    """Sigma level below which layers constrain the sync step.
    ``RUNTIME_VARS.F:212`` (``SIGMA_SYNC_TOP``).

    Only layers up to here take part in choosing the sync step
    (``advstep.F:393``); above it each layer subdivides on its own. Without
    that split a jet aloft would drag the entire model onto its step, which is
    both wasteful and unnecessary -- the fast layer can simply advect more
    often than everything else.
    """


DEFAULT_LIMITS = StepLimits()
"""CMAQ's defaults. Functions here take a ``StepLimits``; this is the usual one."""


@dataclass(frozen=True)
class AdvectionSchedule:
    """How long a sync step is, and how each layer subdivides it."""

    sync_seconds: int
    astep_seconds: NDArray[np.int64]
    """Advection step per layer; each divides ``sync_seconds`` exactly."""

    @property
    def substeps(self) -> NDArray[np.int64]:
        """Advection steps per sync step, per layer."""
        return self.sync_seconds // self.astep_seconds


def wind_index(
    uhat: NDArray[np.floating], vhat: NDArray[np.floating], dx1: float, dx2: float
) -> NDArray[np.float64]:
    """Per-layer inverse advective timescale, ``max(|u|/dx1, |v|/dx2)``.

    Ports ``advstep.F:360-375``. This is the quantity the Courant test is
    applied to: multiply by a candidate step and compare against ``CFL``.

    Reduces over the horizontal, so the result has one entry per layer. In CMAQ
    the reduction is also over MPI ranks (``SUBST_GLOBAL_MAX``) -- the step must
    be the same everywhere, or ranks would fall out of step with each other.
    """
    per_layer_u = np.max(np.abs(np.asarray(uhat)), axis=(0, 1)) / dx1
    per_layer_v = np.max(np.abs(np.asarray(vhat)), axis=(0, 1)) / dx2
    return np.maximum(per_layer_u, per_layer_v)


def sync_top_layer(sigma_faces: NDArray[np.floating], sigma_sync_top: float) -> int:
    """Number of layers that constrain the sync step.

    Ports ``advstep.F:201-217``: the layer whose face sits nearest
    ``SIGMA_SYNC_TOP``. Sigma decreases upward, so this counts from the ground.
    """
    faces = np.asarray(sigma_faces, dtype=np.float64)
    if not faces[-1] <= sigma_sync_top <= faces[1]:
        raise ValueError(
            f"sigma_sync_top {sigma_sync_top} lies outside the grid's [{faces[-1]}, {faces[1]}]"
        )
    # The face closest to the threshold; +1 converts a face index to a count.
    return int(np.argmin(np.abs(faces[1:] - sigma_sync_top))) + 1


def _divisors_descending(total: int) -> list[int]:
    """Divisors of ``total``, largest first."""
    return [d for d in range(total, 0, -1) if total % d == 0]


def _largest_safe_divisor(total: int, rate: float, limit: float) -> int | None:
    """Largest divisor of ``total`` with ``rate * divisor < limit``."""
    for candidate in _divisors_descending(total):
        if rate * candidate < limit:
            return candidate
    return None


def advstep(
    wind: NDArray[np.floating],
    hdiv: NDArray[np.floating],
    output_seconds: int,
    limits: StepLimits = DEFAULT_LIMITS,
    sync_layers: int | None = None,
) -> AdvectionSchedule:
    """Choose the sync step and each layer's advection step.

    Ports ``advstep.F:387-500``. ``wind`` is the per-layer index from
    :func:`wind_index`; ``hdiv`` the per-layer maximum horizontal divergence.
    ``output_seconds`` is the output interval the sync step has to divide.
    ``sync_layers`` is how many layers from the ground take part in choosing
    the sync step -- see :func:`sync_top_layer`. Defaulting it to every layer
    reproduces a column where the whole depth is below ``SIGMA_SYNC_TOP``.

    The search runs in three passes, matching the Fortran:

    1. **Sync step** -- the largest divisor of the output step, no greater than
       ``max_sync``, whose Courant number over *all* layers is safe. CMAQ tries
       divisors from the largest down and takes the first that works.
    2. **Per-layer advection step** -- each layer then subdivides that sync
       step until its own Courant number is safe. Upper layers usually have the
       fastest winds and so the shortest steps, which is the whole reason
       ``ASTEP`` is per layer rather than global.
    3. **Divergence limit** -- each layer's step is shortened further if
       ``hdiv * step`` exceeds ``hdiv_lim``.

    Raises when no step satisfies the constraints, which is where CMAQ calls
    ``M3EXIT``: it means the wind is too fast for the grid, and quietly
    returning an unstable step would be worse than stopping.
    """
    wind = np.asarray(wind, dtype=np.float64)
    hdiv = np.asarray(hdiv, dtype=np.float64)
    if wind.shape != hdiv.shape:
        raise ValueError(f"wind and hdiv must agree per layer, got {wind.shape} and {hdiv.shape}")
    if output_seconds < 1:
        raise ValueError(f"output_seconds must be positive, got {output_seconds}")

    ceiling = min(output_seconds, limits.max_sync)
    # Only the lower layers get a say in the sync step; a fast jet aloft
    # subdivides instead of slowing the whole model down.
    constraining = wind.size if sync_layers is None else sync_layers
    if not 1 <= constraining <= wind.size:
        raise ValueError(f"sync_layers must be within 1..{wind.size}, got {sync_layers}")
    fastest = float(wind[:constraining].max())

    sync = None
    for candidate in _divisors_descending(output_seconds):
        if candidate > ceiling:
            continue
        if fastest * candidate < limits.cfl:
            sync = candidate
            break
        # Below the floor the sync step stops shrinking and the *advection*
        # step subdivides it instead (advstep.F:403-417). One sync step then
        # holds several advection steps, which is what ASTEP exists to express.
        if candidate <= limits.min_sync:
            sync = candidate
            break

    if sync is None:
        raise ValueError(
            f"no sync step divides {output_seconds}s while keeping the Courant number "
            f"below {limits.cfl}: the fastest layer needs {limits.cfl / fastest:.2f}s"
        )

    astep = np.empty(wind.shape, dtype=np.int64)
    for layer, (rate, divergence) in enumerate(zip(wind, hdiv, strict=True)):
        step = _largest_safe_divisor(sync, float(rate), limits.cfl)
        if step is None:
            raise ValueError(
                f"layer {layer} cannot satisfy the Courant condition within a {sync}s sync "
                f"step: it needs {limits.cfl / float(rate):.3f}s"
            )
        if divergence > 0.0:
            step = _largest_safe_divisor(step, float(divergence), limits.hdiv_lim)
            if step is None:
                raise ValueError(
                    f"layer {layer} cannot satisfy the divergence limit within a {sync}s sync "
                    f"step: it needs {limits.hdiv_lim / float(divergence):.3f}s"
                )
        astep[layer] = step

    return AdvectionSchedule(sync_seconds=sync, astep_seconds=astep)

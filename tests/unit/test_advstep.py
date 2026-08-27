"""A3.1 — choosing the sync step and the per-layer advection step.

No Fortran golden: ``advstep.F`` reaches its winds through ``HVELOC``, which
reads meteorology through the I/O API, and the routine's output is a handful of
integers rather than a field. The tests pin the *rules* instead — divisibility,
both limits, and the per-layer behaviour that makes ``ASTEP`` an array.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmaq_jax.advstep import (
    DEFAULT_LIMITS,
    AdvectionSchedule,
    StepLimits,
    advstep,
    sync_top_layer,
    wind_index,
)

HOUR = 3600


def schedule(wind, hdiv=None, output=HOUR, limits=DEFAULT_LIMITS) -> AdvectionSchedule:
    wind = np.atleast_1d(np.asarray(wind, dtype=float))
    if hdiv is None:
        hdiv = np.zeros_like(wind)
    return advstep(wind, np.atleast_1d(np.asarray(hdiv, dtype=float)), output, limits)


class TestWindIndex:
    def test_takes_the_faster_direction_per_layer(self) -> None:
        """``max(|u|/dx1, |v|/dy)`` -- the Courant test cares about whichever
        direction crosses a cell soonest, not about their combination."""
        nc, nr, nl = 4, 3, 2
        u = np.zeros((nc + 1, nr, nl))
        v = np.zeros((nc, nr + 1, nl))
        u[..., 0] = 30.0  # layer 0: zonal wind dominates
        v[..., 1] = 60.0  # layer 1: meridional
        got = wind_index(u, v, dx1=1000.0, dx2=2000.0)
        np.testing.assert_allclose(got, [30.0 / 1000.0, 60.0 / 2000.0])

    def test_reduces_over_the_whole_horizontal(self) -> None:
        """A single fast cell sets the step for its entire layer."""
        u = np.zeros((5, 4, 1))
        v = np.zeros((4, 5, 1))
        u[2, 2, 0] = 100.0
        np.testing.assert_allclose(wind_index(u, v, 1000.0, 1000.0), [0.1])


class TestSyncStep:
    def test_divides_the_output_step(self) -> None:
        """CMAQ writes output on a fixed interval, so the sync step has to land
        on it exactly (``advstep.F:396``)."""
        for wind in (1e-5, 1e-4, 5e-4, 2e-3):
            result = schedule(wind)
            assert HOUR % result.sync_seconds == 0

    def test_respects_the_ceiling(self) -> None:
        """A calm wind would allow the whole output step; ``max_sync`` caps it."""
        result = schedule(1e-9, limits=StepLimits(max_sync=300))
        assert result.sync_seconds <= 300

    def test_takes_the_largest_step_the_courant_condition_allows(self) -> None:
        """Not merely *a* safe step. Choosing a shorter one would be stable but
        would cost proportionally more work for nothing."""
        limits = StepLimits(max_sync=HOUR)
        result = schedule(1.0 / 500.0, limits=limits)
        assert result.sync_seconds * (1.0 / 500.0) < limits.cfl
        larger = [d for d in range(result.sync_seconds + 1, HOUR + 1) if HOUR % d == 0]
        for candidate in larger:
            assert candidate * (1.0 / 500.0) >= limits.cfl, "a larger safe step was available"

    def test_a_fast_wind_shortens_the_step(self) -> None:
        """Both winds must be fast enough to bind, or the ``max_sync`` ceiling
        hides the effect and the test proves nothing."""
        limits = StepLimits(max_sync=HOUR)
        slow = schedule(1e-4, limits=limits).sync_seconds
        fast = schedule(1e-3, limits=limits).sync_seconds
        assert slow * 1e-4 < limits.cfl and fast * 1e-3 < limits.cfl
        assert fast < slow

    def test_below_the_floor_the_advection_step_subdivides_instead(self) -> None:
        """``advstep.F:403-417``: once the sync step reaches ``min_sync`` it
        stops shrinking, and several advection steps run inside it. That is the
        whole reason ``ASTEP`` exists separately from the sync step."""
        limits = StepLimits(min_sync=60, max_sync=HOUR)
        result = schedule(0.05, limits=limits)  # needs ~15 s
        assert result.sync_seconds == 60
        assert result.astep_seconds[0] < 60
        assert int(result.substeps[0]) > 1


class TestPerLayerStep:
    def test_each_layer_divides_the_sync_step(self) -> None:
        result = schedule([1e-4, 1e-3, 5e-3])
        for step in result.astep_seconds:
            assert result.sync_seconds % int(step) == 0

    def test_a_layer_above_the_sync_top_gets_a_shorter_step(self) -> None:
        """This is the whole reason ``ASTEP`` is an array.

        Only layers below ``SIGMA_SYNC_TOP`` help choose the sync step
        (``advstep.F:393``). A jet in the layers above then subdivides that step
        on its own rather than dragging the entire model down to its pace --
        which is what would happen, and what an earlier version of this port
        did, if the sync step were taken over every layer.
        """
        wind = np.array([1e-5, 1e-5, 4e-3])
        result = advstep(wind, np.zeros(3), HOUR, sync_layers=2)
        assert result.astep_seconds[2] < result.astep_seconds[0]
        assert int(result.substeps[0]) == 1
        assert int(result.substeps[2]) > 1

    def test_without_the_split_the_fast_layer_slows_everything(self) -> None:
        """The contrast that makes the split worth having."""
        wind = np.array([1e-5, 1e-5, 4e-3])
        split = advstep(wind, np.zeros(3), HOUR, sync_layers=2)
        merged = advstep(wind, np.zeros(3), HOUR)
        assert merged.sync_seconds < split.sync_seconds

    def test_every_layer_satisfies_the_courant_condition(self) -> None:
        limits = StepLimits()
        wind = np.array([1e-5, 3e-4, 1e-3, 6e-3, 2e-2])
        result = schedule(wind, limits=limits)
        for rate, step in zip(wind, result.astep_seconds, strict=True):
            assert rate * float(step) < limits.cfl


class TestDivergenceLimit:
    def test_divergence_can_shorten_a_courant_safe_step(self) -> None:
        """The reason this limit was added in 2009 (``advstep.F:84``): a wind
        can be slow enough to pass the Courant test while diverging fast enough
        to empty a cell within the step."""
        calm = schedule(1e-5, hdiv=0.0)
        divergent = schedule(1e-5, hdiv=0.05)
        assert divergent.astep_seconds[0] < calm.astep_seconds[0]

    def test_the_limit_is_respected(self) -> None:
        limits = StepLimits()
        hdiv = np.array([0.0, 0.01, 0.05, 0.2])
        result = schedule(np.full(4, 1e-5), hdiv=hdiv, limits=limits)
        for divergence, step in zip(hdiv, result.astep_seconds, strict=True):
            assert divergence * float(step) < limits.hdiv_lim

    def test_zero_divergence_imposes_nothing(self) -> None:
        wind = np.array([1e-4, 1e-3])
        with_zero = schedule(wind, hdiv=np.zeros(2))
        courant_only = schedule(wind)
        np.testing.assert_array_equal(with_zero.astep_seconds, courant_only.astep_seconds)


class TestFailure:
    def test_a_wind_too_fast_for_the_grid_raises(self) -> None:
        """Where CMAQ calls ``M3EXIT``. Returning an unstable step quietly
        would be worse than stopping -- PPM does not merely lose accuracy above
        Courant one, it overflows."""
        with pytest.raises(ValueError, match="Courant"):
            schedule(10.0, limits=StepLimits(min_sync=1, max_sync=HOUR))

    def test_mismatched_layer_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="per layer"):
            advstep(np.zeros(3), np.zeros(4), HOUR)

    def test_a_nonpositive_output_step_raises(self) -> None:
        with pytest.raises(ValueError, match="output_seconds"):
            advstep(np.zeros(2), np.zeros(2), 0)


def test_the_schedule_feeds_the_driver() -> None:
    """The point of the whole module: its output is exactly what ``hadv_step``
    and ``advect_step`` take."""
    nc, nr, nl = 6, 5, 4
    rng = np.random.default_rng(20260910)
    u = rng.normal(0.0, 8.0, (nc + 1, nr, nl))
    v = rng.normal(0.0, 8.0, (nc, nr + 1, nl))
    u[..., -1] *= 6.0  # a jet aloft, as a real profile would have

    wind = wind_index(u, v, dx1=12000.0, dx2=12000.0)
    result = advstep(wind, np.zeros(nl), output_seconds=HOUR)

    assert HOUR % result.sync_seconds == 0
    assert result.astep_seconds.shape == (nl,)
    assert np.all(result.sync_seconds % result.astep_seconds == 0)
    assert int(result.substeps[-1]) >= int(result.substeps[0]), "the jet layer should sub-step"


class TestSyncTopLayer:
    def test_picks_the_face_nearest_the_threshold(self) -> None:
        faces = np.array([1.0, 0.9, 0.8, 0.7, 0.5, 0.0])
        assert sync_top_layer(faces, 0.7) == 3

    def test_a_threshold_outside_the_grid_raises(self) -> None:
        faces = np.array([1.0, 0.9, 0.5, 0.0])
        with pytest.raises(ValueError, match="outside"):
            sync_top_layer(faces, 0.95)

    def test_a_column_entirely_below_the_threshold_uses_every_layer(self) -> None:
        faces = np.linspace(1.0, 0.0, 6)
        assert sync_top_layer(faces, 0.0) == faces.size - 1

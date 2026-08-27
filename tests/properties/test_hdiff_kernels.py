"""B1 — properties of the deformation and diffusivity, beyond the goldens.

A golden pins one field. These pin the *scheme*: what the deformation is
mathematically, and what the diffusivity blend must do at its limits. They are
what would catch a swapped term or an inverted blend that happened to agree with
a single Fortran case.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cmaq_jax.config import DEFAULT_HDIFF, HDiffConstants
from cmaq_jax.hdiff import deformation, eddy_diffusivity, face_coefficients, stable_timestep

NCOLS, NROWS, NLAYS = 9, 8, 2
DX1 = DX2 = 12000.0
SHAPE = (NCOLS + 1, NROWS + 1, NLAYS)

#: Where both cross-gradients are live. ``deform.F:420-421`` zeroes ``du/dy`` on
#: the first and last row and ``dv/dx`` on the first and last column -- different
#: edges -- so a slice that keeps either one is not comparing like with like.
#: Extending this by a single column puts the rotation discrepancy at 40% of the
#: signal, which reads as a broken scheme rather than a bad slice.
INTERIOR = (slice(1, NCOLS - 1), slice(1, NROWS - 1), slice(None))

ROWS = np.arange(NROWS + 1, dtype=np.float64)[None, :, None]
COLS = np.arange(NCOLS + 1, dtype=np.float64)[:, None, None]
ZERO = np.zeros(SHAPE)


def deform(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.asarray(
        deformation(
            np.broadcast_to(u, SHAPE) + 0.0, np.broadcast_to(v, SHAPE) + 0.0, dx1=DX1, dx2=DX2
        )
    )


class TestDeformation:
    def test_solid_body_translation_has_none(self) -> None:
        """A uniform wind has no gradients at all, so the deformation is
        identically zero -- everywhere, not just in the interior. Separates
        "reads the wind" from "differentiates it": a kernel that returned the
        wind itself would fail this and pass a smooth random case.
        """
        assert deform(np.full(SHAPE, 12.0), np.full(SHAPE, -5.0)).max() == 0.0

    @pytest.mark.parametrize("rate", [1.0, 3.0, -2.5])
    def test_pure_shear_recovers_dudy(self, rate: float) -> None:
        got = deform(rate * ROWS, ZERO)
        np.testing.assert_allclose(got[INTERIOR], abs(rate) / DX2, rtol=1e-12)

    @pytest.mark.parametrize("rate", [1.0, 5.0, -4.0])
    def test_pure_stretching_recovers_dudx(self, rate: float) -> None:
        got = deform(rate * COLS, ZERO)
        np.testing.assert_allclose(got[INTERIOR], abs(rate) / DX1, rtol=1e-12)

    def test_rotating_the_frame_leaves_it_unchanged(self) -> None:
        """``sqrt(DF1^2 + DF2^2)`` is the second invariant of the strain-rate
        tensor, so rotating the wind by 90 degrees -- ``(u, v) -> (-v, u)`` --
        must not change it. Catches a swapped ``DF1``/``DF2`` or a sign slip that
        a single golden field cannot distinguish.

        Only on the strict interior: see ``INTERIOR``.
        """
        u, v = 3.0 * ROWS, 2.0 * COLS
        np.testing.assert_allclose(deform(u, v)[INTERIOR], deform(-v, u)[INTERIOR], rtol=1e-12)

    def test_rotation_invariance_is_not_vacuous(self) -> None:
        """Guards the test above: if the rotated field happened to equal the
        original, the comparison would prove nothing."""
        u, v = 3.0 * ROWS, 2.0 * COLS
        assert not np.allclose(np.broadcast_to(u, SHAPE), np.broadcast_to(-v, SHAPE))

    def test_stretching_and_shearing_add_in_quadrature(self) -> None:
        """``DF1`` and ``DF2`` are independent components, so a field carrying
        both must give their Pythagorean sum -- not their total, and not either
        one alone."""
        stretch, shear = 5.0, 3.0
        # u = shear*row + stretch*col makes DF1 = du/dx and DF2 = du/dy.
        got = deform(shear * ROWS + stretch * COLS, ZERO)
        expected = np.hypot(stretch / DX1, shear / DX2)
        np.testing.assert_allclose(got[INTERIOR], expected, rtol=1e-12)

    def test_it_is_never_negative(self) -> None:
        rng = np.random.default_rng(11)
        got = deform(rng.normal(0.0, 8.0, SHAPE), rng.normal(0.0, 8.0, SHAPE))
        assert got.min() >= 0.0


class TestEddyDiffusivity:
    def kha(self, constants: HDiffConstants = DEFAULT_HDIFF) -> float:
        return constants.base_diffusivity(DX1, DX2)

    def eddy(self, deform_value: float, msfd2: float = 1.0) -> float:
        field = np.full((2, 2, 1), deform_value)
        got = eddy_diffusivity(field, np.full((2, 2), msfd2), dx1=DX1, dx2=DX2)
        return float(np.asarray(got)[0, 0, 0])

    def test_zero_deformation_gives_the_floor_not_zero(self) -> None:
        """``KHD = max(KHMIN, 0) = KHMIN``. A calm cell still diffuses, and
        reading the deformation field as the diffusivity field is wrong."""
        kha = self.kha()
        expected = kha * DEFAULT_HDIFF.khmin / (kha + DEFAULT_HDIFF.khmin)
        assert self.eddy(0.0) == pytest.approx(expected)
        assert expected > 0.0

    def test_it_increases_with_deformation(self) -> None:
        values = [self.eddy(d) for d in (0.0, 1e-4, 1e-3, 1e-2, 1.0)]
        assert all(a <= b for a, b in pairwise(values))

    def test_it_saturates_at_the_base_diffusivity(self) -> None:
        """The blend is ``KHA*KHD/(KHA+KHD)``, which tends to ``KHA`` as ``KHD``
        grows. Without that, a sheared cell would get an unbounded diffusivity
        and the stable step would collapse. An upside-down blend passes a mild
        field and fails here.
        """
        kha = self.kha()
        assert self.eddy(1e8) == pytest.approx(kha, rel=1e-5)
        assert self.eddy(1e12) <= kha * (1.0 + 1e-9)

    def test_it_never_exceeds_the_base_diffusivity(self) -> None:
        for d in (0.0, 1e-6, 1e-2, 1.0, 1e6, 1e10):
            assert self.eddy(d) <= self.kha() * (1.0 + 1e-9)

    def test_the_map_factor_scales_it_linearly(self) -> None:
        """``MSFD2`` is ~1 on the benchmark Lambert grid, so dropping it entirely
        would pass any test that only ever uses that grid."""
        assert self.eddy(1e-3, msfd2=3.0) == pytest.approx(3.0 * self.eddy(1e-3))

    def test_a_coarser_grid_gets_less_base_diffusivity(self) -> None:
        """``KHA = (DXB^2)/(dx1*dx2) * KH``: the term stands in for sub-grid
        mixing, and a coarse grid already represents less of it."""
        assert DEFAULT_HDIFF.base_diffusivity(4000.0, 4000.0) == pytest.approx(DEFAULT_HDIFF.kh)
        assert DEFAULT_HDIFF.base_diffusivity(12000.0, 12000.0) < DEFAULT_HDIFF.kh


class TestStableTimestep:
    def test_a_larger_diffusivity_shortens_it(self) -> None:
        small = np.full((4, 4, 1), 100.0)
        large = np.full((4, 4, 1), 1000.0)
        assert stable_timestep(large, large, dx1=DX1, dx2=DX2) < stable_timestep(
            small, small, dx1=DX1, dx2=DX2
        )

    def test_it_takes_the_largest_coefficient_anywhere(self) -> None:
        """One hot cell sets the step for the whole domain, as ``EFFKB``'s
        reduction over the interior does."""
        field = np.full((5, 5, 1), 100.0)
        spiked = field.copy()
        spiked[2, 2, 0] = 5000.0
        assert stable_timestep(spiked, field, dx1=DX1, dx2=DX2) == pytest.approx(
            DEFAULT_HDIFF.cfc * DX1 * DX2 / 5000.0
        )

    def test_it_ignores_the_zeroed_boundary(self) -> None:
        """``K11``'s last row and ``K22``'s last column are zero by construction,
        and a reduction that included them would still be correct -- but one
        that reduced over a *max* of the padded array with a large value parked
        there would not. Pinning the interior reduction states which it is.
        """
        field = np.full((5, 5, 1), 100.0)
        edged = field.copy()
        edged[-1, :, :] = 9.0e9
        edged[:, -1, :] = 9.0e9
        assert stable_timestep(edged, edged, dx1=DX1, dx2=DX2) == pytest.approx(
            stable_timestep(field, field, dx1=DX1, dx2=DX2)
        )


class TestFaceCoefficients:
    def test_each_averages_across_its_own_direction(self) -> None:
        """``K11`` is on x faces and averages over *rows*; ``K22`` on y faces
        averaging over *columns*. Swapping them is the natural mistake and it
        survives any test on a field that is symmetric in the two directions.
        """
        eddyh = np.arange(4 * 3 * 1, dtype=np.float64).reshape((4, 3, 1))
        k11, k22 = (np.asarray(a) for a in face_coefficients(eddyh))
        np.testing.assert_allclose(k11[:, 0, 0], 0.5 * (eddyh[:, 1, 0] + eddyh[:, 0, 0]))
        np.testing.assert_allclose(k22[0, :, 0], 0.5 * (eddyh[0, :, 0] + eddyh[1, :, 0]))

    def test_a_uniform_field_averages_to_itself(self) -> None:
        eddyh = np.full((5, 4, 2), 7.0)
        k11, k22 = (np.asarray(a) for a in face_coefficients(eddyh))
        np.testing.assert_allclose(k11[:, :-1], 7.0)
        np.testing.assert_allclose(k22[:-1], 7.0)

    def test_the_outflow_edges_are_zero(self) -> None:
        eddyh = np.full((5, 4, 2), 7.0)
        k11, k22 = (np.asarray(a) for a in face_coefficients(eddyh))
        assert np.all(k11[:, -1] == 0.0)
        assert np.all(k22[-1] == 0.0)

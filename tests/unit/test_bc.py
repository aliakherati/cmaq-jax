"""A1.2 — filling the ghost region at the ends of a sweep axis.

Ports the boundary blocks of ``x_ppm.F:418-441``. There is no Fortran golden for
this: ``x_ppm.F`` reaches the boundary values through ``RDBCON``, which reads
I/O API files, so the unit under test here is the *logic* -- which side counts
as outflow, that every ghost cell on a side gets the same value, and that the
interior is left alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmaq_jax.bc import fill_halo, zfdbc
from cmaq_jax.config import DEFAULT_PPM

SWP = DEFAULT_PPM.halo_width


def make_con(interior: np.ndarray) -> np.ndarray:
    """Pad an interior profile with deliberately wrong halo values.

    Filling with NaN would hide a bug behind a NaN check; a large finite number
    shows up as an obviously wrong answer instead.
    """
    garbage = np.full((SWP, *interior.shape[1:]), -999.0)
    return np.concatenate([garbage, interior, garbage])


def uniform_vel(ni: int, speed: float) -> np.ndarray:
    return np.full(ni + 1, speed)


class TestSideSelection:
    """Which edge is inflow and which is outflow, by the sign of the wind."""

    def test_wind_from_low_to_high_takes_bcon_at_the_low_edge(self) -> None:
        interior = np.linspace(1.0, 2.0, 8)[:, None]
        con = make_con(interior)
        out = np.asarray(
            fill_halo(con, uniform_vel(8, +1.0)[:, None], np.array([7.0]), np.array([9.0]))
        )

        # Low edge is inflow: the boundary field supplies it.
        np.testing.assert_allclose(out[:SWP, 0], 7.0)
        # High edge is outflow: extrapolated from inside, not the bcon value.
        assert not np.allclose(out[-SWP:, 0], 9.0)

    def test_wind_from_high_to_low_takes_bcon_at_the_high_edge(self) -> None:
        interior = np.linspace(1.0, 2.0, 8)[:, None]
        con = make_con(interior)
        out = np.asarray(
            fill_halo(con, uniform_vel(8, -1.0)[:, None], np.array([7.0]), np.array([9.0]))
        )

        np.testing.assert_allclose(out[-SWP:, 0], 9.0)
        assert not np.allclose(out[:SWP, 0], 7.0)

    def test_convergent_wind_is_inflow_on_both_sides(self) -> None:
        """Wind blowing inward at both edges: both halos come from bcon."""
        interior = np.full((8, 1), 2.0)
        con = make_con(interior)
        vel = np.concatenate([[+1.0], np.zeros(7), [-1.0]])[:, None]
        out = np.asarray(fill_halo(con, vel, np.array([7.0]), np.array([9.0])))
        np.testing.assert_allclose(out[:SWP, 0], 7.0)
        np.testing.assert_allclose(out[-SWP:, 0], 9.0)

    def test_divergent_wind_is_outflow_on_both_sides(self) -> None:
        """Wind blowing outward at both edges: neither halo uses bcon."""
        interior = np.linspace(1.0, 3.0, 8)[:, None]
        con = make_con(interior)
        vel = np.concatenate([[-1.0], np.zeros(7), [+1.0]])[:, None]
        out = np.asarray(fill_halo(con, vel, np.array([7.0]), np.array([9.0])))
        assert not np.allclose(out[:SWP, 0], 7.0)
        assert not np.allclose(out[-SWP:, 0], 9.0)

    def test_zero_wind_is_inflow(self) -> None:
        """The Fortran tests `VELX(1) .LT. 0` and `VELX(NCOLS+1) .GT. 0`, so a
        dead calm falls to the inflow branch on both sides."""
        con = make_con(np.full((8, 1), 2.0))
        out = np.asarray(fill_halo(con, np.zeros((9, 1)), np.array([7.0]), np.array([9.0])))
        np.testing.assert_allclose(out[:SWP, 0], 7.0)
        np.testing.assert_allclose(out[-SWP:, 0], 9.0)


class TestOutflowValue:
    def test_matches_zfdbc_on_the_correct_stencil(self) -> None:
        """x_ppm.F:432 passes the last two interior cells and the last two face
        velocities, in that order -- an easy pair to transpose."""
        interior = np.array([1.0, 2.0, 3.0, 5.0, 8.0])[:, None]
        con = make_con(interior)
        vel = np.array([-1.0, 0.5, 0.5, 0.5, 0.7, 1.2])[:, None]
        out = np.asarray(fill_halo(con, vel, np.array([0.0]), np.array([0.0])))

        expected_hi = float(np.asarray(zfdbc(interior[-1], interior[-2], vel[-1], vel[-2]))[0])
        expected_lo = float(np.asarray(zfdbc(interior[0], interior[1], vel[0], vel[1]))[0])
        np.testing.assert_allclose(out[-SWP:, 0], expected_hi)
        np.testing.assert_allclose(out[:SWP, 0], expected_lo)

    def test_every_ghost_cell_on_a_side_is_identical(self) -> None:
        """CMAQ assigns the whole `1-SWP:0` slice at once, so the
        reconstruction meets a flat approach to the boundary and the limiter
        sees no artificial gradient there."""
        con = make_con(np.linspace(1.0, 4.0, 10)[:, None])
        vel = np.concatenate([[-2.0], np.full(9, 1.0), [2.0]])[:-1][:, None]
        out = np.asarray(fill_halo(con, vel, np.array([0.0]), np.array([0.0])))
        assert len(np.unique(out[:SWP, 0])) == 1
        assert len(np.unique(out[-SWP:, 0])) == 1


class TestStructure:
    def test_interior_is_untouched(self) -> None:
        interior = np.linspace(1.0, 4.0, 12)[:, None]
        con = make_con(interior)
        out = np.asarray(
            fill_halo(con, uniform_vel(12, 1.0)[:, None], np.array([0.0]), np.array([0.0]))
        )
        np.testing.assert_array_equal(out[SWP:-SWP], interior)

    def test_shape_is_preserved(self) -> None:
        con = make_con(np.ones((10, 3)))
        out = fill_halo(con, np.ones((11, 1)), np.zeros(3), np.zeros(3))
        assert out.shape == con.shape

    def test_per_species_boundary_values(self) -> None:
        """Each advected species has its own boundary field; they must not be
        broadcast into one another."""
        interior = np.ones((8, 3))
        con = make_con(interior)
        bcon_lo = np.array([1.0, 2.0, 3.0])
        bcon_hi = np.array([4.0, 5.0, 6.0])
        out = np.asarray(fill_halo(con, np.ones((9, 1)), bcon_lo, bcon_hi))
        np.testing.assert_allclose(out[:SWP], np.broadcast_to(bcon_lo, (SWP, 3)))
        # High edge is outflow under a positive wind, so bcon_hi is not used.
        assert not np.allclose(out[-SWP:], np.broadcast_to(bcon_hi, (SWP, 3)))

    def test_trailing_axes_ride_along(self) -> None:
        """A full grid sweep carries (rows, layers, species) behind the sweep
        axis, and the velocity varies across rows but not species."""
        interior = np.ones((8, 4, 3))
        con = make_con(interior)
        vel = np.ones((9, 4, 1))
        out = fill_halo(con, vel, np.zeros((4, 3)), np.zeros((4, 3)))
        assert out.shape == (8 + 2 * SWP, 4, 3)

    def test_rejects_a_domain_too_small_for_the_stencil(self) -> None:
        """The outflow condition needs two interior cells to extrapolate from."""
        con = make_con(np.ones((1, 1)))
        with pytest.raises(ValueError, match="at least 2"):
            fill_halo(con, np.ones((2, 1)), np.zeros(1), np.zeros(1))

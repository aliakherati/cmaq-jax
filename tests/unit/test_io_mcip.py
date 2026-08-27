"""A3.5 — reading MCIP meteorology in I/O API format.

Against synthetic files, not real MCIP output: CMAQ ships run scripts but not
data, and ``$CMAQ_DATA`` is a separate download. So these pin the reader's
handling of the *format* -- transposes, staggering, time interpolation, header
parsing -- and the one thing they cannot establish is that genuine MCIP files
match the format as written here. See ``docs/plans/subplans/A3-integrate.md``.

The fields in the fixture are deterministic, so every assertion below states
what the value should be rather than only that the read succeeded.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

from cmaq_jax.advstep import StepLimits, advstep, sync_top_layer, wind_index
from cmaq_jax.api import Meteorology, advect_step
from cmaq_jax.hadv import BoundaryConditions
from cmaq_jax.io_mcip import MetFiles, ioapi_datetime, open_met
from cmaq_jax.ppm import nonuniform_mesh
from tests.fixtures.ioapi import MetFixture

pytest.importorskip("netCDF4", reason="reading I/O API needs the 'io' extra")


@pytest.fixture
def met(tmp_path: Path) -> MetFixture:
    return MetFixture(tmp_path)


@pytest.fixture
def legacy_met(tmp_path: Path) -> MetFixture:
    """A pre-MCIPv3.5 file: ``UHAT_JD``/``VHAT_JD``, no C-staggered winds."""
    return MetFixture(tmp_path, c_staggered=False)


def reader(fixture: MetFixture):
    return open_met(
        MetFiles(
            met_cro_3d=fixture.met_cro_3d,
            met_dot_3d=fixture.met_dot_3d,
            grid_cro_2d=fixture.grid_cro_2d,
        )
    )


class TestDateConversion:
    def test_packs_year_and_day_of_year(self) -> None:
        """``2018182`` is day 182 of 2018, not an offset."""
        assert ioapi_datetime(2018182, 0) == datetime(2018, 7, 1)

    def test_reads_hhmmss(self) -> None:
        assert ioapi_datetime(2018001, 133045) == datetime(2018, 1, 1, 13, 30, 45)

    def test_handles_a_leap_year(self) -> None:
        """Day 60 of 2020 is 29 February; of 2019, 1 March. Getting this wrong
        would silently shift a whole run by a day."""
        assert ioapi_datetime(2020060, 0) == datetime(2020, 2, 29)
        assert ioapi_datetime(2019060, 0) == datetime(2019, 3, 1)

    def test_rejects_an_impossible_day(self) -> None:
        with pytest.raises(ValueError, match="day-of-year"):
            ioapi_datetime(2018400, 0)


class TestHeader:
    def test_reads_the_grid_geometry(self, met: MetFixture) -> None:
        with reader(met) as r:
            assert (r.ncols, r.nrows, r.nlays) == (met.ncols, met.nrows, met.nlays)
            assert r.dx1 == 12000.0
            assert r.dx2 == 12000.0

    def test_layer_thicknesses_come_from_vglvls(self, met: MetFixture) -> None:
        """``VGLVLS`` is the sigma face array, so ``ds`` follows from the header
        alone -- no separate configuration, and no chance of the two disagreeing."""
        with reader(met) as r:
            np.testing.assert_allclose(r.ds, np.abs(np.diff(met.sigma_faces)), rtol=1e-6)
            assert r.ds.size == met.nlays
            assert np.all(r.ds > 0.0)

    def test_the_column_is_closed(self, met: MetFixture) -> None:
        """Sigma thicknesses sum to one. The vertical operator relies on it."""
        with reader(met) as r:
            np.testing.assert_allclose(r.ds.sum(), 1.0, rtol=1e-6)

    def test_builds_a_matching_grid_config(self, met: MetFixture) -> None:
        with reader(met) as r:
            cfg = r.grid_config(nspc_adv=5)
        assert (cfg.ncols, cfg.nrows, cfg.nlays) == (met.ncols, met.nrows, met.nlays)
        assert cfg.nspc_adv == 5
        np.testing.assert_allclose(cfg.ds, np.abs(np.diff(met.sigma_faces)), rtol=1e-6)

    def test_reports_the_record_times(self, met: MetFixture) -> None:
        with reader(met) as r:
            assert r.times == met.times


class TestTranspose:
    def test_a_cross_field_comes_back_in_model_order(self, met: MetFixture) -> None:
        """The file stores ``(LAY, ROW, COL)``; the model wants ``(COL, ROW, LAY)``.

        Asserted elementwise rather than by shape, because the fixture is not
        square and a wrong transpose would still have a plausible shape on a
        domain that was.
        """
        with reader(met) as r:
            got = r.density(met.times[0])
        assert got.shape == (met.ncols, met.nrows, met.nlays)
        np.testing.assert_allclose(got, np.transpose(met.density[0], (2, 1, 0)), rtol=1e-6)

    def test_the_fixture_is_not_square(self, met: MetFixture) -> None:
        """Guards the test above: on a square domain it would prove nothing."""
        assert met.ncols != met.nrows


class TestStaggering:
    def test_drops_the_false_dot_points(self, met: MetFixture) -> None:
        """The trap this reader exists to avoid.

        ``MET_DOT_3D`` is ``(NCOLS+1, NROWS+1)``, but C-staggered winds occupy
        only part of it: ``UWINDC`` is on west-east faces, so its last row is a
        false dot point, and ``VWINDC``'s last column likewise
        (``ctmproc.f90:878``, ``init_ctm.f90:1330-1346``). Keeping them would
        hand advection a row of meaningless velocities along one edge.
        """
        with reader(met) as r:
            u, v = r.face_velocities(met.times[0])
        assert u.shape == (met.ncols + 1, met.nrows, met.nlays)
        assert v.shape == (met.ncols, met.nrows + 1, met.nlays)

        stored_u = np.transpose(met.u_dot[0], (2, 1, 0))
        stored_v = np.transpose(met.v_dot[0], (2, 1, 0))
        np.testing.assert_allclose(u, stored_u[:, : met.nrows, :], rtol=1e-6)
        np.testing.assert_allclose(v, stored_v[: met.ncols, :, :], rtol=1e-6)

    def test_the_dropped_points_were_not_the_kept_ones(self, met: MetFixture) -> None:
        """Guards the test above: if the false points happened to duplicate
        their neighbours, cropping the wrong edge would still pass."""
        stored_u = np.transpose(met.u_dot[0], (2, 1, 0))
        assert not np.allclose(stored_u[:, met.nrows - 1, :], stored_u[:, met.nrows, :])

    def test_the_shapes_are_what_advection_wants(self, met: MetFixture) -> None:
        """``hadv_step`` takes exactly these, so a mismatch is caught here rather
        than as a broadcasting error several layers down."""
        with reader(met) as r:
            u, v = r.face_velocities(met.times[0])
            rhoj = r.density(met.times[0])
            cfg = r.grid_config(nspc_adv=3)

        edge = np.zeros((met.nrows, met.nlays, 3))
        bcon = BoundaryConditions(
            edge,
            edge,
            np.zeros((met.ncols, met.nlays, 3)),
            np.zeros((met.ncols, met.nlays, 3)),
        )
        weather = Meteorology(uhat=u, vhat=v, rhoj_met=rhoj, bcon=bcon)
        assert weather.uhat.shape == (cfg.ncols + 1, cfg.nrows, cfg.nlays)
        assert weather.vhat.shape == (cfg.ncols, cfg.nrows + 1, cfg.nlays)
        assert weather.rhoj_met.shape == (cfg.ncols, cfg.nrows, cfg.nlays)

    def test_c_staggered_winds_are_not_divided_by_density(self, met: MetFixture) -> None:
        """``hcontvel.F`` returns ``UWINDC`` and RETURNs immediately -- the
        density division belongs to the older ``UHAT_JD`` path only. Taking the
        fallback here would divide out a density that was never multiplied in.
        """
        with reader(met) as r:
            assert r.has_c_staggered_wind
            u, _ = r.face_velocities(met.times[0])
        stored_u = np.transpose(met.u_dot[0], (2, 1, 0))[:, : met.nrows, :]
        np.testing.assert_allclose(u, stored_u, rtol=1e-6)

    def test_the_legacy_path_divides_by_face_density(self, legacy_met: MetFixture) -> None:
        """Without ``UWINDC`` the reader falls back to ``UHAT_JD``, which is
        velocity times Jacobian times density and must be divided by the
        density interpolated onto the face (``hcontvel.F:329-351``)."""
        with reader(legacy_met) as r:
            assert not r.has_c_staggered_wind
            u, _ = r.face_velocities(legacy_met.times[0])
            rhoj = r.density(legacy_met.times[0])

        stored = np.transpose(legacy_met.u_dot[0], (2, 1, 0))[:, : legacy_met.nrows, :]
        # Interior faces average the two neighbouring cells.
        interior = 0.5 * (rhoj[:-1] + rhoj[1:])
        np.testing.assert_allclose(u[1:-1], stored[1:-1] / interior, rtol=1e-6)
        assert not np.allclose(u, stored), "the fallback did not divide by anything"


class TestTimeInterpolation:
    def test_a_record_time_reads_that_record_exactly(self, met: MetFixture) -> None:
        with reader(met) as r:
            got = r.density(met.times[1])
        np.testing.assert_allclose(got, np.transpose(met.density[1], (2, 1, 0)), rtol=1e-6)

    def test_halfway_between_records_is_the_mean(self, met: MetFixture) -> None:
        """Meteorology is hourly and the sync step is minutes, so almost every
        read lands between records."""
        half = met.times[0] + (met.times[1] - met.times[0]) / 2
        with reader(met) as r:
            got = r.density(half)
        expected = np.transpose(0.5 * (met.density[0] + met.density[1]), (2, 1, 0))
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_interpolation_is_linear_in_time(self, met: MetFixture) -> None:
        """Checked at a weight that is neither 0, 1 nor 1/2, so a nearest-record
        or midpoint implementation cannot pass."""
        span = met.times[1] - met.times[0]
        when = met.times[0] + 0.25 * span
        with reader(met) as r:
            got = r.density(when)
        expected = np.transpose(0.75 * met.density[0] + 0.25 * met.density[1], (2, 1, 0))
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_before_the_first_record_clamps(self, met: MetFixture) -> None:
        """CMAQ's ``REVERT`` branch reuses the nearest step rather than
        extrapolating (``hcontvel.F:221-235``): a stale meteorology is a better
        failure than a linearly extrapolated one."""
        with reader(met) as r:
            got = r.density(met.times[0] - timedelta(hours=5))
        np.testing.assert_allclose(got, np.transpose(met.density[0], (2, 1, 0)), rtol=1e-6)

    def test_after_the_last_record_clamps(self, met: MetFixture) -> None:
        with reader(met) as r:
            got = r.density(met.times[-1] + timedelta(hours=5))
        np.testing.assert_allclose(got, np.transpose(met.density[-1], (2, 1, 0)), rtol=1e-6)

    def test_winds_interpolate_too(self, met: MetFixture) -> None:
        half = met.times[0] + (met.times[1] - met.times[0]) / 2
        with reader(met) as r:
            u, _ = r.face_velocities(half)
        expected = np.transpose(0.5 * (met.u_dot[0] + met.u_dot[1]), (2, 1, 0))
        np.testing.assert_allclose(u, expected[:, : met.nrows, :], rtol=1e-6)


class TestOtherFields:
    def test_reads_the_jacobian(self, met: MetFixture) -> None:
        with reader(met) as r:
            got = r.jacobian(met.times[0])
        np.testing.assert_allclose(got, np.transpose(met.jacobian[0], (2, 1, 0)), rtol=1e-6)

    def test_reads_layer_face_heights(self, met: MetFixture) -> None:
        with reader(met) as r:
            got = r.layer_face_height(met.times[0])
        assert got.shape == (met.ncols, met.nrows, met.nlays)

    def test_reads_the_map_scale_factor(self, met: MetFixture) -> None:
        """Time-independent and 2-D, from the grid file rather than the met one."""
        with reader(met) as r:
            got = r.map_scale_factor_squared()
        assert got.shape == (met.ncols, met.nrows)
        np.testing.assert_allclose(got, np.transpose(met.msfx2[0, 0]), rtol=1e-6)

    def test_the_grid_file_is_optional(self, met: MetFixture) -> None:
        """Advection never uses ``MSFX2`` -- it belongs to coupling -- so a
        reader without the grid file is still complete for transport."""
        files = MetFiles(met_cro_3d=met.met_cro_3d, met_dot_3d=met.met_dot_3d)
        with open_met(files) as r:
            assert r.density(met.times[0]).shape == (met.ncols, met.nrows, met.nlays)
            with pytest.raises(ValueError, match="grid_cro_2d"):
                r.map_scale_factor_squared()


class TestFailure:
    def test_a_missing_file_is_reported_by_name(self, met: MetFixture) -> None:
        with pytest.raises(FileNotFoundError, match="met_cro_3d"):
            MetFiles(met_cro_3d=Path("/nonexistent/METCRO3D.nc"), met_dot_3d=met.met_dot_3d)

    def test_mismatched_domains_are_rejected(self, tmp_path: Path) -> None:
        """A cross and dot file from different domains give a wind field that is
        merely offset -- plausible shape, plausible values, wrong answer. Worth
        refusing rather than trusting."""
        (tmp_path / "small").mkdir()
        (tmp_path / "big").mkdir()
        small = MetFixture(tmp_path / "small", ncols=6, nrows=5)
        big = MetFixture(tmp_path / "big", ncols=9, nrows=8)
        with (
            pytest.raises(ValueError, match="not the same domain"),
            open_met(MetFiles(met_cro_3d=small.met_cro_3d, met_dot_3d=big.met_dot_3d)),
        ):
            pass

    def test_a_missing_variable_lists_what_is_there(self, met: MetFixture) -> None:
        with reader(met) as r, pytest.raises(KeyError, match=r"NOSUCHVAR"):
            r.cross("NOSUCHVAR", met.times[0])


class TestDrivesTheOperator:
    """The claim worth making about a reader: what it returns actually runs.

    Shape checks catch the coarse mistakes, but agreement on shape is not
    agreement on meaning. Feeding the fields through ``advstep`` and
    ``advect_step`` exercises the staggering, the transposes and the units
    together, and constancy preservation is the invariant that fails if any one
    of them is wrong.
    """

    def test_read_meteorology_advects(self, met: MetFixture) -> None:
        mixing_ratio = 0.6
        with reader(met) as r:
            cfg = r.grid_config(nspc_adv=2)
            uhat, vhat = r.face_velocities(met.times[0])
            rhoj = r.density(met.times[0])
            faces = r.sigma_faces

        # A uniform mixing ratio, coupled: slot 0 is q*rho*J, slot 1 is rho*J.
        state = np.stack([mixing_ratio * rhoj, rhoj], axis=-1)
        edge = np.array([mixing_ratio * 2.0, 2.0])
        bcon = BoundaryConditions(
            *(
                np.broadcast_to(edge, (n, cfg.nlays, 2))
                for n in (cfg.nrows, cfg.nrows, cfg.ncols, cfg.ncols)
            )
        )
        weather = Meteorology(uhat=uhat, vhat=vhat, rhoj_met=rhoj, bcon=bcon)

        limits = StepLimits()
        schedule = advstep(
            wind_index(uhat, vhat, cfg.dx1, cfg.dx2),
            np.zeros(cfg.nlays),
            3600,
            limits,
            sync_layers=sync_top_layer(faces, limits.sigma_sync_top),
        )

        step = jax.jit(
            partial(
                advect_step,
                mesh=nonuniform_mesh(cfg.ds),
                cfg=cfg,
                astep_seconds=schedule.astep_seconds,
                sync_seconds=schedule.sync_seconds,
                xyfirst=(True,) * cfg.nlays,
            )
        )
        advected, diagnostics = step(state, weather)
        advected = np.asarray(advected)

        assert np.all(np.isfinite(advected))
        assert np.all(np.isfinite(np.asarray(diagnostics.residual))), (
            "a column exhausted its vertical sub-steps on read meteorology"
        )
        np.testing.assert_allclose(advected[..., 0] / advected[..., -1], mixing_ratio, rtol=1e-9)

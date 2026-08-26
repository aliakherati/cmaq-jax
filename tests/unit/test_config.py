"""A0.2 — configuration objects and the constants they carry."""

import numpy as np
import pytest

from cmaq_jax.config import (
    DEFAULT_PPM,
    GridConfig,
    PPMConstants,
    sigma_layer_thickness,
)


class TestPPMConstants:
    def test_defaults_match_fortran(self) -> None:
        """Values as written in reference/fortran/. See field docstrings."""
        c = DEFAULT_PPM
        assert c.two_thirds == pytest.approx(2.0 / 3.0)  # hppm.F:169
        assert c.sixth == pytest.approx(1.0 / 6.0)  # hppm.F:170
        assert c.halo_width == 3  # hppm.F:147, SWP
        assert c.zfdbc_small_wind == pytest.approx(1.0e-3)  # zfdbc.f:29
        assert c.velocity_flux_tolerance == pytest.approx(1.0e-3)  # vppm.F:145, EPSF
        assert c.cfl_safety == pytest.approx(0.9)  # zadvppmwrf.F:393
        assert c.min_substep_seconds == pytest.approx(1.0)  # zadvppmwrf.F:426
        assert c.max_substeps == 30  # zadvppmwrf.F:126, MAXITER

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            DEFAULT_PPM.sixth = 0.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"halo_width": 2}, "halo_width"),
            ({"velocity_adjust_iterations": 0}, "velocity_adjust_iterations"),
            ({"max_substeps": 0}, "max_substeps"),
            ({"cfl_safety": 0.0}, "cfl_safety"),
            ({"cfl_safety": 1.5}, "cfl_safety"),
        ],
    )
    def test_rejects_invalid(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            PPMConstants(**kwargs)  # type: ignore[arg-type]


class TestSigmaLayerThickness:
    def test_matches_fortran_definition(self) -> None:
        """DS(L) = ABS(X3FACE_GD(L) - X3FACE_GD(L-1)), zadvppmwrf.F:246."""
        # CMAQ sigma faces run 1.0 at the surface down to 0.0 at the top.
        faces = np.array([1.0, 0.98, 0.94, 0.86, 0.70, 0.0])
        ds = sigma_layer_thickness(faces)
        np.testing.assert_allclose(ds, [0.02, 0.04, 0.08, 0.16, 0.70])

    def test_positive_regardless_of_face_ordering(self) -> None:
        """CMAQ takes an absolute value, so ascending faces work too."""
        descending = sigma_layer_thickness(np.array([1.0, 0.6, 0.0]))
        ascending = sigma_layer_thickness(np.array([0.0, 0.4, 1.0]))
        np.testing.assert_allclose(descending, [0.4, 0.6])
        np.testing.assert_allclose(ascending, [0.4, 0.6])

    def test_thicknesses_sum_to_full_depth(self) -> None:
        faces = np.array([1.0, 0.9, 0.5, 0.0])
        assert sigma_layer_thickness(faces).sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [np.array([1.0]), np.array([[1.0, 0.0]])])
    def test_rejects_bad_shape(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError, match="x3face"):
            sigma_layer_thickness(bad)


def make_grid(**overrides: object) -> GridConfig:
    kwargs: dict[str, object] = {
        "ncols": 4,
        "nrows": 3,
        "ds": sigma_layer_thickness(np.array([1.0, 0.9, 0.6, 0.0])),
        "dx1": 12000.0,
        "dx2": 12000.0,
        "nspc_adv": 5,
    }
    kwargs.update(overrides)
    return GridConfig(**kwargs)  # type: ignore[arg-type]


class TestGridConfig:
    def test_nlays_derived_from_ds(self) -> None:
        assert make_grid().nlays == 3

    def test_rhoj_is_the_last_species_slot(self) -> None:
        """x_ppm.F:312 — ADV_MAP(N_SPC_ADV) = RHOJ_LOC."""
        grid = make_grid(nspc_adv=5)
        assert grid.rhoj_index == 4

    def test_uniform_ds_detection(self) -> None:
        uniform = make_grid(ds=np.array([0.25, 0.25, 0.25, 0.25]))
        stretched = make_grid(ds=np.array([0.05, 0.15, 0.30, 0.50]))
        assert uniform.uniform_ds
        assert not stretched.uniform_ds

    def test_ds_coerced_to_float64_array(self) -> None:
        grid = make_grid(ds=[0.5, 0.5])
        assert isinstance(grid.ds, np.ndarray)
        assert grid.ds.dtype == np.float64

    def test_carries_ppm_constants(self) -> None:
        assert make_grid().ppm.halo_width == 3

    def test_default_dtype_is_float64(self) -> None:
        """We deliberately run wider than the Fortran; see README deviations."""
        assert make_grid().dtype == "float64"

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"ncols": 0}, "ncols"),
            ({"nrows": 0}, "nrows"),
            ({"dx1": 0.0}, "dx1"),
            ({"dx2": -1.0}, "dx2"),
            ({"nspc_adv": 0}, "rho"),
            ({"ds": np.array([0.5, 0.0])}, "positive"),
            ({"ds": np.array([[0.5, 0.5]])}, "ds must be 1-D"),
        ],
    )
    def test_rejects_invalid(self, overrides: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            make_grid(**overrides)

"""A0.4 — the committed Fortran goldens are well-formed and behave like PPM.

These tests read ``data/goldens/*.npz`` and need no Fortran toolchain. They do
not test the JAX port (that arrives in A0.5-A0.7); they guard the *harness*, so
that a mis-specified case or a broken binary round-trip is caught before any
port is compared against it.

The properties asserted here are the ones PPM guarantees by construction:
monotonicity, positivity, and constancy preservation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"
SWP = 3

pytestmark = pytest.mark.goldens


def _names(prefix: str) -> list[str]:
    return sorted(p.stem for p in GOLDENS.glob(f"{prefix}_*.npz"))


def _load(name: str) -> dict[str, Any]:
    with np.load(GOLDENS / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


HPPM_CASES = _names("hppm")
VPPM_CASES = _names("vppm")


def _is_nondivergent(name: str) -> bool:
    """True when the face velocities are uniform, so the flow neither
    converges nor diverges anywhere in the row."""
    return bool(np.ptp(_load(name)["vel_in"]) == 0.0)


NONDIVERGENT_HPPM_CASES = [n for n in HPPM_CASES if _is_nondivergent(n)]


def test_goldens_exist() -> None:
    """A missing goldens directory means the whole regression suite is vacuous."""
    assert HPPM_CASES, f"no hppm goldens in {GOLDENS}; run scripts/generate_goldens.py"
    assert VPPM_CASES, f"no vppm goldens in {GOLDENS}; run scripts/generate_goldens.py"
    assert NONDIVERGENT_HPPM_CASES, "no non-divergent case to test monotonicity against"
    assert len(NONDIVERGENT_HPPM_CASES) < len(HPPM_CASES), "no divergent case in the suite"


class TestHppmGoldens:
    @pytest.mark.parametrize("name", HPPM_CASES)
    def test_shapes_and_dtype(self, name: str) -> None:
        g = _load(name)
        ni, nspcs = int(g["ni"]), int(g["nspcs"])
        assert g["con_in"].shape == (ni + 2 * SWP, nspcs)
        assert g["con_out"].shape == (ni + 2 * SWP, nspcs)
        assert g["vel_in"].shape == (ni + 1,)
        for key in ("f_lo_in", "f_lo_out", "f_hi_in", "f_hi_out"):
            assert g[key].shape == (nspcs,)
        # CMAQ's default REAL. Goldens must not drift to float64.
        assert g["con_out"].dtype == np.float32

    @pytest.mark.parametrize("name", HPPM_CASES)
    def test_output_is_finite(self, name: str) -> None:
        assert np.all(np.isfinite(_load(name)["con_out"]))

    @pytest.mark.parametrize("name", HPPM_CASES)
    def test_halo_untouched(self, name: str) -> None:
        """HPPM updates cells 1..NI only; the caller refills the halo.

        hppm.F:443 loops ``DO I = 1, NI``. If a golden shows a modified ghost
        cell, the harness is writing outside the interior.
        """
        g = _load(name)
        np.testing.assert_array_equal(g["con_out"][:SWP], g["con_in"][:SWP])
        np.testing.assert_array_equal(g["con_out"][-SWP:], g["con_in"][-SWP:])

    @pytest.mark.parametrize("name", NONDIVERGENT_HPPM_CASES)
    def test_monotonic_no_new_extrema(self, name: str) -> None:
        """PPM's limiter (Colella & Woodward eqs. 1.8, 1.10) forbids new extrema.

        Restricted to non-divergent winds, and deliberately so. The advected
        variable is in coupled units (rho*J*q), so under a divergent wind a cell
        legitimately falls below the global input minimum -- mass really does
        leave through both faces and rho*J itself drops. Applying a
        no-new-extrema bound to the coupled variable there would be asserting
        something false about the physics, not about the scheme. What survives
        divergence is constancy of the mixing ratio, which
        ``test_constancy_preservation`` covers.
        """
        g = _load(name)
        interior = g["con_out"][SWP:-SWP]
        lo = g["con_in"].min(axis=0)
        hi = g["con_in"].max(axis=0)
        tol = 1e-5 * np.maximum(np.abs(hi), 1.0)
        assert np.all(interior >= lo - tol), f"{name}: undershoot"
        assert np.all(interior <= hi + tol), f"{name}: overshoot"

    @pytest.mark.parametrize("name", HPPM_CASES)
    def test_positivity(self, name: str) -> None:
        """Non-negative in, non-negative out. All cases start non-negative."""
        g = _load(name)
        assert np.all(g["con_in"] >= 0.0), f"{name}: test case is not non-negative"
        assert np.all(g["con_out"][SWP:-SWP] >= 0.0)

    def test_zero_wind_is_exactly_unchanged(self) -> None:
        g = _load("hppm_zero_wind")
        assert np.all(g["vel_in"] == 0.0)
        np.testing.assert_array_equal(g["con_out"], g["con_in"])

    def test_row_and_column_orientation_agree(self) -> None:
        """ORI only selects which boundary-PE query is made; with the same
        inputs the numbers must be identical."""
        col = _load("hppm_smooth_positive_wind")
        row = _load("hppm_row_orientation")
        np.testing.assert_array_equal(col["con_in"], row["con_in"])
        np.testing.assert_array_equal(col["con_out"], row["con_out"])

    def test_constancy_preservation(self) -> None:
        """The CMAQ-specific invariant.

        With the state in coupled units (slot s = rho*J*q_s, last slot = rho*J)
        a uniform mixing ratio must survive a divergent wind. This is what the
        rho*J ride-along buys, and the test most likely to catch a coupling
        error in the port.
        """
        g = _load("hppm_constancy_divergent_wind")
        con_in = g["con_in"][SWP:-SWP]
        con_out = g["con_out"][SWP:-SWP]
        rhoj_in, rhoj_out = con_in[:, -1], con_out[:, -1]

        # The wind must actually diverge, or the test proves nothing.
        vel = g["vel_in"]
        assert np.ptp(vel) > 1.0, "constancy case needs a divergent wind"

        for spc in range(con_in.shape[1] - 1):
            q_in = con_in[:, spc] / rhoj_in
            assert np.allclose(q_in, q_in[0], rtol=1e-6), "input q is not uniform"
            q_out = con_out[:, spc] / rhoj_out
            np.testing.assert_allclose(q_out, q_in[0], rtol=1e-6)


class TestVppmGoldens:
    @pytest.mark.parametrize("name", VPPM_CASES)
    def test_shapes_and_dtype(self, name: str) -> None:
        g = _load(name)
        ni, nspcs = int(g["ni"]), int(g["nspcs"])
        assert g["con_in"].shape == (ni, nspcs)
        assert g["con_out"].shape == (ni, nspcs)
        assert g["ds"].shape == (ni,)
        assert g["flx_in"].shape == (ni + 1,)
        assert g["vel_in"].shape == (ni + 1,)
        assert g["vel_out"].shape == (ni + 1,)
        assert g["con_out"].dtype == np.float32

    @pytest.mark.parametrize("name", VPPM_CASES)
    def test_output_is_finite(self, name: str) -> None:
        g = _load(name)
        assert np.all(np.isfinite(g["con_out"]))
        assert np.all(np.isfinite(g["vel_out"]))

    @pytest.mark.parametrize("name", VPPM_CASES)
    def test_surface_is_impermeable(self, name: str) -> None:
        """zadvppmwrf.F:339 sets VEL(1) = FLX(1) = 0; nothing crosses the ground."""
        g = _load(name)
        assert g["flx_in"][0] == 0.0
        assert g["vel_in"][0] == 0.0
        assert g["vel_out"][0] == 0.0

    @pytest.mark.parametrize("name", VPPM_CASES)
    def test_velocity_adjustment_is_small(self, name: str) -> None:
        """vppm.F:200-246 nudges the face velocity so the PPM flux of the rho*J
        column matches the donor-cell flux. Inputs built by ``_upwind_velocity``
        are already nearly consistent, so the correction should be a fine
        adjustment -- a large one means the case is badly posed."""
        g = _load(name)
        scale = max(float(np.abs(g["vel_in"]).max()), 1e-30)
        assert np.abs(g["vel_out"] - g["vel_in"]).max() <= 0.1 * scale

    @pytest.mark.parametrize("name", VPPM_CASES)
    def test_positivity(self, name: str) -> None:
        g = _load(name)
        assert np.all(g["con_in"] >= 0.0), f"{name}: test case is not non-negative"
        assert np.all(g["con_out"] >= 0.0)

    def test_zero_flux_is_exactly_unchanged(self) -> None:
        g = _load("vppm_zero_flux")
        assert np.all(g["flx_in"] == 0.0)
        np.testing.assert_array_equal(g["con_out"], g["con_in"])
        np.testing.assert_array_equal(g["vel_out"], g["vel_in"])

    def test_stretched_grid_is_actually_stretched(self) -> None:
        """The non-uniform mesh coefficients (vppm.F:450-468) are only
        exercised if ds really varies."""
        ds = _load("vppm_smooth_stretched_ds")["ds"]
        assert ds.max() / ds.min() > 3.0

    def test_constancy_preservation(self) -> None:
        g = _load("vppm_constancy_coupled")
        con_in, con_out = g["con_in"], g["con_out"]
        rhoj_in, rhoj_out = con_in[:, -1], con_out[:, -1]
        for spc in range(con_in.shape[1] - 1):
            q_in = con_in[:, spc] / rhoj_in
            assert np.allclose(q_in, q_in[0], rtol=1e-6), "input q is not uniform"
            q_out = con_out[:, spc] / rhoj_out
            np.testing.assert_allclose(q_out, q_in[0], rtol=1e-6)

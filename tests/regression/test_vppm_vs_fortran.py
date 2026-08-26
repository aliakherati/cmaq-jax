"""A2.2/A2.3 — the JAX vertical column solve against the Fortran.

The ``vppm_*`` goldens capture both outputs of ``VPPM``: the advected column and
the **adjusted face velocities**. The second is not incidental — ``vppm.F``
rescales each face velocity until its PPM flux matches the mass flux diagnosed
from the rho*J budget, and that corrected velocity is what every species is then
advected with.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from atmos_jax_common.real4 import downcast_to_real4

from cmaq_jax.config import DEFAULT_PPM
from cmaq_jax.ppm import nonuniform_mesh, ppm_parabola_nonuniform
from cmaq_jax.vadv import vppm, vppm_adjust_velocity

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

RTOL = 1e-6
ATOL = 1e-7

PRECISIONS = [np.float32, np.float64]

pytestmark = pytest.mark.goldens

CASES = sorted(p.stem for p in GOLDENS.glob("vppm_*.npz"))


def _load(name: str) -> dict[str, Any]:
    with np.load(GOLDENS / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _run(golden: dict[str, Any], dtype: type = np.float64):
    ds = np.asarray(golden["ds"], dtype=dtype)
    return vppm(
        np.asarray(golden["con_in"], dtype=dtype),
        np.asarray(golden["vel_in"], dtype=dtype),
        np.asarray(golden["flx_in"], dtype=dtype),
        ds,
        nonuniform_mesh(ds),
        dt=dtype(golden["dt"]),
    )


def test_cases_present() -> None:
    assert CASES, f"no vppm goldens in {GOLDENS}; run scripts/generate_goldens.py"


@pytest.mark.parametrize("dtype", PRECISIONS, ids=["f32", "f64"])
@pytest.mark.parametrize("name", CASES)
def test_concentrations_match_fortran(name: str, dtype: type) -> None:
    golden = _load(name)
    con, _ = _run(golden, dtype)
    expected = np.asarray(golden["con_out"], dtype=np.float64)
    scale = max(float(np.abs(expected).max()), 1.0)
    np.testing.assert_allclose(
        downcast_to_real4(np.asarray(con)), expected, rtol=RTOL, atol=ATOL * scale
    )


@pytest.mark.parametrize("dtype", PRECISIONS, ids=["f32", "f64"])
@pytest.mark.parametrize("name", CASES)
def test_adjusted_velocity_matches_fortran(name: str, dtype: type) -> None:
    """The velocity is an output, not just an input. Getting the concentrations
    right with the wrong velocity would mean the flux matching was being
    compensated for somewhere else."""
    golden = _load(name)
    _, adjusted = _run(golden, dtype)
    expected = np.asarray(golden["vel_out"], dtype=np.float64)
    scale = max(float(np.abs(expected).max()), 1e-30)
    np.testing.assert_allclose(
        downcast_to_real4(np.asarray(adjusted.vel)), expected, rtol=1e-5, atol=1e-6 * scale
    )


@pytest.mark.parametrize("name", CASES)
def test_adjustment_converges(name: str) -> None:
    """Every face CMAQ adjusts must reach its own tolerance within the fixed
    iteration count. A residual above ``EPSF`` is what ``M3EXIT`` would have
    been raised for."""
    _, adjusted = _run(_load(name))
    worst = float(np.asarray(adjusted.residual).max())
    assert worst <= DEFAULT_PPM.velocity_flux_tolerance, f"{name}: residual {worst:.2e}"


@pytest.mark.parametrize("name", CASES)
def test_converges_well_inside_the_iteration_cap(name: str) -> None:
    """Half the configured iterations should already suffice.

    The update is a sqrt-Newton on a near-quadratic, so it converges fast. If
    this starts failing, the fixed count is closer to the edge than intended
    and the margin over CMAQ's cap of 50 has quietly gone.
    """
    golden = _load(name)
    ds = np.asarray(golden["ds"], dtype=np.float64)
    halved = replace(
        DEFAULT_PPM, velocity_adjust_iterations=DEFAULT_PPM.velocity_adjust_iterations // 2
    )
    density = ppm_parabola_nonuniform(
        np.asarray(golden["con_in"], dtype=np.float64)[:, -1], nonuniform_mesh(ds)
    )
    adjusted = vppm_adjust_velocity(
        np.asarray(golden["vel_in"], dtype=np.float64),
        np.asarray(golden["flx_in"], dtype=np.float64),
        density,
        ds,
        dt=float(golden["dt"]),
        ppm=halved,
    )
    worst = float(np.asarray(adjusted.residual).max())
    assert worst <= DEFAULT_PPM.velocity_flux_tolerance, (
        f"{name}: residual {worst:.2e} at half iterations"
    )


def test_impermeable_ground() -> None:
    """zadvppmwrf.F:339 pins VEL(1) = FLX(1) = 0, so nothing crosses the surface
    and the bottom face must stay exactly at rest."""
    for name in CASES:
        golden = _load(name)
        _, adjusted = _run(golden)
        assert float(np.asarray(adjusted.vel)[0]) == 0.0, name


def test_zero_flux_column_is_untouched() -> None:
    golden = _load("vppm_zero_flux")
    con, adjusted = _run(golden)
    np.testing.assert_array_equal(np.asarray(con), golden["con_in"].astype(np.float64))
    np.testing.assert_array_equal(np.asarray(adjusted.vel), golden["vel_in"].astype(np.float64))


def test_constancy_through_the_column() -> None:
    """A uniform mixing ratio survives vertical transport.

    Built in float64 here rather than reused from the golden, whose float32
    input is only uniform to ~1e-7.
    """
    nlays = 12
    ds = np.diff(np.linspace(0.0, 1.0, nlays + 1) ** 1.7)
    rhoj = 1.0 + 0.5 * np.linspace(1.0, 0.2, nlays)
    q = np.array([0.75, 2.0, 0.5])
    con = np.stack([qq * rhoj for qq in q] + [rhoj], axis=-1)

    flx = 2.0e-4 * np.sin(np.pi * np.arange(nlays + 1) / nlays)
    flx[0] = 0.0
    vel = np.zeros(nlays + 1)
    for lvl in range(1, nlays):
        vel[lvl] = flx[lvl] / (rhoj[lvl - 1] if flx[lvl] >= 0.0 else rhoj[lvl])
    vel[nlays] = flx[nlays] / rhoj[nlays - 1]

    out, _ = vppm(con, vel, flx, ds, nonuniform_mesh(ds), dt=60.0)
    out = np.asarray(out)
    for spc, q_expected in enumerate(q):
        np.testing.assert_allclose(out[:, spc] / out[:, -1], q_expected, rtol=1e-9)

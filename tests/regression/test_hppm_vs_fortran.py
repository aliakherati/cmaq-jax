"""A0.5/A0.7 — the JAX uniform-spacing PPM sweep against the Fortran goldens.

The port runs in float64 while CMAQ runs in float32, so a bit-for-bit match is
not the bar. Comparison is done after downcasting the JAX result to float32 via
``atmos_jax_common.real4``, at a tolerance that leaves no room for a real
numerical difference to hide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from atmos_jax_common.real4 import downcast_to_real4

from cmaq_jax.config import DEFAULT_PPM
from cmaq_jax.ppm import ppm_advect_uniform

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"
SWP = DEFAULT_PPM.halo_width

# float32 has ~7 decimal digits. Anything above this is a genuine disagreement,
# not accumulated rounding across the handful of operations PPM performs.
RTOL = 1e-6
ATOL = 1e-6

pytestmark = pytest.mark.goldens

CASES = sorted(p.stem for p in GOLDENS.glob("hppm_*.npz"))


def _load(name: str) -> dict[str, Any]:
    with np.load(GOLDENS / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _run(golden: dict[str, Any]) -> np.ndarray:
    """Advect the golden's input with the JAX kernel, in float64."""
    con = np.asarray(golden["con_in"], dtype=np.float64)
    # vel varies along the sweep axis only; add a species axis to broadcast.
    vel = np.asarray(golden["vel_in"], dtype=np.float64)[:, None]
    out = ppm_advect_uniform(con, vel, float(golden["dt"]), float(golden["ds"]))
    return np.asarray(out)


def test_cases_present() -> None:
    assert CASES, f"no hppm goldens in {GOLDENS}; run scripts/generate_goldens.py"


@pytest.mark.parametrize("name", CASES)
def test_matches_fortran(name: str) -> None:
    golden = _load(name)
    got = downcast_to_real4(_run(golden))
    expected = np.asarray(golden["con_out"], dtype=np.float64)

    # Scale the tolerance by the field magnitude; concentrations here are O(1)
    # but the API should not depend on that.
    scale = max(float(np.abs(expected).max()), 1.0)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL * scale)


@pytest.mark.parametrize("name", CASES)
def test_halo_untouched(name: str) -> None:
    """hppm.F:443 updates cells 1..NI only. Refilling the halo is the caller's
    job, and the port must not quietly do it here."""
    golden = _load(name)
    got = _run(golden)
    con_in = np.asarray(golden["con_in"], dtype=np.float64)
    np.testing.assert_array_equal(got[:SWP], con_in[:SWP])
    np.testing.assert_array_equal(got[-SWP:], con_in[-SWP:])


def test_zero_wind_is_exactly_unchanged() -> None:
    """No transport at all: not merely close, but identical."""
    golden = _load("hppm_zero_wind")
    con_in = np.asarray(golden["con_in"], dtype=np.float64)
    np.testing.assert_array_equal(_run(golden), con_in)


def test_constancy_preservation() -> None:
    """A uniform mixing ratio survives a divergent wind.

    The state is in coupled units (slot s = rho*J*q_s, last slot = rho*J), so
    this only holds because rho*J is advected by the same scheme. It is the
    test most likely to catch a coupling error.

    Inputs are built here in float64 rather than reused from the golden. The
    golden's ``con_in`` is float32, so the mixing ratio it encodes is itself
    only uniform to ~1e-7 -- testing against it would cap the tolerance at the
    Fortran's precision instead of the port's.
    """
    ni, swp = 24, SWP
    x = np.arange(ni, dtype=np.float64)
    rhoj = 1.0 + 0.4 * np.sin(2.0 * np.pi * x / ni)
    q = np.array([0.75, 2.0])

    interior = np.stack([*(qq * rhoj for qq in q), rhoj], axis=1)
    con = np.concatenate(
        [np.repeat(interior[:1], swp, axis=0), interior, np.repeat(interior[-1:], swp, axis=0)]
    )
    vel = 120.0 * np.sin(2.0 * np.pi * np.arange(ni + 1, dtype=np.float64) / (ni + 1))
    assert np.ptp(vel) > 1.0, "constancy needs a divergent wind to mean anything"

    out = np.asarray(ppm_advect_uniform(con, vel[:, None], 60.0, 12000.0))[swp:-swp]

    for spc, q_expected in enumerate(q):
        np.testing.assert_allclose(out[:, spc] / out[:, -1], q_expected, rtol=1e-12)


def test_single_species_matches_multi_species() -> None:
    """The species axis is inert: advecting one species alone must give the
    same answer as advecting it alongside others."""
    multi = _load("hppm_smooth_positive_wind")
    single = _load("hppm_single_species")
    np.testing.assert_array_equal(single["con_in"][:, 0], multi["con_in"][:, 0])
    np.testing.assert_allclose(_run(single)[:, 0], _run(multi)[:, 0], rtol=1e-12)


def test_accepts_extra_trailing_axes() -> None:
    """A full grid sweep carries (rows, layers, species) behind the sweep axis.

    Broadcasting them must give the same answer as looping, which is what makes
    the whole-array formulation valid.
    """
    golden = _load("hppm_smooth_positive_wind")
    con = np.asarray(golden["con_in"], dtype=np.float64)
    vel = np.asarray(golden["vel_in"], dtype=np.float64)

    ncells, nspcs = con.shape
    nrows = 4
    # Stack the same row several times with a distinct scale factor each.
    scales = np.array([1.0, 2.0, 0.5, 3.0])
    stacked = con[:, None, :] * scales[None, :, None]

    got = np.asarray(
        ppm_advect_uniform(
            stacked,
            vel[:, None, None],
            float(golden["dt"]),
            float(golden["ds"]),
        )
    )
    assert got.shape == (ncells, nrows, nspcs)

    # PPM is homogeneous of degree one, so scaling the input scales the output.
    reference = _run(golden)
    for row, scale in enumerate(scales):
        np.testing.assert_allclose(got[:, row, :], reference * scale, rtol=1e-12)

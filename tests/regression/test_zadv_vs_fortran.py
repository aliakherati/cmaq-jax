"""A2.1/A2.4 — the JAX vertical driver against the Fortran.

These goldens come from running ``zadvppmwrf.F`` and ``vppm.F`` unmodified, with
only the meteorology source stubbed. They pin what the column solve alone does
not: the flux diagnosed from the density budget, and the per-column CFL
sub-stepping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from atmos_jax_common.real4 import downcast_to_real4

from cmaq_jax.config import sigma_layer_thickness
from cmaq_jax.ppm import nonuniform_mesh
from cmaq_jax.vadv import diagnose_flux, face_velocity_from_flux, zadv

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

RTOL = 1e-6
ATOL = 1e-7

PRECISIONS = [np.float32, np.float64]

pytestmark = pytest.mark.goldens

CASES = sorted(p.stem for p in GOLDENS.glob("zadv_*.npz"))


def _load(name: str) -> dict[str, Any]:
    with np.load(GOLDENS / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _seconds(hhmmss: int) -> int:
    v = int(hhmmss)
    return (v // 10000) * 3600 + (v // 100 % 100) * 60 + v % 100


def _run(golden: dict[str, Any], dtype: type = np.float64):
    """CMAQ stores CGRID layer-third; the port works layer-first."""
    ds = sigma_layer_thickness(np.asarray(golden["faces"], dtype=np.float64))
    con = np.moveaxis(np.asarray(golden["cgrid_in"], dtype=dtype), 2, 0)
    met = np.moveaxis(np.asarray(golden["rhoj_met"], dtype=dtype), 2, 0)
    out, diag = zadv(
        con,
        met,
        ds.astype(dtype),
        nonuniform_mesh(ds.astype(dtype)),
        dt=float(_seconds(golden["tstep"][1])),
    )
    return np.moveaxis(np.asarray(out), 0, 2), diag


def test_cases_present() -> None:
    assert CASES, f"no zadv goldens in {GOLDENS}; run scripts/generate_goldens.py"


@pytest.mark.parametrize("dtype", PRECISIONS, ids=["f32", "f64"])
@pytest.mark.parametrize("name", CASES)
def test_matches_fortran(name: str, dtype: type) -> None:
    golden = _load(name)
    got, _ = _run(golden, dtype)
    expected = np.asarray(golden["cgrid_out"], dtype=np.float64)
    scale = max(float(np.abs(expected).max()), 1.0)
    np.testing.assert_allclose(downcast_to_real4(got), expected, rtol=RTOL, atol=ATOL * scale)


def test_the_suite_covers_both_cfl_regimes() -> None:
    """A suite that never exceeds Courant 1 would leave the sub-stepping loop
    entirely untested, and it would still pass."""
    counts = {name: int(np.asarray(_run(_load(name))[1].substeps).max()) for name in CASES}
    assert min(counts.values()) == 1, f"no single-pass case: {counts}"
    assert max(counts.values()) > 1, f"no sub-stepped case: {counts}"


@pytest.mark.parametrize("name", CASES)
def test_every_column_finishes(name: str) -> None:
    """A column with time left over never completed its sync step. The
    diagnostic reports that as an infinite residual, which is the fixed-count
    loop's stand-in for CMAQ's M3EXIT."""
    _, diag = _run(_load(name))
    assert np.all(np.isfinite(np.asarray(diag.residual))), "a column ran out of sub-steps"


def test_substep_count_tracks_the_courant_number() -> None:
    """More overshoot means more splitting. If these ever decouple, the
    sub-step length is not following the CFL limit."""
    ordered = [
        (float(np.asarray(d.max_courant).max()), int(np.asarray(d.substeps).max()))
        for d in (
            _run(_load(n))[1]
            for n in ("zadv_gentle_stretched", "zadv_substepped", "zadv_heavily_substepped")
        )
    ]
    courants = [c for c, _ in ordered]
    steps = [s for _, s in ordered]
    assert courants == sorted(courants), courants
    assert steps == sorted(steps), steps
    assert courants[0] < 1.0 <= courants[1]


def test_no_mismatch_means_no_transport() -> None:
    """When the transported density already equals the meteorology there is
    nothing for the vertical flux to correct, so the column must not move."""
    golden = _load("zadv_no_mismatch")
    ds = sigma_layer_thickness(np.asarray(golden["faces"], dtype=np.float64))
    con = np.moveaxis(np.asarray(golden["cgrid_in"], dtype=np.float64), 2, 0)
    met = np.moveaxis(np.asarray(golden["rhoj_met"], dtype=np.float64), 2, 0)

    flx = np.asarray(diagnose_flux(met, con[..., -1], ds, 180.0))
    assert np.abs(flx).max() < 1e-12, "a matched column should diagnose no flux"

    got, _ = _run(golden)
    np.testing.assert_allclose(got, golden["cgrid_in"].astype(np.float64), rtol=1e-12)


def test_ground_is_impermeable() -> None:
    """zadvppmwrf.F:341 pins FLX(1) = 0 and the velocity follows."""
    for name in CASES:
        golden = _load(name)
        ds = sigma_layer_thickness(np.asarray(golden["faces"], dtype=np.float64))
        con = np.moveaxis(np.asarray(golden["cgrid_in"], dtype=np.float64), 2, 0)
        met = np.moveaxis(np.asarray(golden["rhoj_met"], dtype=np.float64), 2, 0)
        flx = np.asarray(diagnose_flux(met, con[..., -1], ds, 180.0))
        vel = np.asarray(face_velocity_from_flux(flx, con[..., -1]))
        np.testing.assert_array_equal(flx[0], np.zeros_like(flx[0]), err_msg=name)
        np.testing.assert_array_equal(vel[0], np.zeros_like(vel[0]), err_msg=name)


def test_constancy_through_the_column() -> None:
    """A uniform mixing ratio survives the whole vertical operator.

    Built in float64 rather than reused from the golden, whose float32 input is
    only uniform to ~1e-7.
    """
    nlays = 12
    ds = sigma_layer_thickness(np.linspace(1.0, 0.0, nlays + 1) ** 0.625)
    rng = np.random.default_rng(20260905)
    rhoj = 1.5 + 0.4 * rng.random((nlays, 3, 2))
    met = rhoj * (1.0 + 0.15 * np.sin(np.linspace(0.0, 2.0 * np.pi, nlays))[:, None, None])
    q = np.array([0.75, 3.0])
    con = np.stack([*(qq * rhoj for qq in q), rhoj], axis=-1)

    out, _ = zadv(con, met, ds, nonuniform_mesh(ds), dt=180.0)
    out = np.asarray(out)
    for spc, q_expected in enumerate(q):
        np.testing.assert_allclose(out[..., spc] / out[..., -1], q_expected, rtol=1e-9)


@pytest.mark.parametrize("name", CASES)
def test_positivity(name: str) -> None:
    got, _ = _run(_load(name))
    assert np.all(got >= 0.0), f"{name}: negative concentration"

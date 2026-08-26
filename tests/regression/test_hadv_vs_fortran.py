"""A1.3-A1.5 — the JAX horizontal-advection driver against the Fortran.

These goldens come from running CMAQ's whole chain unmodified -- ``hadvppm.F``,
``x_ppm.F``, ``y_ppm.F``, ``hcontvel.F``, ``zfdbc.f``, ``hppm.F`` -- with only
the data sources stubbed. They therefore pin the three things the sweep alone
cannot: per-layer sub-stepping, the X-Y/Y-X alternation, and layer independence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from atmos_jax_common.real4 import downcast_to_real4

from cmaq_jax.config import GridConfig
from cmaq_jax.hadv import BoundaryConditions, hadv
from cmaq_jax.velocity import face_velocity

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

# Measured worst case across all eight driver cases is 1.05e-7, under one
# float32 ULP, so the whole chain agrees with CMAQ to the reference's own
# precision. 1e-6 leaves ~10x headroom while staying tight enough to catch a
# real regression.
RTOL = 1e-6
ATOL = 1e-7

pytestmark = pytest.mark.goldens

CASES = sorted(p.stem for p in GOLDENS.glob("hadv_*.npz"))


def _load(name: str) -> dict[str, Any]:
    with np.load(GOLDENS / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _hhmmss_to_seconds(value: int) -> int:
    value = int(value)
    return (value // 10000) * 3600 + (value // 100 % 100) * 60 + value % 100


def _split_bcon(flat: np.ndarray, ncols: int, nrows: int) -> BoundaryConditions:
    """Cut CMAQ's flat boundary ring into named edges.

    Offsets from ``x_ppm.F:208`` and ``y_ppm.F:203``, converted from CMAQ's
    1-based ``OFFSET + COL`` / ``OFFSET + ROW`` indexing to Python slices.
    """
    return BoundaryConditions(
        south=flat[0:ncols],
        east=flat[ncols + 1 : ncols + 1 + nrows],
        north=flat[ncols + nrows + 3 : 2 * ncols + nrows + 3],
        west=flat[2 * ncols + nrows + 4 : 2 * ncols + 2 * nrows + 4],
    )


def _run(golden: dict[str, Any]) -> np.ndarray:
    cgrid = np.asarray(golden["cgrid_in"], dtype=np.float64)
    ncols, nrows, nlays, nspc = cgrid.shape

    cfg = GridConfig(
        ncols=ncols,
        nrows=nrows,
        ds=np.full(nlays, 1.0 / nlays),  # unused by HADV; vertical is A2
        dx1=float(golden["xcell"]),
        dx2=float(golden["ycell"]),
        nspc_adv=nspc,
    )

    # C-staggered winds: hcontvel.F returns them unchanged (see velocity.py).
    uhat = face_velocity(np.asarray(golden["uwindc"], dtype=np.float64), axis=0)
    vhat = face_velocity(np.asarray(golden["vwindc"], dtype=np.float64), axis=1)

    bcon = _split_bcon(
        np.asarray(golden["bcon"], dtype=np.float64).transpose(0, 2, 1), ncols, nrows
    )

    astep = np.array([_hhmmss_to_seconds(a) for a in golden["astep"]])
    sync = _hhmmss_to_seconds(golden["tstep"][1])
    # hadvppm.F's XYFIRST is a SAVEd array initialised .TRUE. on first call.
    xyfirst = (True,) * nlays

    for _ in range(int(golden["ncalls"])):
        cgrid, xyfirst = hadv(
            cgrid,
            uhat,
            vhat,
            bcon,
            cfg=cfg,
            astep_seconds=astep,
            sync_seconds=sync,
            xyfirst=xyfirst,
        )
    return np.asarray(cgrid)


def test_cases_present() -> None:
    assert CASES, f"no hadv goldens in {GOLDENS}; run scripts/generate_goldens.py"


@pytest.mark.parametrize("name", CASES)
def test_matches_fortran(name: str) -> None:
    golden = _load(name)
    expected = np.asarray(golden["cgrid_out"], dtype=np.float64)
    got = downcast_to_real4(_run(golden))
    scale = max(float(np.abs(expected).max()), 1.0)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL * scale)


def test_layers_stay_independent() -> None:
    """Advecting a two-layer grid must give the same answer as advecting each
    layer alone. Nothing in horizontal advection couples them, and the
    per-layer ASTEP grouping is exactly where that could break."""
    golden = _load("hadv_mixed_layer_astep")
    full = _run(golden)

    for layer in range(full.shape[2]):
        single = dict(golden)
        single["cgrid_in"] = golden["cgrid_in"][:, :, layer : layer + 1]
        single["uwindc"] = golden["uwindc"][:, :, layer : layer + 1]
        single["vwindc"] = golden["vwindc"][:, :, layer : layer + 1]
        single["bcon"] = golden["bcon"][:, :, layer : layer + 1]
        single["astep"] = golden["astep"][layer : layer + 1]
        np.testing.assert_allclose(_run(single)[:, :, 0], full[:, :, layer], rtol=1e-12)


def test_alternation_is_not_a_repeat() -> None:
    """Two calls sweep X-Y then Y-X. Doing X-Y twice must differ, or the saved
    flag is not being carried between calls."""
    golden = _load("hadv_xy_alternation")
    alternating = _run(golden)

    once = dict(golden)
    once["ncalls"] = np.int32(1)
    first = _run(once)

    repeated = dict(golden)
    repeated["cgrid_in"] = first.astype(np.float32)
    repeated["ncalls"] = np.int32(1)
    repeat_same_order = _run(repeated)

    assert not np.allclose(alternating, repeat_same_order, rtol=1e-6)


def test_constancy_through_both_sweeps() -> None:
    """A uniform mixing ratio survives both sweeps and the boundaries.

    Inputs are built here in float64 rather than reused from the golden. The
    golden's ``cgrid_in`` is float32, so the mixing ratio it encodes is itself
    only uniform to ~1e-7; testing against it would cap the tolerance at the
    Fortran's precision instead of the port's.
    """
    ncols, nrows, nlays = 8, 6, 2
    q = np.array([1.0, 3.0])
    nspc = q.size + 1

    rng = np.random.default_rng(20260901)
    rhoj = 1.5 + 0.4 * rng.random((ncols, nrows, nlays))
    cgrid = np.stack([qq * rhoj for qq in q] + [rhoj], axis=-1)

    rhoj_bndy = 2.0
    edge = np.stack(
        [qq * np.full((nlays,), rhoj_bndy) for qq in q] + [np.full(nlays, rhoj_bndy)], axis=-1
    )
    bcon = BoundaryConditions(
        west=np.broadcast_to(edge, (nrows, nlays, nspc)),
        east=np.broadcast_to(edge, (nrows, nlays, nspc)),
        south=np.broadcast_to(edge, (ncols, nlays, nspc)),
        north=np.broadcast_to(edge, (ncols, nlays, nspc)),
    )

    # Divergent in both directions, or the test is trivially satisfied.
    uhat = 20.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, ncols + 1))[:, None, None]
    uhat = np.broadcast_to(uhat, (ncols + 1, nrows, nlays))
    vhat = 15.0 * np.cos(np.linspace(0.0, 2.0 * np.pi, nrows + 1))[None, :, None]
    vhat = np.broadcast_to(vhat, (ncols, nrows + 1, nlays))
    assert np.ptp(uhat) > 1.0 and np.ptp(vhat) > 1.0

    cfg = GridConfig(
        ncols=ncols,
        nrows=nrows,
        ds=np.full(nlays, 1.0 / nlays),
        dx1=12000.0,
        dx2=12000.0,
        nspc_adv=nspc,
    )

    out, _ = hadv(
        cgrid,
        uhat,
        vhat,
        bcon,
        cfg=cfg,
        astep_seconds=np.full(nlays, 180),
        sync_seconds=180,
        xyfirst=(True,) * nlays,
    )
    out = np.asarray(out)

    for spc, q_expected in enumerate(q):
        np.testing.assert_allclose(out[..., spc] / out[..., -1], q_expected, rtol=1e-11)


def test_positivity_under_outflow_on_every_edge() -> None:
    """Outflow on every edge is where an unclamped boundary extrapolation would
    show up as a negative concentration."""
    assert np.all(_run(_load("hadv_outflow_all_edges")) >= 0.0)

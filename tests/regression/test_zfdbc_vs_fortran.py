"""A1.1 — the JAX zero-flux-divergence boundary condition against the Fortran.

``zfdbc.f`` is stateless and dependency-free, so the goldens cover every branch
of it exhaustively rather than at a handful of sample points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from atmos_jax_common.real4 import downcast_to_real4

from cmaq_jax.bc import zfdbc
from cmaq_jax.config import DEFAULT_PPM

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"
SMALL = DEFAULT_PPM.zfdbc_small_wind

pytestmark = pytest.mark.goldens

CASES = sorted(p.stem for p in GOLDENS.glob("zfdbc_*.npz"))

# CMAQ runs in float32; we default to float64. Both are supported compute paths,
# so every golden comparison runs twice. Measured worst case against the
# goldens: float64-then-downcast and native float32 agree with the Fortran to
# comparable accuracy, and float32 is often closer -- unsurprisingly, since it
# is doing the same arithmetic in the same precision as the reference.
PRECISIONS = [np.float32, np.float64]


def _load(name: str) -> dict[str, Any]:
    with np.load(GOLDENS / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def test_cases_present() -> None:
    assert CASES, f"no zfdbc goldens in {GOLDENS}; run scripts/generate_goldens.py"


@pytest.mark.parametrize("dtype", PRECISIONS, ids=["f32", "f64"])
@pytest.mark.parametrize("name", CASES)
def test_matches_fortran(name: str, dtype: type) -> None:
    g = _load(name)
    result = zfdbc(*(np.asarray(g[k], dtype=dtype) for k in ("c1", "c2", "v1", "v2")))
    assert result.dtype == dtype, f"zfdbc did not preserve {dtype}, gave {result.dtype}"
    got = downcast_to_real4(np.asarray(result))
    np.testing.assert_allclose(got, g["result"].astype(np.float64), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", CASES)
def test_never_negative(name: str) -> None:
    """The Fortran's MAX(0, ...) clamp is what keeps an outflow edge from
    manufacturing negative concentrations."""
    g = _load(name)
    result = np.asarray(zfdbc(g["c1"], g["c2"], g["v1"], g["v2"]))
    assert np.all(result >= 0.0)


@pytest.mark.parametrize("name", CASES)
def test_no_nan_from_the_guarded_division(name: str) -> None:
    """`jnp.where` evaluates both branches, so the v1 division has to stay
    finite even where the result is discarded -- otherwise a nan leaks into the
    gradient."""
    g = _load(name)
    assert np.all(np.isfinite(np.asarray(zfdbc(g["c1"], g["c2"], g["v1"], g["v2"]))))


class TestBranches:
    """The three outcomes of zfdbc.f:32-40, stated directly."""

    def test_small_wind_passes_the_edge_value_through(self) -> None:
        c1 = np.array([2.0, 5.0])
        got = zfdbc(
            c1, np.array([9.0, 0.0]), np.array([SMALL / 2, -SMALL / 2]), np.array([1.0, 1.0])
        )
        np.testing.assert_allclose(np.asarray(got), c1)

    def test_diverging_wind_passes_the_edge_value_through(self) -> None:
        """v1 and v2 of opposite sign means the flow splits at the edge, where
        an extrapolation along the flux has no meaning."""
        c1 = np.array([2.0, 2.0])
        got = zfdbc(c1, np.array([5.0, 5.0]), np.array([1.0, -1.0]), np.array([-1.0, 1.0]))
        np.testing.assert_allclose(np.asarray(got), c1)

    def test_zero_v2_passes_through(self) -> None:
        """v1*v2 == 0 is not > 0, so the Fortran takes the else branch."""
        got = zfdbc(np.array([2.0]), np.array([5.0]), np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(np.asarray(got), [2.0])

    def test_extrapolates_along_the_gradient(self) -> None:
        """c1 - (v2/v1)(c2 - c1): with v1 == v2 this is a plain linear
        extrapolation one cell beyond the edge."""
        got = zfdbc(np.array([2.0]), np.array([3.0]), np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(np.asarray(got), [1.0])  # 2 - 1*(3-2)

    def test_clamps_at_zero(self) -> None:
        """A steep inward gradient would extrapolate negative; the clamp holds
        it at zero."""
        got = zfdbc(np.array([1.0]), np.array([50.0]), np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(np.asarray(got), [0.0])

    def test_threshold_is_inclusive(self) -> None:
        """zfdbc.f uses `ABS(V1) .GE. SMALL`, so exactly SMALL extrapolates."""
        at_threshold = zfdbc(np.array([2.0]), np.array([3.0]), np.array([SMALL]), np.array([SMALL]))
        below = zfdbc(
            np.array([2.0]), np.array([3.0]), np.array([SMALL * 0.999]), np.array([SMALL])
        )
        np.testing.assert_allclose(np.asarray(at_threshold), [1.0])
        np.testing.assert_allclose(np.asarray(below), [2.0])

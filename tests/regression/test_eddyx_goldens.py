"""C1.3 — the vertical eddy diffusivity against its Fortran golden.

`eddyx.F` compiled unmodified against a minimal `ASX_DATA_MOD`
(`reference/harness/stubs_asx.f90`) holding only the twelve met arrays it reads.

The parameterization switches between three regimes, and one random field would
exercise all of them at once while distinguishing none. Each golden isolates
one, so a failure says which branch broke.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmaq_jax.config import DEFAULT_ACM2
from cmaq_jax.vdiff import VerticalMeteorology, eddy_diffusivity

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

#: Measured worst is 3.0 float32 ULPs, on the cloudy case where the moist
#: correction chains several products and quotients.
ULP_BUDGET = 8.0

EPS32 = float(np.finfo(np.float32).eps)

FIELDS = (
    "pbl",
    "ustar",
    "moli",
    "zf",
    "zh",
    "kzmin",
    "thetav",
    "ta",
    "qv",
    "qc",
    "uwind",
    "vwind",
)


def names() -> list[str]:
    found = sorted(p.stem[len("eddyx_") :] for p in GOLDENS.glob("eddyx_*.npz"))
    if not found:
        pytest.skip("no eddyx goldens committed")
    return found


def run(case: np.lib.npyio.NpzFile, dtype: type) -> np.ndarray:
    met = VerticalMeteorology(*(case[k].astype(dtype) for k in FIELDS))
    return np.asarray(eddy_diffusivity(met, c_staggered=bool(case["cstaguv"])))


@pytest.mark.goldens
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", names())
def test_matches_the_fortran(name: str, dtype: type) -> None:
    with np.load(GOLDENS / f"eddyx_{name}.npz") as case:
        got = run(case, dtype)
        expected = case["eddyv"].astype(np.float64)
    worst = float(np.abs(expected - got).max()) / max(float(np.abs(expected).max()), 1.0) / EPS32
    assert worst <= ULP_BUDGET, f"eddyx_{name}: {worst:.2f} float32 ULPs"


@pytest.mark.goldens
class TestPhysics:
    """What the goldens cannot say: that the parameterization is the one it
    claims to be. These are checkable in closed form."""

    def test_the_neutral_surface_layer_is_exact(self) -> None:
        """With ``1/L = 0`` the stability function is exactly 1, so the surface
        term reduces to ``κ·u*·z·(1 − z/h)²`` — no free parameters, no fitting.
        If this does not hold, the port is not this parameterization.
        """
        with np.load(GOLDENS / "eddyx_neutral.npz") as case:
            got = run(case, np.float64)
            zf = case["zf"].astype(np.float64)
            pbl = case["pbl"].astype(np.float64)[..., None]
            ustar = case["ustar"].astype(np.float64)[..., None]

        z = zf[..., :-1]
        expected = DEFAULT_ACM2.karman * ustar * z * (1.0 - z / pbl) ** 2
        below = z < pbl
        # Only where the similarity term wins, which below the PBL it does here.
        np.testing.assert_allclose(got[..., :-1][below], expected[below], rtol=1e-5)

    def test_the_top_layer_has_no_diffusivity(self) -> None:
        """Kz lives on layer interfaces, of which there are ``nlays - 1``."""
        for name in names():
            with np.load(GOLDENS / f"eddyx_{name}.npz") as case:
                assert np.all(case["eddyv"][..., -1] == 0.0), name

    def test_unstable_mixes_more_than_neutral_than_stable(self) -> None:
        """The ordering the stability function exists to produce."""
        values = {}
        for name in ("unstable", "neutral", "stable"):
            with np.load(GOLDENS / f"eddyx_{name}.npz") as case:
                values[name] = float(case["eddyv"].max())
        assert values["stable"] < values["neutral"] < values["unstable"]

    def test_the_floor_is_respected(self) -> None:
        """``KZMIN`` is a floor, not a ceiling — a ``max`` written as a ``min``
        would still produce a plausible-looking field."""
        with np.load(GOLDENS / "eddyx_kzmin_floor.npz") as case:
            eddyv = case["eddyv"]
            kzmin = float(case["kzmin"].max())
            pbl = case["pbl"].astype(np.float64)[..., None]
            z = case["zf"].astype(np.float64)[..., :-1]
        below = z < pbl
        assert eddyv[..., :-1][below].min() >= kzmin - 1e-5

    def test_the_cap_is_respected(self) -> None:
        for name in names():
            with np.load(GOLDENS / f"eddyx_{name}.npz") as case:
                assert case["eddyv"].max() <= DEFAULT_ACM2.eddy_max * (1.0 + 1e-6), name


@pytest.mark.goldens
class TestTheCasesDiscriminate:
    """Guards. A case that cannot tell two code paths apart is not testing one."""

    def test_the_wind_stencils_actually_differ(self) -> None:
        """The C-staggered and B-staggered branches average a different number
        of points. On a *spatially uniform* wind they are algebraically
        identical — ``0.25·(2du)² == (1/16)·(4du)²`` — so the case has to vary
        the wind across the domain or it silently tests nothing.

        It did, at first. This guard is why the case was rebuilt.
        """
        with np.load(GOLDENS / "eddyx_b_staggered.npz") as case:
            b_staggered = case["eddyv"]
        with np.load(GOLDENS / "eddyx_sheared.npz") as case:
            c_staggered = case["eddyv"]
        assert not np.allclose(b_staggered, c_staggered), (
            "the two wind stencils give identical answers, so neither is tested"
        )

    def test_the_moist_correction_changes_something(self) -> None:
        with np.load(GOLDENS / "eddyx_cloudy.npz") as case:
            cloudy = case["eddyv"]
            assert case["qc"].max() > DEFAULT_ACM2.qc_threshold
        with np.load(GOLDENS / "eddyx_sheared.npz") as case:
            dry = case["eddyv"]
        assert not np.allclose(cloudy, dry), "the moist correction did nothing"

    def test_the_richardson_branches_are_both_reached(self) -> None:
        """``Ri >= 0`` uses the ``FH`` polynomial, ``Ri < 0`` a square root.
        Two cases with opposite lapse rates should land on opposite sides."""
        with np.load(GOLDENS / "eddyx_sheared.npz") as case:
            stable = case["eddyv"].max()
        with np.load(GOLDENS / "eddyx_unstable_richardson.npz") as case:
            unstable = case["eddyv"].max()
        assert unstable > 10.0 * stable, (
            f"the two Richardson branches gave similar answers: {stable}, {unstable}"
        )

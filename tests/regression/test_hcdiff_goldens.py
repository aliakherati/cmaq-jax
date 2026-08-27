"""B1 — the diffusivity chain against its Fortran goldens.

Each stage is compared separately (deformation, face coefficients, stable step)
rather than only the end of the chain, so a failure says which stage broke. The
goldens come from `deform.F` and `hcdiff3d.F` compiled unmodified; see
`scripts/generate_goldens.py`.

Both precisions are checked. CMAQ itself runs float32, and the port defaults to
float64 — a stage that only agrees in one of them is not ported, it is tuned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmaq_jax.config import DEFAULT_HDIFF
from cmaq_jax.hdiff import (
    contravariant_winds,
    deformation,
    eddy_diffusivity,
    face_coefficients,
    halo_density,
    stable_timestep,
)

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

#: The goldens are float32 Fortran output, so ULPs are measured against float32.
#: 8 leaves room for the density division in ``variable_density`` -- measured at
#: 5.4 -- while staying far below anything that would indicate a real
#: disagreement. Deformation itself is bit-exact on every case but one.
ULP_BUDGET = 8.0

#: The grid the goldens were generated on (``stubs_driver.f90:38-39``).
DX1 = DX2 = 12000.0

EPS32 = float(np.finfo(np.float32).eps)


def golden_names() -> list[str]:
    names = sorted(p.stem[len("hcdiff_") :] for p in GOLDENS.glob("hcdiff_*.npz"))
    if not names:
        pytest.skip("no hcdiff goldens committed")
    return names


def ulps(expected: np.ndarray, got: np.ndarray) -> float:
    """Worst elementwise difference in float32 ULPs, scaled by array magnitude."""
    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(got, dtype=np.float64)
    scale = max(float(np.abs(left).max()), 1.0)
    return float(np.abs(left - right).max()) / scale / EPS32


def chain(case: np.lib.npyio.NpzFile, dtype: type) -> dict[str, np.ndarray]:
    """Run the whole hcdiff3d chain at ``dtype`` from a golden's inputs."""
    cast = lambda name: case[name].astype(dtype)  # noqa: E731
    densj = halo_density(cast("densa_j"), cast("densa_j_bnd"))
    u, v = contravariant_winds(cast("uhat_jd"), cast("vhat_jd"), densj)
    deform = deformation(u, v, dx1=DX1, dx2=DX2)
    eddyh = eddy_diffusivity(deform, cast("msfd2"), dx1=DX1, dx2=DX2)
    k11, k22 = face_coefficients(eddyh)
    return {
        "deform": np.asarray(deform),
        "k11bar": np.asarray(k11),
        "k22bar": np.asarray(k22),
        "dt": np.asarray(stable_timestep(k11, k22, dx1=DX1, dx2=DX2)),
    }


@pytest.mark.goldens
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", golden_names())
@pytest.mark.parametrize("field", ["deform", "k11bar", "k22bar", "dt"])
def test_matches_the_fortran(name: str, field: str, dtype: type) -> None:
    with np.load(GOLDENS / f"hcdiff_{name}.npz") as case:
        got = chain(case, dtype)
        worst = ulps(case[field], got[field])
    assert worst <= ULP_BUDGET, f"hcdiff_{name}.{field}: {worst:.2f} float32 ULPs"


@pytest.mark.goldens
def test_the_deformation_stage_is_essentially_exact() -> None:
    """Deformation is arithmetic on differences, with no cancellation-prone
    step, so it should agree far more tightly than the chain's budget. Pinning
    that separately means a regression there cannot hide inside the looser
    tolerance the density division needs.
    """
    for name in golden_names():
        with np.load(GOLDENS / f"hcdiff_{name}.npz") as case:
            worst = ulps(case["deform"], chain(case, np.float64)["deform"])
        assert worst <= 1.0, f"hcdiff_{name}.deform drifted to {worst:.2f} ULPs"


@pytest.mark.goldens
class TestBoundaryConventions:
    """Three different edges get zeroed, and conflating them is the easy error.

    Asserted against the Fortran output rather than against the port, so these
    say what CMAQ does, not merely that the port is self-consistent.
    """

    def test_the_deformation_pad_is_zero(self) -> None:
        """``deform.F:337-343`` zeroes the full ``(ncols+1, nrows+1)`` extent."""
        with np.load(GOLDENS / "hcdiff_smooth_random.npz") as case:
            deform = case["deform"]
        assert np.all(deform[-1, :, :] == 0.0), "last column of deformation is not zero"
        assert np.all(deform[:, -1, :] == 0.0), "last row of deformation is not zero"

    def test_the_cross_gradients_are_zeroed_on_opposite_edges(self) -> None:
        """``deform.F:420-421``: ``du/dy`` vanishes on the first and last *row*,
        ``dv/dx`` on the first and last *column*. The shear case isolates it --
        with ``v = 0`` the deformation is exactly ``|du/dy|``, so the zeroed rows
        show up as zeros and everything else does not.
        """
        with np.load(GOLDENS / "hcdiff_shear_dudy.npz") as case:
            deform = case["deform"]
        ncols_p1, nrows_p1, _ = deform.shape
        nrows = nrows_p1 - 1
        assert np.all(deform[: ncols_p1 - 1, 0, :] == 0.0), "row 1 should carry no du/dy"
        assert np.all(deform[: ncols_p1 - 1, nrows - 1, :] == 0.0), "last row likewise"
        assert np.all(deform[: ncols_p1 - 1, 1 : nrows - 1, :] > 0.0), (
            "the interior rows should all carry the shear"
        )

    def test_no_flux_crosses_the_domain_edge(self) -> None:
        """``hcdiff3d.F:216,226`` zeroes ``K11`` on the last row and ``K22`` on
        the last column -- a *different* pair of edges from the two above."""
        with np.load(GOLDENS / "hcdiff_smooth_random.npz") as case:
            k11, k22 = case["k11bar"], case["k22bar"]
        assert np.all(k11[:, -1, :] == 0.0)
        assert np.all(k22[-1, :, :] == 0.0)

    def test_zero_deformation_does_not_mean_zero_diffusivity(self) -> None:
        """The trap the arithmetic sets. ``KHD = max(KHMIN, ACOEF*0) = KHMIN``,
        so a cell with no deformation still diffuses at
        ``KHA*KHMIN/(KHA+KHMIN)``. Reading "deformation is zero here" as
        "diffusivity is zero here" would look right on a plot and be wrong.
        """
        with np.load(GOLDENS / "hcdiff_uniform_wind.npz") as case:
            deform, k11 = case["deform"], case["k11bar"]
        assert np.all(deform == 0.0), "uniform wind should have no deformation at all"
        kha = DEFAULT_HDIFF.base_diffusivity(DX1, DX2)
        expected = kha * DEFAULT_HDIFF.khmin / (kha + DEFAULT_HDIFF.khmin)
        interior = k11[:-1, :-1, :]
        np.testing.assert_allclose(interior, expected, rtol=1e-6)
        assert expected > 0.0

"""C2 — the ACM2 driver against its Fortran golden.

`vdiffacmx.F` compiled unmodified, together with `tri.F` and `matrix1.F`, over
cases chosen one per code path: stable (convective stage skipped entirely),
convective at three CBL depths, deposition, emissions, deep sub-stepping, and
one with everything on.

Deposition velocities and emission fluxes are inputs, which is this port's scope
boundary — see `docs/plans/PLAN-vdiff.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmaq_jax.vdiff import (
    ColumnState,
    SurfaceExchange,
    acm2_setup,
    column_geometry,
    substep_counts,
    vdiff_step,
)

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

#: Measured worst is 9.4 float32 ULPs on `substepped`, which runs 46 sub-steps
#: and so has the most passes for rounding to accumulate over. Every other case
#: is under 2.
ULP_BUDGET = 24.0

EPS32 = float(np.finfo(np.float32).eps)


def names() -> list[str]:
    found = sorted(p.stem[len("vdiff_") :] for p in GOLDENS.glob("vdiff_*.npz"))
    if not found:
        pytest.skip("no vdiff goldens committed")
    return found


def ulps(expected: np.ndarray, got: np.ndarray) -> float:
    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(got, dtype=np.float64)
    return float(np.abs(left - right).max()) / max(float(np.abs(left).max()), 1.0) / EPS32


def unpack(case: np.lib.npyio.NpzFile, dtype: type):
    """The golden's CMAQ-order arrays, in this package's order.

    CMAQ carries `SEDDY` layer-first and the species arrays species-first; the
    package uses `(ncols, nrows, nlays, nspc)` throughout. The transpose belongs
    at the boundary, which is here.
    """
    cast = lambda name: case[name].astype(dtype)  # noqa: E731
    state = ColumnState(
        seddy=np.transpose(cast("seddy"), (1, 2, 0)),
        zf=cast("zf"),
        zh=cast("zh"),
        pbl=cast("pbl"),
        lpbl=case["lpbl"].astype(np.int32),
        hol=cast("hol"),
        dens1=cast("dens1"),
        rdepvht=cast("rdepvht"),
        convective=case["convct"] != 0,
    )
    surface = SurfaceExchange(
        depv=np.transpose(cast("depv"), (1, 2, 0)),
        pldv=np.transpose(cast("pldv"), (1, 2, 0)),
        emis=np.transpose(cast("vdemis"), (2, 3, 1, 0)),
    )
    conc = np.transpose(cast("cngrd"), (2, 3, 1, 0))
    return conc, state, surface


def run(case: np.lib.npyio.NpzFile, dtype: type) -> tuple[np.ndarray, np.ndarray]:
    conc, state, surface = unpack(case, dtype)
    dtsec = float(case["dtsec"])
    bound = int(np.asarray(substep_counts(state, dtsec)).max())
    diffused, ddep = vdiff_step(conc, state, surface, dtsec=dtsec, max_substeps=bound)
    return np.asarray(diffused), np.asarray(ddep)


@pytest.mark.goldens
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", names())
def test_concentrations_match_the_fortran(name: str, dtype: type) -> None:
    with np.load(GOLDENS / f"vdiff_{name}.npz") as case:
        diffused, _ = run(case, dtype)
        expected = np.transpose(case["cngrd_out"], (2, 3, 1, 0))
    worst = ulps(expected, diffused)
    assert worst <= ULP_BUDGET, f"vdiff_{name}: {worst:.2f} float32 ULPs"


@pytest.mark.goldens
@pytest.mark.parametrize("name", names())
def test_dry_deposition_matches_the_fortran(name: str) -> None:
    """Checked separately from the concentrations because it is accumulated
    rather than solved, and an error in it does not show up in the field.

    It caught one: the evasion term (`- DTS*DENS1*PLDV`) appears in *both* halves
    of the Crank-Nicolson step for a plain species, and only the
    heterogeneous-HONO branches omit it from the second. Copying their form cost
    exactly `THBAR * DTS * DENS1 * PLDV`, which is invisible without emissions —
    the concentrations were already matching to 0.4 ULPs.
    """
    with np.load(GOLDENS / f"vdiff_{name}.npz") as case:
        _, ddep = run(case, np.float64)
        expected = np.transpose(case["ddep"], (1, 2, 0))
    worst = ulps(expected, ddep)
    assert worst <= ULP_BUDGET, f"vdiff_{name} ddep: {worst:.2f} float32 ULPs"


@pytest.mark.goldens
def test_the_emission_cases_would_catch_a_missing_evasion_term() -> None:
    """Guards the test above. If no case had a nonzero `PLDV`, the evasion bug
    would have passed every deposition check."""
    with np.load(GOLDENS / "vdiff_with_emissions.npz") as case:
        assert case["pldv"].max() > 0.0
        assert case["vdemis"].max() > 0.0


@pytest.mark.goldens
class TestTheACM2Split:
    """The reallocation that makes ACM2 asymmetric, and the easiest thing to
    miss when reading `vdiffacmx.F`."""

    @pytest.mark.parametrize("name", names())
    def test_the_diffusivity_split_matches(self, name: str) -> None:
        """Inside the CBL the convective stage takes a fraction `FNL` of the
        eddy diffusivity and carries it non-locally, leaving `(1 - FNL)` for the
        local stage. The harness returns the modified array precisely so this
        can be checked, since it is an in-place change CMAQ never reports.
        """
        with np.load(GOLDENS / f"vdiff_{name}.npz") as case:
            zf = case["zf"][0, 0].astype(np.float64)
            zh = case["zh"][0, 0].astype(np.float64)
            setup = acm2_setup(
                case["seddy"][:, 0, 0].astype(np.float64),
                column_geometry(zf, zh),
                pbl=np.float64(case["pbl"][0, 0]),
                zf=zf,
                lpbl=np.int32(case["lpbl"][0, 0]),
                hol=np.float64(case["hol"][0, 0]),
                convective=np.bool_(case["convct"][0, 0] != 0),
                dtsec=float(case["dtsec"]),
            )
            expected = case["seddy_out"][:, 0, 0].astype(np.float64)
        np.testing.assert_allclose(np.asarray(setup.seddy), expected, rtol=1e-6)

    def test_a_convective_column_actually_splits_it(self) -> None:
        """Guard: if the split were a no-op the test above would pass trivially."""
        with np.load(GOLDENS / "vdiff_convective.npz") as case:
            assert not np.allclose(case["seddy"], case["seddy_out"])

    def test_a_stable_column_leaves_it_alone(self) -> None:
        """And the converse — with no convection there is no plume to feed."""
        with np.load(GOLDENS / "vdiff_stable.npz") as case:
            np.testing.assert_array_equal(case["seddy"], case["seddy_out"])


@pytest.mark.goldens
def test_the_substep_count_agrees() -> None:
    """`NLP = int(DTSEC/DTLIM + 0.99)` is a ceiling. Getting it wrong changes
    the answer without changing its shape, and one case is meant to exercise
    deep sub-stepping — assert that it does."""
    counts = {}
    for name in names():
        with np.load(GOLDENS / f"vdiff_{name}.npz") as case:
            _, state, _ = unpack(case, np.float64)
            counts[name] = int(np.asarray(substep_counts(state, float(case["dtsec"]))).max())
    assert counts["substepped"] > 10, f"the sub-stepping case runs {counts['substepped']} steps"
    assert counts["stable"] == 1

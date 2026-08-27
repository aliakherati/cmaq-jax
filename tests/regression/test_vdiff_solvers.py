"""C1 — the two ACM2 solvers against their Fortran goldens.

`tri.F` and `matrix1.F` compiled unmodified. These are pure linear algebra, so a
mismatch is unambiguous — there is no meteorology or unit convention to blame.

Each solver is checked two ways: against the golden, and by *residual* — assemble
the matrix, multiply the solution back, compare to the right-hand side. The
residual test is the stronger claim, because it does not depend on the Fortran
being right, and it is what catches a transposed sub/super-diagonal that a
single golden on a nearly symmetric matrix would let through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmaq_jax.vdiff import solve_acm1, solve_tridiagonal

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

#: Measured worst is 0.98 float32 ULPs across all nine cases, so this is
#: generous. Both solvers are short recurrences with no cancellation-prone step.
ULP_BUDGET = 4.0

EPS32 = float(np.finfo(np.float32).eps)


def names(prefix: str) -> list[str]:
    found = sorted(p.stem[len(prefix) + 1 :] for p in GOLDENS.glob(f"{prefix}_*.npz"))
    if not found:
        pytest.skip(f"no {prefix} goldens committed")
    return found


def ulps(expected: np.ndarray, got: np.ndarray) -> float:
    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(got, dtype=np.float64)
    return float(np.abs(left - right).max()) / max(float(np.abs(left).max()), 1.0) / EPS32


def tri_matrix(case: np.lib.npyio.NpzFile) -> np.ndarray:
    """The dense matrix, from CMAQ's storage: row k holds sub[k], diag[k], sup[k]."""
    diag = case["diag"].astype(np.float64)
    nlays = diag.size
    matrix = np.zeros((nlays, nlays))
    matrix[np.arange(nlays), np.arange(nlays)] = diag
    matrix[np.arange(1, nlays), np.arange(nlays - 1)] = case["sub"].astype(np.float64)[1:]
    matrix[np.arange(nlays - 1), np.arange(1, nlays)] = case["sup"].astype(np.float64)[:-1]
    return matrix


def acm1_matrix(case: np.lib.npyio.NpzFile) -> np.ndarray:
    """The dense ACM1 matrix over the active rows. Tridiagonal plus first column."""
    kl = int(case["kl"])
    col = case["col"].astype(np.float64)
    diag = case["diag"].astype(np.float64)
    sup = case["sup"].astype(np.float64)
    matrix = np.zeros((kl, kl))
    matrix[0, 0] = diag[0]
    if kl > 1:
        matrix[0, 1] = sup[1]
    for k in range(1, kl):
        matrix[k, 0] += col[k]
        matrix[k, k] += diag[k]
        if k < kl - 1:
            matrix[k, k + 1] = sup[k + 1]
    return matrix


@pytest.mark.goldens
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", names("tri"))
def test_tri_matches_the_fortran(name: str, dtype: type) -> None:
    with np.load(GOLDENS / f"tri_{name}.npz") as case:
        got = solve_tridiagonal(*(case[k].astype(dtype) for k in ("sub", "diag", "sup", "rhs")))
        worst = ulps(case["x"], np.asarray(got))
    assert worst <= ULP_BUDGET, f"tri_{name}: {worst:.2f} float32 ULPs"


@pytest.mark.goldens
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", names("matrix1"))
def test_acm1_matches_the_fortran(name: str, dtype: type) -> None:
    with np.load(GOLDENS / f"matrix1_{name}.npz") as case:
        got = solve_acm1(
            *(case[k].astype(dtype) for k in ("col", "diag", "sup", "rhs")),
            int(case["kl"]),
        )
        worst = ulps(case["x"], np.asarray(got))
    assert worst <= ULP_BUDGET, f"matrix1_{name}: {worst:.2f} float32 ULPs"


@pytest.mark.goldens
@pytest.mark.parametrize("name", names("tri"))
def test_tri_actually_solves_the_system(name: str) -> None:
    """The claim that does not depend on the Fortran being right."""
    with np.load(GOLDENS / f"tri_{name}.npz") as case:
        got = np.asarray(
            solve_tridiagonal(*(case[k].astype(np.float64) for k in ("sub", "diag", "sup", "rhs")))
        )
        residual = np.abs(tri_matrix(case) @ got.T - case["rhs"].astype(np.float64).T).max()
        scale = float(np.abs(case["rhs"]).max())
    assert residual < 1e-10 * max(scale, 1.0), f"tri_{name}: residual {residual:.3e}"


@pytest.mark.goldens
@pytest.mark.parametrize("name", names("matrix1"))
def test_acm1_actually_solves_the_system(name: str) -> None:
    with np.load(GOLDENS / f"matrix1_{name}.npz") as case:
        kl = int(case["kl"])
        got = np.asarray(
            solve_acm1(
                *(case[k].astype(np.float64) for k in ("col", "diag", "sup", "rhs")),
                kl,
            )
        )
        rhs = case["rhs"].astype(np.float64)
        residual = np.abs(acm1_matrix(case) @ got[:, :kl].T - rhs[:, :kl].T).max()
        scale = float(np.abs(rhs).max())
    assert residual < 1e-10 * max(scale, 1.0), f"matrix1_{name}: residual {residual:.3e}"


@pytest.mark.goldens
def test_the_asymmetric_case_is_actually_asymmetric() -> None:
    """Guards the residual tests. A symmetric matrix lets a transposed
    sub/super-diagonal pass, and that is the natural error when porting a banded
    solver — so at least one case has to be able to tell them apart."""
    with np.load(GOLDENS / "tri_asymmetric.npz") as case:
        sub, sup = case["sub"][1:], case["sup"][:-1]
    assert not np.allclose(sub, sup), "the asymmetric case has equal off-diagonals"


@pytest.mark.goldens
def test_transposing_the_off_diagonals_would_be_caught() -> None:
    """The above, demonstrated rather than asserted: swapping them changes the
    answer, so the tests genuinely discriminate."""
    with np.load(GOLDENS / "tri_asymmetric.npz") as case:
        args = [case[k].astype(np.float64) for k in ("sub", "diag", "sup", "rhs")]
        correct = np.asarray(solve_tridiagonal(*args))
        # Swap, preserving which entries are the unused poison values.
        swapped_sub = np.concatenate([args[0][:1], args[2][:-1]])
        swapped_sup = np.concatenate([args[0][1:], args[2][-1:]])
        swapped = np.asarray(solve_tridiagonal(swapped_sub, args[1], swapped_sup, args[3]))
    assert not np.allclose(correct, swapped)


@pytest.mark.goldens
class TestAlphaUnderflow:
    """C0's open question, answered by measurement.

    ``matrix1.F`` accumulates ``ALPHA = prod(-sup/diag)`` down the convective
    boundary layer and divides by a sum weighted with it. The plan flagged
    float32 underflow as a risk worth checking rather than assuming.
    """

    def test_alpha_really_does_reach_the_float32_floor(self) -> None:
        """Guards the test below: if the case were not extreme, its passing
        would say nothing."""
        with np.load(GOLDENS / "matrix1_alpha_underflow.npz") as case:
            diag = case["diag"].astype(np.float64)
            sup = case["sup"].astype(np.float64)
            kl = int(case["kl"])
        alpha, smallest = 1.0, 1.0
        for layer in range(1, kl):
            alpha = -alpha * sup[layer] / diag[layer]
            smallest = min(smallest, abs(alpha))
        assert smallest < 1e-30, f"alpha only reached {smallest:.2e}"
        assert smallest < 10 * float(np.finfo(np.float32).tiny)

    def test_it_is_harmless(self) -> None:
        """And the answer is that it does not matter.

        ``alpha`` weights contributions to ``gama`` that are already negligible
        beside ``diag[0]``, so losing them to underflow costs nothing. Solving in
        float32 — where the underflow actually happens — agrees with the Fortran
        as closely as any other case.
        """
        with np.load(GOLDENS / "matrix1_alpha_underflow.npz") as case:
            got = solve_acm1(
                *(case[k].astype(np.float32) for k in ("col", "diag", "sup", "rhs")),
                int(case["kl"]),
            )
            worst = ulps(case["x"], np.asarray(got))
        assert worst <= ULP_BUDGET
        assert np.all(np.isfinite(np.asarray(got)))


@pytest.mark.goldens
@pytest.mark.parametrize("name", names("matrix1"))
def test_rows_above_the_cbl_top_are_zero(name: str) -> None:
    """``MATRIX1`` only touches rows ``1..KL``; the caller zeroes ``X`` first.
    Reproducing that matters because the driver adds the two stages together,
    and a nonzero value up there would be added to the local stage's answer.
    """
    with np.load(GOLDENS / f"matrix1_{name}.npz") as case:
        kl = int(case["kl"])
        got = np.asarray(
            solve_acm1(
                *(case[k].astype(np.float64) for k in ("col", "diag", "sup", "rhs")),
                kl,
            )
        )
    assert np.all(got[:, kl:] == 0.0)

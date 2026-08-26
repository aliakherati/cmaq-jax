"""A0.6 — the JAX non-uniform PPM reconstruction against the Fortran goldens.

These goldens come from calling ``vppm.F``'s inner ``PPM`` subroutine directly,
so they pin the parabola alone -- independent of the flux-matching velocity
adjustment that ``VPPM`` wraps around it (that lands in A2.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from atmos_jax_common.real4 import downcast_to_real4

from cmaq_jax.ppm import (
    nonuniform_mesh,
    ppm_parabola_nonuniform,
    ppm_parabola_uniform,
)

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

RTOL = 1e-6

EPS32 = float(np.finfo(np.float32).eps)
# All four outputs are O(|cn|) quantities computed by the Fortran in float32, so
# their absolute error scales with the *input* magnitude, not with their own.
# That distinction matters for c6 = 6*(cn - (cl+cr)/2): on a linear profile it
# is a difference of nearly-equal numbers, so its true value is ~0 while its
# rounding error is still set by |cn|. Scaling the tolerance by max|c6| would
# make it absurdly tight exactly where cancellation is worst.
#
# Measured worst case across every golden, in units of EPS32*max|cn|:
#   cl 0.80, cr 0.80, dc 1.58, c6 4.64
# 16 leaves ~3x headroom on the worst field while still catching any real
# algorithmic difference, which would be orders of magnitude larger.
ULP_BUDGET = 16.0


def _atol(cn: np.ndarray) -> float:
    return ULP_BUDGET * EPS32 * max(float(np.abs(cn).max()), 1.0)


pytestmark = pytest.mark.goldens

CASES = sorted(p.stem for p in GOLDENS.glob("coeffs_*.npz"))

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
    assert CASES, f"no coeffs goldens in {GOLDENS}; run scripts/generate_goldens.py"


@pytest.mark.parametrize("dtype", PRECISIONS, ids=["f32", "f64"])
@pytest.mark.parametrize("name", CASES)
def test_matches_fortran(name: str, dtype: type) -> None:
    golden = _load(name)
    cn = np.asarray(golden["cn"], dtype=dtype)
    mesh = nonuniform_mesh(np.asarray(golden["ds"], dtype=dtype))
    got = ppm_parabola_nonuniform(cn, mesh)
    atol = _atol(cn)

    for field in ("cl", "cr", "dc", "c6"):
        np.testing.assert_allclose(
            downcast_to_real4(np.asarray(getattr(got, field))),
            np.asarray(golden[field], dtype=np.float64),
            rtol=RTOL,
            atol=atol,
            err_msg=f"{name}: {field}",
        )


@pytest.mark.parametrize("name", CASES)
def test_parabola_is_self_consistent(name: str) -> None:
    """dc and c6 are defined by cl, cr and the cell value (eq. 1.5)."""
    golden = _load(name)
    cn = np.asarray(golden["cn"], dtype=np.float64)
    mesh = nonuniform_mesh(np.asarray(golden["ds"], dtype=np.float64))
    p = ppm_parabola_nonuniform(cn, mesh)

    np.testing.assert_allclose(np.asarray(p.dc), np.asarray(p.cr) - np.asarray(p.cl), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(p.c6),
        6.0 * (cn - 0.5 * (np.asarray(p.cl) + np.asarray(p.cr))),
        rtol=1e-12,
    )


@pytest.mark.parametrize("name", CASES)
def test_edge_values_bracket_the_cell(name: str) -> None:
    """After monotonisation the cell mean lies between its two edge values.

    This is what stops PPM producing new extrema, so it must hold for every
    profile including the discontinuous ones.
    """
    golden = _load(name)
    cn = np.asarray(golden["cn"], dtype=np.float64)
    mesh = nonuniform_mesh(np.asarray(golden["ds"], dtype=np.float64))
    p = ppm_parabola_nonuniform(cn, mesh)

    lo = np.minimum(np.asarray(p.cl), np.asarray(p.cr))
    hi = np.maximum(np.asarray(p.cl), np.asarray(p.cr))
    tol = 1e-12 * np.maximum(np.abs(cn), 1.0)
    assert np.all(cn >= lo - tol)
    assert np.all(cn <= hi + tol)


def test_constant_profile_is_flat() -> None:
    """A constant field must reconstruct exactly flat, with no slope or
    curvature, or vertical transport would manufacture structure."""
    golden = _load("coeffs_constant_stretched")
    cn = np.asarray(golden["cn"], dtype=np.float64)
    mesh = nonuniform_mesh(np.asarray(golden["ds"], dtype=np.float64))
    p = ppm_parabola_nonuniform(cn, mesh)

    np.testing.assert_allclose(np.asarray(p.cl), cn, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(p.cr), cn, rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(p.dc), np.zeros_like(cn))
    np.testing.assert_array_equal(np.asarray(p.c6), np.zeros_like(cn))


def test_reduces_to_the_uniform_scheme() -> None:
    """On an evenly spaced grid the non-uniform form must reproduce the uniform
    one, which is a genuine cross-check between two independent ports.

    Compared on the interior only: the uniform kernel takes a haloed array and
    extrapolates through it, whereas a CMAQ column has hard ends and drops to
    reduced order in the outermost two cells (``vppm.F:475-480``).
    """
    n = 16
    z = np.arange(n, dtype=np.float64)
    cn = 1.0 + 0.4 * np.sin(2.0 * np.pi * z / n) + 0.15 * np.cos(5.0 * np.pi * z / n)
    ds = np.full(n, 0.0625)

    nonuniform = ppm_parabola_nonuniform(cn, nonuniform_mesh(ds))

    # Give the uniform kernel a halo wide enough that its own edge effects stay
    # outside the window we compare.
    swp = 3
    padded = np.concatenate([np.repeat(cn[:1], swp), cn, np.repeat(cn[-1:], swp)])
    uniform = ppm_parabola_uniform(padded)

    interior = slice(2, n - 2)
    for field in ("cl", "cr", "dc", "c6"):
        np.testing.assert_allclose(
            np.asarray(getattr(nonuniform, field))[interior],
            np.asarray(getattr(uniform, field))[swp : swp + n][interior],
            rtol=1e-10,
            err_msg=field,
        )


def test_rejects_too_few_layers() -> None:
    """vppm.F's interior loops go degenerate below four layers and leave part
    of CM unset; refuse rather than return quietly wrong numbers."""
    with pytest.raises(ValueError, match="at least 4 layers"):
        nonuniform_mesh(np.array([0.3, 0.3, 0.4]))


def test_accepts_extra_trailing_axes() -> None:
    """Columns carry a species axis; broadcasting must match the 1-D result."""
    golden = _load("coeffs_smooth_stretched")
    cn = np.asarray(golden["cn"], dtype=np.float64)
    mesh = nonuniform_mesh(np.asarray(golden["ds"], dtype=np.float64))

    scales = np.array([1.0, 3.0, 0.25])
    stacked = cn[:, None] * scales[None, :]
    got = ppm_parabola_nonuniform(stacked, mesh)
    reference = ppm_parabola_nonuniform(cn, mesh)

    for spc, scale in enumerate(scales):
        for field in ("cl", "cr", "dc", "c6"):
            np.testing.assert_allclose(
                np.asarray(getattr(got, field))[:, spc],
                np.asarray(getattr(reference, field)) * scale,
                rtol=1e-12,
                err_msg=f"{field} species {spc}",
            )

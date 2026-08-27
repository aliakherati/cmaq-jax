"""B2.1 — the horizontal-diffusion driver against its Fortran golden.

`hdiff.F` compiled unmodified, over four cases chosen mainly for how many
sub-steps they force: 1, 2, 63 and 147. The sub-step count is what the two
behaviours worth pinning both depend on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmaq_jax.config import GridConfig, sigma_layer_thickness
from cmaq_jax.hdiff import (
    contravariant_winds,
    deformation,
    eddy_diffusivity,
    face_coefficients,
    halo_density,
    hdiff_step,
    stable_timestep,
    substep_count,
)

GOLDENS = Path(__file__).resolve().parents[2] / "data" / "goldens"

#: Measured worst is 1.8 float32 ULPs, on the 147-sub-step case where rounding
#: has the most passes to accumulate over. 8 leaves room without hiding a real
#: disagreement, which would be orders of magnitude larger.
ULP_BUDGET = 8.0

EPS32 = float(np.finfo(np.float32).eps)


def golden_names() -> list[str]:
    names = sorted(p.stem[len("hdiff_") :] for p in GOLDENS.glob("hdiff_*.npz"))
    if not names:
        pytest.skip("no hdiff goldens committed")
    return names


def run_port(case: np.lib.npyio.NpzFile, dtype: type) -> tuple[np.ndarray, int]:
    """The whole chain the driver needs, at ``dtype``."""
    cast = lambda name: case[name].astype(dtype)  # noqa: E731
    dx = float(case["dx"])
    sync = float(case["sync_seconds"])
    ncols, nrows, nlays, nspc = case["cgrid_in"].shape

    densj = halo_density(cast("densa_j"), cast("densa_j_bnd"))
    u, v = contravariant_winds(cast("uhat_jd"), cast("vhat_jd"), densj)
    eddyh = eddy_diffusivity(deformation(u, v, dx1=dx, dx2=dx), cast("msfd2"), dx1=dx, dx2=dx)
    k11, k22 = face_coefficients(eddyh)
    nsteps = substep_count(sync, float(stable_timestep(k11, k22, dx1=dx, dx2=dx)))

    cfg = GridConfig(
        ncols=ncols,
        nrows=nrows,
        ds=sigma_layer_thickness(np.linspace(1.0, 0.0, nlays + 1)),
        dx1=dx,
        dx2=dx,
        nspc_adv=nspc,
        dtype="float64" if dtype is np.float64 else "float32",  # type: ignore[arg-type]
    )
    got = hdiff_step(cast("cgrid_in"), densj, k11, k22, cfg=cfg, sync_seconds=sync, nsteps=nsteps)
    return np.asarray(got), nsteps


@pytest.mark.goldens
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", golden_names())
def test_matches_the_fortran(name: str, dtype: type) -> None:
    with np.load(GOLDENS / f"hdiff_{name}.npz") as case:
        got, _ = run_port(case, dtype)
        expected = case["cgrid_out"].astype(np.float64)
    scale = max(float(np.abs(expected).max()), 1.0)
    worst = float(np.abs(expected - got.astype(np.float64)).max()) / scale / EPS32
    assert worst <= ULP_BUDGET, f"hdiff_{name}: {worst:.2f} float32 ULPs"


@pytest.mark.goldens
@pytest.mark.parametrize("name", golden_names())
def test_the_substep_count_agrees(name: str) -> None:
    """``NSTEPS = int(DTSEC/DT) + 1`` is a host-side decision in both, and an
    off-by-one there changes the answer without changing its shape."""
    with np.load(GOLDENS / f"hdiff_{name}.npz") as case:
        _, nsteps = run_port(case, np.float64)
        assert nsteps == int(case["nsteps"])


@pytest.mark.goldens
def test_the_cases_actually_reach_the_substepping_path() -> None:
    """Guards every test above.

    On the 12 km benchmark grid the stable step is ~2e5 s, so no sync step ever
    subdivides and ``NSTEPS`` is 1 — under which the frozen halo is exact and
    the sub-step loop is untested. These cases only exercise it because they use
    a finer grid, and if that stopped being true the suite would still pass
    while testing nothing.
    """
    counts = {}
    for name in golden_names():
        with np.load(GOLDENS / f"hdiff_{name}.npz") as case:
            counts[name] = int(case["nsteps"])
    assert max(counts.values()) >= 100, f"no case sub-steps deeply: {counts}"
    assert min(counts.values()) == 1, f"no single-step case to isolate the update: {counts}"


@pytest.mark.goldens
class TestDriverBehaviour:
    """The two things a tidier rewrite would silently change.

    Asserted against the Fortran output, so these state what CMAQ does rather
    than that the port agrees with itself.
    """

    @pytest.mark.parametrize("name", golden_names())
    def test_rho_j_is_not_diffused(self, name: str) -> None:
        """``DIFF_MAP`` (``hdiff.F:276-292``) covers the transported species with
        no ``+ 1`` for density — unlike ``ADV_MAP``, which includes it. Density
        is a coefficient here, not a tracer. Carrying it over from advection by
        analogy would smooth the meteorology, and the field would still look
        entirely plausible.
        """
        with np.load(GOLDENS / f"hdiff_{name}.npz") as case:
            np.testing.assert_array_equal(case["cgrid_out"][..., -1], case["cgrid_in"][..., -1])

    @pytest.mark.parametrize("name", golden_names())
    def test_the_species_actually_changed(self, name: str) -> None:
        """Guards the test above: if diffusion did nothing at all, an untouched
        density slot would prove nothing."""
        with np.load(GOLDENS / f"hdiff_{name}.npz") as case:
            before, after = case["cgrid_in"][..., :-1], case["cgrid_out"][..., :-1]
        assert not np.allclose(before, after), "diffusion left the species unchanged"

    def test_a_frozen_halo_is_what_the_fortran_does(self) -> None:
        """``hdiff.F`` fills ``HALO_*`` once before the ``DO 344`` loop and
        reloads only the interior each sub-step, so the zero-gradient boundary
        is exact only on the first pass.

        Refreshing the halo every sub-step is the obvious tidy-up and it is a
        different scheme. Over 147 sub-steps the two diverge, so the port
        agreeing with the Fortran to under 2 ULPs on ``deep_substepping`` is
        the evidence — this test pins that the case is deep enough for the
        distinction to bite.
        """
        with np.load(GOLDENS / "hdiff_deep_substepping.npz") as case:
            assert int(case["nsteps"]) > 100
            got, _ = run_port(case, np.float64)
            expected = case["cgrid_out"].astype(np.float64)
        scale = max(float(np.abs(expected).max()), 1.0)
        worst = float(np.abs(expected - got).max()) / scale / EPS32
        assert worst <= 2.0, f"drifted to {worst:.2f} ULPs over deep sub-stepping"

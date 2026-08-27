"""B2.2 — properties of the horizontal-diffusion operator.

What the golden cannot say: that the scheme conserves mass, leaves a uniform
field alone, and cannot manufacture an extremum. These are the tests that would
catch a coupling error carried over from advection by analogy — where the
convention is genuinely different.
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np
import pytest

from cmaq_jax.config import DEFAULT_HDIFF, GridConfig, sigma_layer_thickness
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

NCOLS, NROWS, NLAYS, NTRNS = 8, 7, 3, 2


def setup(
    *,
    q: np.ndarray | None = None,
    dx: float = 4000.0,
    sync_seconds: float = 3600.0,
    uniform_density: bool = False,
    seed: int = 20260827,
):
    """A complete diffusion problem, plus everything ``hdiff_step`` needs."""
    rng = np.random.default_rng(seed)
    shape = (NCOLS + 1, NROWS + 1, NLAYS)
    rows = np.arange(NROWS + 1, dtype=np.float64)[None, :, None]
    cols = np.arange(NCOLS + 1, dtype=np.float64)[:, None, None]

    rho = (
        np.full((NCOLS, NROWS, NLAYS), 2.0)
        if uniform_density
        else 1.5 + 0.4 * rng.random((NCOLS, NROWS, NLAYS))
    )
    mean_rho = float(rho.mean())
    ring = np.full((2 * (NCOLS + NROWS + 2), NLAYS), mean_rho)

    u = (40.0 * rows + 15.0 * cols + rng.normal(0.0, 3.0, shape)) * mean_rho
    v = (25.0 * cols + rng.normal(0.0, 3.0, shape)) * mean_rho
    msfd2 = np.ones((NCOLS + 1, NROWS + 1))

    if q is None:
        q = 1.0 + rng.random((NCOLS, NROWS, NLAYS, NTRNS))
    state = np.concatenate([q * rho[..., None], rho[..., None]], axis=-1)

    densj = halo_density(rho, ring)
    uu, vv = contravariant_winds(u, v, densj)
    eddyh = eddy_diffusivity(deformation(uu, vv, dx1=dx, dx2=dx), msfd2, dx1=dx, dx2=dx)
    k11, k22 = face_coefficients(eddyh)
    nsteps = substep_count(sync_seconds, float(stable_timestep(k11, k22, dx1=dx, dx2=dx)))

    cfg = GridConfig(
        ncols=NCOLS,
        nrows=NROWS,
        ds=sigma_layer_thickness(np.linspace(1.0, 0.0, NLAYS + 1)),
        dx1=dx,
        dx2=dx,
        nspc_adv=state.shape[-1],
    )
    return cfg, state, densj, k11, k22, nsteps, sync_seconds


def run(**kwargs) -> tuple[np.ndarray, np.ndarray, int]:
    cfg, state, densj, k11, k22, nsteps, sync = setup(**kwargs)
    got = hdiff_step(state, densj, k11, k22, cfg=cfg, sync_seconds=sync, nsteps=nsteps)
    return np.asarray(got), state, nsteps


class TestUniformity:
    """The analogue of advection's constancy preservation, and the test most
    likely to catch a coupling error — because the coupling convention here is
    the opposite of advection's."""

    @pytest.mark.parametrize("uniform_density", [True, False])
    def test_a_uniform_mixing_ratio_is_untouched(self, uniform_density: bool) -> None:
        """Every flux is proportional to a mixing-ratio gradient, so a uniform
        ``q`` must come back exactly uniform — including where the density is
        not uniform, which is the case that separates "diffuses q" from
        "diffuses the coupled concentration".
        """
        value = 0.75
        q = np.full((NCOLS, NROWS, NLAYS, NTRNS), value)
        got, _state, _ = run(q=q, uniform_density=uniform_density)
        np.testing.assert_allclose(got[..., :-1] / got[..., -1:], value, rtol=1e-12)

    def test_diffusing_the_coupled_field_instead_would_be_visible(self) -> None:
        """Guards the test above: with a varying density, a uniform mixing ratio
        is a *non*-uniform coupled field, so an implementation that diffused the
        coupled field directly would smooth it and fail."""
        _, state, _ = run(q=np.full((NCOLS, NROWS, NLAYS, NTRNS), 0.75))
        coupled = state[..., 0]
        assert coupled.std() > 0.01 * coupled.mean(), "density is too flat to distinguish"


class TestMass:
    def test_a_single_substep_conserves_mass_exactly(self) -> None:
        """The update is a difference of face fluxes, so interior terms cancel
        telescopically and only the boundary survives. On the first sub-step the
        seeded halo makes the boundary gradient exactly zero, so nothing leaves
        the domain at all and the total is preserved to rounding.
        """
        got, state, nsteps = run(dx=12000.0, sync_seconds=180.0)
        assert nsteps == 1, "this test is only sharp for a single sub-step"
        for spc in range(NTRNS):
            np.testing.assert_allclose(got[..., spc].sum(), state[..., spc].sum(), rtol=1e-12)

    def test_the_frozen_halo_leaks_mass_over_many_substeps(self) -> None:
        """After the first pass the halo is stale, so the boundary gradient is
        no longer identically zero and mass crosses the domain edge.

        This is CMAQ's behaviour, not a defect in the port, and it is not small.
        Measured directly from the Fortran goldens:

            sub-steps    mass drift
                    1    +1.9e-10   (exact, to rounding)
                    2    -3.0e-04
                   66    -1.85e-02
                  155    -2.23e-02

        So an hour of deep sub-stepping loses ~2% of the tracer. The sign is
        consistent: diffusion lifts the edge cells above the frozen halo value,
        the gradient there points outward, and mass flows out. Asserting a
        single loose tolerance across both regimes would hide this; the exact
        first-sub-step case above and this one together say which is which.
        """
        got, state, nsteps = run(dx=1000.0, sync_seconds=3600.0)
        assert nsteps > 50
        drift = [
            (got[..., spc].sum() - state[..., spc].sum()) / state[..., spc].sum()
            for spc in range(NTRNS)
        ]
        assert all(d < 0.0 for d in drift), f"expected outward leakage, got {drift}"
        assert all(abs(d) < 0.05 for d in drift), f"leakage larger than the Fortran's: {drift}"


class TestBoundedness:
    def test_no_new_extrema(self) -> None:
        """Diffusion averages, so it cannot manufacture a value outside the
        initial range of the mixing ratio."""
        rng = np.random.default_rng(3)
        q = rng.random((NCOLS, NROWS, NLAYS, NTRNS)) + 0.5
        got, _state, _ = run(q=q, dx=4000.0)
        ratio = got[..., :-1] / got[..., -1:]
        assert ratio.min() >= q.min() - 1e-12
        assert ratio.max() <= q.max() + 1e-12

    def test_positivity(self) -> None:
        spike = np.zeros((NCOLS, NROWS, NLAYS, NTRNS))
        spike[NCOLS // 2, NROWS // 2, 1, 0] = 25.0
        got, _, _ = run(q=spike, dx=1000.0)
        assert got[..., :-1].min() >= 0.0

    def test_a_spike_spreads(self) -> None:
        """The operator has to actually do something, or every test above is
        satisfied by returning the input."""
        spike = np.zeros((NCOLS, NROWS, NLAYS, NTRNS))
        spike[NCOLS // 2, NROWS // 2, 1, 0] = 25.0
        got, state, _ = run(q=spike, dx=1000.0)
        peak_before = state[NCOLS // 2, NROWS // 2, 1, 0]
        peak_after = got[NCOLS // 2, NROWS // 2, 1, 0]
        assert peak_after < peak_before, "the spike did not diffuse at all"


class TestDensitySlot:
    def test_rho_j_comes_back_unchanged(self) -> None:
        """``DIFF_MAP`` has no density slot. Diffusing it — the natural mistake
        after porting advection, where it *is* transported — would smooth the
        meteorology."""
        got, state, _ = run(dx=1000.0)
        np.testing.assert_array_equal(got[..., -1], state[..., -1])


class TestJitAndGradients:
    def test_it_jits(self) -> None:
        cfg, state, densj, k11, k22, nsteps, sync = setup(dx=4000.0)
        step = jax.jit(partial(hdiff_step, cfg=cfg, sync_seconds=sync, nsteps=nsteps))
        got = np.asarray(step(state, densj, k11, k22))
        assert np.all(np.isfinite(got))

    def test_gradients_match_finite_differences(self) -> None:
        """The payoff that justifies JAX over a Fortran rewrite. Diffusion is
        linear in the state, so there is no limiter kink to worry about here and
        the check can be tight."""
        cfg, state, densj, k11, k22, nsteps, sync = setup(dx=4000.0)

        def total(field):
            out = hdiff_step(field, densj, k11, k22, cfg=cfg, sync_seconds=sync, nsteps=nsteps)
            return (out[..., 0] ** 2).sum()

        grad = np.asarray(jax.grad(total)(state))

        eps = 1e-5
        rng = np.random.default_rng(0)
        for _ in range(4):
            i = tuple(rng.integers(0, n) for n in state.shape)
            up, down = state.copy(), state.copy()
            up[i] += eps
            down[i] -= eps
            numeric = (float(total(up)) - float(total(down))) / (2 * eps)
            assert numeric == pytest.approx(float(grad[i]), rel=1e-6, abs=1e-8)


def test_the_constants_are_cmaq_defaults() -> None:
    assert DEFAULT_HDIFF.kh == 2000.0
    assert DEFAULT_HDIFF.khmin == 200.0
    assert DEFAULT_HDIFF.cfc == 0.300

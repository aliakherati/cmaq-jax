"""A3.4 — gradients through the full advection operator.

This is the payoff that justifies a JAX port over a CUDA-Fortran rewrite. CMAQ
gets sensitivities from DDM-3D, a separate and partially maintained code path;
here ``jax.grad`` differentiates the same code that does the forward run.

Every check compares against central finite differences, so a gradient that is
merely *finite* cannot pass. Getting here required fixing four places where a
masked branch still poisoned the reverse pass — see ``cmaq_jax.vadv`` — and the
NaN tests below exist to keep them fixed.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cmaq_jax.api import Meteorology, advect_step
from cmaq_jax.config import DEFAULT_PPM, GridConfig, sigma_layer_thickness
from cmaq_jax.hadv import BoundaryConditions, hadv_step
from cmaq_jax.ppm import nonuniform_mesh
from cmaq_jax.vadv import zadv

NCOLS, NROWS, NLAYS, NSPC = 6, 5, 8, 2
SYNC = 180


def setup(mismatch: float = 0.02, substeps: int = 4):
    """A small but complete state: both operators, several layers, rho*J last."""
    ds = sigma_layer_thickness(np.linspace(1.0, 0.0, NLAYS + 1) ** 0.625)
    cfg = GridConfig(
        ncols=NCOLS,
        nrows=NROWS,
        ds=ds,
        dx1=12000.0,
        dx2=12000.0,
        nspc_adv=NSPC,
        ppm=replace(DEFAULT_PPM, max_substeps=substeps),
    )
    rng = np.random.default_rng(20260909)
    rhoj = 1.5 + 0.4 * rng.random((NCOLS, NROWS, NLAYS))
    state = jnp.asarray(np.stack([2.0 * rhoj, rhoj], axis=-1))
    uhat = jnp.full((NCOLS + 1, NROWS, NLAYS), 20.0)
    vhat = jnp.full((NCOLS, NROWS + 1, NLAYS), 15.0)
    edge = np.array([4.0, 2.0])
    bcon = BoundaryConditions(
        *(
            jnp.asarray(np.broadcast_to(edge, (n, NLAYS, NSPC)))
            for n in (NROWS, NROWS, NCOLS, NCOLS)
        )
    )
    met = Meteorology(
        uhat=uhat, vhat=vhat, rhoj_met=jnp.asarray(rhoj * (1.0 + mismatch)), bcon=bcon
    )
    return cfg, nonuniform_mesh(ds), state, met


def step(state, uhat, vhat, cfg, *, mesh, met):
    replaced = Meteorology(uhat=uhat, vhat=vhat, rhoj_met=met.rhoj_met, bcon=met.bcon)
    out, _ = advect_step(
        state,
        replaced,
        mesh,
        cfg=cfg,
        astep_seconds=np.full(NLAYS, SYNC),
        sync_seconds=SYNC,
        xyfirst=(True,) * NLAYS,
    )
    return out


def central_difference(fn, array, index, eps=1e-5) -> float:
    return float((fn(array.at[index].add(eps)) - fn(array.at[index].add(-eps))) / (2.0 * eps))


class TestGradientsAreFinite:
    """Every place a masked branch could poison the reverse pass.

    The forward value is correct in all of these; only the gradient breaks, so
    a forward-only test would pass while ``jax.grad`` returned NaN everywhere.
    """

    @pytest.mark.parametrize("mismatch", [0.0, 0.02, 0.3, 0.8])
    def test_across_cfl_regimes(self, mismatch: float) -> None:
        """Including a mismatch large enough to force sub-stepping, and one of
        exactly zero, where the diagnosed flux vanishes entirely."""
        cfg, mesh, state, met = setup(mismatch)
        grads = jax.grad(
            lambda s, u, v: jnp.sum(step(s, u, v, cfg, mesh=mesh, met=met) ** 2), argnums=(0, 1, 2)
        )(state, met.uhat, met.vhat)
        for name, g in zip(("state", "uhat", "vhat"), grads, strict=True):
            assert jnp.all(jnp.isfinite(g)), f"{name} gradient has NaN at mismatch {mismatch}"

    def test_finished_columns_do_not_poison_the_gradient(self) -> None:
        """The sub-step loop keeps calling the column solve with ``dt = 0`` once
        a column is done. That makes flux and target both zero, and a relative
        floor alone would then divide 0 by 0 -- masked in the forward pass,
        fatal in the reverse one. Many sub-steps make this near-certain.
        """
        cfg, mesh, state, met = setup(mismatch=0.02, substeps=12)
        grad = jax.grad(
            lambda s: jnp.sum(step(s, met.uhat, met.vhat, cfg, mesh=mesh, met=met) ** 2)
        )(state)
        assert jnp.all(jnp.isfinite(grad))

    def test_vertical_alone(self) -> None:
        cfg, mesh, state, met = setup(mismatch=0.2)
        column = jnp.moveaxis(state, 2, 0)
        column_met = jnp.moveaxis(met.rhoj_met, 2, 0)
        grad = jax.grad(
            lambda c: jnp.sum(
                zadv(c, column_met, cfg.ds, mesh, dt=float(SYNC), ppm=cfg.ppm)[0] ** 2
            )
        )(column)
        assert jnp.all(jnp.isfinite(grad))

    def test_horizontal_alone(self) -> None:
        cfg, _, state, met = setup()
        grad = jax.grad(
            lambda u: jnp.sum(
                hadv_step(
                    state,
                    u,
                    met.vhat,
                    met.bcon,
                    cfg=cfg,
                    astep_seconds=np.full(NLAYS, SYNC),
                    sync_seconds=SYNC,
                    xyfirst=(True,) * NLAYS,
                )
                ** 2
            )
        )(met.uhat)
        assert jnp.all(jnp.isfinite(grad))


class TestGradientsAreCorrect:
    """Finite differences, not just finiteness."""

    @pytest.mark.parametrize("index", [(3, 2, 4), (0, 1, 0), (6, 4, 7)])
    def test_wrt_zonal_wind(self, index: tuple[int, ...]) -> None:
        cfg, mesh, state, met = setup()

        def loss(u):
            return jnp.sum(step(state, u, met.vhat, cfg, mesh=mesh, met=met) ** 2)

        analytic = float(jax.grad(loss)(met.uhat)[index])
        numeric = central_difference(loss, met.uhat, index)
        assert abs(analytic - numeric) <= 1e-4 * max(abs(numeric), 1.0)

    @pytest.mark.parametrize("index", [(2, 3, 5), (0, 0, 0)])
    def test_wrt_meridional_wind(self, index: tuple[int, ...]) -> None:
        cfg, mesh, state, met = setup()

        def loss(v):
            return jnp.sum(step(state, met.uhat, v, cfg, mesh=mesh, met=met) ** 2)

        analytic = float(jax.grad(loss)(met.vhat)[index])
        numeric = central_difference(loss, met.vhat, index)
        assert abs(analytic - numeric) <= 1e-4 * max(abs(numeric), 1.0)

    @pytest.mark.parametrize("index", [(2, 3, 5, 0), (1, 1, 1, 1), (0, 0, 0, 0)])
    def test_wrt_initial_state(self, index: tuple[int, ...]) -> None:
        """Index ``(..., 1)`` is the rho*J slot, which reaches the answer twice
        over: once as an advected field and once through the flux diagnosis."""
        cfg, mesh, state, met = setup()

        def loss(s):
            return jnp.sum(step(s, met.uhat, met.vhat, cfg, mesh=mesh, met=met) ** 2)

        analytic = float(jax.grad(loss)(state)[index])
        numeric = central_difference(loss, state, index)
        assert abs(analytic - numeric) <= 1e-4 * max(abs(numeric), 1.0)


class TestComposition:
    def test_jit_and_grad_compose(self) -> None:
        cfg, mesh, state, met = setup()

        def loss(s):
            return jnp.sum(step(s, met.uhat, met.vhat, cfg, mesh=mesh, met=met) ** 2)

        eager = jax.grad(loss)(state)
        compiled = jax.jit(jax.grad(loss))(state)
        np.testing.assert_allclose(np.asarray(eager), np.asarray(compiled), rtol=1e-12)

    def test_gradient_of_a_multi_step_run(self) -> None:
        """Sensitivity to the initial state after several sync steps -- the
        shape a parameter-estimation or adjoint problem actually takes."""
        cfg, mesh, state, met = setup()

        def loss(s):
            current = s
            for _ in range(3):
                current = step(current, met.uhat, met.vhat, cfg, mesh=mesh, met=met)
            return jnp.sum(current[..., 0] ** 2)

        index = (2, 2, 3, 0)
        analytic = float(jax.grad(loss)(state)[index])
        numeric = central_difference(loss, state, index)
        assert jnp.all(jnp.isfinite(jax.grad(loss)(state)))
        assert abs(analytic - numeric) <= 1e-4 * max(abs(numeric), 1.0)


class TestTheOperatorHasAKinkAtZero:
    """PPM's limiter and the outflow condition both clamp at zero.

    A tracer sitting exactly at zero is at a corner of the operator: the
    gradient there is a one-sided derivative, and a *central* finite difference
    straddles the corner and matches neither side. That is a property of the
    scheme rather than a defect in the port, but it is a trap for anyone
    validating gradients from a clean initial state -- the natural first thing
    to try.

    It takes a few steps to appear. Measured worst ``|grad - fd|`` as a
    fraction of peak sensitivity:

    ==========  =======  =======  =======
    background  1 step   4 steps  8 steps
    ==========  =======  =======  =======
    0.0         0.0      8.4e-2   2.9e-1
    1.0         9.2e-12  6.0e-11  1.1e-10
    ==========  =======  =======  =======

    One step from an all-zero field agrees exactly: the perturbation has not
    spread far enough to flip a limiter branch yet. That is why these tests run
    several steps, and why an earlier single-step version of this reported no
    disagreement at all.
    """

    STEPS = 4

    def _loss(self, cfg, mesh, met):
        def loss(initial):
            current = initial
            for _ in range(self.STEPS):
                current = step(current, met.uhat, met.vhat, cfg, mesh=mesh, met=met)
            return current[3, 2, 4, 0]

        return loss

    @staticmethod
    def _seed(state, background: float):
        rhoj = state[..., -1]
        return jnp.stack([background * rhoj, rhoj], axis=-1)

    def test_a_positive_background_agrees_with_finite_differences(self) -> None:
        cfg, mesh, state, met = setup()
        loss = self._loss(cfg, mesh, met)
        seeded = self._seed(state, 1.0)
        gradient = jax.grad(loss)(seeded)
        peak = float(jnp.abs(gradient).max())

        for index in ((2, 2, 3, 0), (1, 3, 2, 0), (3, 1, 5, 0)):
            numeric = central_difference(loss, seeded, index)
            assert abs(float(gradient[index]) - numeric) <= 1e-6 * peak

    def test_a_zero_background_does_not_and_that_is_expected(self) -> None:
        """Documented rather than worked around.

        If this ever starts agreeing, the limiter has stopped clamping and
        something more important than a finite-difference check is wrong.
        """
        cfg, mesh, state, met = setup()
        loss = self._loss(cfg, mesh, met)
        seeded = self._seed(state, 0.0)
        gradient = jax.grad(loss)(seeded)
        assert jnp.all(jnp.isfinite(gradient)), "the gradient must stay finite at the kink"

        peak = float(jnp.abs(gradient).max())
        worst = max(
            abs(float(gradient[index]) - central_difference(loss, seeded, index))
            for index in ((2, 2, 3, 0), (1, 3, 2, 0), (3, 1, 5, 0))
        )
        assert worst > 1e-3 * peak, (
            "a central difference now agrees at zero concentration; the limiter's "
            "clamp appears to have gone"
        )

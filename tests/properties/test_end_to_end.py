"""A3.3 — properties of the complete advection operator.

The horizontal and vertical suites each check their own operator. These check
the pair, over many sync steps, driven by a schedule ``advstep`` chose — which
is the configuration a real run is actually in, and the one where an interaction
between the two would show up.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import numpy as np
import pytest

from cmaq_jax.advstep import StepLimits, advstep, sync_top_layer, wind_index
from cmaq_jax.api import Meteorology, advance_xyfirst, advect_step
from cmaq_jax.config import DEFAULT_PPM, GridConfig, sigma_layer_thickness
from cmaq_jax.hadv import BoundaryConditions, hadv_step
from cmaq_jax.ppm import nonuniform_mesh
from cmaq_jax.vadv import zadv

NCOLS, NROWS, NLAYS = 16, 14, 10
OUTPUT_SECONDS = 3600


def build(
    *,
    coupled_q: list[float] | None = None,
    tracers: list[np.ndarray] | None = None,
    boundary_q: list[float] | None = None,
    wind_scale: float = 1.0,
    mismatch: float = 0.01,
    substeps: int = 6,
    dtype: str = "float64",
):
    """A complete state plus the schedule ``advstep`` picks for its wind."""
    faces = np.linspace(1.0, 0.0, NLAYS + 1) ** 0.625
    ds = sigma_layer_thickness(faces)

    rng = np.random.default_rng(20260912)
    rhoj = 1.5 + 0.4 * rng.random((NCOLS, NROWS, NLAYS))

    if coupled_q is not None:
        fields = [q * rhoj for q in coupled_q]
    elif tracers is not None:
        fields = [t * rhoj for t in tracers]
    else:
        fields = [1.0 + rng.random((NCOLS, NROWS, NLAYS))]
    state = np.stack([*fields, rhoj], axis=-1)
    nspc = state.shape[-1]

    cfg = GridConfig(
        ncols=NCOLS,
        nrows=NROWS,
        ds=ds,
        dx1=12000.0,
        dx2=12000.0,
        nspc_adv=nspc,
        dtype=dtype,  # type: ignore[arg-type]
        ppm=replace(DEFAULT_PPM, max_substeps=substeps),
    )

    profile = np.linspace(1.0, 3.0, NLAYS)
    uhat = wind_scale * 9.0 * profile[None, None, :] * np.ones((NCOLS + 1, NROWS, 1))
    vhat = wind_scale * 7.0 * profile[None, None, :] * np.ones((NCOLS, NROWS + 1, 1))

    if boundary_q is None:
        edge = np.zeros(nspc)
        edge[-1] = 2.0
    else:
        edge = np.array([*(q * 2.0 for q in boundary_q), 2.0])
    bcon = BoundaryConditions(
        *(np.broadcast_to(edge, (n, NLAYS, nspc)) for n in (NROWS, NROWS, NCOLS, NCOLS))
    )
    met = Meteorology(uhat=uhat, vhat=vhat, rhoj_met=rhoj * (1.0 + mismatch), bcon=bcon)

    limits = StepLimits()
    schedule = advstep(
        wind_index(uhat, vhat, cfg.dx1, cfg.dx2),
        np.zeros(NLAYS),
        OUTPUT_SECONDS,
        limits,
        sync_layers=sync_top_layer(faces, limits.sigma_sync_top),
    )
    return cfg, nonuniform_mesh(ds), state, met, schedule


def run(cfg, mesh, state, met, schedule, *, nsteps: int):
    """Advance `nsteps` sync steps, jitting each alternation phase once."""
    phases = {}
    current, xyfirst = state, (True,) * NLAYS
    worst_residual = 0.0
    for _ in range(nsteps):
        if xyfirst not in phases:
            phases[xyfirst] = jax.jit(
                partial(
                    advect_step,
                    mesh=mesh,
                    cfg=cfg,
                    astep_seconds=schedule.astep_seconds,
                    sync_seconds=schedule.sync_seconds,
                    xyfirst=xyfirst,
                )
            )
        current, diag = phases[xyfirst](current, met)
        xyfirst = advance_xyfirst(xyfirst, schedule.astep_seconds, schedule.sync_seconds)
        worst_residual = max(worst_residual, float(np.asarray(diag.residual).max()))
    return np.asarray(current), worst_residual


class TestStability:
    def test_the_advstep_schedule_keeps_the_run_bounded(self) -> None:
        """The point of ``advstep``: a schedule it approves must not blow up.

        PPM above Courant one does not degrade gracefully -- it overflows -- so
        a run that stays finite over many steps is real evidence the CFL search
        is doing its job.
        """
        cfg, mesh, state, met, schedule = build(wind_scale=3.0)
        final, residual = run(cfg, mesh, state, met, schedule, nsteps=30)
        assert np.all(np.isfinite(final))
        assert final.max() < 1e3 * state.max(), "the run is growing without bound"
        assert np.isfinite(residual), "a column ran out of vertical sub-steps"

    def test_positivity_over_many_steps(self) -> None:
        rng = np.random.default_rng(7)
        spike = np.zeros((NCOLS, NROWS, NLAYS))
        spike[NCOLS // 2, NROWS // 2, NLAYS // 2] = 20.0
        cfg, mesh, state, met, schedule = build(tracers=[spike], wind_scale=2.0)
        final, _ = run(cfg, mesh, state, met, schedule, nsteps=25)
        assert final.min() >= 0.0, f"negative concentration: {final.min()}"
        del rng


class TestConstancy:
    @pytest.mark.parametrize("wind_scale", [1.0, 3.0])
    def test_a_uniform_mixing_ratio_survives_both_operators(self, wind_scale: float) -> None:
        """The invariant that ties the whole operator together.

        It has to hold through the horizontal sweeps, their alternation, the
        vertical flux diagnosis and its sub-stepping -- all of which move rho*J
        and every species with it. Any one of them dropping the ride-along
        breaks this and nothing else would notice.
        """
        q = [0.75, 3.0]
        cfg, mesh, state, met, schedule = build(coupled_q=q, boundary_q=q, wind_scale=wind_scale)
        final, _ = run(cfg, mesh, state, met, schedule, nsteps=20)
        for spc, expected in enumerate(q):
            np.testing.assert_allclose(final[..., spc] / final[..., -1], expected, rtol=1e-9)

    def test_constancy_holds_in_float32(self) -> None:
        """CMAQ's own precision, and the likeliest choice on a GPU."""
        q = [0.75, 3.0]
        cfg, mesh, state, met, schedule = build(
            coupled_q=q, boundary_q=q, wind_scale=2.0, dtype="float32"
        )
        final, _ = run(cfg, mesh, state, met, schedule, nsteps=10)
        assert final.dtype == np.float32
        for spc, expected in enumerate(q):
            np.testing.assert_allclose(final[..., spc] / final[..., -1], expected, rtol=1e-5)


class TestMass:
    def test_a_compact_plume_conserves_mass(self) -> None:
        """Nothing crosses the vertical boundaries at all, and the horizontal
        edges only matter if material reaches them -- so with a plume kept clear
        of the sides and a clean inflow, the total is exactly preserved.

        Stated the other way round: any loss here is a leak in the operator, not
        physics, because there is nowhere for the mass to have gone.
        """
        centre = np.zeros((NCOLS, NROWS, NLAYS))
        cols, rows, lays = np.meshgrid(
            np.arange(NCOLS), np.arange(NROWS), np.arange(NLAYS), indexing="ij"
        )
        centre = np.exp(
            -(
                ((cols - NCOLS / 2) / 1.5) ** 2
                + ((rows - NROWS / 2) / 1.5) ** 2
                + ((lays - NLAYS / 2) / 1.5) ** 2
            )
        )
        centre[centre < 1e-12] = 0.0

        # Slow enough, and short enough, that the plume stays clear of every
        # wall -- the assertion below refuses to draw a conclusion otherwise.
        cfg, mesh, state, met, schedule = build(tracers=[centre], mismatch=0.0, wind_scale=0.05)
        final, _ = run(cfg, mesh, state, met, schedule, nsteps=4)

        def mass(field: np.ndarray) -> float:
            return float(np.einsum("l,crl->", cfg.ds, field))

        total = mass(state[..., 0])
        edge_total = (
            final[0, :, :, 0].sum()
            + final[-1, :, :, 0].sum()
            + final[:, 0, :, 0].sum()
            + final[:, -1, :, 0].sum()
        )
        # Relative, and generous, because what reaches the wall is a diffusion
        # tail rather than the plume. The two scale together -- measured, an
        # edge fraction of 4.5e-6 comes with a mass drift of 4.2e-9 -- so the
        # guard is set well above the tail and the tolerance well above what
        # the tail can carry off.
        assert edge_total < 1e-4 * total, "the plume itself reached a wall"
        np.testing.assert_allclose(mass(final[..., 0]), total, rtol=1e-7)


class TestDensity:
    def test_the_vertical_step_narrows_the_gap_the_horizontal_one_opens(self) -> None:
        """What the vertical operator exists for.

        The comparison is against the *same run without it*, not against the
        starting gap. Holding the meteorology fixed while horizontal advection
        keeps transporting means the gap grows on its own -- in a real run the
        met field evolves alongside -- so "the gap shrank" is the wrong
        question and an earlier version of this test asked it and failed.
        """
        cfg, mesh, state, met, schedule = build(mismatch=0.05, wind_scale=2.0)

        def gap(field: np.ndarray) -> float:
            return float(np.abs(field[..., -1] - np.asarray(met.rhoj_met)).max())

        horizontal_only = state
        both = state
        for _ in range(6):
            horizontal_only = np.asarray(
                hadv_step(
                    horizontal_only,
                    met.uhat,
                    met.vhat,
                    met.bcon,
                    cfg=cfg,
                    astep_seconds=schedule.astep_seconds,
                    sync_seconds=schedule.sync_seconds,
                    xyfirst=(True,) * NLAYS,
                )
            )
            both, _ = advect_step(
                both,
                met,
                mesh,
                cfg=cfg,
                astep_seconds=schedule.astep_seconds,
                sync_seconds=schedule.sync_seconds,
                xyfirst=(True,) * NLAYS,
            )
            both = np.asarray(both)

        assert gap(both) < gap(horizontal_only), (
            f"the vertical step did not help: {gap(both):.4g} with it, "
            f"{gap(horizontal_only):.4g} without"
        )


class TestOrder:
    def test_horizontal_runs_before_vertical(self) -> None:
        """``sciproc.F`` runs HADV then ZADV, and the order is load-bearing: the
        vertical flux is diagnosed from a gap that horizontal advection opens.
        Swapping them gives a different answer, so this is not a free choice.
        """
        cfg, mesh, state, met, schedule = build(mismatch=0.05, wind_scale=2.0)
        forward, _ = run(cfg, mesh, state, met, schedule, nsteps=1)

        # Same inputs, vertical applied to the untouched state instead.
        column = np.moveaxis(state, 2, 0)
        column_met = np.moveaxis(np.asarray(met.rhoj_met), 2, 0)
        vertical_first, _ = zadv(
            column, column_met, cfg.ds, mesh, dt=float(schedule.sync_seconds), ppm=cfg.ppm
        )
        assert not np.allclose(np.moveaxis(np.asarray(vertical_first), 0, 2), forward, rtol=1e-6)

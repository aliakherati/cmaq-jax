"""A1.6 — scheme properties through the real horizontal-advection driver.

The A0 property tests drove the bare 1-D kernel with periodic wrapping. These
drive :func:`cmaq_jax.hadv.hadv_step`, so they exercise what the driver adds:
real boundary conditions, both sweeps, the X-Y/Y-X alternation and per-layer
sub-stepping.
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np
import pytest

from cmaq_jax.config import GridConfig
from cmaq_jax.hadv import BoundaryConditions, advance_xyfirst, hadv_step

DX = 1000.0
ROTATION_PERIOD = 3600.0
ROTATION_N = 48
ROTATION_DT = 8


def solid_body_wind(n: int, nlays: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Rigid rotation about the domain centre, on the staggered faces.

    Discretely non-divergent: ``u`` varies only with row and ``v`` only with
    column, so every cell's flux divergence is exactly zero. That makes rho*J
    a conserved constant and turns any change in it into a bug signal.
    """
    length = n * DX
    omega = 2.0 * np.pi / ROTATION_PERIOD
    centres = (np.arange(n) + 0.5) * DX
    u = np.broadcast_to((-omega * (centres - length / 2))[None, :, None], (n + 1, n, nlays))
    v = np.broadcast_to((omega * (centres - length / 2))[:, None, None], (n, n + 1, nlays))
    return np.array(u), np.array(v)


def cone(n: int, centre_frac: tuple[float, float], radius_frac: float) -> np.ndarray:
    length = n * DX
    centres = (np.arange(n) + 0.5) * DX
    x, y = np.meshgrid(centres, centres, indexing="ij")
    r = np.hypot(x - centre_frac[0] * length, y - centre_frac[1] * length)
    radius = radius_frac * length
    return np.where(r <= radius, 1.0 - r / radius, 0.0)


def make_state(
    tracers: list[np.ndarray], rhoj: np.ndarray, bcon_values: list[float], rhoj_bcon: float
) -> tuple[np.ndarray, BoundaryConditions]:
    """Assemble a coupled CGRID and matching boundary ring.

    Slot -1 is rho*J, as CMAQ requires; the tracers arrive already coupled.
    """
    cgrid = np.stack([*tracers, rhoj], axis=-1)
    ncols, nrows, nlays, nspc = cgrid.shape
    edge = np.array([*bcon_values, rhoj_bcon])
    return cgrid, BoundaryConditions(
        west=np.broadcast_to(edge, (nrows, nlays, nspc)),
        east=np.broadcast_to(edge, (nrows, nlays, nspc)),
        south=np.broadcast_to(edge, (ncols, nlays, nspc)),
        north=np.broadcast_to(edge, (ncols, nlays, nspc)),
    )


def run(
    cgrid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bcon: BoundaryConditions,
    *,
    nsteps: int,
    dt: int,
    n: int,
) -> np.ndarray:
    """Advance `nsteps` sync steps, jitting each alternation phase once.

    The flags alternate with period two, so two traces cover every step.
    Without the jit each step is dispatch-bound and costs ~1000x the
    arithmetic in it.
    """
    courant = max(np.abs(u).max(), np.abs(v).max()) * dt / DX
    assert courant < 1.0, (
        f"test setup has Courant {courant:.2f}; PPM is unstable above 1 and the "
        "result will overflow rather than merely be inaccurate"
    )

    nlays = cgrid.shape[2]
    cfg = GridConfig(
        ncols=n, nrows=n, ds=np.full(nlays, 1.0 / nlays), dx1=DX, dx2=DX, nspc_adv=cgrid.shape[-1]
    )
    astep = np.full(nlays, dt)
    steps = {
        phase: jax.jit(
            partial(hadv_step, cfg=cfg, astep_seconds=astep, sync_seconds=dt, xyfirst=phase)
        )
        for phase in ((True,) * nlays, (False,) * nlays)
    }

    state = cgrid
    phase = (True,) * nlays
    for _ in range(nsteps):
        state = steps[phase](state, u, v, bcon)
        phase = advance_xyfirst(phase, astep, dt)
    return np.asarray(state)


@pytest.fixture(scope="module")
def rotated() -> tuple[np.ndarray, np.ndarray]:
    """One full revolution of a cone. Shared, since it is the slowest run here."""
    n = ROTATION_N
    u, v = solid_body_wind(n)
    blob = cone(n, (0.5, 0.75), 0.12)[:, :, None]
    rhoj = np.ones((n, n, 1))
    cgrid, bcon = make_state([blob * rhoj], rhoj, [0.0], 1.0)
    nsteps = round(ROTATION_PERIOD / ROTATION_DT)
    return cgrid, run(cgrid, u, v, bcon, nsteps=nsteps, dt=ROTATION_DT, n=n)


class TestSolidBodyRotation:
    """A cone carried once around by a rigid rotation. The exact answer after a
    full turn is the initial field."""

    N = ROTATION_N

    def test_returns_to_where_it_started(self, rotated: tuple[np.ndarray, np.ndarray]) -> None:
        """Centroid displacement is the phase error -- the serious defect. Edge
        smearing is expected; transporting at the wrong speed is not."""
        initial, final = rotated
        n = self.N
        centres = (np.arange(n) + 0.5) * DX
        x, y = np.meshgrid(centres, centres, indexing="ij")

        def centroid(field: np.ndarray) -> tuple[float, float]:
            w = field[:, :, 0, 0]
            return float((x * w).sum() / w.sum()), float((y * w).sum() / w.sum())

        (x0, y0), (x1, y1) = centroid(initial), centroid(final)
        assert np.hypot(x1 - x0, y1 - y0) / DX < 0.1, "phase error exceeds a tenth of a cell"

    def test_density_is_exactly_unchanged(self, rotated: tuple[np.ndarray, np.ndarray]) -> None:
        """The wind is discretely non-divergent, so rho*J must not move at all
        -- not merely stay close."""
        initial, final = rotated
        np.testing.assert_allclose(final[..., -1], initial[..., -1], rtol=1e-12)

    def test_no_new_extrema(self, rotated: tuple[np.ndarray, np.ndarray]) -> None:
        initial, final = rotated
        assert final[..., 0].min() >= -1e-12
        assert final[..., 0].max() <= initial[..., 0].max() + 1e-12

    def test_mass_conserved(self, rotated: tuple[np.ndarray, np.ndarray]) -> None:
        """The cone stays clear of the edges and the inflow is clean, so
        nothing enters or leaves."""
        initial, final = rotated
        total = initial[..., 0].sum()
        edge_mass = (
            final[0, :, 0, 0].sum()
            + final[-1, :, 0, 0].sum()
            + final[:, 0, 0, 0].sum()
            + final[:, -1, 0, 0].sum()
        )
        # Relative, not absolute: after 450 steps a diffusion tail reaches the
        # edge at the 1e-8-of-total level, which is not the cone arriving.
        assert edge_mass < 1e-6 * total, "the cone reached a boundary; the test is invalid"
        np.testing.assert_allclose(final[..., 0].sum(), total, rtol=1e-6)

    def test_diffuses_but_does_not_amplify(self, rotated: tuple[np.ndarray, np.ndarray]) -> None:
        """A full turn should cost some peak to numerical diffusion, and only
        cost it. No loss at all would mean the cone never actually moved."""
        initial, final = rotated
        ratio = final[..., 0].max() / initial[..., 0].max()
        assert 0.5 < ratio < 1.0, f"peak ratio {ratio} is not a plausible diffusion loss"


class TestBoundaries:
    def test_inflow_carries_the_boundary_value(self) -> None:
        """A clean domain with a dirty boundary must fill from the edge the
        wind enters through, and only that edge."""
        n, dt = 24, 20  # Courant = 30 * 20 / 1000 = 0.6
        u = np.full((n + 1, n, 1), 30.0)
        v = np.zeros((n, n + 1, 1))
        rhoj = np.ones((n, n, 1))
        cgrid, bcon = make_state([np.zeros((n, n, 1))], rhoj, [4.0], 1.0)

        out = run(cgrid, u, v, bcon, nsteps=20, dt=dt, n=n)
        q = out[..., 0, 0] / out[..., 0, -1]
        assert q[0, 0] > 1.0, "west edge is inflow and should have filled"
        assert q[-1, 0] < 1e-9, "east edge is outflow and should still be clean"

    def test_outflow_does_not_reflect(self) -> None:
        """A blob leaving through an outflow edge should leave. If the boundary
        reflected, mass would pile up against it instead."""
        n, dt = 24, 15  # Courant = 40 * 15 / 1000 = 0.6
        u = np.full((n + 1, n, 1), 40.0)
        v = np.zeros((n, n + 1, 1))
        rhoj = np.ones((n, n, 1))
        blob = np.zeros((n, n, 1))
        blob[n // 2 :, :, :] = 1.0
        cgrid, bcon = make_state([blob * rhoj], rhoj, [0.0], 1.0)

        out = run(cgrid, u, v, bcon, nsteps=120, dt=dt, n=n)
        assert out[..., 0].sum() < 0.05 * cgrid[..., 0].sum(), "material did not leave"
        assert out[..., 0].min() >= -1e-12, "reflection produced an undershoot"


class TestConstancy:
    def test_uniform_mixing_ratio_survives_many_steps(self) -> None:
        """The CMAQ invariant, through both sweeps, real boundaries and the
        alternation, over enough steps for a slow leak to show."""
        n, dt = 32, 20
        u, v = solid_body_wind(n)
        # Break the non-divergence so the test is not trivially satisfied.
        u = u + 12.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, n + 1))[:, None, None]

        rng = np.random.default_rng(20260902)
        rhoj = 1.5 + 0.4 * rng.random((n, n, 1))
        q = [0.5, 3.0]
        cgrid, bcon = make_state([qq * rhoj for qq in q], rhoj, [qq * 2.0 for qq in q], 2.0)

        out = run(cgrid, u, v, bcon, nsteps=60, dt=dt, n=n)
        for spc, q_expected in enumerate(q):
            np.testing.assert_allclose(out[..., spc] / out[..., -1], q_expected, rtol=1e-10)


class TestLayers:
    def test_layers_with_different_astep_do_not_interact(self) -> None:
        """Two layers, different sub-step counts, advanced together must match
        each advanced alone. The layer grouping is where this could break."""
        n = 24
        u, v = solid_body_wind(n, nlays=2)
        blob = cone(n, (0.5, 0.7), 0.12)[:, :, None]
        rhoj = np.ones((n, n, 2))
        tracer = np.concatenate([blob, 2.0 * blob], axis=2)
        cgrid, bcon = make_state([tracer * rhoj], rhoj, [0.0], 1.0)

        cfg = GridConfig(
            ncols=n, nrows=n, ds=np.full(2, 0.5), dx1=DX, dx2=DX, nspc_adv=cgrid.shape[-1]
        )
        astep = np.array([30, 60])
        together = np.asarray(
            hadv_step(
                cgrid,
                u,
                v,
                bcon,
                cfg=cfg,
                astep_seconds=astep,
                sync_seconds=60,
                xyfirst=(True, True),
            )
        )

        for layer in range(2):
            one = GridConfig(
                ncols=n, nrows=n, ds=np.full(1, 1.0), dx1=DX, dx2=DX, nspc_adv=cgrid.shape[-1]
            )
            alone = np.asarray(
                hadv_step(
                    cgrid[:, :, layer : layer + 1],
                    u[:, :, layer : layer + 1],
                    v[:, :, layer : layer + 1],
                    BoundaryConditions(
                        *(getattr(bcon, e)[:, layer : layer + 1] for e in bcon._fields)
                    ),
                    cfg=one,
                    astep_seconds=astep[layer : layer + 1],
                    sync_seconds=60,
                    xyfirst=(True,),
                )
            )
            np.testing.assert_allclose(alone[:, :, 0], together[:, :, layer], rtol=1e-12)


class TestFloat32:
    """The invariants must survive CMAQ's own precision, not just float64.

    float32 is what CMAQ runs in and what a GPU is likeliest to want for memory
    bandwidth, so the guarantees have to hold there. The tolerances are looser
    by roughly the ratio of the two epsilons, and nothing else changes.
    """

    N = 32
    DT = 20

    @staticmethod
    def _cfg(n: int, nlays: int, nspc: int) -> GridConfig:
        return GridConfig(
            ncols=n,
            nrows=n,
            ds=np.full(nlays, 1.0 / nlays),
            dx1=DX,
            dx2=DX,
            nspc_adv=nspc,
            dtype="float32",
        )

    def _advect(
        self,
        cgrid: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        bcon: BoundaryConditions,
        nsteps: int,
    ) -> np.ndarray:
        cfg = self._cfg(self.N, cgrid.shape[2], cgrid.shape[-1])
        astep = np.full(cgrid.shape[2], self.DT)
        state, phase = cgrid, (True,) * cgrid.shape[2]
        for _ in range(nsteps):
            state = hadv_step(
                state, u, v, bcon, cfg=cfg, astep_seconds=astep, sync_seconds=self.DT, xyfirst=phase
            )
            phase = advance_xyfirst(phase, astep, self.DT)
        assert state.dtype == np.float32
        return np.asarray(state)

    def test_positivity_and_monotonicity(self) -> None:
        """A sharp spike on a clean background: any undershoot is immediately
        negative, which is how a limiter failure shows up in float32."""
        n = self.N
        u, v = solid_body_wind(n)
        spike = np.zeros((n, n, 1))
        spike[n // 2, 3 * n // 4, 0] = 10.0
        rhoj = np.ones((n, n, 1))
        cgrid, bcon = make_state([spike * rhoj], rhoj, [0.0], 1.0)

        out = self._advect(cgrid.astype(np.float32), u, v, bcon, nsteps=40)
        assert out[..., 0].min() >= 0.0, f"negative concentration: {out[..., 0].min()}"
        assert out[..., 0].max() <= 10.0 + 1e-4

    def test_density_is_still_exactly_unchanged(self) -> None:
        """The wind is discretely non-divergent, so rho*J must hold in float32
        too -- to float32 round-off, not to float64's."""
        n = self.N
        u, v = solid_body_wind(n)
        rhoj = np.ones((n, n, 1))
        cgrid, bcon = make_state([np.zeros((n, n, 1))], rhoj, [0.0], 1.0)
        out = self._advect(cgrid.astype(np.float32), u, v, bcon, nsteps=40)
        np.testing.assert_allclose(out[..., -1], 1.0, rtol=1e-6)

    def test_constancy_holds_at_float32_precision(self) -> None:
        """The CMAQ invariant, in CMAQ's precision, under a divergent wind."""
        n = self.N
        u, v = solid_body_wind(n)
        u = u + 12.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, n + 1))[:, None, None]

        rng = np.random.default_rng(20260903)
        rhoj = 1.5 + 0.4 * rng.random((n, n, 1))
        q = [0.5, 3.0]
        cgrid, bcon = make_state([qq * rhoj for qq in q], rhoj, [qq * 2.0 for qq in q], 2.0)

        out = self._advect(cgrid.astype(np.float32), u, v, bcon, nsteps=40)
        for spc, q_expected in enumerate(q):
            np.testing.assert_allclose(out[..., spc] / out[..., -1], q_expected, rtol=1e-5)

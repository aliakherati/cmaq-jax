"""B2.3 — the full transport chain: HADV, ZADV, HDIFF.

Each operator has its own suite. These check the composition, over many sync
steps, which is the configuration a real run is in and where an interaction
between them would show up.
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np

from cmaq_jax.advstep import StepLimits, advstep, sync_top_layer, wind_index
from cmaq_jax.api import Diffusivity, Meteorology, advance_xyfirst, advect_step, transport_step
from cmaq_jax.config import GridConfig, sigma_layer_thickness
from cmaq_jax.hadv import BoundaryConditions
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
from cmaq_jax.ppm import nonuniform_mesh

NCOLS, NROWS, NLAYS = 12, 10, 6
DX = 4000.0


def build(*, coupled_q: list[float] | None = None, seed: int = 20260827):
    rng = np.random.default_rng(seed)
    faces = np.linspace(1.0, 0.0, NLAYS + 1) ** 0.625
    rhoj = 1.5 + 0.4 * rng.random((NCOLS, NROWS, NLAYS))

    if coupled_q is not None:
        fields = [q * rhoj for q in coupled_q]
    else:
        fields = [1.0 + rng.random((NCOLS, NROWS, NLAYS))]
    state = np.stack([*fields, rhoj], axis=-1)
    nspc = state.shape[-1]

    cfg = GridConfig(
        ncols=NCOLS,
        nrows=NROWS,
        ds=sigma_layer_thickness(faces),
        dx1=DX,
        dx2=DX,
        nspc_adv=nspc,
    )

    profile = np.linspace(1.0, 3.0, NLAYS)
    uhat = 6.0 * profile[None, None, :] * np.ones((NCOLS + 1, NROWS, 1))
    vhat = 4.0 * profile[None, None, :] * np.ones((NCOLS, NROWS + 1, 1))

    if coupled_q is None:
        edge = np.zeros(nspc)
        edge[-1] = 2.0
    else:
        edge = np.array([*(q * 2.0 for q in coupled_q), 2.0])
    bcon = BoundaryConditions(
        *(np.broadcast_to(edge, (n, NLAYS, nspc)) for n in (NROWS, NROWS, NCOLS, NCOLS))
    )
    met = Meteorology(uhat=uhat, vhat=vhat, rhoj_met=rhoj, bcon=bcon)

    # Diffusion inputs. The dot-grid winds carry shear, so the diffusivity is
    # not merely sitting on its floor.
    dot = (NCOLS + 1, NROWS + 1, NLAYS)
    rows = np.arange(NROWS + 1, dtype=np.float64)[None, :, None]
    cols = np.arange(NCOLS + 1, dtype=np.float64)[:, None, None]
    mean_rho = float(rhoj.mean())
    u_jd = (30.0 * rows + 12.0 * cols + rng.normal(0.0, 2.0, dot)) * mean_rho
    v_jd = (18.0 * cols + rng.normal(0.0, 2.0, dot)) * mean_rho
    ring = np.full((2 * (NCOLS + NROWS + 2), NLAYS), mean_rho)

    densj = halo_density(rhoj, ring)
    uu, vv = contravariant_winds(u_jd, v_jd, densj)
    eddyh = eddy_diffusivity(
        deformation(uu, vv, dx1=DX, dx2=DX), np.ones((NCOLS + 1, NROWS + 1)), dx1=DX, dx2=DX
    )
    k11, k22 = face_coefficients(eddyh)
    diffusivity = Diffusivity(densj=densj, k11=k11, k22=k22)

    limits = StepLimits()
    schedule = advstep(
        wind_index(uhat, vhat, cfg.dx1, cfg.dx2),
        np.zeros(NLAYS),
        3600,
        limits,
        sync_layers=sync_top_layer(faces, limits.sigma_sync_top),
    )
    nsteps = substep_count(
        float(schedule.sync_seconds), float(stable_timestep(k11, k22, dx1=DX, dx2=DX))
    )
    return cfg, state, met, diffusivity, nonuniform_mesh(cfg.ds), schedule, nsteps


def run(nsteps_out: int, **kwargs):
    cfg, state, met, diffusivity, mesh, schedule, nsub = build(**kwargs)
    phases: dict[tuple[bool, ...], object] = {}
    current, xyfirst = state, (True,) * NLAYS
    for _ in range(nsteps_out):
        if xyfirst not in phases:
            phases[xyfirst] = jax.jit(
                partial(
                    transport_step,
                    mesh=mesh,
                    cfg=cfg,
                    astep_seconds=schedule.astep_seconds,
                    sync_seconds=schedule.sync_seconds,
                    xyfirst=xyfirst,
                    diffusion_substeps=nsub,
                )
            )
        current, _ = phases[xyfirst](current, met, diffusivity)  # type: ignore[operator]
        xyfirst = advance_xyfirst(xyfirst, schedule.astep_seconds, schedule.sync_seconds)
    return np.asarray(current), state


def test_the_whole_chain_jits_and_stays_bounded() -> None:
    final, state = run(12)
    assert np.all(np.isfinite(final))
    assert final.max() < 1e3 * state.max(), "the run is growing without bound"


def test_uniformity_is_controlled_by_the_density_mismatch() -> None:
    """Uniform mixing ratio is preserved *exactly* by advection alone and by
    diffusion alone — but not by the composition, and that is CMAQ's structure
    rather than a defect.

    ``hdiff.F:309`` gets its density from ``RHO_J``, which reads ``DENSA_J``
    from the meteorology file. It does **not** use the advected rho*J that
    advection has just been transporting in the last ``CGRID`` slot. So
    diffusion divides the coupled concentration by one density while the
    concentration is coupled to another, and a field that is uniform in the
    advected density is not quite uniform in the meteorological one. The two
    agree only to the extent ZADV has closed the gap, which is what ZADV is for.

    Measured here, holding the meteorology fixed across steps so the gap grows:
    a density mismatch of 18-26% comes with a uniformity error of 1.2-2.2%, a
    ratio near 0.08 and stable over ten steps. So the error is *controlled by*
    the mismatch rather than accumulating on its own, which is the invariant
    worth asserting.
    """
    q = [0.75]
    cfg, state, met, diffusivity, mesh, schedule, nsub = build(coupled_q=q)
    step = jax.jit(
        partial(
            transport_step,
            mesh=mesh,
            cfg=cfg,
            astep_seconds=schedule.astep_seconds,
            sync_seconds=schedule.sync_seconds,
            xyfirst=(True,) * NLAYS,
            diffusion_substeps=nsub,
        )
    )
    current = state
    for _ in range(10):
        current, _ = step(current, met, diffusivity)
    final = np.asarray(current)

    rhoj_met = np.asarray(met.rhoj_met)
    gap = np.abs(final[..., -1] - rhoj_met).max() / rhoj_met.mean()
    error = np.abs(final[..., 0] / final[..., -1] - q[0]).max() / q[0]

    assert gap > 1e-3, "the meteorology tracked too well for this test to say anything"
    assert error < 0.25 * gap, f"uniformity error {error:.2e} outgrew the gap {gap:.2e}"


def test_uniformity_is_exact_when_the_densities_agree() -> None:
    """The mechanism behind the test above, isolated.

    Give diffusion the density the state is actually coupled to — rather than
    the meteorological one CMAQ hands it — and the uniformity error vanishes.
    That identifies the mismatch as the cause, rather than leaving it as a
    plausible story about an error that might have any number of sources.

    This is deliberately *not* what ``transport_step`` does; it is a diagnostic.
    """
    q = 0.75
    cfg, state, _met, diffusivity, _mesh, schedule, nsub = build(coupled_q=[q])
    ring_value = float(np.asarray(state[..., -1]).mean())
    ring = np.full((2 * (NCOLS + NROWS + 2), NLAYS), ring_value)
    densj = halo_density(np.asarray(state[..., -1]), ring)

    diffused = np.asarray(
        hdiff_step(
            state,
            densj,
            diffusivity.k11,
            diffusivity.k22,
            cfg=cfg,
            sync_seconds=float(schedule.sync_seconds),
            nsteps=nsub,
        )
    )
    np.testing.assert_allclose(diffused[..., 0] / diffused[..., -1], q, rtol=1e-12)


def test_diffusion_actually_ran() -> None:
    """Guards the tests above: a no-op diffusion would satisfy them too."""
    cfg, state, met, diffusivity, mesh, schedule, nsub = build()
    common = {
        "mesh": mesh,
        "cfg": cfg,
        "astep_seconds": schedule.astep_seconds,
        "sync_seconds": schedule.sync_seconds,
        "xyfirst": (True,) * NLAYS,
    }
    advected, _ = advect_step(state, met, **common)
    both, _ = transport_step(state, met, diffusivity, diffusion_substeps=nsub, **common)
    assert not np.allclose(np.asarray(advected), np.asarray(both)), (
        "adding diffusion changed nothing"
    )


def test_rho_j_still_tracks_the_meteorology() -> None:
    """Advection transports rho*J; diffusion must leave it alone. If diffusion
    smoothed it, the advected density would drift away from the met field over
    several steps.
    """
    cfg, state, met, diffusivity, mesh, schedule, nsub = build()
    step = jax.jit(
        partial(
            transport_step,
            mesh=mesh,
            cfg=cfg,
            astep_seconds=schedule.astep_seconds,
            sync_seconds=schedule.sync_seconds,
            xyfirst=(True,) * NLAYS,
            diffusion_substeps=nsub,
        )
    )
    current = state
    for _ in range(5):
        current, _ = step(current, met, diffusivity)
    gap = np.abs(np.asarray(current)[..., -1] - np.asarray(met.rhoj_met)).max()
    assert gap < 0.5 * float(np.asarray(met.rhoj_met).mean())

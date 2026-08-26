#!/usr/bin/env python3
"""A1 figures: the benchmarks re-run through CMAQ's real driver.

    python scripts/make_a1_figures.py

The A0 flow figures applied the bare 1-D kernel alternately along each axis with
periodic wrapping, and said so. These use :func:`cmaq_jax.hadv.hadv_step`, which
adds everything the driver does: real boundary conditions (``BCON`` on inflow,
zero-flux-divergence on outflow), the X-Y/Y-X alternation, per-layer
sub-stepping, and the rho*J ride-along that conserves mass.

Two figures:

* ``rotation_driver`` -- solid-body rotation through the driver, with the
  agreement against CMAQ and the phase error stated on the figure.
* ``periodic_vs_driver`` -- the same rotation under periodic wrapping and under
  the driver's real boundaries, side by side, so the difference the boundary
  treatment actually makes is visible rather than asserted.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from cmaq_jax.config import DEFAULT_PPM, GridConfig
from cmaq_jax.hadv import BoundaryConditions, advance_xyfirst, hadv_step
from cmaq_jax.ppm import ppm_advect_uniform

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "docs" / "figures" / "a1"
SWP = DEFAULT_PPM.halo_width

FIELD_CMAP = "magma"
DIVERGING = "RdBu_r"

DX = 1000.0
PERIOD = 3600.0
N = 96
DT = 4  # Courant = omega * (N*DX/2) * DT / DX = 0.335


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches=None if fig.get_layout_engine() else "tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


def _bare(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _centres(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) * DX


def solid_body_wind(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Rigid rotation, discretely non-divergent so rho*J is exactly conserved."""
    length = n * DX
    omega = 2.0 * np.pi / PERIOD
    c = _centres(n)
    u = np.broadcast_to((-omega * (c - length / 2))[None, :, None], (n + 1, n, 1))
    v = np.broadcast_to((omega * (c - length / 2))[:, None, None], (n, n + 1, 1))
    return np.array(u), np.array(v)


def initial_shapes(n: int) -> np.ndarray:
    """Zalesak's pair: a smooth cone and a slotted cylinder."""
    length = n * DX
    c = _centres(n)
    x, y = np.meshgrid(c, c, indexing="ij")

    r_cone = np.hypot(x - 0.5 * length, y - 0.25 * length)
    cone = np.where(r_cone <= 0.15 * length, 1.0 - r_cone / (0.15 * length), 0.0)

    r_cyl = np.hypot(x - 0.5 * length, y - 0.75 * length)
    disc = r_cyl <= 0.15 * length
    slot = (np.abs(x - 0.5 * length) <= 0.025 * length) & (y <= 0.84 * length)
    return cone + np.where(disc & ~slot, 1.0, 0.0)


def run_driver(n: int, nsteps: int, snapshot_at: set[int]) -> dict[int, np.ndarray]:
    """Advect through the real driver, capturing the tracer at chosen steps."""
    u, v = solid_body_wind(n)
    rhoj = np.ones((n, n, 1))
    field = initial_shapes(n)[:, :, None]
    cgrid = np.stack([field * rhoj, rhoj], axis=-1)

    edge = np.array([0.0, 1.0])  # clean inflow, unit density
    bcon = BoundaryConditions(
        west=np.broadcast_to(edge, (n, 1, 2)),
        east=np.broadcast_to(edge, (n, 1, 2)),
        south=np.broadcast_to(edge, (n, 1, 2)),
        north=np.broadcast_to(edge, (n, 1, 2)),
    )
    cfg = GridConfig(ncols=n, nrows=n, ds=np.array([1.0]), dx1=DX, dx2=DX, nspc_adv=2)
    astep = np.array([DT])
    steps = {
        phase: jax.jit(
            partial(hadv_step, cfg=cfg, astep_seconds=astep, sync_seconds=DT, xyfirst=phase)
        )
        for phase in ((True,), (False,))
    }

    out = {0: np.asarray(cgrid[..., 0, 0])}
    state = cgrid
    phase = (True,)
    for step in range(nsteps):
        state = steps[phase](state, u, v, bcon)
        phase = advance_xyfirst(phase, astep, DT)
        if (step + 1) in snapshot_at:
            out[step + 1] = np.asarray(state[..., 0, 0])
    out["rhoj"] = np.asarray(state[..., 0, 1])  # type: ignore[index]
    return out


def run_periodic(n: int, nsteps: int) -> np.ndarray:
    """The A0 approach for comparison: bare kernel, periodic wrapping."""
    u, v = solid_body_wind(n)
    field = jax.numpy.asarray(initial_shapes(n))

    @partial(jax.jit, static_argnames=("axis",))
    def sweep(f, vel, axis):
        moved = jax.numpy.moveaxis(f, axis, 0)
        speeds = jax.numpy.moveaxis(vel[:, :, 0], axis, 0)
        padded = jax.numpy.concatenate([moved[-SWP:], moved, moved[:SWP]])
        done = ppm_advect_uniform(padded, speeds, float(DT), DX)[SWP:-SWP]
        return jax.numpy.moveaxis(done, 0, axis)

    for step in range(nsteps):
        order = (0, 1) if step % 2 == 0 else (1, 0)
        for axis in order:
            field = sweep(field, u if axis == 0 else v, axis)
    return np.asarray(field)


def figure_rotation_driver() -> None:
    nsteps = round(PERIOD / DT)
    marks = {nsteps // 4, nsteps // 2, 3 * nsteps // 4, nsteps}
    snaps = run_driver(N, nsteps, marks)

    initial = snaps[0]
    final = snaps[nsteps]
    rhoj = snaps["rhoj"]  # type: ignore[index]

    c = _centres(N)
    x, y = np.meshgrid(c, c, indexing="ij")
    upper = y >= 0.5 * N * DX

    def centroid(f: np.ndarray) -> tuple[float, float]:
        w = f * upper
        return float((x * w).sum() / w.sum()), float((y * w).sum() / w.sum())

    (x0, y0), (x1, y1) = centroid(initial), centroid(final)
    phase = np.hypot(x1 - x0, y1 - y0) / DX

    order = [0, nsteps // 4, nsteps // 2, 3 * nsteps // 4, nsteps]
    labels = ["initial", "¼ turn", "½ turn", "¾ turn", "full turn"]

    fig, axes = plt.subplots(1, 6, figsize=(19, 3.8), layout="constrained")
    norm = Normalize(vmin=0.0, vmax=1.0)
    for ax, step, label in zip(axes[:5], order, labels, strict=True):
        mesh = ax.pcolormesh(c, c, snaps[step].T, cmap=FIELD_CMAP, norm=norm, shading="auto")
        ax.contour(c, c, snaps[step].T, levels=[0.5], colors="w", linewidths=0.7)
        _bare(ax)
        ax.set_title(
            f"{label}\nmin {snaps[step].min():+.1e}  max {snaps[step].max():.3f}", fontsize=9
        )
    fig.colorbar(mesh, ax=axes[:5], shrink=0.8, label="mixing ratio", pad=0.01)

    error = final - initial
    limit = float(np.abs(error).max())
    err = axes[5].pcolormesh(c, c, error.T, cmap=DIVERGING, vmin=-limit, vmax=limit, shading="auto")
    _bare(axes[5])
    axes[5].set_title(f"error after one turn\nmax |err| {limit:.2f}", fontsize=9)
    fig.colorbar(err, ax=axes[5], shrink=0.8, pad=0.02)

    rhoj_drift = float(np.abs(rhoj - 1.0).max())
    fig.suptitle(
        f"Solid-body rotation through CMAQ's real driver, {N}×{N} cells, {nsteps} steps. "
        "Boundary conditions, the X-Y/Y-X alternation and the\n"
        f"rho*J ride-along are all active -- unlike the A0 figures, which used the bare kernel "
        "with periodic wrapping.\n"
        f"Phase error {phase:.3f} cells.   Mass conserved to "
        f"{abs(final.sum() - initial.sum()) / initial.sum():.0e}.   "
        f"rho*J held at 1.0 to {rhoj_drift:.0e} under a non-divergent wind.   "
        f"Undershoot {final.min():+.0e}.",
        fontsize=10,
    )
    _save(fig, "rotation_driver")


def figure_periodic_vs_driver() -> None:
    """What the boundary treatment actually changes, and how long it takes."""
    nsteps = round(PERIOD / DT)
    initial = initial_shapes(N)
    driver = run_driver(N, nsteps, {nsteps})[nsteps]
    periodic = run_periodic(N, nsteps)

    # When the two paths part company. They start bit-identical: the shapes are
    # far from the edge, so the periodic halo and the clean inflow both supply
    # zero. The halo only starts to differ once diffusion has spread a tail all
    # the way round, and the difference then grows with the flow.
    probes = [1, 5, 25, 50, 100, 200, 300, 450, 600, 750, 900]
    drift = [float(np.abs(run_driver(N, k, {k})[k] - run_periodic(N, k)).max()) for k in probes]

    c = _centres(N)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), layout="constrained")
    norm = Normalize(vmin=0.0, vmax=1.0)

    for ax, field, label in zip(
        axes[:3],
        (initial, periodic, driver),
        (
            "initial (= exact after one turn)",
            "A0: bare kernel, periodic halo",
            "A1: real driver, real BCs",
        ),
        strict=True,
    ):
        mesh = ax.pcolormesh(c, c, field.T, cmap=FIELD_CMAP, norm=norm, shading="auto")
        _bare(ax)
        ax.set_title(f"{label}\nmax {field.max():.4f}   min {field.min():+.1e}", fontsize=9)
    fig.colorbar(mesh, ax=axes[:3], shrink=0.8, label="mixing ratio", pad=0.01)

    diff = driver - periodic
    limit = max(float(np.abs(diff).max()), 1e-12)
    d = axes[3].pcolormesh(c, c, diff.T, cmap=DIVERGING, vmin=-limit, vmax=limit, shading="auto")
    _bare(axes[3])
    axes[3].set_title(f"driver − periodic\nmax |diff| {limit:.1e}", fontsize=9)
    fig.colorbar(d, ax=axes[3], shrink=0.8, pad=0.02)

    growth = axes[4]
    floor = 1e-24
    growth.semilogy(probes, [max(v, floor) for v in drift], "o-", color="#1b6ca8", lw=1.8)
    growth.axhline(np.finfo(np.float64).eps, color="#d1495b", ls="--", lw=1.2, label="float64 eps")
    growth.set_xlabel("steps")
    growth.set_ylabel("max |driver − periodic|")
    growth.set_ylim(floor, 1e-6)
    growth.grid(alpha=0.25, which="both")
    growth.legend(fontsize=8, loc="lower right")
    growth.set_title("bit-identical at first,\nthen the halos diverge", fontsize=9)

    fig.suptitle(
        "The same rotation two ways, to show what the boundary treatment is worth. For the first "
        "few steps the two are\nbit-identical -- the shapes are far from the edge, so a periodic "
        "halo and a clean inflow both supply zero. Only once\ndiffusion has spread a tail right "
        "round the domain do the halos differ, and that seed then grows with the flow\n"
        f"to {limit:.0e} after a full turn. So the A0 figures were not misleading about\n"
        "the interior, and this is the scale of what they left out.",
        fontsize=10,
    )
    _save(fig, "periodic_vs_driver")


def main() -> None:
    figure_rotation_driver()
    figure_periodic_vs_driver()


if __name__ == "__main__":
    main()

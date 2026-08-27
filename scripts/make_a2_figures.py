#!/usr/bin/env python3
"""A2 figures: the vertical operator.

    python scripts/make_a2_figures.py

Three figures:

* ``vertical_transport`` -- what a column does over many sync steps, as a
  layer-versus-time Hovmoller. A GIF would show the same profile line moving;
  the Hovmoller shows the whole history at once and is easier to read.
* ``flux_diagnosis`` -- how a density mismatch becomes a face flux, a face
  velocity and a Courant number.
* ``substepping`` -- when the driver splits the sync step, and by how much.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from cmaq_jax.config import sigma_layer_thickness
from cmaq_jax.ppm import nonuniform_mesh
from cmaq_jax.vadv import diagnose_flux, face_velocity_from_flux, zadv

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "docs" / "figures" / "a2"

FIELD_CMAP = "magma"
DIVERGING = "RdBu_r"
PORT = "#1b6ca8"
ACCENT = "#d1495b"
MUTED = "#8d99ae"

NLAYS = 24
DT = 180.0


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches=None if fig.get_layout_engine() else "tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


def sigma_grid(nlays: int = NLAYS, stretch: float = 0.625) -> tuple[np.ndarray, np.ndarray]:
    """CMAQ-like sigma faces and the thicknesses they imply."""
    faces = np.linspace(1.0, 0.0, nlays + 1) ** stretch
    return faces, sigma_layer_thickness(faces)


def _height_axis(ds: np.ndarray) -> np.ndarray:
    """Layer-centre height as a fraction of the column, for plotting.

    Sigma runs 1 at the ground to 0 at the top, so this flips it into
    something that reads upward.
    """
    edges = np.concatenate([[0.0], np.cumsum(ds)])
    return 0.5 * (edges[:-1] + edges[1:])


def figure_flux_diagnosis() -> None:
    """A density mismatch, and everything the driver derives from it."""
    _, ds = sigma_grid()
    height = _height_axis(ds)

    rng = np.random.default_rng(20260907)
    rhoj = 1.5 + 0.4 * rng.random(NLAYS)
    mismatch = 0.25 * np.sin(np.linspace(0.0, 2.0 * np.pi, NLAYS))
    met = rhoj * (1.0 + mismatch)

    flx = np.asarray(diagnose_flux(met[:, None], rhoj[:, None], ds, DT))[:, 0]
    vel = np.asarray(face_velocity_from_flux(flx[:, None], rhoj[:, None]))[:, 0]
    face_height = np.concatenate([[0.0], np.cumsum(ds)])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), layout="constrained")

    axes[0].plot(rhoj, height, "o-", color=PORT, ms=3, label="transported rho*J")
    axes[0].plot(met, height, "o-", color=ACCENT, ms=3, label="meteorology")
    axes[0].set_xlabel("rho*J")
    axes[0].set_ylabel("height (fraction of column)")
    axes[0].set_title("the mismatch to be closed", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(flx, face_height, "-", color=PORT, lw=1.8)
    axes[1].axvline(0.0, color="k", lw=0.8, alpha=0.4)
    axes[1].plot([flx[0]], [face_height[0]], "o", color=ACCENT, ms=7, mfc="none")
    axes[1].plot([flx[-1]], [face_height[-1]], "o", color=ACCENT, ms=7, mfc="none")
    axes[1].set_xlabel("face mass flux")
    axes[1].set_title(
        f"diagnosed flux\nboth ends pinned: |top| / |max| = {abs(flx[-1]) / abs(flx).max():.0e}",
        fontsize=10,
    )

    axes[2].plot(vel, face_height, "-", color=PORT, lw=1.8)
    axes[2].axvline(0.0, color="k", lw=0.8, alpha=0.4)
    axes[2].set_xlabel("face velocity (sigma / s)")
    axes[2].set_title("velocity, upwinded on flux sign", fontsize=10)

    courant = np.zeros_like(vel)
    courant[1:-1] = np.where(vel[1:-1] > 0.0, vel[1:-1] * DT / ds[:-1], -vel[1:-1] * DT / ds[1:])
    courant[-1] = abs(vel[-1]) * DT / ds[-1]
    axes[3].plot(np.abs(courant), face_height, "-", color=PORT, lw=1.8)
    axes[3].axvline(1.0, color=ACCENT, ls="--", lw=1.2, label="CFL limit")
    axes[3].set_xlabel("Courant number")
    axes[3].set_title(f"max Courant {np.abs(courant).max():.2f}", fontsize=10)
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, 1.0)

    fig.suptitle(
        "Vertical advection has no wind field to read. The flux is *diagnosed* from how far the "
        "transported density has\ndrifted from the meteorology, so that advecting rho*J closes the "
        "gap. The ground is pinned closed by construction;\nthe model top closes itself, because "
        "the sigma thicknesses sum to one and the flux recurrence cancels there.",
        fontsize=10,
    )
    _save(fig, "flux_diagnosis")


def figure_vertical_transport() -> None:
    """What one sync step does, and what repeating it converges to.

    A layer-versus-time Hovmoller was the obvious choice and it is the wrong
    one. The flux here is *diagnosed from the density mismatch*, so once a step
    has closed the gap there is nothing left to drive transport and the picture
    goes flat. In a real run horizontal advection refreshes that mismatch every
    sync step; in isolation the operator is a relaxation, and this shows it
    relaxing.
    """
    _, ds = sigma_grid()
    height = _height_axis(ds)
    mesh = nonuniform_mesh(ds)

    rng = np.random.default_rng(20260907)
    rhoj = 1.5 + 0.4 * rng.random(NLAYS)
    met = rhoj * (1.0 + 0.25 * np.sin(np.linspace(0.0, 2.0 * np.pi, NLAYS)))

    tracer = np.exp(-(((np.arange(NLAYS) - NLAYS / 3.0) / 2.0) ** 2))
    con = np.stack([tracer * rhoj, rhoj], axis=-1)[:, None, :]
    initial = np.asarray(con).copy()

    gaps, masses = [float(np.abs(rhoj - met).max())], [float(np.sum(tracer * rhoj * ds))]
    after_one = None
    for step in range(20):
        out, _ = zadv(con, met[:, None], ds, mesh, dt=DT)
        con = np.asarray(out)
        if step == 0:
            after_one = con.copy()
        gaps.append(float(np.abs(con[:, 0, -1] - met).max()))
        masses.append(float(np.sum(con[:, 0, 0] * ds)))

    drift = np.abs(np.array(masses) - masses[0]) / masses[0]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), layout="constrained")

    axes[0].plot(initial[:, 0, -1], height, "o-", color=PORT, ms=3, label="before")
    axes[0].plot(after_one[:, 0, -1], height, "o-", color=ACCENT, ms=3, label="after one step")
    axes[0].plot(met, height, "--", color=MUTED, lw=1.6, label="meteorology")
    axes[0].set_xlabel("rho*J")
    axes[0].set_ylabel("height (fraction of column)")
    axes[0].set_title("density moves toward the met field", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        initial[:, 0, 0] / initial[:, 0, -1], height, "-", color=PORT, lw=1.8, label="before"
    )
    axes[1].plot(
        after_one[:, 0, 0] / after_one[:, 0, -1],
        height,
        "-",
        color=ACCENT,
        lw=1.8,
        label="after one step",
    )
    axes[1].plot(con[:, 0, 0] / con[:, 0, -1], height, ":", color="k", lw=1.6, label="after 20")
    axes[1].set_xlabel("mixing ratio")
    axes[1].set_title("the tracer is carried with it", fontsize=10)
    axes[1].legend(fontsize=8)

    # The gap does not go to zero, and the floor is not arbitrary: vertical
    # flux conserves column mass, so it can redistribute a mismatch but cannot
    # remove the part that is a uniform column-mean offset.
    column_offset = abs(float(np.sum(met * ds)) - float(np.sum(rhoj * ds)))
    axes[2].semilogy(range(len(gaps)), np.maximum(gaps, 1e-18), "o-", color=PORT, ms=3)
    axes[2].axhline(column_offset, color=ACCENT, ls="--", lw=1.2, label="column-mean mismatch")
    axes[2].set_xlabel("sync steps applied")
    axes[2].set_ylabel("max |transported rho*J - met|")
    axes[2].set_title(
        f"gap falls {gaps[0]:.3f} -> {gaps[-1]:.3f}\nand stops at the column-mean offset",
        fontsize=10,
    )
    axes[2].legend(fontsize=8)

    axes[3].semilogy(range(len(drift)), np.maximum(drift, 1e-18), color=PORT, lw=1.8)
    axes[3].axhline(np.finfo(np.float64).eps, color=ACCENT, ls="--", lw=1.2, label="float64 eps")
    axes[3].set_xlabel("sync steps applied")
    axes[3].set_ylabel("relative column-mass drift")
    axes[3].set_title(f"mass held to {drift.max():.0e}", fontsize=10)
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)

    fig.suptitle(
        "Vertical advection is a relaxation, not a wind. Its flux is diagnosed from the gap "
        "between the transported density and\n"
        "the meteorology, so applying it repeatedly with a fixed met field closes that gap and "
        "the transport stops. It stops short of\n"
        "zero, at exactly the column-mean mismatch: the flux conserves column mass, so it can "
        "redistribute an offset but never\n"
        "remove one. In a full run horizontal advection reopens the gap every sync step.",
        fontsize=10,
    )
    _save(fig, "vertical_transport")


def figure_substepping() -> None:
    """When the driver splits the sync step, and by how much."""
    _, ds = sigma_grid()
    mesh = nonuniform_mesh(ds)
    rng = np.random.default_rng(20260908)

    amplitudes = np.linspace(0.0, 1.2, 25)
    courants, steps = [], []
    for amp in amplitudes:
        rhoj = 1.5 + 0.4 * rng.random(NLAYS)
        met = rhoj * (1.0 + amp * np.sin(np.linspace(0.0, 2.0 * np.pi, NLAYS)))
        con = np.stack([rhoj, rhoj], axis=-1)[:, None, :]
        _, diag = zadv(con, met[:, None], ds, mesh, dt=DT)
        courants.append(float(np.asarray(diag.max_courant).max()))
        steps.append(int(np.asarray(diag.substeps).max()))

    # A grid where columns need different numbers of sub-steps.
    ncols, nrows = 12, 10
    amp_field = np.linspace(0.0, 1.0, ncols)[:, None] * np.linspace(0.3, 1.0, nrows)[None, :]
    rhoj = 1.5 + 0.4 * rng.random((NLAYS, ncols, nrows))
    met = rhoj * (
        1.0 + amp_field[None, :, :] * np.sin(np.linspace(0.0, 2.0 * np.pi, NLAYS))[:, None, None]
    )
    con = np.stack([rhoj, rhoj], axis=-1)
    _, grid_diag = zadv(con, met, ds, mesh, dt=DT)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), layout="constrained")

    axes[0].plot(amplitudes, courants, "o-", color=PORT, ms=4)
    axes[0].axhline(1.0, color=ACCENT, ls="--", lw=1.2, label="CFL limit")
    axes[0].set_xlabel("density mismatch amplitude")
    axes[0].set_ylabel("max Courant number")
    axes[0].set_title("mismatch drives the Courant number", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].step(courants, steps, where="post", color=PORT, lw=1.8)
    axes[1].axvline(1.0, color=ACCENT, ls="--", lw=1.2)
    axes[1].set_xlabel("max Courant number")
    axes[1].set_ylabel("sub-steps taken")
    axes[1].set_yticks(range(1, max(steps) + 1))
    axes[1].set_title("splitting follows the overshoot", fontsize=10)
    axes[1].grid(alpha=0.25)

    counts = np.asarray(grid_diag.substeps)
    img = axes[2].pcolormesh(
        counts.T,
        cmap="viridis",
        norm=TwoSlopeNorm(vmin=1, vcenter=max(2, counts.mean()), vmax=max(counts.max(), 3)),
        shading="auto",
    )
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("column")
    axes[2].set_ylabel("row")
    axes[2].set_title(f"sub-steps across a grid: {counts.min()} to {counts.max()}", fontsize=10)
    fig.colorbar(img, ax=axes[2], shrink=0.85, label="sub-steps")

    fig.suptitle(
        "Each column decides its own sub-stepping, and neighbours needing one step or four\n"
        "advance together in a single fixed-count loop with finished columns masked off -- so a "
        "whole grid\nruns at once despite the ragged work.",
        fontsize=10,
    )
    _save(fig, "substepping")


def main() -> None:
    figure_flux_diagnosis()
    figure_vertical_transport()
    figure_substepping()


if __name__ == "__main__":
    main()

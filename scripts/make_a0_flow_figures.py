#!/usr/bin/env python3
"""Two-dimensional advection benchmarks, rendered into docs/figures/a0/.

    python scripts/make_a0_flow_figures.py

The figures in ``make_a0_figures.py`` justify the port numerically -- error
against the Fortran, convergence order, limiter behaviour. These show what the
scheme actually *does* to a field, on the two benchmarks anyone who works with
transport schemes will recognise:

* **Solid-body rotation** (Zalesak 1979). A smooth cone and a slotted cylinder
  carried once around the domain. The exact answer is the initial condition, so
  everything visible is scheme error: how far the cone flattens, whether the
  slot survives, and whether ringing appears around the sharp edges.
* **Deformational swirl** (LeVeque 1996). A vortex stretches a blob into a thin
  filament and then, because the flow reverses in time, pulls it back. Whatever
  fails to return is irreversible mixing the scheme introduced.
* A 3-D surface view of the rotation, where over- and undershoot are obvious in
  a way a contour plot hides.

**These use the A0 1-D kernel applied alternately along each axis with periodic
wrapping.** That is not CMAQ's HADV driver: no boundary conditions, no
contravariant velocity, no per-layer sub-stepping, and the sweep order
alternates on a fixed schedule rather than by CMAQ's per-layer ``XYFIRST``
parity. Those arrive in A1. The numerics exercised here -- reconstruction,
limiter, flux, update -- are exactly the ones validated against the Fortran.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize

from cmaq_jax.config import DEFAULT_PPM
from cmaq_jax.ppm import ppm_advect_uniform

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "docs" / "figures" / "a0"
SWP = DEFAULT_PPM.halo_width

FIELD_CMAP = "magma"
DIVERGING = "RdBu_r"
PORT = "#1b6ca8"
ACCENT = "#d1495b"


def _save_gif(anim: FuncAnimation, fig: plt.Figure, name: str, fps: int) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.gif"
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path.relative_to(REPO)} ({size_mb:.1f} MB)")


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches=None if fig.get_layout_engine() else "tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


# --------------------------------------------------------------------------
# A minimal two-dimensional driver
# --------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("axis",))
def _sweep(field: jax.Array, vel: jax.Array, dt: float, ds: float, *, axis: int) -> jax.Array:
    """One PPM sweep along `axis` of a periodic 2-D field.

    The kernel always sweeps axis 0, so the requested axis is rotated to the
    front and back again. `vel` holds face velocities along that axis and
    carries the other axis as its trailing dimension, which broadcasts.
    """
    swept = jnp.moveaxis(field, axis, 0)
    speeds = jnp.moveaxis(vel, axis, 0)
    padded = jnp.concatenate([swept[-SWP:], swept, swept[:SWP]])
    updated = ppm_advect_uniform(padded, speeds, dt, ds)[SWP:-SWP]
    return jnp.moveaxis(updated, 0, axis)


def _step(
    field: jax.Array,
    u: jax.Array,
    v: jax.Array,
    dt: float,
    ds: float,
    *,
    x_first: bool,
) -> jax.Array:
    """Directionally split step. Alternating the order cancels the leading
    splitting error, which is why CMAQ alternates too (``hadvppm.F:215-251``)."""
    order = (0, 1) if x_first else (1, 0)
    speeds = (u, v)
    for axis in order:
        field = _sweep(field, speeds[axis], dt, ds, axis=axis)
    return field


def _cell_centres(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def _faces(n: int) -> np.ndarray:
    return np.arange(n + 1) / n


def _bare(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------


def _cone(x: np.ndarray, y: np.ndarray, centre: tuple[float, float], radius: float) -> np.ndarray:
    """A smooth cone: the easy shape. Measures numerical diffusion."""
    r = np.hypot(x - centre[0], y - centre[1])
    return np.where(r <= radius, 1.0 - r / radius, 0.0)


def _slotted_cylinder(
    x: np.ndarray,
    y: np.ndarray,
    centre: tuple[float, float],
    radius: float,
    slot_halfwidth: float,
) -> np.ndarray:
    """Zalesak's slotted cylinder: the hard shape.

    Vertical sides, a flat top and a narrow slot. Every feature is a
    discontinuity, so a scheme without a limiter rings visibly, and one with too
    much diffusion fills the slot in.
    """
    r = np.hypot(x - centre[0], y - centre[1])
    disc = r <= radius
    slot = (np.abs(x - centre[0]) <= slot_halfwidth) & (y <= centre[1] + 0.6 * radius)
    return np.where(disc & ~slot, 1.0, 0.0)


def _cosine_bell(
    x: np.ndarray, y: np.ndarray, centre: tuple[float, float], radius: float
) -> np.ndarray:
    """Smooth, compactly supported blob -- LeVeque's deformation-test shape."""
    r = np.hypot(x - centre[0], y - centre[1])
    return np.where(r <= radius, 0.25 * (1.0 + np.cos(np.pi * r / radius)) ** 2, 0.0)


# --------------------------------------------------------------------------
# Solid-body rotation
# --------------------------------------------------------------------------

# The cone sits below the domain midline and the cylinder above it, and both
# return to their starting places after a whole number of turns. That lets each
# shape be measured separately at t=0 and t=T, which a single global max cannot
# do -- the cylinder pins it at 1.0 and hides the cone's diffusion entirely.
CONE_CENTRE = (0.5, 0.25)
CYLINDER_CENTRE = (0.5, 0.75)
SHAPE_RADIUS = 0.15


def _run_rotation(n: int, nsteps: int) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """One full revolution, capturing the field at each quarter turn."""
    omega = 2.0 * np.pi  # period 1.0
    ds = 1.0 / n
    dt = 1.0 / nsteps

    x, y = np.meshgrid(_cell_centres(n), _cell_centres(n), indexing="ij")
    initial = _cone(x, y, CONE_CENTRE, SHAPE_RADIUS) + _slotted_cylinder(
        x, y, CYLINDER_CENTRE, SHAPE_RADIUS, 0.025
    )

    xc = _cell_centres(n)
    u = jnp.asarray(-omega * (xc[None, :] - 0.5) * np.ones((n + 1, 1)))
    v = jnp.asarray(omega * (xc[:, None] - 0.5) * np.ones((1, n + 1)))

    field = jnp.asarray(initial)
    snapshots = [np.asarray(field)]
    masses = [float(jnp.sum(field))]
    for step in range(nsteps):
        field = _step(field, u, v, dt, ds, x_first=(step % 2 == 0))
        masses.append(float(jnp.sum(field)))
        if (step + 1) % (nsteps // 4) == 0:
            snapshots.append(np.asarray(field))
    return initial, snapshots, masses


def _shape_peaks(field: np.ndarray, n: int) -> tuple[float, float]:
    """Peak of the cone and of the cylinder, split at the domain midline."""
    lower = field[:, : n // 2]
    upper = field[:, n // 2 :]
    return float(lower.max()), float(upper.max())


def _phase_error_cells(initial: np.ndarray, final: np.ndarray, n: int) -> float:
    """How far the cylinder's centroid moved over the revolution, in cells.

    Separates the two things an L-infinity error conflates: a shape that came
    back in the wrong *place* (phase error, the serious defect) from one that
    came back in the right place with smeared edges (diffusion, expected at a
    discontinuity).
    """
    xc = _cell_centres(n)
    x, y = np.meshgrid(xc, xc, indexing="ij")
    upper = y >= 0.5

    def centroid(field: np.ndarray) -> tuple[float, float]:
        weight = field * upper
        total = weight.sum()
        return float((x * weight).sum() / total), float((y * weight).sum() / total)

    (x0, y0), (x1, y1) = centroid(initial), centroid(final)
    return float(np.hypot(x1 - x0, y1 - y0) * n)


def _slot_fill(initial: np.ndarray, final: np.ndarray, n: int) -> float:
    """Fraction of the slot that diffusion has filled in."""
    xc = _cell_centres(n)
    x, y = np.meshgrid(xc, xc, indexing="ij")
    inside_disc = np.hypot(x - CYLINDER_CENTRE[0], y - CYLINDER_CENTRE[1]) <= SHAPE_RADIUS
    slot = inside_disc & (initial < 0.5)
    return float(final[slot].mean())


def figure_rotation_2d() -> None:
    n, nsteps = 128, 1000
    courant = 2.0 * np.pi * 0.5 * (1.0 / nsteps) * n
    initial, snapshots, masses = _run_rotation(n, nsteps)
    final = snapshots[-1]

    cone_0, _ = _shape_peaks(initial, n)
    cone_t, cyl_t = _shape_peaks(final, n)
    mass_error = abs(masses[-1] - masses[0]) / masses[0]
    phase = _phase_error_cells(initial, final, n)
    slot = _slot_fill(initial, final, n)

    labels = ["initial", "¼ turn", "½ turn", "¾ turn", "full turn"]
    fig, axes = plt.subplots(1, 6, figsize=(19, 3.8), layout="constrained")
    norm = Normalize(vmin=0.0, vmax=1.0)
    centres = _cell_centres(n)

    for ax, field, label in zip(axes[:5], snapshots, labels, strict=True):
        mesh = ax.pcolormesh(centres, centres, field.T, cmap=FIELD_CMAP, norm=norm, shading="auto")
        ax.contour(centres, centres, field.T, levels=[0.5], colors="w", linewidths=0.7)
        _bare(ax)
        ax.set_title(f"{label}\nmin {field.min():+.1e}   max {field.max():.3f}", fontsize=9)
    fig.colorbar(mesh, ax=axes[:5], shrink=0.8, label="concentration", pad=0.01)

    error = final - initial
    limit = float(np.abs(error).max())
    err_mesh = axes[5].pcolormesh(
        centres, centres, error.T, cmap=DIVERGING, vmin=-limit, vmax=limit, shading="auto"
    )
    _bare(axes[5])
    axes[5].set_title(
        f"error after one turn\nmax |err| {limit:.2f}, all of it on edges",
        fontsize=9,
    )
    fig.colorbar(err_mesh, ax=axes[5], shrink=0.8, pad=0.02)

    fig.suptitle(
        "Solid-body rotation of a cone and a slotted cylinder (Zalesak), "
        f"{n}×{n} cells, Courant {courant:.2f}, {nsteps} steps. After a full turn the exact "
        "answer is the initial field.\n"
        f"Phase error {phase:.3f} cells — the shapes come back where they started; the error "
        "panel is edge diffusion, not displacement.   "
        f"Slot {100 * slot:.0f}% filled.\n"
        f"Cone peak {cone_0:.2f} → {cone_t:.2f} ({100 * (1 - cone_t / cone_0):.0f}% lost), "
        f"cylinder plateau holds at {cyl_t:.3f}.   "
        f"Undershoot {final.min():+.0e}, overshoot {max(final.max() - 1.0, 0.0):+.0e}.   "
        f"Mass conserved to {mass_error:.0e}.",
        fontsize=10,
    )
    _save(fig, "rotation_2d")


def figure_rotation_3d() -> None:
    """The same result as a surface, where ringing would be unmistakable."""
    n, nsteps = 128, 1000
    initial, snapshots, _ = _run_rotation(n, nsteps)
    final = snapshots[-1]

    x, y = np.meshgrid(_cell_centres(n), _cell_centres(n), indexing="ij")
    fig = plt.figure(figsize=(13, 5.4), layout="constrained")

    for idx, (field, label) in enumerate(
        ((initial, "initial"), (final, "after one full revolution"))
    ):
        cone_peak, cyl_peak = _shape_peaks(field, n)
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        # The ambient zero field is drawn, not masked. Masking it out looks
        # tidier but deletes the cylinder's vertical walls: at t=0 every cell is
        # exactly 1.0 or exactly 0.0, so with the zeros gone there are no
        # intermediate values left to draw a side from, and the shape the figure
        # exists to show collapses to a floating disc.
        ax.plot_surface(
            x,
            y,
            field,
            cmap=FIELD_CMAP,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_zlim(0.0, 1.1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_zticks([0, 0.5, 1])
        ax.view_init(elev=30, azim=-125)
        ax.set_title(
            f"{label}\ncylinder plateau {cyl_peak:.3f}    cone peak {cone_peak:.3f}"
            f"    field minimum {field.min():+.0e}",
            fontsize=10,
        )

    fig.suptitle(
        "The same rotation as a surface. The cone visibly loses height to numerical diffusion and "
        "the slot fills in\npartway, but the cylinder keeps vertical sides and a flat top, and "
        "nothing dips below zero. An unlimited\nhigh-order scheme would ring around every one of "
        "those edges.",
        fontsize=10,
    )
    _save(fig, "rotation_3d")


# --------------------------------------------------------------------------
# Deformational swirl
# --------------------------------------------------------------------------


def _deformation_velocities(n: int, t: float, period: float) -> tuple[jax.Array, jax.Array]:
    """LeVeque's time-reversing swirl, sampled on the staggered faces.

    The ``cos(pi t / T)`` factor flips the flow at the half-period, so the exact
    solution at ``t = T`` is the initial condition again. A longer period lets
    the vortex wind the blob further before reversing.
    """
    xc, yc = _cell_centres(n), _cell_centres(n)
    xf, yf = _faces(n), _faces(n)
    reverse = np.cos(np.pi * t / period)
    u = np.sin(np.pi * xf[:, None]) ** 2 * np.sin(2.0 * np.pi * yc[None, :]) * reverse
    v = -(np.sin(np.pi * yf[None, :]) ** 2) * np.sin(2.0 * np.pi * xc[:, None]) * reverse
    return jnp.asarray(u), jnp.asarray(v)


def figure_deformation_2d() -> None:
    n, period = 128, 3.0
    ds = 1.0 / n
    nsteps = int(period / (0.4 * ds))
    dt = period / nsteps

    x, y = np.meshgrid(_cell_centres(n), _cell_centres(n), indexing="ij")
    initial = _cosine_bell(x, y, (0.5, 0.75), 0.15)

    field = jnp.asarray(initial)
    capture = {nsteps // 4: "winding up", nsteps // 2: "maximum deformation", nsteps: "returned"}
    snapshots: list[tuple[str, np.ndarray]] = [("initial", initial)]
    peaks = [float(initial.max())]
    masses = [float(initial.sum())]
    # sum(c^2) falls whenever the scheme mixes, while sum(c) cannot. Tracking
    # both separates "the field spread out" from "material went missing".
    variances = [float((initial**2).sum())]

    for step in range(nsteps):
        u, v = _deformation_velocities(n, (step + 0.5) * dt, period)
        field = _step(field, u, v, dt, ds, x_first=(step % 2 == 0))
        peaks.append(float(jnp.max(field)))
        masses.append(float(jnp.sum(field)))
        variances.append(float(jnp.sum(field**2)))
        if (step + 1) in capture:
            snapshots.append((capture[step + 1], np.asarray(field)))

    final = snapshots[-1][1]
    error = final - initial
    limit = float(np.abs(error).max())
    mass_drift = float(np.abs(np.array(masses) - masses[0]).max() / masses[0])
    centres = _cell_centres(n)

    fig = plt.figure(figsize=(15.5, 8.2), layout="constrained")
    grid = fig.add_gridspec(2, 4)
    norm = Normalize(vmin=0.0, vmax=float(initial.max()))

    field_axes = [fig.add_subplot(grid[0, col]) for col in range(4)]
    for ax, (label, snapshot) in zip(field_axes, snapshots, strict=True):
        mesh = ax.pcolormesh(
            centres, centres, snapshot.T, cmap=FIELD_CMAP, norm=norm, shading="auto"
        )
        _bare(ax)
        ax.set_title(f"{label}\npeak {snapshot.max():.3f}", fontsize=10)
    fig.colorbar(mesh, ax=field_axes, shrink=0.85, label="concentration", pad=0.01)

    # One column, so it lines up with the field panels above; a two-column span
    # would leave a gap, since an equal-aspect field cannot fill the extra width.
    err_ax = fig.add_subplot(grid[1, 0])
    err_mesh = err_ax.pcolormesh(
        centres, centres, error.T, cmap=DIVERGING, vmin=-limit, vmax=limit, shading="auto"
    )
    _bare(err_ax)
    err_ax.set_title(
        f"error after return\nmax |err| {limit:.3f} ({100 * limit / initial.max():.0f}% of peak)",
        fontsize=10,
    )
    fig.colorbar(err_mesh, ax=err_ax, shrink=0.85, pad=0.02)

    diag = fig.add_subplot(grid[1, 1:4])
    time = np.arange(len(peaks)) * dt
    diag.plot(time, np.array(peaks) / peaks[0], color=PORT, lw=2.0, label="peak amplitude")
    diag.plot(
        time,
        np.array(variances) / variances[0],
        color=ACCENT,
        lw=2.0,
        label=r"$\sum c^2$  (mixing measure)",
    )
    diag.axhline(1.0, color="0.6", lw=1.0, ls="-")
    diag.axvline(period / 2, color="k", ls=":", lw=1.2)
    diag.text(period / 2, 0.02, "  flow reverses", fontsize=9, va="bottom")
    diag.set_xlabel("time")
    diag.set_ylabel("fraction of initial value")
    diag.set_ylim(0.0, 1.08)
    diag.set_xlim(0.0, period)
    diag.grid(alpha=0.25)
    diag.legend(fontsize=9, loc="lower left")
    diag.set_title(
        f"amplitude and $\\sum c^2$ fall and never recover;\n"
        f"mass holds to {mass_drift:.0e} (machine precision)",
        fontsize=10,
    )

    fig.suptitle(
        f"Deformational swirl (LeVeque), {n}×{n} cells, period {period:g}. A vortex winds the blob "
        "into a filament thinner than the grid,\nthen reverses and unwinds it. The exact answer at "
        "the end is the initial field, so the error panel is entirely irreversible mixing.\n"
        f"The peak recovers only to {final.max() / initial.max():.0%} -- but no mass is lost, "
        "which is the distinction the right-hand panel makes.",
        fontsize=10,
    )
    _save(fig, "deformation_2d")


# --------------------------------------------------------------------------
# Animations
# --------------------------------------------------------------------------

# Frames are subsampled from the run rather than recomputed at coarse steps:
# the Courant number has to stay below 1 for the scheme to be stable, so the
# time step is set by the physics and only the *rendering* cadence is free.
GIF_FRAMES = 72
GIF_FPS = 18


def _animate_field(
    frames: list[np.ndarray],
    titles: list[str],
    n: int,
    *,
    vmax: float,
    suptitle: str,
    name: str,
) -> None:
    """Render a sequence of 2-D fields to a GIF."""
    centres = _cell_centres(n)
    fig, ax = plt.subplots(figsize=(5.2, 5.6), layout="constrained")
    norm = Normalize(vmin=0.0, vmax=vmax)

    mesh = ax.pcolormesh(centres, centres, frames[0].T, cmap=FIELD_CMAP, norm=norm, shading="auto")
    _bare(ax)
    fig.colorbar(mesh, ax=ax, shrink=0.85, label="concentration")
    title = ax.set_title(titles[0], fontsize=10)
    fig.suptitle(suptitle, fontsize=9)

    def update(index: int) -> tuple[object, ...]:
        mesh.set_array(frames[index].T.ravel())
        title.set_text(titles[index])
        return mesh, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=1000 // GIF_FPS)
    _save_gif(anim, fig, name, GIF_FPS)


def gif_rotation() -> None:
    n, nsteps = 128, 1000
    omega = 2.0 * np.pi
    ds = 1.0 / n
    dt = 1.0 / nsteps

    x, y = np.meshgrid(_cell_centres(n), _cell_centres(n), indexing="ij")
    initial = _cone(x, y, CONE_CENTRE, SHAPE_RADIUS) + _slotted_cylinder(
        x, y, CYLINDER_CENTRE, SHAPE_RADIUS, 0.025
    )
    xc = _cell_centres(n)
    u = jnp.asarray(-omega * (xc[None, :] - 0.5) * np.ones((n + 1, 1)))
    v = jnp.asarray(omega * (xc[:, None] - 0.5) * np.ones((1, n + 1)))

    every = nsteps // GIF_FRAMES
    field = jnp.asarray(initial)
    frames, titles = [initial], ["0.00 turns   max 1.000   min +0.0e+00"]
    for step in range(nsteps):
        field = _step(field, u, v, dt, ds, x_first=(step % 2 == 0))
        if (step + 1) % every == 0:
            snapshot = np.asarray(field)
            frames.append(snapshot)
            titles.append(
                f"{(step + 1) / nsteps:.2f} turns   "
                f"max {snapshot.max():.3f}   min {snapshot.min():+.1e}"
            )

    _animate_field(
        frames,
        titles,
        n,
        vmax=1.0,
        suptitle=(
            "Solid-body rotation (Zalesak). After one full turn the exact answer is\n"
            "the initial field. Watch the slot: it narrows but does not close, and no\n"
            "ringing appears around the sharp edges."
        ),
        name="rotation",
    )


def gif_deformation() -> None:
    n, period = 128, 3.0
    ds = 1.0 / n
    nsteps = int(period / (0.4 * ds))
    dt = period / nsteps

    x, y = np.meshgrid(_cell_centres(n), _cell_centres(n), indexing="ij")
    initial = _cosine_bell(x, y, (0.5, 0.75), 0.15)

    every = max(nsteps // GIF_FRAMES, 1)
    field = jnp.asarray(initial)
    frames, titles = [initial], [f"t = 0.00   peak {initial.max():.3f}"]
    for step in range(nsteps):
        u, v = _deformation_velocities(n, (step + 0.5) * dt, period)
        field = _step(field, u, v, dt, ds, x_first=(step % 2 == 0))
        if (step + 1) % every == 0:
            snapshot = np.asarray(field)
            elapsed = (step + 1) * dt
            phase = "unwinding" if elapsed > period / 2 else "winding up"
            frames.append(snapshot)
            titles.append(f"t = {elapsed:.2f}   {phase}   peak {snapshot.max():.3f}")

    _animate_field(
        frames,
        titles,
        n,
        vmax=float(initial.max()),
        suptitle=(
            "Deformational swirl (LeVeque). The flow reverses at the half-period, so\n"
            "the exact answer at the end is the initial blob. What does not come back\n"
            "is mixing the scheme introduced once the filament thinned past the grid."
        ),
        name="deformation",
    )


def main() -> None:
    figure_rotation_2d()
    figure_rotation_3d()
    figure_deformation_2d()
    gif_rotation()
    gif_deformation()


if __name__ == "__main__":
    main()

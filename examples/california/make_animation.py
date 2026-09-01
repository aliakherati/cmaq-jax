#!/usr/bin/env python3
"""Animate the California run, and draw a summary figure.

    python examples/california/make_animation.py

Reads ``data/run_<stamp>.npz`` from `run_advection.py` and writes into
``figures/``:

* ``california_july2018.gif`` -- the month, three-hourly, both tracers with the
  wind field over them.
* ``california_summary.png`` -- monthly mean, the sources, and a few snapshots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"


def coastline(ax: plt.Axes) -> None:
    """Draw coast and state lines, if they have been fetched."""
    path = DATA / "coastline.npz"
    if not path.exists():
        return
    with np.load(path) as lines:
        for key in lines.files:
            line = lines[key]
            ax.plot(
                line[:, 0],
                line[:, 1],
                color="white",
                lw=0.7 if key.startswith("coast") else 0.4,
                alpha=0.85 if key.startswith("coast") else 0.45,
                zorder=5,
            )


def frame_axes(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray) -> None:
    coastline(ax)
    ax.set_xlim(lon.min() + 1.0, lon.max() - 1.0)
    ax.set_ylim(lat.min() + 1.5, lat.max() - 1.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="201807")
    parser.add_argument("--stride", type=int, default=1, help="frames to skip")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args(argv)

    path = DATA / f"run_{args.stamp}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path.name}\nRun: python examples/california/run_advection.py")

    with np.load(path) as run:
        frames = run["frames"]  # (ntimes, ncols, nrows, 2)
        times = [str(t) for t in run["times"]]
        lat, lon = run["lat"], run["lon"]
        urban, agricultural = run["urban"], run["agricultural"]
        u, v = run["u"], run["v"]

    FIGURES.mkdir(parents=True, exist_ok=True)
    select = slice(None, None, args.stride)
    frames, times, u, v = frames[select], times[select], u[select], v[select]
    ceiling = [float(np.percentile(frames[..., k], 99.9)) for k in (0, 1)]
    floor = [c / 300.0 for c in ceiling]

    # Thin the wind vectors: one arrow per three cells is legible, every cell is
    # a grey mat.
    step = 3
    qx, qy = lon[::step, ::step], lat[::step, ::step]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))
    meshes, quivers = [], []
    labels = ("urban / traffic tracer", "agricultural tracer")
    for k, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            lon,
            lat,
            np.maximum(frames[0, ..., k], floor[k]),
            norm=LogNorm(vmin=floor[k], vmax=ceiling[k]),
            cmap="inferno",
            shading="nearest",
        )
        meshes.append(mesh)
        quivers.append(
            ax.quiver(
                qx,
                qy,
                u[0, ::step, ::step],
                v[0, ::step, ::step],
                color="deepskyblue",
                alpha=0.55,
                scale=260,
                width=0.0025,
                zorder=6,
            )
        )
        frame_axes(ax, lon, lat)
        ax.set_title(labels[k], fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.046, label="tracer (arbitrary units)")

    title = fig.suptitle("", fontsize=12)

    def draw(index: int):
        for k in (0, 1):
            meshes[k].set_array(np.maximum(frames[index, ..., k], floor[k]).ravel())
            quivers[k].set_UVC(u[index, ::step, ::step], v[index, ::step, ::step])
        title.set_text(
            f"California, {times[index][:16].replace('T', ' ')} UTC  —  "
            "CMAQ PPM advection on NARR winds (1000 mb)"
        )
        return [*meshes, *quivers, title]

    fig.tight_layout()
    animation = FuncAnimation(fig, draw, frames=len(frames), interval=1000 // args.fps)
    out = FIGURES / f"california_july{args.stamp[:4]}.gif"
    print(f"rendering {len(frames)} frames -> {out.name}")
    animation.save(out, writer=PillowWriter(fps=args.fps), dpi=90)
    plt.close(fig)
    print(f"  {out.stat().st_size / 1e6:.1f} MB")

    summary(
        frames, times, lat, lon, urban=urban, agricultural=agricultural, stamp=args.stamp
    )
    return 0


def summary(frames, times, lat, lon, *, urban, agricultural, stamp: str) -> None:
    """Monthly mean, the sources that produced it, and three snapshots."""
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.2))

    ax = axes[0, 0]
    ax.pcolormesh(lon, lat, urban + agricultural, cmap="viridis", shading="nearest")
    frame_axes(ax, lon, lat)
    ax.set_title("where the tracers enter\n(urban peaks + agricultural band)", fontsize=10)

    total = frames.mean(axis=0)
    for k, (ax, label) in enumerate(
        zip(axes[0, 1:], ("urban / traffic", "agricultural"), strict=True)
    ):
        ceiling = float(np.percentile(total[..., k], 99.9))
        mesh = ax.pcolormesh(
            lon,
            lat,
            np.maximum(total[..., k], ceiling / 200.0),
            norm=LogNorm(vmin=ceiling / 200.0, vmax=ceiling),
            cmap="inferno",
            shading="nearest",
        )
        frame_axes(ax, lon, lat)
        ax.set_title(f"July mean, {label}", fontsize=10)
        fig.colorbar(mesh, ax=ax, fraction=0.046)

    picks = [len(frames) // 8, len(frames) // 2, -1]
    ceiling = float(np.percentile(frames[..., 1], 99.9))
    for ax, index in zip(axes[1], picks, strict=True):
        mesh = ax.pcolormesh(
            lon,
            lat,
            np.maximum(frames[index, ..., 1], ceiling / 300.0),
            norm=LogNorm(vmin=ceiling / 300.0, vmax=ceiling),
            cmap="inferno",
            shading="nearest",
        )
        frame_axes(ax, lon, lat)
        ax.set_title(f"agricultural, {times[index][:13].replace('T', ' ')}h", fontsize=10)
        fig.colorbar(mesh, ax=ax, fraction=0.046)

    fig.suptitle(
        "Central Valley tracers through July 2018, advected by NARR winds with CMAQ's PPM scheme.\n"
        "Transport only — no chemistry, deposition or vertical mixing.",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGURES / f"california_summary_{stamp}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    sys.exit(main())

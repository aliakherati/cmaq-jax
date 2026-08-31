#!/usr/bin/env python3
# ruff: noqa: E402
"""Render an animated plume map and diagnostic summary for a transport run.

Example:

    .venv/bin/python examples/conus404/make_visualizations.py \
        --run examples/conus404/output/transport_20180726_00_8km.npz \
        --diagnostics examples/conus404/output/transport_20180726_00_8km.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Matplotlib and fontconfig otherwise try unwritable user caches in a managed
# workspace and can emit thousands of warnings while rendering GIF frames.
_CACHE = Path(tempfile.gettempdir()) / "cmaq-jax-visualization-cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_BOUNDARIES = HERE / "data" / "boundaries.npz"
DEFAULT_FIGURES = HERE / "figures"


def _boundaries(ax: plt.Axes, path: Path) -> None:
    if not path.exists():
        return
    with np.load(path) as lines:
        for name in lines.files:
            line = lines[name]
            ax.plot(
                line[:, 0],
                line[:, 1],
                color="white" if name.startswith("coast") else "#d8d8d8",
                linewidth=0.9 if name.startswith("coast") else 0.5,
                alpha=0.9 if name.startswith("coast") else 0.65,
                zorder=5,
            )


def _map_axes(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, boundaries: Path) -> None:
    _boundaries(ax, boundaries)
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(float(lat.min()), float(lat.max()))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_facecolor("#202020")


def _read_diagnostics(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, np.ndarray] = {
        "time_utc": np.asarray([datetime.fromisoformat(row["time_utc"]) for row in rows])
    }
    for name in rows[0]:
        if name != "time_utc":
            result[name] = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    return result


def _plume_scale(frames: np.ndarray, cell_area_km2: float) -> tuple[np.ndarray, float, float]:
    areal = frames / cell_area_km2
    floor, ceiling = _positive_scale(areal)
    return areal, floor, ceiling


def _positive_scale(field: np.ndarray) -> tuple[float, float]:
    positive = field[field > 0.0]
    if positive.size == 0:
        raise ValueError("transport output has no positive CO mass to visualize")
    ceiling = float(np.percentile(positive, 99.5))
    floor = max(float(np.percentile(positive, 1.0)), ceiling / 10_000.0)
    return floor, ceiling


def _animate(
    *,
    output: Path,
    frames: np.ndarray,
    column_mass_frames: np.ndarray,
    times: list[datetime],
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dx: float,
    dy: float,
    boundaries: Path,
    interpolation: int,
    fps: int,
    quantity: str,
) -> None:
    if quantity == "column":
        display, floor, ceiling = _plume_scale(frames, dx * dy / 1.0e6)
        title_prefix = "California fire CO enhancement"
        colorbar_label = "vertically integrated CO enhancement (kg km⁻²)"
        footer = (
            "CONUS404 winds · CMAQ HADV + ZADV · inert tracer; "
            "no chemistry, deposition, or plume rise"
        )
    elif quantity == "ground":
        display = frames
        floor, ceiling = _positive_scale(display)
        title_prefix = "California fire CO — lowest model layer"
        colorbar_label = "lowest-layer CO enhancement (ppbv)"
        footer = (
            "Lowest WRF layer · CONUS404 winds · inert tracer; "
            "no chemistry, deposition, plume rise, or vertical mixing"
        )
    else:  # pragma: no cover - internal call contract
        raise ValueError(f"unknown animation quantity {quantity!r}")
    positions = np.linspace(0.0, len(times) - 1, (len(times) - 1) * interpolation + 1)
    arrow_stride = max(1, min(lon.shape) // 18)

    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.maximum(display[0], floor),
        shading="nearest",
        cmap="inferno",
        norm=LogNorm(vmin=floor, vmax=ceiling),
    )
    quiver = ax.quiver(
        lon[::arrow_stride, ::arrow_stride],
        lat[::arrow_stride, ::arrow_stride],
        u[0, ::arrow_stride, ::arrow_stride],
        v[0, ::arrow_stride, ::arrow_stride],
        color="#60d8ff",
        alpha=0.7,
        width=0.0024,
        scale=380,
        zorder=6,
    )
    _map_axes(ax, lon, lat, boundaries)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)
    title = ax.set_title(
        f"{title_prefix} — 2018-07-26 00:00 UTC\n"
        f"{dx / 1000:.0f} km grid · 0.0 metric tons in domain"
    )
    subtitle = fig.text(
        0.5,
        0.015,
        footer,
        ha="center",
        fontsize=8,
    )

    def draw(frame_index: int) -> tuple:
        position = positions[frame_index]
        lower = min(int(np.floor(position)), len(times) - 1)
        upper = min(lower + 1, len(times) - 1)
        weight = position - lower
        field = (1.0 - weight) * display[lower] + weight * display[upper]
        column_mass = (
            (1.0 - weight) * column_mass_frames[lower]
            + weight * column_mass_frames[upper]
        )
        wind_u = (1.0 - weight) * u[lower] + weight * u[upper]
        wind_v = (1.0 - weight) * v[lower] + weight * v[upper]
        stamp = times[lower] + timedelta(
            seconds=weight * (times[upper] - times[lower]).total_seconds()
        )
        mesh.set_array(np.maximum(field, floor).ravel())
        quiver.set_UVC(
            wind_u[::arrow_stride, ::arrow_stride],
            wind_v[::arrow_stride, ::arrow_stride],
        )
        metric_tons = column_mass.sum() / 1000.0
        title.set_text(
            f"{title_prefix} — {stamp:%Y-%m-%d %H:%M} UTC\n"
            f"{dx / 1000:.0f} km grid · {metric_tons:.1f} metric tons in domain"
        )
        return mesh, quiver, title, subtitle

    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
    images: list[Image.Image] = []
    for frame_index in range(len(positions)):
        draw(frame_index)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
        disposal=2,
        optimize=False,
    )
    plt.close(fig)


def _summary(
    *,
    output: Path,
    frames: np.ndarray,
    daily_emission: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dx: float,
    dy: float,
    boundaries: Path,
    diagnostics: dict[str, np.ndarray],
) -> None:
    areal, floor, ceiling = _plume_scale(frames, dx * dy / 1.0e6)
    source_areal = daily_emission / (dx * dy / 1.0e6)
    source_positive = source_areal[source_areal > 0.0]
    source_floor = max(float(source_positive.min()), float(source_positive.max()) / 10_000.0)
    times = diagnostics["time_utc"]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.4))
    source_ax, plume_ax, mass_ax, closure_ax = axes.ravel()

    source_mesh = source_ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_less_equal(source_areal, 0.0),
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=source_floor, vmax=float(source_positive.max())),
    )
    _map_axes(source_ax, lon, lat, boundaries)
    source_ax.set_title("FINNv1.5 daily CO source")
    fig.colorbar(source_mesh, ax=source_ax, label="kg CO km⁻² day⁻¹", pad=0.02)

    plume_mesh = plume_ax.pcolormesh(
        lon,
        lat,
        np.maximum(areal[-1], floor),
        shading="nearest",
        cmap="inferno",
        norm=LogNorm(vmin=floor, vmax=ceiling),
    )
    _map_axes(plume_ax, lon, lat, boundaries)
    plume_ax.set_title("Final vertically integrated CO enhancement")
    fig.colorbar(plume_mesh, ax=plume_ax, label="kg CO km⁻²", pad=0.02)

    mass_ax.plot(times, diagnostics["emitted_kg"] / 1000.0, label="cumulative emitted")
    mass_ax.plot(times, diagnostics["tracer_mass_kg"] / 1000.0, label="in domain")
    mass_ax.plot(
        times,
        diagnostics["inferred_boundary_loss_kg"] / 1000.0,
        label="inferred boundary loss",
    )
    mass_ax.set_title("CO mass budget")
    mass_ax.set_ylabel("metric tons CO")
    mass_ax.set_xlabel("UTC")
    mass_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    mass_ax.legend(frameon=False)
    mass_ax.grid(alpha=0.25)

    closure_ax.plot(times, diagnostics["rhoj_relative_l1"], label="rhoJ relative L1")
    closure_ax.plot(
        times,
        diagnostics["max_vertical_flux_residual"],
        label="vertical flux residual",
    )
    closure_ax.axhline(1.0e-3, color="0.4", linestyle="--", linewidth=1, label="CMAQ tolerance")
    closure_ax.set_yscale("log")
    closure_ax.set_ylabel("dimensionless")
    closure_ax.set_xlabel("UTC")
    closure_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    closure_ax.grid(alpha=0.25)
    centroid_ax = closure_ax.twinx()
    centroid_ax.plot(
        times[1:],
        diagnostics["vertical_centroid_m_msl"][1:],
        color="tab:green",
        label="vertical centroid",
    )
    centroid_ax.set_ylabel("CO vertical centroid (m MSL)", color="tab:green")
    lines = closure_ax.lines[:3] + centroid_ax.lines
    closure_ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    closure_ax.set_title("Mass-coordinate and vertical diagnostics")

    min_tracer = float(np.min(diagnostics["min_coupled_tracer"]))
    negative_mass = float(np.max(diagnostics["negative_tracer_mass_kg"]))
    fig.suptitle(
        f"CONUS404 California transport validation at {dx / 1000:.0f} km\n"
        f"minimum tracer {min_tracer:.2e}; maximum negative mass {negative_mass:.2e} kg",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--interpolation",
        type=int,
        default=4,
        help="visual subframes between each pair of saved model frames",
    )
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    if args.interpolation < 1 or args.fps < 1:
        parser.error("--interpolation and --fps must be positive")

    with np.load(args.run) as run:
        frames = np.asarray(run["column_mass_frames_kg"], dtype=np.float64)
        surface_ppbv = (
            np.asarray(run["surface_co_frames_ppbv"], dtype=np.float64)
            if "surface_co_frames_ppbv" in run.files
            else None
        )
        times = [datetime.fromisoformat(str(value)) for value in run["frame_times"]]
        u = np.asarray(run["u_surface"], dtype=np.float64)
        v = np.asarray(run["v_surface"], dtype=np.float64)
        daily_emission = np.asarray(run["daily_emission_kg"], dtype=np.float64)
        latitude = np.asarray(run["latitude"], dtype=np.float64)
        longitude = np.asarray(run["longitude"], dtype=np.float64)
        dx = float(run["dx"])
        dy = float(run["dy"])
    diagnostics = _read_diagnostics(args.diagnostics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run.stem
    gif = args.output_dir / f"{stem}.gif"
    png = args.output_dir / f"{stem}_summary.png"
    print(f"rendering {gif}")
    _animate(
        output=gif,
        frames=frames,
        column_mass_frames=frames,
        times=times,
        u=u,
        v=v,
        lon=longitude,
        lat=latitude,
        dx=dx,
        dy=dy,
        boundaries=args.boundaries,
        interpolation=args.interpolation,
        fps=args.fps,
        quantity="column",
    )
    ground_gif = None
    if surface_ppbv is not None:
        ground_gif = args.output_dir / f"{stem}_ground_level.gif"
        print(f"rendering {ground_gif}")
        _animate(
            output=ground_gif,
            frames=surface_ppbv,
            column_mass_frames=frames,
            times=times,
            u=u,
            v=v,
            lon=longitude,
            lat=latitude,
            dx=dx,
            dy=dy,
            boundaries=args.boundaries,
            interpolation=args.interpolation,
            fps=args.fps,
            quantity="ground",
        )
    print(f"rendering {png}")
    _summary(
        output=png,
        frames=frames,
        daily_emission=daily_emission,
        lon=longitude,
        lat=latitude,
        dx=dx,
        dy=dy,
        boundaries=args.boundaries,
        diagnostics=diagnostics,
    )
    ground_note = (
        f", {ground_gif} ({ground_gif.stat().st_size / 1e6:.1f} MB)"
        if ground_gif is not None
        else ""
    )
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB){ground_note} and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

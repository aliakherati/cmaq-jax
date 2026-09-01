#!/usr/bin/env python3
# ruff: noqa: E402
"""Render full-CONUS column and lowest-layer CO GIFs plus a summary."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_CACHE = Path(tempfile.gettempdir()) / "cmaq-jax-visualization-cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

CONUS404 = Path(__file__).resolve().parents[1] / "conus404"
sys.path.insert(0, str(CONUS404))

from make_visualizations import (
    _animate,
    _cell_edges,
    _map_axes,
    _plume_scale,
    _positive_scale,
    _read_diagnostics,
)

HERE = Path(__file__).resolve().parent
DEFAULT_BOUNDARIES = CONUS404 / "data" / "boundaries.npz"
DEFAULT_FIGURES = HERE / "figures"


def _format_time_axis(ax: plt.Axes, times: np.ndarray) -> None:
    duration_hours = (times[-1] - times[0]).total_seconds() / 3600.0
    if duration_hours > 48.0:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def _ground_snapshot(
    *,
    output: Path,
    frames: np.ndarray,
    times: list,
    lon: np.ndarray,
    lat: np.ndarray,
    dx: float,
    boundaries: Path,
) -> None:
    floor, ceiling = _positive_scale(frames)
    figure_color = "#080d15"
    map_color = "#101923"
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(map_color)
    fig, ax = plt.subplots(figsize=(9.6, 7.0), facecolor=figure_color)
    mesh = ax.pcolormesh(
        _cell_edges(lon),
        _cell_edges(lat),
        np.ma.masked_less_equal(frames[-1], 0.0),
        shading="flat",
        cmap=cmap,
        norm=LogNorm(vmin=floor, vmax=ceiling),
    )
    _map_axes(ax, lon, lat, boundaries)
    ax.set_facecolor(map_color)
    ax.set_axis_off()
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    for spine in colorbar.ax.spines.values():
        spine.set_edgecolor("#718096")
    colorbar.ax.tick_params(colors="#dce7f2", labelsize=9)
    colorbar.set_label(
        "lowest-layer CO enhancement (ppbv)", color="#dce7f2", labelpad=12
    )
    fig.text(
        0.055,
        0.952,
        "EPA projected-2023 CO — lowest MCIP layer",
        color="#f4f8fb",
        fontsize=19,
        ha="left",
        va="top",
    )
    fig.text(
        0.055,
        0.913,
        f"EPA 2016v3 2023gf emissions  •  {dx / 1000:.0f} km  •  35 layers  •  CMAQ/JAX",
        color="#9fb2c5",
        fontsize=9,
        ha="left",
        va="top",
    )
    ax.text(
        0.018,
        0.965,
        f"MET  {times[-1]:%d %b %Y  ·  %H:%M UTC}".upper(),
        transform=ax.transAxes,
        color="#f4f8fb",
        fontsize=11,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": figure_color,
            "edgecolor": "none",
            "alpha": 0.82,
        },
        zorder=10,
    )
    ax.text(
        0.018,
        0.035,
        f"FINAL PEAK  {frames[-1].max():,.1f} ppbv",
        transform=ax.transAxes,
        color="#f4f8fb",
        fontsize=10,
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": figure_color,
            "edgecolor": "none",
            "alpha": 0.82,
        },
        zorder=10,
    )
    fig.text(
        0.5,
        0.022,
        "2016 MCIP winds · inert enhancement · no background, chemistry, deposition, or turbulence",
        color="#8799aa",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.025, right=0.91, bottom=0.07, top=0.875)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _summary(
    *,
    output: Path,
    frames: np.ndarray,
    emitted_co: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dx: float,
    dy: float,
    boundaries: Path,
    diagnostics: dict[str, np.ndarray],
) -> None:
    areal, floor, ceiling = _plume_scale(frames, dx * dy / 1.0e6)
    source_areal = emitted_co / (dx * dy / 1.0e6)
    positive = source_areal[source_areal > 0.0]
    source_floor = max(float(np.percentile(positive, 1.0)), float(positive.max()) / 10_000.0)
    times = diagnostics["time_utc"]
    duration_hours = round((times[-1] - times[0]).total_seconds() / 3600.0)
    duration_label = (
        f"{duration_hours // 24} days"
        if duration_hours % 24 == 0
        else f"{duration_hours} hours"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    source_ax, plume_ax, mass_ax, closure_ax = axes.ravel()
    lon_edges = _cell_edges(lon)
    lat_edges = _cell_edges(lat)

    source_mesh = source_ax.pcolormesh(
        lon_edges,
        lat_edges,
        np.ma.masked_less_equal(source_areal, 0.0),
        shading="flat",
        cmap="viridis",
        norm=LogNorm(vmin=source_floor, vmax=float(positive.max())),
    )
    _map_axes(source_ax, lon, lat, boundaries)
    source_ax.set_title(f"EPA 2023gf merged CO emitted over {duration_label}")
    fig.colorbar(source_mesh, ax=source_ax, label="kg CO km⁻² over period", pad=0.02)

    plume_mesh = plume_ax.pcolormesh(
        lon_edges,
        lat_edges,
        np.maximum(areal[-1], floor),
        shading="flat",
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
    mass_ax.set_xlabel("2016 meteorological time (UTC)")
    _format_time_axis(mass_ax, times)
    mass_ax.legend(frameon=False)
    mass_ax.grid(alpha=0.25)

    closure_ax.plot(times, diagnostics["rhoj_relative_l1"], label="rhoJ relative L1")
    closure_ax.plot(
        times,
        diagnostics["max_vertical_flux_residual"],
        label="vertical flux residual",
    )
    closure_ax.axhline(1.0e-3, color="0.4", linestyle="--", linewidth=1)
    closure_ax.set_yscale("log")
    closure_ax.set_ylabel("dimensionless")
    closure_ax.set_xlabel("2016 meteorological time (UTC)")
    _format_time_axis(closure_ax, times)
    closure_ax.grid(alpha=0.25)
    centroid_ax = closure_ax.twinx()
    centroid_ax.plot(
        times[1:],
        diagnostics["vertical_centroid_m_msl"][1:],
        color="tab:green",
        label="vertical centroid",
    )
    centroid_ax.set_ylabel("CO vertical centroid (m MSL)", color="tab:green")
    lines = closure_ax.lines[:2] + centroid_ax.lines
    closure_ax.legend(lines, [line.get_label() for line in lines], frameon=False)
    closure_ax.set_title("Mass-coordinate and vertical diagnostics")

    min_tracer = float(np.min(diagnostics["min_coupled_tracer"]))
    negative_mass = float(np.max(diagnostics["negative_tracer_mass_kg"]))
    fig.suptitle(
        "Full-CONUS 12 km inert CO transport\n"
        f"EPA 2016v3 projected-2023 emissions · {times[0]:%b %d}–"
        f"{times[-1]:%b %d, %Y} MCIP meteorology\n"
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
    parser.add_argument("--interpolation", type=int, default=1)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if args.interpolation < 1 or args.fps < 1:
        parser.error("--interpolation and --fps must be positive")

    with np.load(args.run) as run:
        frames = np.asarray(run["column_mass_frames_kg"], dtype=np.float64)
        surface = np.asarray(run["surface_co_frames_ppbv"], dtype=np.float64)
        times = [
            np.datetime64(value).astype("datetime64[us]").astype(object)
            for value in run["frame_times"]
        ]
        u = np.asarray(run["u_surface"], dtype=np.float64)
        v = np.asarray(run["v_surface"], dtype=np.float64)
        emitted_co = np.asarray(run["emitted_co_kg"], dtype=np.float64)
        latitude = np.asarray(run["latitude"], dtype=np.float64)
        longitude = np.asarray(run["longitude"], dtype=np.float64)
        dx = float(run["dx"])
        dy = float(run["dy"])
    diagnostics = _read_diagnostics(args.diagnostics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run.stem
    column_gif = args.output_dir / f"{stem}.gif"
    ground_gif = args.output_dir / f"{stem}_ground_level.gif"
    ground_png = args.output_dir / f"{stem}_ground_level.png"
    summary = args.output_dir / f"{stem}_summary.png"
    footer = (
        "EPA projected-2023 merged emissions · 2016 MCIP winds · CMAQ HADV + ZADV · "
        "inert enhancement; no chemistry, deposition, or turbulence"
    )

    print(f"rendering {column_gif}")
    _animate(
        output=column_gif,
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
        title_override="EPA projected-2023 CO enhancement",
        footer_override=footer,
        presentation=True,
    )
    print(f"rendering {ground_gif}")
    _animate(
        output=ground_gif,
        frames=surface,
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
        title_override="EPA projected-2023 CO — lowest MCIP layer",
        footer_override=footer,
        presentation=True,
    )
    print(f"rendering {ground_png}")
    _ground_snapshot(
        output=ground_png,
        frames=surface,
        times=times,
        lon=longitude,
        lat=latitude,
        dx=dx,
        boundaries=args.boundaries,
    )
    print(f"rendering {summary}")
    _summary(
        output=summary,
        frames=frames,
        emitted_co=emitted_co,
        lon=longitude,
        lat=latitude,
        dx=dx,
        dy=dy,
        boundaries=args.boundaries,
        diagnostics=diagnostics,
    )
    print(f"wrote {column_gif}, {ground_gif}, {ground_png}, and {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

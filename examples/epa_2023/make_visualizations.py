#!/usr/bin/env python3
# ruff: noqa: E402
"""Render diagnostic and satellite-style full-CONUS CO visualizations."""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

_CACHE = Path(tempfile.gettempdir()) / "cmaq-jax-visualization-cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
from PIL import Image, ImageEnhance
from pyproj import CRS, Transformer

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
DEFAULT_BOUNDARIES = HERE / "data" / "north_america_boundaries.npz"
DEFAULT_FIGURES = HERE / "figures"
DEFAULT_BASEMAP = HERE / "data" / "world.topo.200407.3x5400x2700.jpg"
MCIP_X_ORIGIN_M = -2_556_000.0
MCIP_Y_ORIGIN_M = -1_728_000.0
SATELLITE_BOUNDS_M = (-3_050_000.0, 3_250_000.0, -2_450_000.0, 3_850_000.0)
MCIP_CRS = CRS.from_proj4(
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 "
    "+R=6370000 +units=m +no_defs"
)
TO_LONLAT = Transformer.from_crs(MCIP_CRS, CRS.from_epsg(4326), always_xy=True)
FROM_LONLAT = Transformer.from_crs(CRS.from_epsg(4326), MCIP_CRS, always_xy=True)


def _haze_colormap() -> ListedColormap:
    """Weather-map palette with opacity increasing with CO enhancement."""
    colors = LinearSegmentedColormap.from_list(
        "co_haze_colors",
        [
            (0.00, "#f7f2df"),
            (0.35, "#efe7d0"),
            (0.55, "#e1c09e"),
            (0.72, "#c77955"),
            (0.87, "#7b2f2d"),
            (1.00, "#28111a"),
        ],
    )(np.linspace(0.0, 1.0, 256))
    colors[:, 3] = np.interp(
        np.linspace(0.0, 1.0, 256),
        [0.0, 0.35, 0.55, 0.72, 0.87, 1.0],
        [0.04, 0.10, 0.20, 0.38, 0.66, 0.91],
    )
    cmap = ListedColormap(colors, name="co_haze")
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    return cmap


def _load_basemap(
    path: Path,
    bounds: tuple[float, float, float, float],
    size: int = 1200,
) -> np.ndarray:
    """Warp and tone the Plate Carree Blue Marble image onto the MCIP LCC grid."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        image = ImageEnhance.Color(image).enhance(0.74)
        image = ImageEnhance.Contrast(image).enhance(0.98)
        image = ImageEnhance.Brightness(image).enhance(0.90)
        plate_carree = np.asarray(image, dtype=np.float32)

    west, east, south, north = bounds
    x = np.linspace(west, east, size, dtype=np.float64)
    y = np.linspace(north, south, size, dtype=np.float64)
    projected_x, projected_y = np.meshgrid(x, y)
    longitude, latitude = TO_LONLAT.transform(projected_x, projected_y)
    source_x = (longitude + 180.0) / 360.0 * (plate_carree.shape[1] - 1)
    source_y = (90.0 - latitude) / 180.0 * (plate_carree.shape[0] - 1)
    valid = (
        np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 0.0)
        & (source_x <= plate_carree.shape[1] - 1)
        & (source_y >= 0.0)
        & (source_y <= plate_carree.shape[0] - 1)
    )
    source_x = np.clip(source_x, 0.0, plate_carree.shape[1] - 1)
    source_y = np.clip(source_y, 0.0, plate_carree.shape[0] - 1)
    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, plate_carree.shape[1] - 1)
    y1 = np.minimum(y0 + 1, plate_carree.shape[0] - 1)
    x_weight = (source_x - x0)[..., None]
    y_weight = (source_y - y0)[..., None]
    top = (1.0 - x_weight) * plate_carree[y0, x0] + x_weight * plate_carree[y0, x1]
    bottom = (1.0 - x_weight) * plate_carree[y1, x0] + x_weight * plate_carree[y1, x1]
    warped = (1.0 - y_weight) * top + y_weight * bottom
    warped[~valid] = 8.0
    return np.asarray(np.clip(warped, 0.0, 255.0), dtype=np.uint8)


def _satellite_boundaries(ax: plt.Axes, path: Path) -> None:
    if not path.exists():
        return
    with np.load(path) as lines:
        for name in lines.files:
            line = lines[name]
            x, y = FROM_LONLAT.transform(line[:, 0], line[:, 1])
            coast = name.startswith("coast")
            ax.plot(
                x,
                y,
                color="#f3eadb" if coast else "#2d2b29",
                linewidth=0.66 if coast else 0.42,
                alpha=0.56 if coast else 0.52,
                zorder=5,
            )


def _source_hotspots(
    emitted_co: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the top 0.1% of positive cumulative gridded source cells."""
    positive = emitted_co[emitted_co > 0.0]
    if positive.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    threshold = float(np.percentile(positive, 99.9))
    selected = emitted_co >= threshold
    magnitude = emitted_co[selected]
    scale = np.log1p(magnitude / threshold)
    sizes = 10.0 + 15.0 * scale / max(float(scale.max()), 1.0)
    x_index, y_index = np.nonzero(selected)
    x = MCIP_X_ORIGIN_M + (x_index + 0.5) * dx
    y = MCIP_Y_ORIGIN_M + (y_index + 0.5) * dy
    return x, y, sizes


def _satellite_animation(  # noqa: PLR0915
    *,
    output: Path | None,
    snapshot: Path,
    frames: np.ndarray,
    times: list,
    emitted_co: np.ndarray,
    dx: float,
    dy: float,
    boundaries: Path,
    basemap: Path,
    interpolation: int,
    fps: int,
    preview_frame: int | None,
) -> None:
    """Render an uncluttered satellite-map animation inspired by smoke maps."""
    display = frames / (dx * dy / 1.0e6)
    positive = display[display > 0.0]
    if positive.size == 0:
        raise ValueError("transport output has no positive CO mass to visualize")
    ceiling = float(np.percentile(positive, 99.7))
    floor = max(float(np.percentile(positive, 25.0)), ceiling / 250.0)
    positions = np.linspace(0.0, len(times) - 1, (len(times) - 1) * interpolation + 1)
    if preview_frame is not None:
        if not 0 <= preview_frame < len(times):
            raise ValueError(f"preview frame must be in [0, {len(times) - 1}]")
        positions = np.asarray([float(preview_frame)])

    bounds = SATELLITE_BOUNDS_M
    background = _load_basemap(basemap, bounds)
    cmap = _haze_colormap()
    norm = LogNorm(vmin=floor, vmax=ceiling, clip=True)
    x_edges = MCIP_X_ORIGIN_M + np.arange(frames.shape[1] + 1) * dx
    y_edges = MCIP_Y_ORIGIN_M + np.arange(frames.shape[2] + 1) * dy
    source_x, source_y, source_sizes = _source_hotspots(emitted_co, dx, dy)

    fig, ax = plt.subplots(figsize=(9.0, 9.0), facecolor="#11100f")
    ax.imshow(background, extent=bounds, origin="upper", interpolation="bilinear", zorder=0)
    first = np.ma.masked_less(display[int(positions[0])], floor)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        first.T,
        shading="flat",
        cmap=cmap,
        norm=norm,
        antialiased=False,
        edgecolors="none",
        linewidth=0.0,
        snap=True,
        zorder=3,
    )
    ax.scatter(
        source_x,
        source_y,
        s=source_sizes,
        marker="s",
        color="#ff1818",
        edgecolors="none",
        alpha=0.96,
        zorder=7,
    )
    _satellite_boundaries(ax, boundaries)
    ax.set(xlim=bounds[:2], ylim=bounds[2:])
    ax.set_aspect("auto")
    ax.set_axis_off()

    ax.text(
        0.035,
        0.958,
        "PROJECTED-2023 CO TRANSPORT",
        transform=ax.transAxes,
        color="#fffaf1",
        fontsize=17,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "square,pad=0.55",
            "facecolor": "#11100f",
            "edgecolor": "none",
            "alpha": 0.78,
        },
        zorder=10,
    )
    time_text = ax.text(
        0.038,
        0.902,
        "",
        transform=ax.transAxes,
        color="#f5ecdc",
        fontsize=10,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "square,pad=0.42",
            "facecolor": "#11100f",
            "edgecolor": "none",
            "alpha": 0.70,
        },
        zorder=10,
    )
    state_text = ax.text(
        0.965,
        0.958,
        "",
        transform=ax.transAxes,
        color="#fffaf1",
        fontsize=9,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "square,pad=0.48",
            "facecolor": "#11100f",
            "edgecolor": "none",
            "alpha": 0.72,
        },
        zorder=10,
    )

    legend_ax = ax.inset_axes([0.038, 0.080, 0.29, 0.016], zorder=11)
    legend_ax.imshow(
        np.linspace(0.0, 1.0, 256)[None, :],
        aspect="auto",
        cmap=cmap,
        extent=(0.0, 1.0, 0.0, 1.0),
    )
    legend_ax.set_axis_off()
    ax.text(
        0.038,
        0.103,
        "CO ENHANCEMENT  ·  kg km⁻²",
        transform=ax.transAxes,
        color="#fffaf1",
        fontsize=8,
        ha="left",
        va="bottom",
        zorder=12,
    )
    ax.text(
        0.038,
        0.065,
        f"{floor:.2g}",
        transform=ax.transAxes,
        color="#fffaf1",
        fontsize=7,
        ha="left",
        va="top",
        zorder=12,
    )
    ax.text(
        0.328,
        0.065,
        f"{ceiling:.2g}",
        transform=ax.transAxes,
        color="#fffaf1",
        fontsize=7,
        ha="right",
        va="top",
        zorder=12,
    )
    ax.scatter(
        [0.565],
        [0.085],
        transform=ax.transAxes,
        s=24,
        marker="s",
        color="#ff1818",
        edgecolors="none",
        zorder=12,
    )
    ax.text(
        0.585,
        0.085,
        "TOP 0.1% EMITTING GRID CELLS",
        transform=ax.transAxes,
        color="#fffaf1",
        fontsize=7.5,
        ha="left",
        va="center",
        zorder=12,
    )
    ax.text(
        0.5,
        0.027,
        "INERT ENHANCEMENT  ·  NO BACKGROUND, CHEMISTRY, DEPOSITION, OR TURBULENT MIXING",
        transform=ax.transAxes,
        color="#f0e5d4",
        fontsize=7,
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "square,pad=0.38",
            "facecolor": "#11100f",
            "edgecolor": "none",
            "alpha": 0.68,
        },
        zorder=11,
    )
    ax.plot(
        [0.035, 0.965],
        [0.014, 0.014],
        transform=ax.transAxes,
        color="#d6c5ad",
        linewidth=1.7,
        alpha=0.44,
        solid_capstyle="round",
        zorder=12,
    )
    (progress,) = ax.plot(
        [0.035, 0.035],
        [0.014, 0.014],
        transform=ax.transAxes,
        color="#ff4a32",
        linewidth=2.5,
        solid_capstyle="round",
        zorder=13,
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    images: list[Image.Image] = []
    snapshot_image: Image.Image | None = None
    for render_index, position in enumerate(positions):
        lower = min(int(np.floor(position)), len(times) - 1)
        upper = min(lower + 1, len(times) - 1)
        weight = position - lower
        field = (1.0 - weight) * display[lower] + weight * display[upper]
        stamp = times[lower] + timedelta(
            seconds=weight * (times[upper] - times[lower]).total_seconds()
        )
        mesh.set_array(np.ma.masked_less(field.T, floor).ravel())
        time_text.set_text(f"{stamp:%d %b %Y  ·  %H:%M UTC}  ·  12 km  ·  35 layers".upper())
        domain_mass = (
            (1.0 - weight) * frames[lower].sum() + weight * frames[upper].sum()
        ) / 1000.0
        state_text.set_text(
            f"IN DOMAIN  {domain_mass:,.0f} t\n"
            f"PEAK  {float(field.max()):,.0f} kg km⁻²"
        )
        fraction = position / max(len(times) - 1, 1)
        progress.set_xdata([0.035, 0.035 + 0.93 * fraction])

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100, facecolor=fig.get_facecolor())
        buffer.seek(0)
        frame = Image.open(buffer).convert("RGB")
        if preview_frame is not None or render_index == len(positions) - 1:
            snapshot_image = frame.copy()
        if output is not None:
            images.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))

    if snapshot_image is None:  # pragma: no cover - positions is never empty
        raise RuntimeError("no satellite frame was rendered")
    snapshot_image.save(snapshot)
    if output is not None:
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / fps),
            loop=0,
            disposal=2,
            optimize=False,
        )
    plt.close(fig)


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


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--basemap", type=Path, default=DEFAULT_BASEMAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--interpolation", type=int, default=1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--satellite-style",
        action="store_true",
        help="also render the Blue Marble CO presentation GIF and final frame",
    )
    parser.add_argument(
        "--satellite-only",
        action="store_true",
        help="render only the Blue Marble presentation products",
    )
    parser.add_argument(
        "--satellite-preview-frame",
        type=int,
        help="render one exact model frame to the satellite PNG without making a GIF",
    )
    args = parser.parse_args()
    if args.interpolation < 1 or args.fps < 1:
        parser.error("--interpolation and --fps must be positive")
    if args.satellite_preview_frame is not None:
        args.satellite_only = True
    if args.satellite_only:
        args.satellite_style = True
    if not args.satellite_only and args.diagnostics is None:
        parser.error("--diagnostics is required unless --satellite-only is used")
    if args.satellite_style and not args.basemap.exists():
        parser.error(
            f"satellite basemap not found: {args.basemap}; "
            "run examples/epa_2023/download_basemap.py"
        )

    with np.load(args.run) as run:
        visual_dtype = np.float32 if args.satellite_only else np.float64
        frames = np.asarray(run["column_mass_frames_kg"], dtype=visual_dtype)
        times = [
            np.datetime64(value).astype("datetime64[us]").astype(object)
            for value in run["frame_times"]
        ]
        emitted_co = np.asarray(run["emitted_co_kg"], dtype=visual_dtype)
        latitude = np.asarray(run["latitude"], dtype=np.float64)
        longitude = np.asarray(run["longitude"], dtype=np.float64)
        dx = float(run["dx"])
        dy = float(run["dy"])
        if not args.satellite_only:
            surface = np.asarray(run["surface_co_frames_ppbv"], dtype=np.float64)
            u = np.asarray(run["u_surface"], dtype=np.float64)
            v = np.asarray(run["v_surface"], dtype=np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run.stem
    satellite_gif = args.output_dir / f"{stem}_satellite.gif"
    satellite_png = args.output_dir / f"{stem}_satellite.png"
    if args.satellite_style:
        gif_output = None if args.satellite_preview_frame is not None else satellite_gif
        target_label = satellite_png if gif_output is None else satellite_gif
        print(f"rendering {target_label}")
        _satellite_animation(
            output=gif_output,
            snapshot=satellite_png,
            frames=frames,
            times=times,
            emitted_co=emitted_co,
            dx=dx,
            dy=dy,
            boundaries=args.boundaries,
            basemap=args.basemap,
            interpolation=args.interpolation,
            fps=args.fps,
            preview_frame=args.satellite_preview_frame,
        )
        print(f"wrote {satellite_png}" + (f" and {satellite_gif}" if gif_output else ""))
        if args.satellite_only:
            return 0

    if args.diagnostics is None:  # pragma: no cover - checked above
        raise RuntimeError("diagnostics path is required")
    diagnostics = _read_diagnostics(args.diagnostics)
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

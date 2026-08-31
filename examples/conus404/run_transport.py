#!/usr/bin/env python3
# ruff: noqa: PLR0915, PLR0917
"""Run 3-D CMAQ advection with CONUS404 winds and FINN fire CO.

This is an inert enhancement-tracer experiment, not a smoke forecast.  It runs
HADV followed by ZADV on all 50 WRF layers, keeps rho*J coupled as the last
transported slot, uses clean tracer inflow, and never resets atmospheric mass.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import (
    clean_boundary_conditions,
    coarsen_cells,
    coarsen_u,
    coarsen_v,
    coupled_co_to_ppbv,
    emission_tendency,
    grid_finn_co,
    horizontal_divergence,
    negative_tracer_mass_kg,
    read_finn_co,
    rhoj_from_dry_air_mass,
    tracer_mass_kg,
    vertical_centroid_m,
)

from cmaq_jax.advstep import (
    DEFAULT_LIMITS,
    advstep,
    sync_top_layer,
    wind_index,
)
from cmaq_jax.api import Meteorology, advance_xyfirst, advect_step
from cmaq_jax.config import DEFAULT_PPM, GridConfig, sigma_layer_thickness
from cmaq_jax.ppm import nonuniform_mesh

DEFAULT_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


class MetSubset:
    """Read a local subset file and expose arrays in CMAQ model order."""

    def __init__(self, path: Path) -> None:
        try:
            from netCDF4 import Dataset  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on installed extra
            raise ImportError(
                "the experiment needs netCDF4; install with: uv pip install -e '.[io]'"
            ) from exc
        self.path = path
        self.dataset = Dataset(path)
        self.dx = float(self.dataset.dx)
        self.dy = float(self.dataset.dy)
        self.sigma_face = np.asarray(self.dataset["sigma_face"][:], dtype=np.float64)
        self.times = [
            datetime.fromtimestamp(value.item(), tz=UTC)
            for value in np.asarray(self.dataset["time"][:])
        ]
        self.bbox = (
            float(self.dataset.west),
            float(self.dataset.east),
            float(self.dataset.south),
            float(self.dataset.north),
        )
        self.latitude_native = np.asarray(self.dataset["latitude"][:], dtype=np.float64).T
        self.longitude_native = np.asarray(self.dataset["longitude"][:], dtype=np.float64).T
        self.map_factor_native = np.asarray(self.dataset["map_factor"][:], dtype=np.float64).T

    def close(self) -> None:
        self.dataset.close()

    def u(self, index: int, factor: int) -> np.ndarray:
        raw = np.asarray(self.dataset["u"][index], dtype=np.float32).transpose(2, 1, 0)
        return coarsen_u(raw, factor)

    def v(self, index: int, factor: int) -> np.ndarray:
        raw = np.asarray(self.dataset["v"][index], dtype=np.float32).transpose(2, 1, 0)
        return coarsen_v(raw, factor)

    def rhoj(self, index: int, factor: int) -> np.ndarray:
        dry_mass = np.asarray(self.dataset["dry_air_mass"][index], dtype=np.float64).T
        native = rhoj_from_dry_air_mass(dry_mass, self.map_factor_native, self.nlays)
        return coarsen_cells(native, factor, extensive=False).astype(np.float32)

    def zface(self, factor: int) -> np.ndarray:
        native = np.asarray(self.dataset["zface"][:], dtype=np.float64).transpose(2, 1, 0)
        return coarsen_cells(native, factor, extensive=False)

    @property
    def nlays(self) -> int:
        return self.sigma_face.size - 1


def _diagnostic_row(
    *,
    stamp: datetime,
    state: np.ndarray,
    target_rhoj: np.ndarray,
    cfg: GridConfig,
    zface: np.ndarray,
    emitted_kg: float,
    inferred_boundary_loss_kg: float,
    max_courant: float,
    max_residual: float,
    max_vertical_substeps: int,
) -> dict[str, Any]:
    tracer = state[..., 0]
    rhoj = state[..., -1]
    mass = tracer_mass_kg(tracer, cfg.ds, cfg.dx1 * cfg.dx2)
    density_denominator = float(np.abs(target_rhoj).sum())
    density_relative_l1 = float(np.abs(rhoj - target_rhoj).sum() / density_denominator)
    density_relative_max = float(np.max(np.abs(rhoj - target_rhoj) / target_rhoj))
    return {
        "time_utc": stamp.isoformat(),
        "tracer_mass_kg": mass,
        "emitted_kg": emitted_kg,
        "inferred_boundary_loss_kg": inferred_boundary_loss_kg,
        "budget_residual_kg": emitted_kg - inferred_boundary_loss_kg - mass,
        "min_coupled_tracer": float(tracer.min()),
        "negative_tracer_mass_kg": negative_tracer_mass_kg(
            tracer, cfg.ds, cfg.dx1 * cfg.dx2
        ),
        "min_mixing_ratio_kg_kg": float(np.min(tracer / rhoj)),
        "rhoj_relative_l1": density_relative_l1,
        "rhoj_relative_max": density_relative_max,
        "vertical_centroid_m_msl": vertical_centroid_m(tracer, cfg.ds, zface),
        "max_vertical_courant": max_courant,
        "max_vertical_flux_residual": max_residual,
        "max_vertical_substeps": max_vertical_substeps,
    }


def _write_outputs(
    output: Path,
    diagnostics: list[dict[str, Any]],
    state: np.ndarray,
    column_frames: list[np.ndarray],
    surface_ppbv_frames: list[np.ndarray],
    frame_times: list[datetime],
    u_frames: list[np.ndarray],
    v_frames: list[np.ndarray],
    daily_emission_kg: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cfg: GridConfig,
    coarsen: int,
) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    npz_path = output.with_suffix(".npz")
    if csv_path.exists() or npz_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output}.[csv|npz]")

    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    column_mass = np.einsum("ijl,l->ij", state[..., 0], cfg.ds) * cfg.dx1 * cfg.dx2
    np.savez_compressed(
        npz_path,
        column_mass_kg=column_mass.astype(np.float32),
        column_mass_frames_kg=np.asarray(column_frames, dtype=np.float32),
        surface_co_frames_ppbv=np.asarray(surface_ppbv_frames, dtype=np.float32),
        frame_times=np.asarray([stamp.isoformat() for stamp in frame_times]),
        u_surface=np.asarray(u_frames, dtype=np.float32),
        v_surface=np.asarray(v_frames, dtype=np.float32),
        daily_emission_kg=np.asarray(daily_emission_kg, dtype=np.float32),
        latitude=latitude.astype(np.float32),
        longitude=longitude.astype(np.float32),
        dx=np.float64(cfg.dx1),
        dy=np.float64(cfg.dx2),
        coarsen=np.int64(coarsen),
    )
    return csv_path, npz_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--met", type=Path, required=True, help="output of download_conus404.py")
    parser.add_argument("--finn", type=Path, required=True, help="daily FINN each-fire txt.gz")
    parser.add_argument(
        "--coarsen", type=int, default=1, help="grid factor: 1=4 km, 2=8 km, 4=16 km"
    )
    parser.add_argument(
        "--vertical-max-substeps",
        type=int,
        default=DEFAULT_PPM.max_substeps,
        help="CMAQ default is 30; lowering is only for small development checks",
    )
    parser.add_argument(
        "--frame-minutes",
        type=int,
        default=60,
        help="save model-resolved animation frames at this interval",
    )
    parser.add_argument("--output", type=Path, default=None, help="output stem")
    args = parser.parse_args()
    if args.coarsen < 1:
        parser.error("--coarsen must be positive")
    if args.vertical_max_substeps < 1:
        parser.error("--vertical-max-substeps must be positive")
    if args.frame_minutes < 1:
        parser.error("--frame-minutes must be positive")

    met = MetSubset(args.met)
    try:
        if len(met.times) < 2:
            raise ValueError("meteorology subset needs at least two hourly records")
        intervals = [
            int((b - a).total_seconds())
            for a, b in zip(met.times[:-1], met.times[1:], strict=True)
        ]
        if any(seconds <= 0 for seconds in intervals):
            raise ValueError("meteorology timestamps must increase")
        frame_seconds = args.frame_minutes * 60
        if any(seconds % frame_seconds for seconds in intervals):
            raise ValueError(
                f"the {args.frame_minutes}-minute frame interval must divide every "
                f"meteorology interval, which is {intervals} seconds"
            )

        ds = sigma_layer_thickness(met.sigma_face)
        latitude = coarsen_cells(met.latitude_native, args.coarsen, extensive=False)
        longitude = coarsen_cells(met.longitude_native, args.coarsen, extensive=False)
        zface = met.zface(args.coarsen)
        ncols, nrows = latitude.shape
        ppm = replace(DEFAULT_PPM, max_substeps=args.vertical_max_substeps)
        cfg = GridConfig(
            ncols=ncols,
            nrows=nrows,
            ds=ds,
            dx1=met.dx * args.coarsen,
            dx2=met.dy * args.coarsen,
            nspc_adv=2,
            dtype="float32",
            ppm=ppm,
        )
        mesh = nonuniform_mesh(jnp.asarray(ds, dtype=jnp.float32))

        fires = read_finn_co(args.finn, met.bbox)
        daily_native = grid_finn_co(fires, met.latitude_native, met.longitude_native)
        daily_co = coarsen_cells(daily_native, args.coarsen, extensive=True)
        source = jnp.asarray(
            emission_tendency(daily_co, ds, cfg.dx1 * cfg.dx2), dtype=jnp.float32
        )
        daily_total = float(daily_co.sum())

        rho0 = met.rhoj(0, args.coarsen)
        state = np.zeros((ncols, nrows, met.nlays, 2), dtype=np.float32)
        state[..., -1] = rho0
        current: Any = jnp.asarray(state)
        xyfirst = (True,) * met.nlays
        compiled: dict[tuple[Any, ...], Any] = {}
        sync_layers = sync_top_layer(met.sigma_face, DEFAULT_LIMITS.sigma_sync_top)

        diagnostics = [
            _diagnostic_row(
                stamp=met.times[0],
                state=state,
                target_rhoj=rho0,
                cfg=cfg,
                zface=zface,
                emitted_kg=0.0,
                inferred_boundary_loss_kg=0.0,
                max_courant=0.0,
                max_residual=0.0,
                max_vertical_substeps=0,
            )
        ]
        cumulative_emitted = 0.0
        cumulative_boundary_loss = 0.0
        previous_mass = 0.0
        column_frames = [np.zeros((ncols, nrows), dtype=np.float32)]
        surface_ppbv_frames = [np.zeros((ncols, nrows), dtype=np.float32)]
        frame_times = [met.times[0]]
        u_frames: list[np.ndarray] = []
        v_frames: list[np.ndarray] = []
        total_sync_steps = 0
        started = time.perf_counter()

        print(
            f"domain {ncols} x {nrows} x {met.nlays} at {cfg.dx1 / 1000:.0f} km; "
            f"{len(fires.latitude)} FINN fires, {daily_total / 1000:.1f} metric tons CO/day"
        )
        for index, interval in enumerate(intervals):
            u0 = met.u(index, args.coarsen)
            u1 = met.u(index + 1, args.coarsen)
            v0 = met.v(index, args.coarsen)
            v1 = met.v(index + 1, args.coarsen)
            rho0 = met.rhoj(index, args.coarsen)
            rho1 = met.rhoj(index + 1, args.coarsen)
            if index == 0:
                u_frames.append(0.5 * (u0[:-1, :, 0] + u0[1:, :, 0]))
                v_frames.append(0.5 * (v0[:, :-1, 0] + v0[:, 1:, 0]))

            wind = np.maximum(
                wind_index(u0, v0, cfg.dx1, cfg.dx2),
                wind_index(u1, v1, cfg.dx1, cfg.dx2),
            )
            hdiv = np.maximum(
                horizontal_divergence(u0, v0, cfg.dx1, cfg.dx2),
                horizontal_divergence(u1, v1, cfg.dx1, cfg.dx2),
            )
            schedule = advstep(
                wind,
                hdiv,
                frame_seconds,
                DEFAULT_LIMITS,
                sync_layers=sync_layers,
            )
            count = interval // schedule.sync_seconds
            total_sync_steps += count
            interval_courant = 0.0
            interval_residual = 0.0
            interval_vertical_substeps = 0

            for substep in range(count):
                weight = (substep + 0.5) / count
                uhat = u0 + weight * (u1 - u0)
                vhat = v0 + weight * (v1 - v0)
                rhoj = rho0 + weight * (rho1 - rho0)
                weather = Meteorology(
                    uhat=uhat,
                    vhat=vhat,
                    rhoj_met=rhoj,
                    bcon=clean_boundary_conditions(rhoj),
                )
                key = (
                    xyfirst,
                    schedule.sync_seconds,
                    tuple(int(value) for value in schedule.astep_seconds),
                )
                if key not in compiled:
                    compiled[key] = jax.jit(
                        partial(
                            advect_step,
                            mesh=mesh,
                            cfg=cfg,
                            astep_seconds=schedule.astep_seconds,
                            sync_seconds=schedule.sync_seconds,
                            xyfirst=xyfirst,
                        )
                    )

                half_source = source * np.float32(0.5 * schedule.sync_seconds)
                current = current + half_source
                current, vertical = compiled[key](current, weather)
                current = current + half_source
                xyfirst = advance_xyfirst(
                    xyfirst, schedule.astep_seconds, schedule.sync_seconds
                )

                interval_courant = max(
                    interval_courant,
                    float(np.max(np.asarray(jax.device_get(vertical.max_courant)))),
                )
                interval_residual = max(
                    interval_residual,
                    float(np.max(np.asarray(jax.device_get(vertical.residual)))),
                )
                interval_vertical_substeps = max(
                    interval_vertical_substeps,
                    int(np.max(np.asarray(jax.device_get(vertical.substeps)))),
                )
                elapsed_in_interval = (substep + 1) * schedule.sync_seconds
                if elapsed_in_interval % frame_seconds == 0:
                    frame_state = np.asarray(jax.device_get(current), dtype=np.float32)
                    column_frames.append(
                        np.einsum("ijl,l->ij", frame_state[..., 0], cfg.ds)
                        * cfg.dx1
                        * cfg.dx2
                    )
                    surface_ppbv_frames.append(
                        coupled_co_to_ppbv(
                            frame_state[..., 0, 0], frame_state[..., 0, -1]
                        ).astype(np.float32)
                    )
                    frame_times.append(
                        met.times[index] + timedelta(seconds=elapsed_in_interval)
                    )
                    end_weight = elapsed_in_interval / interval
                    frame_u = u0 + end_weight * (u1 - u0)
                    frame_v = v0 + end_weight * (v1 - v0)
                    u_frames.append(0.5 * (frame_u[:-1, :, 0] + frame_u[1:, :, 0]))
                    v_frames.append(0.5 * (frame_v[:, :-1, 0] + frame_v[:, 1:, 0]))

            state = np.asarray(jax.device_get(current), dtype=np.float32)
            mass = tracer_mass_kg(state[..., 0], ds, cfg.dx1 * cfg.dx2)
            interval_emitted = daily_total * interval / 86_400.0
            cumulative_emitted += interval_emitted
            cumulative_boundary_loss += previous_mass + interval_emitted - mass
            previous_mass = mass
            row = _diagnostic_row(
                stamp=met.times[index + 1],
                state=state,
                target_rhoj=rho1,
                cfg=cfg,
                zface=zface,
                emitted_kg=cumulative_emitted,
                inferred_boundary_loss_kg=cumulative_boundary_loss,
                max_courant=interval_courant,
                max_residual=interval_residual,
                max_vertical_substeps=interval_vertical_substeps,
            )
            diagnostics.append(row)
            print(
                f"{met.times[index + 1]:%Y-%m-%d %H:%M} UTC  mass={mass:.1f} kg  "
                f"min={row['min_coupled_tracer']:.3e}  "
                f"rhoJ-L1={row['rhoj_relative_l1']:.3e}  "
                f"z-CFL={interval_courant:.3f}  z-res={interval_residual:.3e}"
            )

        elapsed = time.perf_counter() - started
        stem = args.output or DEFAULT_OUTPUT / (
            f"transport_{met.times[0]:%Y%m%d_%H}_{cfg.dx1 / 1000:.0f}km"
        )
        csv_path, npz_path = _write_outputs(
            stem,
            diagnostics,
            state,
            column_frames,
            surface_ppbv_frames,
            frame_times,
            u_frames,
            v_frames,
            daily_co,
            latitude,
            longitude,
            cfg,
            args.coarsen,
        )
        print(
            f"{total_sync_steps} sync steps in {elapsed:.1f} s; wrote {csv_path} and {npz_path}"
        )
    finally:
        met.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

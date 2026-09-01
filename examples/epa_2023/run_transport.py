#!/usr/bin/env python3
# ruff: noqa: PLR0912, PLR0915, PLR0917
"""Run full-CONUS 12 km CO advection for EPA's projected-2023 case.

This is an inert enhancement-tracer experiment, not a complete air-quality
simulation.  It transports the EPA 2016v3 platform's hourly merged CO source
with its matching 2016 MCIP meteorology on all 35 layers.  Clean inflow means
there is no background CO, chemistry, deposition, or turbulent mixing.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from datetime import date, datetime, timedelta
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import (
    clean_boundary_conditions,
    coupled_co_to_ppbv,
    grid_signature,
    horizontal_divergence,
    negative_tracer_mass_kg,
    read_hourly_co,
    read_latitude_longitude,
    surface_emission_tendency,
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
from cmaq_jax.config import DEFAULT_PPM
from cmaq_jax.io_mcip import MetFiles, open_met
from cmaq_jax.ppm import nonuniform_mesh

DEFAULT_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
DEFAULT_START = date(2016, 7, 15)


def _daily_paths(data_dir: Path, day: date) -> tuple[Path, Path, Path, Path]:
    stamp = day.strftime("%y%m%d")
    yyyymmdd = day.strftime("%Y%m%d")
    return (
        data_dir / f"METCRO3D.12US1.35L.{stamp}",
        data_dir / f"METDOT3D.12US1.35L.{stamp}",
        data_dir / f"GRIDCRO2D.12US1.35L.{stamp}",
        data_dir / f"emis_mole_all_{yyyymmdd}_12US1_withbeis_withrwc_2023gf_16j.ncf",
    )


def _diagnostic_row(
    *,
    stamp: Any,
    state: np.ndarray,
    target_rhoj: np.ndarray,
    ds: np.ndarray,
    cell_area: float,
    layer_height: np.ndarray,
    emitted_kg: float,
    inferred_boundary_loss_kg: float,
    max_courant: float,
    max_residual: float,
    max_vertical_substeps: int,
) -> dict[str, Any]:
    tracer = state[..., 0]
    rhoj = state[..., -1]
    mass = tracer_mass_kg(tracer, ds, cell_area)
    return {
        "time_utc": stamp.isoformat(),
        "tracer_mass_kg": mass,
        "emitted_kg": emitted_kg,
        "inferred_boundary_loss_kg": inferred_boundary_loss_kg,
        "budget_residual_kg": emitted_kg - inferred_boundary_loss_kg - mass,
        "min_coupled_tracer": float(tracer.min()),
        "negative_tracer_mass_kg": negative_tracer_mass_kg(tracer, ds, cell_area),
        "min_mixing_ratio_kg_kg": float(np.min(tracer / rhoj)),
        "rhoj_relative_l1": float(np.abs(rhoj - target_rhoj).sum() / target_rhoj.sum()),
        "rhoj_relative_max": float(np.max(np.abs(rhoj - target_rhoj) / target_rhoj)),
        "vertical_centroid_m_msl": vertical_centroid_m(tracer, ds, layer_height),
        "max_vertical_courant": max_courant,
        "max_vertical_flux_residual": max_residual,
        "max_vertical_substeps": max_vertical_substeps,
    }


def _write_outputs(
    output: Path,
    diagnostics: list[dict[str, Any]],
    state: np.ndarray,
    column_frames: list[np.ndarray],
    surface_frames: list[np.ndarray],
    frame_times: list[Any],
    u_frames: list[np.ndarray],
    v_frames: list[np.ndarray],
    emitted_grid: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    ds: np.ndarray,
    dx: float,
    dy: float,
    elapsed_seconds: float,
    sync_steps: int,
    start_time: datetime,
    end_time: datetime,
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

    column_mass = np.einsum("ijl,l->ij", state[..., 0], ds) * dx * dy
    np.savez_compressed(
        npz_path,
        column_mass_kg=column_mass.astype(np.float32),
        column_mass_frames_kg=np.asarray(column_frames, dtype=np.float32),
        surface_co_frames_ppbv=np.asarray(surface_frames, dtype=np.float32),
        frame_times=np.asarray([stamp.isoformat() for stamp in frame_times]),
        u_surface=np.asarray(u_frames, dtype=np.float32),
        v_surface=np.asarray(v_frames, dtype=np.float32),
        emitted_co_kg=np.asarray(emitted_grid, dtype=np.float32),
        latitude=latitude.astype(np.float32),
        longitude=longitude.astype(np.float32),
        dx=np.float64(dx),
        dy=np.float64(dy),
        emissions_scenario=np.asarray("EPA 2016v3 2023gf projected emissions"),
        meteorology_case=np.asarray(
            f"EPA MCIP WRFv3.8 {start_time:%Y-%m-%d} through {end_time:%Y-%m-%d}"
        ),
        transport_processes=np.asarray("HADV+ZADV; no chemistry/deposition/turbulence"),
        backend=np.asarray(jax.default_backend()),
        elapsed_seconds=np.float64(elapsed_seconds),
        sync_steps=np.int64(sync_steps),
    )
    return csv_path, npz_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help="first 2016 meteorological day",
    )
    parser.add_argument("--met-cro", type=Path, default=None)
    parser.add_argument("--met-dot", type=Path, default=None)
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--emissions", type=Path, default=None)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--frame-minutes", type=int, default=15)
    parser.add_argument(
        "--vertical-max-substeps",
        type=int,
        default=DEFAULT_PPM.max_substeps,
    )
    parser.add_argument("--output", type=Path, default=None, help="output stem")
    args = parser.parse_args()
    if args.hours < 1:
        parser.error("--hours must be positive")
    if args.frame_minutes < 1 or 60 % args.frame_minutes:
        parser.error("--frame-minutes must be a positive divisor of 60")
    if args.vertical_max_substeps < 1:
        parser.error("--vertical-max-substeps must be positive")
    file_days = (args.hours + 23) // 24
    final_file_day = args.start + timedelta(days=file_days - 1)
    if args.start.year != 2016 or final_file_day.year != 2016:
        parser.error("EPA's 2023gf inputs are indexed by their 2016 meteorological dates")
    overrides = (args.met_cro, args.met_dot, args.emissions)
    if file_days > 1 and any(path is not None for path in overrides):
        parser.error("--met-cro, --met-dot, and --emissions only override a single-day run")

    _, _, first_grid, _ = _daily_paths(args.data_dir, args.start)
    grid_path = args.grid or first_grid
    latitude, longitude = read_latitude_longitude(grid_path)
    ppm = replace(DEFAULT_PPM, max_substeps=args.vertical_max_substeps)
    frame_seconds = args.frame_minutes * 60
    compiled: dict[tuple[Any, ...], Any] = {}
    diagnostics: list[dict[str, Any]] = []
    column_frames: list[np.ndarray] = []
    surface_frames: list[np.ndarray] = []
    frame_times: list[datetime] = []
    u_frames: list[np.ndarray] = []
    v_frames: list[np.ndarray] = []
    cumulative_emitted = 0.0
    cumulative_boundary_loss = 0.0
    previous_mass = 0.0
    total_sync_steps = 0
    cfg: Any = None
    ds: np.ndarray | None = None
    mesh: Any = None
    sync_layers: int | None = None
    cell_area = 0.0
    current: Any = None
    xyfirst: tuple[bool, ...] = ()
    state: np.ndarray | None = None
    emitted_grid: np.ndarray | None = None
    u0: np.ndarray | None = None
    v0: np.ndarray | None = None
    rho0: np.ndarray | None = None
    started = time.perf_counter()

    for day_index in range(file_days):
        day = args.start + timedelta(days=day_index)
        day_hours = min(24, args.hours - 24 * day_index)
        met_cro_default, met_dot_default, _, emissions_default = _daily_paths(
            args.data_dir, day
        )
        met_cro = args.met_cro or met_cro_default
        met_dot = args.met_dot or met_dot_default
        emissions_path = args.emissions or emissions_default
        emissions = read_hourly_co(emissions_path)
        files = MetFiles(met_cro, met_dot, grid_path)

        with open_met(files) as met:
            times = met.times[: day_hours + 1]
            if len(times) != day_hours + 1:
                raise ValueError(
                    f"{met_cro.name} has only {len(met.times) - 1} hourly intervals"
                )
            if tuple(times) != emissions.times[: day_hours + 1]:
                raise ValueError(
                    f"meteorology and emissions times differ: {times[:2]} vs "
                    f"{emissions.times[:2]}"
                )
            from netCDF4 import Dataset  # noqa: PLC0415

            with Dataset(met_cro) as dataset:
                met_grid = grid_signature(dataset)
            if met_grid != emissions.grid:
                raise ValueError(
                    f"meteorology and emissions grids differ: {met_grid} vs {emissions.grid}"
                )
            if latitude.shape != (met.ncols, met.nrows):
                raise ValueError(
                    f"grid LAT/LON are {latitude.shape}, expected {met.ncols, met.nrows}"
                )

            day_cfg = met.grid_config(nspc_adv=2, dtype="float32", ppm=ppm)
            day_ds = np.asarray(day_cfg.ds, dtype=np.float64)
            day_rho0 = met.density(times[0]).astype(np.float32)
            day_u0_raw, day_v0_raw = met.face_velocities(times[0])
            day_u0 = day_u0_raw.astype(np.float32)
            day_v0 = day_v0_raw.astype(np.float32)

            if day_index == 0:
                cfg = day_cfg
                ds = day_ds
                mesh = nonuniform_mesh(jnp.asarray(ds, dtype=jnp.float32))
                sync_layers = sync_top_layer(
                    met.sigma_faces, DEFAULT_LIMITS.sigma_sync_top
                )
                cell_area = cfg.dx1 * cfg.dx2
                u0, v0, rho0 = day_u0, day_v0, day_rho0
                layer_height = met.cross("ZH", times[0]).astype(np.float32)
                state = np.zeros(
                    (met.ncols, met.nrows, met.nlays, 2), dtype=np.float32
                )
                state[..., -1] = rho0
                current = jnp.asarray(state)
                xyfirst = (True,) * met.nlays
                diagnostics.append(
                    _diagnostic_row(
                        stamp=times[0],
                        state=state,
                        target_rhoj=rho0,
                        ds=ds,
                        cell_area=cell_area,
                        layer_height=layer_height,
                        emitted_kg=0.0,
                        inferred_boundary_loss_kg=0.0,
                        max_courant=0.0,
                        max_residual=0.0,
                        max_vertical_substeps=0,
                    )
                )
                column_frames.append(
                    np.zeros((met.ncols, met.nrows), dtype=np.float32)
                )
                surface_frames.append(
                    np.zeros((met.ncols, met.nrows), dtype=np.float32)
                )
                frame_times.append(times[0])
                u_frames.append(0.5 * (u0[:-1, :, 0] + u0[1:, :, 0]))
                v_frames.append(0.5 * (v0[:, :-1, 0] + v0[:, 1:, 0]))
                emitted_grid = np.zeros((met.ncols, met.nrows), dtype=np.float64)
                print(
                    f"backend={jax.default_backend()}; domain {met.ncols} x "
                    f"{met.nrows} x {met.nlays} at {cfg.dx1 / 1000:.0f} km; "
                    f"continuous projected-2023 CO run for {args.hours} hours"
                )
            else:
                assert cfg is not None and ds is not None
                assert u0 is not None and v0 is not None and rho0 is not None
                if frame_times[-1] != times[0]:
                    raise ValueError(
                        f"daily files are not continuous: {frame_times[-1]} to {times[0]}"
                    )
                if (
                    (day_cfg.ncols, day_cfg.nrows, day_cfg.nlays)
                    != (cfg.ncols, cfg.nrows, cfg.nlays)
                    or day_cfg.dx1 != cfg.dx1
                    or day_cfg.dx2 != cfg.dx2
                    or not np.allclose(day_ds, ds, rtol=0.0, atol=0.0)
                ):
                    raise ValueError(f"MCIP grid or layers changed in {met_cro.name}")
                if not np.allclose(day_rho0, rho0, rtol=2.0e-6, atol=1.0e-6):
                    difference = float(np.max(np.abs(day_rho0 - rho0)))
                    raise ValueError(
                        f"MCIP density is discontinuous at {times[0]}: max |Δ|={difference}"
                    )
                u0, v0, rho0 = day_u0, day_v0, day_rho0

            assert cfg is not None and ds is not None and sync_layers is not None
            assert state is not None and emitted_grid is not None
            assert u0 is not None and v0 is not None and rho0 is not None
            daily_tons = (
                emissions.kilograms_per_second[:day_hours].sum() * 3600.0 / 1000.0
            )
            print(f"{day:%Y-%m-%d}: {day_hours} h forcing, {daily_tons:,.1f} t CO")

            for index, (start, end) in enumerate(pairwise(times)):
                interval = int((end - start).total_seconds())
                if interval != 3600:
                    raise ValueError(
                        f"expected hourly meteorology, got {interval} s at {start}"
                    )
                u1_raw, v1_raw = met.face_velocities(end)
                u1 = u1_raw.astype(np.float32)
                v1 = v1_raw.astype(np.float32)
                rho1 = met.density(end).astype(np.float32)

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
                if interval % schedule.sync_seconds:
                    raise ValueError(
                        f"sync step {schedule.sync_seconds} does not divide {interval}"
                    )
                count = interval // schedule.sync_seconds
                total_sync_steps += count
                rate = emissions.kilograms_per_second[index]
                source = jnp.asarray(
                    surface_emission_tendency(rate, ds, cell_area), dtype=jnp.float32
                )
                emitted_grid += rate * interval
                interval_emitted = float(rate.sum() * interval)
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
                        frame_state = np.asarray(
                            jax.device_get(current), dtype=np.float32
                        )
                        column_frames.append(
                            np.einsum("ijl,l->ij", frame_state[..., 0], ds)
                            * cell_area
                        )
                        surface_frames.append(
                            coupled_co_to_ppbv(
                                frame_state[..., 0, 0], frame_state[..., 0, -1]
                            ).astype(np.float32)
                        )
                        frame_times.append(
                            start + timedelta(seconds=elapsed_in_interval)
                        )
                        end_weight = elapsed_in_interval / interval
                        frame_u = u0 + end_weight * (u1 - u0)
                        frame_v = v0 + end_weight * (v1 - v0)
                        u_frames.append(
                            0.5 * (frame_u[:-1, :, 0] + frame_u[1:, :, 0])
                        )
                        v_frames.append(
                            0.5 * (frame_v[:, :-1, 0] + frame_v[:, 1:, 0])
                        )

                state = np.asarray(jax.device_get(current), dtype=np.float32)
                mass = tracer_mass_kg(state[..., 0], ds, cell_area)
                cumulative_emitted += interval_emitted
                cumulative_boundary_loss += previous_mass + interval_emitted - mass
                previous_mass = mass
                layer_height = met.cross("ZH", end).astype(np.float32)
                row = _diagnostic_row(
                    stamp=end,
                    state=state,
                    target_rhoj=rho1,
                    ds=ds,
                    cell_area=cell_area,
                    layer_height=layer_height,
                    emitted_kg=cumulative_emitted,
                    inferred_boundary_loss_kg=cumulative_boundary_loss,
                    max_courant=interval_courant,
                    max_residual=interval_residual,
                    max_vertical_substeps=interval_vertical_substeps,
                )
                diagnostics.append(row)
                print(
                    f"{end:%Y-%m-%d %H:%M} UTC  mass={mass / 1000:.1f} t  "
                    f"emitted={cumulative_emitted / 1000:.1f} t  "
                    f"min={row['min_coupled_tracer']:.2e}  "
                    f"rhoJ-L1={row['rhoj_relative_l1']:.3e}  "
                    f"z-res={interval_residual:.3e}"
                )
                u0, v0, rho0 = u1, v1, rho1

    elapsed = time.perf_counter() - started
    assert cfg is not None and ds is not None and state is not None
    assert emitted_grid is not None and frame_times
    stem = args.output or DEFAULT_OUTPUT / (
        f"transport_2023gf_{frame_times[0]:%Y%m%d}_{args.hours:02d}h_12km"
    )
    csv_path, npz_path = _write_outputs(
        stem,
        diagnostics,
        state,
        column_frames,
        surface_frames,
        frame_times,
        u_frames,
        v_frames,
        emitted_grid,
        latitude,
        longitude,
        ds,
        cfg.dx1,
        cfg.dx2,
        elapsed,
        total_sync_steps,
        frame_times[0],
        frame_times[-1],
    )
    print(
        f"{total_sync_steps} sync steps in {elapsed:.1f} s; "
        f"wrote {csv_path} and {npz_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

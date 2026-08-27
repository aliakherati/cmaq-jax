#!/usr/bin/env python3
"""Time a full advection step.

    python -m cmaq_jax.bench                       # benchmark-sized domain
    python -m cmaq_jax.bench --ncols 50 --nrows 50

Reports milliseconds per sync step, and the throughput that implies. The
default size is CMAQ's 2018 12NE3 benchmark domain -- 100x105 cells, 35 layers
-- with a species count typical of CB6.

Runs on whatever device JAX finds. Nothing here is CPU- or GPU-specific.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from cmaq_jax.advstep import StepLimits, advstep, sync_top_layer, wind_index
from cmaq_jax.api import Meteorology, advect_step
from cmaq_jax.config import DEFAULT_PPM, GridConfig, sigma_layer_thickness
from cmaq_jax.hadv import BoundaryConditions
from cmaq_jax.ppm import nonuniform_mesh


def build(ncols: int, nrows: int, nlays: int, *, nspc: int, dtype: str, substeps: int):
    """A synthetic state of the right shape and a plausible wind profile."""
    faces = np.linspace(1.0, 0.0, nlays + 1) ** 0.625
    ds = sigma_layer_thickness(faces)
    cfg = GridConfig(
        ncols=ncols,
        nrows=nrows,
        ds=ds,
        dx1=12000.0,
        dx2=12000.0,
        nspc_adv=nspc,
        dtype=dtype,  # type: ignore[arg-type]
        ppm=replace(DEFAULT_PPM, max_substeps=substeps),
    )

    rng = np.random.default_rng(20260911)
    rhoj = 1.5 + 0.4 * rng.random((ncols, nrows, nlays))
    tracers = [(0.5 + rng.random()) * rhoj for _ in range(nspc - 1)]
    state = np.stack([*tracers, rhoj], axis=-1)

    # Wind increasing with height, as a real profile does.
    profile = np.linspace(1.0, 4.0, nlays)
    uhat = 8.0 * profile[None, None, :] * np.ones((ncols + 1, nrows, 1))
    vhat = 6.0 * profile[None, None, :] * np.ones((ncols, nrows + 1, 1))

    edge = np.array([*(1.0 for _ in range(nspc - 1)), 2.0])

    # Put everything on the device up front. Handing the step numpy arrays
    # would make JAX convert them on every call, and the transfer would land
    # inside the measurement rather than the arithmetic.
    def device(array: np.ndarray) -> jax.Array:
        return jnp.asarray(array, dtype=cfg.numpy_dtype)

    bcon = BoundaryConditions(
        *(device(np.broadcast_to(edge, (n, nlays, nspc))) for n in (nrows, nrows, ncols, ncols))
    )
    met = Meteorology(uhat=device(uhat), vhat=device(vhat), rhoj_met=device(rhoj * 1.01), bcon=bcon)
    return cfg, nonuniform_mesh(device(ds)), device(state), met, faces


def time_step(fn, *args, repeats: int) -> float:
    """Milliseconds per call, excluding compilation."""
    warm = fn(*args)
    jax.block_until_ready(jax.tree.leaves(warm)[0])
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(*args)
    jax.block_until_ready(jax.tree.leaves(result)[0])
    return (time.perf_counter() - start) / repeats * 1e3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncols", type=int, default=100)
    parser.add_argument("--nrows", type=int, default=105)
    parser.add_argument("--nlays", type=int, default=35)
    parser.add_argument("--nspc", type=int, default=80, help="advected slots, rho*J included")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)

    cfg, mesh, state, met, faces = build(
        args.ncols,
        args.nrows,
        args.nlays,
        nspc=args.nspc,
        dtype=args.dtype,
        substeps=DEFAULT_PPM.max_substeps,
    )

    # Let advstep pick the schedule, as a real run would.
    limits = StepLimits()
    wind = wind_index(met.uhat, met.vhat, cfg.dx1, cfg.dx2)
    layers = sync_top_layer(faces, limits.sigma_sync_top)
    schedule = advstep(wind, np.zeros(args.nlays), 3600, limits, sync_layers=layers)

    print(f"device        : {jax.devices()[0]}")
    print(f"domain        : {args.ncols} x {args.nrows} x {args.nlays}, {args.nspc} species")
    print(f"precision     : {args.dtype}")
    print(f"sync step     : {schedule.sync_seconds} s  (sync-top layer {layers}/{args.nlays})")
    print(
        f"advection step: {schedule.astep_seconds.min()}-{schedule.astep_seconds.max()} s, "
        f"{schedule.substeps.min()}-{schedule.substeps.max()} sub-steps per layer"
    )

    step = jax.jit(
        partial(
            advect_step,
            mesh=mesh,
            cfg=cfg,
            astep_seconds=schedule.astep_seconds,
            sync_seconds=schedule.sync_seconds,
            xyfirst=(True,) * args.nlays,
        )
    )
    milliseconds = time_step(step, state, met, repeats=args.repeats)

    cells = args.ncols * args.nrows * args.nlays * args.nspc
    print(f"\nper sync step : {milliseconds:8.2f} ms")
    print(f"throughput    : {cells / milliseconds / 1e3:8.2f} M cell-species / s")
    print(
        f"simulated day : {milliseconds * 86400 / schedule.sync_seconds / 1e3:8.2f} s of wall time"
    )

    # max_substeps is the largest single lever on cost; show what it buys.
    print("\ncost of the vertical sub-step cap (see PPMConstants.max_substeps):")
    for cap in (DEFAULT_PPM.max_substeps, 8, 4, 1):
        capped = replace(cfg, ppm=replace(cfg.ppm, max_substeps=cap))
        fn = jax.jit(
            partial(
                advect_step,
                mesh=mesh,
                cfg=capped,
                astep_seconds=schedule.astep_seconds,
                sync_seconds=schedule.sync_seconds,
                xyfirst=(True,) * args.nlays,
            )
        )
        elapsed = time_step(fn, state, met, repeats=max(args.repeats // 2, 3))
        _, diag = fn(state, met)
        needed = int(np.asarray(diag.substeps).max())
        ok = bool(np.all(np.isfinite(np.asarray(diag.residual))))
        print(
            f"  max_substeps={cap:3d}: {elapsed:8.2f} ms   "
            f"columns needed {needed}   converged: {ok}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

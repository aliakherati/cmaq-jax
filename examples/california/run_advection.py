#!/usr/bin/env python3
"""Advect two Central Valley tracers through a month of NARR winds.

    python examples/california/run_advection.py

Writes ``data/run_<stamp>.npz`` for the animation script.

**What this runs.** `cmaq_jax.hadv.hadv_step` — CMAQ's PPM horizontal advection,
the code validated against unmodified Fortran in `tests/regression/`. The winds
are NARR, unmodified and unregridded, on NARR's own Lambert grid.

**What it does not run, and why.** Vertical advection (`zadv`) is left out. Its
flux diagnosis assumes a column closed at both ends — sigma thicknesses summing
to one — and this domain is a five-layer slab through the lower troposphere, not
a whole atmosphere. Running `zadv` on a slab would produce numbers, and they
would not mean anything. Horizontal advection is the part that moves a plume
across California, which is what the demo is about.

**The tracers are tracers.** Two inert species with plausible source *patterns* —
urban centres for traffic, a band along the valley floor for agriculture — at
arbitrary units. Nothing here is an emissions inventory, and with no chemistry,
deposition or vertical mixing the concentrations are transport only.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import domain as cal

from cmaq_jax.advstep import DEFAULT_LIMITS, advstep, wind_index
from cmaq_jax.config import GridConfig
from cmaq_jax.hadv import BoundaryConditions, advance_xyfirst, hadv_step

DATA = Path(__file__).resolve().parent / "data"

#: Sigma thickness of each of the five pressure layers, from
#: sigma = (p - ptop)/(psfc - ptop) with psfc = 1013.25 mb and ptop = 100 mb.
DS = np.full(5, 0.0274)

#: Emission rate per hour, in tracer units, at the peak of each mask. Chosen so
#: a month of accumulation lands in a range that plots readably; the absolute
#: value carries no meaning.
EMISSION_PER_HOUR = 1.0

#: Uniform air density times Jacobian. Advection carries this as an extra
#: species -- it is CMAQ's mass-conservation mechanism -- so it must be present
#: even though nothing here varies it.
RHOJ = 1.0


def restore_density(state: np.ndarray) -> np.ndarray:
    """Reset rho*J to its background, holding the mixing ratio fixed.

    This stands in for vertical advection, and it is needed rather than
    cosmetic. NARR's horizontal winds are divergent -- in the atmosphere that
    divergence is balanced by vertical motion. Advection carries rho*J as a
    species, so with no vertical operator to absorb the divergence the advected
    density wanders: measured over two days it spread from 1.0 to 0.20-2.81, and
    over a month it would run away entirely.

    CMAQ closes the same gap with `zadv`, which diagnoses a vertical flux from
    exactly this mismatch and transports every species along it. That needs a
    column closed at both ends, which a five-layer slab is not, so instead the
    density is put back and the coupled concentrations are rescaled with it.
    The mixing ratio `c / rho*J` -- the physical quantity, and what gets plotted
    -- is untouched by the rescaling.

    What this does *not* reproduce is the vertical redistribution `zadv` would
    also do. There is no vertical exchange between the five layers here at all.
    """
    mixing_ratio = state[..., :-1] / state[..., -1:]
    restored = state.copy()
    restored[..., :-1] = mixing_ratio * RHOJ
    restored[..., -1] = RHOJ
    return restored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", type=float, default=31.0, help="shorten the run while iterating")
    args = parser.parse_args(argv)

    field = cal.load(args.year, args.month)
    ncols, nrows, nlays = field.shape
    nspc = 3  # urban, agricultural, and rho*J in the last slot

    cfg = GridConfig(
        ncols=ncols,
        nrows=nrows,
        ds=DS,
        dx1=field.dx,
        dx2=field.dx,
        nspc_adv=nspc,
    )

    # Clean inflow on every side: whatever leaves does not come back.
    edge = np.zeros(nspc)
    edge[-1] = RHOJ
    bcon = BoundaryConditions(
        *(
            np.broadcast_to(edge, (n, nlays, nspc)).astype(np.float64)
            for n in (nrows, nrows, ncols, ncols)
        )
    )

    state = np.zeros((ncols, nrows, nlays, nspc))
    state[..., -1] = RHOJ

    # Emissions go into the lowest layer only, scaled to the sync step below.
    source = np.zeros((ncols, nrows, nlays, nspc))
    source[:, :, 0, 0] = field.urban
    source[:, :, 0, 1] = field.agricultural

    interval = (field.times[1] - field.times[0]).total_seconds()
    wanted = min(len(field.times) - 1, int(args.days * 86400 / interval))
    print(
        f"domain {ncols} x {nrows} x {nlays} at {field.dx / 1000:.1f} km, "
        f"{wanted} x {interval / 3600:.0f} h from {field.times[0]:%Y-%m-%d %H:%M}"
    )

    frames = [state[..., 0, :2].copy()]
    stamps = [field.times[0]]
    compiled: dict[tuple, object] = {}
    xyfirst = (True,) * nlays
    started = time.perf_counter()
    total_sync = 0

    for index in range(wanted):
        # NARR is three-hourly; step the model on a CFL-safe sub-interval and
        # interpolate the winds linearly across it, as CMAQ does between met
        # records.
        u0, u1 = field.u[index], field.u[index + 1]
        v0, v1 = field.v[index], field.v[index + 1]

        schedule = advstep(
            wind_index(*cal.face_velocities(u0, v0), cfg.dx1, cfg.dx2),
            np.zeros(nlays),
            int(interval),
            DEFAULT_LIMITS,
        )
        steps = int(interval) // schedule.sync_seconds
        total_sync += steps

        for step in range(steps):
            weight = (step + 0.5) / steps
            uhat, vhat = cal.face_velocities(u0 + weight * (u1 - u0), v0 + weight * (v1 - v0))
            key = (xyfirst, schedule.sync_seconds, tuple(schedule.astep_seconds))
            if key not in compiled:
                compiled[key] = jax.jit(
                    partial(
                        hadv_step,
                        cfg=cfg,
                        astep_seconds=schedule.astep_seconds,
                        sync_seconds=schedule.sync_seconds,
                        xyfirst=xyfirst,
                    )
                )
            advected = compiled[key](state, uhat, vhat, bcon)  # type: ignore[operator]
            state = np.array(advected, dtype=np.float64)
            state = restore_density(state)
            state += source * (EMISSION_PER_HOUR * schedule.sync_seconds / 3600.0)
            xyfirst = advance_xyfirst(xyfirst, schedule.astep_seconds, schedule.sync_seconds)

        frames.append(state[..., 0, :2].copy())
        stamps.append(field.times[index + 1])
        if index % 24 == 0:
            print(
                f"  {field.times[index]:%m-%d %H:%M}  "
                f"peak urban {state[..., 0, 0].max():8.2f}  "
                f"agri {state[..., 0, 1].max():8.2f}  "
                f"rho*J {state[..., -1].min():.4f}-{state[..., -1].max():.4f}"
            )

    elapsed = time.perf_counter() - started
    print(
        f"\n{total_sync} sync steps in {elapsed:.1f} s "
        f"({1000 * elapsed / total_sync:.1f} ms per step)"
    )

    out = DATA / f"run_{args.year}{args.month:02d}.npz"
    np.savez_compressed(
        out,
        frames=np.array(frames, dtype=np.float32),
        times=np.array([s.isoformat() for s in stamps]),
        lat=field.lat,
        lon=field.lon,
        urban=field.urban,
        agricultural=field.agricultural,
        u=field.u[: wanted + 1, :, :, 0].astype(np.float32),
        v=field.v[: wanted + 1, :, :, 0].astype(np.float32),
        dx=field.dx,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

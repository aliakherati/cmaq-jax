#!/usr/bin/env python3
"""Fetch a month of NARR winds over California.

    python examples/california/download_met.py

Downloads to ``examples/california/data/``, which is git-ignored — the data is
~40 MB and reproducible from this script.

**Why NARR.** It is a 32 km reanalysis on a Lambert conformal grid covering
North America, 3-hourly, with monthly files and a THREDDS subset service that
will crop to a bounding box. The projection matters: CMAQ runs on Lambert
conformal too, so the reanalysis grid *is* a usable model grid and no
regridding is needed. Nothing here interpolates the meteorology, which means
the winds driving the demo are exactly what the reanalysis says they were.

**What 32 km does and does not resolve.** It carries the synoptic flow and the
regional sea breeze, which is what moves pollution out of the San Joaquin Valley.
It does *not* resolve the valley itself — the valley is ~80 km wide, so about
two and a half cells — so do not read valley channeling into the result. A finer
model grid would not fix that: it would interpolate the same 32 km wind field.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
BASE = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/NARR"

#: A box around California with room for the Pacific inflow to the west and the
#: Great Basin to the east, so material can leave the domain rather than piling
#: up against a wall.
WEST, EAST, SOUTH, NORTH = -128.0, -112.0, 30.0, 44.0

#: Pressure levels standing in for a boundary layer and the air just above it.
#: NARR carries 29 levels; these five span roughly the lowest 1.5 km.
LEVELS = (1000, 975, 950, 925, 900)


def fetch(variable: str, year: int, month: int, level: int, out: Path) -> None:
    """One variable at one level for one month, cropped to the box."""
    days = 31 if month in (1, 3, 5, 7, 8, 10, 12) else 30 if month != 2 else 28
    query = (
        f"?var={variable}"
        f"&north={NORTH}&west={WEST}&east={EAST}&south={SOUTH}"
        f"&vertCoord={level}"
        f"&time_start={year}-{month:02d}-01T00:00:00Z"
        f"&time_end={year}-{month:02d}-{days:02d}T21:00:00Z"
        "&accept=netcdf4"
    )
    url = f"{BASE}/pressure/{variable}.{year}{month:02d}.nc{query}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"  {out.name}: already present ({out.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  {out.name}: fetching…", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=600) as response:
            out.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"\n{variable} @ {level} mb failed: HTTP {error.code}\n{url}") from error
    print(f" {out.stat().st_size / 1e6:.1f} MB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--month", type=int, default=7)
    args = parser.parse_args(argv)

    print(
        f"NARR winds, {args.year}-{args.month:02d}, box "
        f"({SOUTH}..{NORTH} N, {WEST}..{EAST} E), levels {LEVELS} mb"
    )
    for level in LEVELS:
        for variable in ("uwnd", "vwnd"):
            fetch(
                variable,
                args.year,
                args.month,
                level,
                DATA / f"{variable}_{level}mb_{args.year}{args.month:02d}.nc",
            )
    print(f"\nwrote to {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

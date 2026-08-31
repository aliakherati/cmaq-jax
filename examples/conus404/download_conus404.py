#!/usr/bin/env python3
# ruff: noqa: PLR0915
"""Subset hourly CONUS404 meteorology for a California transport run.

The RDA THREDDS server is queried with OPeNDAP hyperslabs, so only the requested
native grid window is transferred.  The output keeps WRF's C staggering and all
50 vertical layers; no horizontal or vertical interpolation is performed.

Example (seven records for a six-hour run):

    .venv/bin/python examples/conus404/download_conus404.py \
        --start 2018-07-26T00:00:00 --hours 6
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

BASE = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d559000"
CONSTANTS_URL = f"{BASE}/INVARIANT/wrfconstants_usgs404.nc"
DEFAULT_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_BBOX = (-124.75, -114.0, 32.0, 42.25)


def _open(url: str) -> Any:
    try:
        from pydap.client import open_url  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on installed extra
        raise ImportError(
            "CONUS404 download needs pydap; install with: uv pip install -e '.[io]'"
        ) from exc
    return open_url(url, protocol="dap2")


def _read(variable: Any, key: Any, *, drop_time: bool = False) -> NDArray[np.float32]:
    # Pydap's BaseType implements ``__array__`` without numpy's optional dtype
    # argument, so conversion and casting must be two distinct operations.
    values = np.asarray(variable[key]).astype(np.float32, copy=False)
    if drop_time:
        if values.shape[0] != 1:
            raise ValueError(f"expected one time record, got {values.shape}")
        values = values[0]
    return values


def _window(
    latitude: NDArray[np.floating],
    longitude: NDArray[np.floating],
    bbox: tuple[float, float, float, float],
) -> tuple[slice, slice]:
    west, east, south, north = bbox
    inside = (
        (longitude >= west)
        & (longitude <= east)
        & (latitude >= south)
        & (latitude <= north)
    )
    rows, columns = np.nonzero(inside)
    if rows.size == 0:
        raise ValueError(f"bbox {bbox} does not intersect the CONUS404 grid")
    return slice(int(rows.min()), int(rows.max()) + 1), slice(
        int(columns.min()), int(columns.max()) + 1
    )


def _water_year(stamp: datetime) -> int:
    return stamp.year + int(stamp.month >= 10)


def _hourly_url(kind: str, stamp: datetime) -> str:
    encoded_stamp = stamp.strftime("%Y-%m-%d_%H%%3A%M%%3A%S")
    return (
        f"{BASE}/wy{_water_year(stamp)}/{stamp:%Y%m}/"
        f"wrf{kind}_d01_{encoded_stamp}.nc"
    )


def _global(dataset: Any, name: str) -> Any:
    try:
        return dataset.attributes[name]
    except KeyError as exc:
        raise ValueError(f"CONUS404 constants are missing global attribute {name}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=datetime.fromisoformat,
        default=datetime(2018, 7, 26),
        help="first UTC record (ISO 8601, aligned to an hour)",
    )
    parser.add_argument("--hours", type=int, default=6, help="run duration; downloads hours+1")
    parser.add_argument("--west", type=float, default=DEFAULT_BBOX[0])
    parser.add_argument("--east", type=float, default=DEFAULT_BBOX[1])
    parser.add_argument("--south", type=float, default=DEFAULT_BBOX[2])
    parser.add_argument("--north", type=float, default=DEFAULT_BBOX[3])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.hours < 1:
        parser.error("--hours must be at least 1")
    if args.start.minute or args.start.second or args.start.microsecond:
        parser.error("--start must be aligned to an exact hour")
    start = (
        args.start.replace(tzinfo=UTC)
        if args.start.tzinfo is None
        else args.start.astimezone(UTC)
    )
    bbox = (args.west, args.east, args.south, args.north)
    if not args.west < args.east or not args.south < args.north:
        parser.error("the west/east and south/north bounds must increase")

    output = args.output or DEFAULT_DATA / (
        f"conus404_{start:%Y%m%d_%H}_{args.hours:02d}h.nc"
    )
    if output.exists():
        print(f"already present: {output} ({output.stat().st_size / 1e6:.1f} MB)")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")

    constants = _open(CONSTANTS_URL)
    full_lat = np.asarray(constants["XLAT"][:]).astype(np.float64, copy=False)
    full_lon = np.asarray(constants["XLONG"][:]).astype(np.float64, copy=False)
    row_slice, column_slice = _window(full_lat, full_lon, bbox)
    ny = row_slice.stop - row_slice.start
    nx = column_slice.stop - column_slice.start

    latitude = full_lat[row_slice, column_slice].astype(np.float32)
    longitude = full_lon[row_slice, column_slice].astype(np.float32)
    map_factor = _read(
        constants["MAPFAC_M"], (slice(0, 1), row_slice, column_slice), drop_time=True
    )
    sigma_face = _read(constants["ZNW"], (slice(0, 1), slice(None)), drop_time=True)
    dx = float(_global(constants, "DX"))
    dy = float(_global(constants, "DY"))
    hybrid_opt = int(_global(constants, "HYBRID_OPT"))
    if sigma_face.shape != (51,):
        raise ValueError(f"expected all 51 CONUS404 sigma faces, got {sigma_face.shape}")
    if hybrid_opt != -1:
        raise ValueError(
            f"expected CONUS404's non-hybrid WRF coordinate (HYBRID_OPT=-1), got {hybrid_opt}"
        )

    print(
        f"native window x={column_slice.start}:{column_slice.stop}, "
        f"y={row_slice.start}:{row_slice.stop} -> {nx} x {ny} x 50"
    )
    print(f"bbox requested {bbox}; grid-cell centres span lon {longitude.min():.2f}.."
          f"{longitude.max():.2f}, lat {latitude.min():.2f}..{latitude.max():.2f}")

    try:
        from netCDF4 import Dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on installed extra
        raise ImportError(
            "writing the subset needs netCDF4; install with: uv pip install -e '.[io]'"
        ) from exc

    try:
        with Dataset(temporary, "w", format="NETCDF4") as target:
            target.createDimension("time", args.hours + 1)
            target.createDimension("bottom_top", 50)
            target.createDimension("bottom_top_stag", 51)
            target.createDimension("south_north", ny)
            target.createDimension("south_north_stag", ny + 1)
            target.createDimension("west_east", nx)
            target.createDimension("west_east_stag", nx + 1)

            target.title = "Native-grid CONUS404 subset for cmaq-jax transport validation"
            target.source = "CONUS404 USGS/RDA ds559.0"
            target.constants_url = CONSTANTS_URL
            target.dx = dx
            target.dy = dy
            target.west = args.west
            target.east = args.east
            target.south = args.south
            target.north = args.north
            target.i_start = column_slice.start
            target.j_start = row_slice.start
            target.hybrid_opt = hybrid_opt

            compression = {"zlib": True, "complevel": 2, "shuffle": True}
            target.createVariable("latitude", "f4", ("south_north", "west_east"))[:] = latitude
            target.createVariable("longitude", "f4", ("south_north", "west_east"))[:] = longitude
            target.createVariable(
                "map_factor", "f4", ("south_north", "west_east"), **compression
            )[:] = map_factor
            target.createVariable("sigma_face", "f4", ("bottom_top_stag",))[:] = sigma_face
            time_var = target.createVariable("time", "i8", ("time",))
            time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
            u_var = target.createVariable(
                "u",
                "f4",
                ("time", "bottom_top", "south_north", "west_east_stag"),
                chunksizes=(1, 50, min(ny, 64), min(nx + 1, 128)),
                **compression,
            )
            v_var = target.createVariable(
                "v",
                "f4",
                ("time", "bottom_top", "south_north_stag", "west_east"),
                chunksizes=(1, 50, min(ny + 1, 64), min(nx, 128)),
                **compression,
            )
            mass_var = target.createVariable(
                "dry_air_mass",
                "f4",
                ("time", "south_north", "west_east"),
                chunksizes=(1, min(ny, 128), min(nx, 128)),
                **compression,
            )
            mass_var.units = "Pa"
            mass_var.description = "CONUS404 MU, which stores dry airmass in column (MU+MUB)"
            z_var = target.createVariable(
                "zface",
                "f4",
                ("bottom_top_stag", "south_north", "west_east"),
                chunksizes=(51, min(ny, 64), min(nx, 128)),
                **compression,
            )
            z_var.units = "m MSL"

            for index in range(args.hours + 1):
                stamp = start + timedelta(hours=index)
                print(f"[{index + 1}/{args.hours + 1}] {stamp:%Y-%m-%d %H:%M} UTC", flush=True)
                fields3d = _open(_hourly_url("3d", stamp))
                fields2d = _open(_hourly_url("2d", stamp))
                u_var[index] = _read(
                    fields3d["U"],
                    (
                        slice(0, 1),
                        slice(None),
                        row_slice,
                        slice(column_slice.start, column_slice.stop + 1),
                    ),
                    drop_time=True,
                )
                v_var[index] = _read(
                    fields3d["V"],
                    (
                        slice(0, 1),
                        slice(None),
                        slice(row_slice.start, row_slice.stop + 1),
                        column_slice,
                    ),
                    drop_time=True,
                )
                mass_var[index] = _read(
                    fields2d["MU"], (slice(0, 1), row_slice, column_slice), drop_time=True
                )
                if index == 0:
                    z_var[:] = _read(
                        fields3d["Z"],
                        (slice(0, 1), slice(None), row_slice, column_slice),
                        drop_time=True,
                    )
                time_var[index] = int(stamp.timestamp())
                target.sync()
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

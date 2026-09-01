"""The California domain: grid, winds, emissions.

Shared by the run script and the animation so that both describe the same
domain and neither can drift from the other.

The grid **is** the NARR grid. NARR is Lambert conformal, which is what CMAQ
runs on, so the reanalysis cells are used directly as model cells and nothing
regrids the meteorology. The winds driving this demo are exactly what the
reanalysis says they were.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pyproj
from numpy.typing import NDArray

DATA = Path(__file__).resolve().parent / "data"
LEVELS = (1000, 975, 950, 925, 900)

#: Central Valley sources. Urban sites stand in for traffic; the agricultural
#: band is spread along the valley floor between them. Both are tracers, not
#: real inventories -- the point is where material enters, not how much.
CITIES = {
    "Sacramento": (38.58, -121.49, 0.7),
    "Stockton": (37.96, -121.29, 0.5),
    "Modesto": (37.64, -120.99, 0.5),
    "Fresno": (36.74, -119.79, 1.0),
    "Bakersfield": (35.37, -119.02, 1.0),
}

#: Corners of a band down the San Joaquin and Sacramento valleys, used to place
#: the agricultural source.
VALLEY_AXIS = ((39.5, -121.9), (35.0, -118.9))


@dataclass(frozen=True)
class Domain:
    """Grid geometry, hourly winds, and the emission masks."""

    lat: NDArray[np.float64]  # (ncols, nrows) cell centres
    lon: NDArray[np.float64]
    dx: float  # cell width, m
    times: list[datetime]
    u: NDArray[np.float64]  # (ntimes, ncols, nrows, nlays) cell-centre winds
    v: NDArray[np.float64]
    urban: NDArray[np.float64]  # (ncols, nrows) emission weight
    agricultural: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (*self.lat.shape, self.u.shape[-1])


def _read_level(variable: str, level: int, stamp: str) -> NDArray[np.float64]:
    path = DATA / f"{variable}_{level}mb_{stamp}.nc"
    if not path.exists():
        raise SystemExit(f"missing {path.name}\nRun: python examples/california/download_met.py")
    with netCDF4.Dataset(path) as nc:
        # (time, level, y, x) -> (time, x, y); the level axis has one entry.
        return np.transpose(np.asarray(nc.variables[variable][:, 0]), (0, 2, 1)).astype(np.float64)


def load(year: int = 2018, month: int = 7) -> Domain:
    stamp = f"{year}{month:02d}"
    with netCDF4.Dataset(DATA / f"uwnd_{LEVELS[0]}mb_{stamp}.nc") as nc:
        x = np.asarray(nc.variables["x"][:], dtype=np.float64)
        y = np.asarray(nc.variables["y"][:], dtype=np.float64)
        raw_time = nc.variables["time"]
        hours = np.asarray(raw_time[:], dtype=np.float64)
        epoch = datetime(1800, 1, 1)
        times = [epoch + timedelta(hours=float(h)) for h in hours]
        mapping = nc.variables["Lambert_Conformal"]
        crs = pyproj.CRS.from_cf(
            {
                "grid_mapping_name": "lambert_conformal_conic",
                "latitude_of_projection_origin": float(mapping.latitude_of_projection_origin),
                "longitude_of_central_meridian": float(mapping.longitude_of_central_meridian),
                "standard_parallel": float(mapping.standard_parallel),
                # The file's attributes carry these in km, the axes in metres.
                "false_easting": float(mapping.false_easting) * 1000.0,
                "false_northing": float(mapping.false_northing) * 1000.0,
                "earth_radius": float(mapping.earth_radius),
            }
        )

    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")  # (ncols, nrows)
    lon, lat = transformer.transform(grid_x, grid_y)

    u = np.stack([_read_level("uwnd", level, stamp) for level in LEVELS], axis=-1)
    v = np.stack([_read_level("vwnd", level, stamp) for level in LEVELS], axis=-1)

    urban, agricultural = emission_masks(lat, lon)
    return Domain(
        lat=lat,
        lon=lon,
        dx=float(np.diff(x).mean()),
        times=times,
        u=u,
        v=v,
        urban=urban,
        agricultural=agricultural,
    )


def emission_masks(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Where the two tracers enter the domain.

    Urban: a narrow Gaussian on each city, so most of the mass lands in one or
    two cells. Agricultural: a broad band along the valley floor, which is how
    area sources like livestock and tillage actually look.

    At 32 km the valley is under three cells wide, so neither pattern resolves
    it. They place material in the right cells; they do not claim to be an
    inventory.
    """
    urban = np.zeros_like(lat)
    for city_lat, city_lon, weight in CITIES.values():
        distance = np.hypot((lat - city_lat) * 111.0, (lon - city_lon) * 89.0)
        urban += weight * np.exp(-((distance / 35.0) ** 2))

    (lat0, lon0), (lat1, lon1) = VALLEY_AXIS
    # Distance from each cell to the valley axis segment, in km.
    axis = np.array([lat1 - lat0, lon1 - lon0])
    axis_km = np.array([axis[0] * 111.0, axis[1] * 89.0])
    length = float(np.hypot(*axis_km))
    offset = np.stack([(lat - lat0) * 111.0, (lon - lon0) * 89.0], axis=-1)
    along = np.clip((offset @ axis_km) / length**2, 0.0, 1.0)
    nearest = along[..., None] * axis_km
    across = np.linalg.norm(offset - nearest, axis=-1)
    agricultural = np.exp(-((across / 60.0) ** 2))

    return urban / urban.max(), agricultural / agricultural.max()


def face_velocities(
    u: NDArray[np.float64], v: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cell-centre winds averaged onto cell faces.

    ``hadv_step`` wants C-staggered velocities: ``u`` on the ``ncols+1`` faces
    normal to x, ``v`` on the ``nrows+1`` faces normal to y. Domain-edge faces
    take the single adjacent cell's value, which is a zero-gradient
    extrapolation -- the same convention the port uses elsewhere.
    """
    padded_u = np.concatenate([u[:1], u, u[-1:]], axis=0)
    uhat = 0.5 * (padded_u[:-1] + padded_u[1:])
    padded_v = np.concatenate([v[:, :1], v, v[:, -1:]], axis=1)
    vhat = 0.5 * (padded_v[:, :-1] + padded_v[:, 1:])
    return uhat, vhat

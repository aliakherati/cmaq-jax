"""Small, testable pieces of the CONUS404 + FINN transport experiment.

Arrays in this module use CMAQ/JAX order, ``(column, row, layer)``, unless a
docstring says otherwise.  CONUS404 files use WRF order and are transposed by
``run_transport.py`` at the I/O boundary.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cmaq_jax.hadv import BoundaryConditions

GRAVITY = 9.81  # m s-2; MCIP metvars2ctm.f90 uses the same value.
CO_MOLAR_MASS = 28.01e-3  # kg mol-1
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class FinnCO:
    """FINN fire locations and daily CO emissions."""

    latitude: NDArray[np.float64]
    longitude: NDArray[np.float64]
    moles_per_day: NDArray[np.float64]

    @property
    def kilograms_per_day(self) -> NDArray[np.float64]:
        """Daily CO mass at each fire location."""
        return self.moles_per_day * CO_MOLAR_MASS


def rhoj_from_dry_air_mass(
    dry_air_mass: NDArray[np.floating],
    map_factor: NDArray[np.floating],
    nlays: int,
) -> NDArray[np.float64]:
    """Reconstruct MCIP ``DENSA_J`` from CONUS404 WRF fields.

    For WRF's non-hybrid terrain coordinate, MCIP computes
    ``JACOBM = (MU+MUB)/(g*rho)`` and
    ``DENSA_J = rho*JACOBM/MAPFAC_M**2``.  CONUS404's processed ``MU`` variable
    already stores ``MU+MUB``, so density cancels and the result is the same in
    all layers.  It is nevertheless expanded over all 50 layers because rho*J
    rides through the advection scheme as the final transported slot.
    """
    column_mass = np.asarray(dry_air_mass, dtype=np.float64)
    scale = np.asarray(map_factor, dtype=np.float64)
    if column_mass.shape != scale.shape or column_mass.ndim != 2:
        raise ValueError(
            "dry_air_mass and map_factor must be matching 2-D fields, got "
            f"{column_mass.shape} and {scale.shape}"
        )
    if nlays < 1:
        raise ValueError(f"nlays must be positive, got {nlays}")
    if np.any(column_mass <= 0.0) or np.any(scale <= 0.0):
        raise ValueError("dry_air_mass and map_factor must be positive")

    rhoj_2d = column_mass / (GRAVITY * scale**2)
    return np.broadcast_to(rhoj_2d[..., None], (*rhoj_2d.shape, nlays)).copy()


def clean_boundary_conditions(rhoj: NDArray[np.floating]) -> BoundaryConditions:
    """Zero CO enhancement at inflow, with real rho*J on every edge."""
    density = np.asarray(rhoj)
    if density.ndim != 3:
        raise ValueError(f"rhoj must be (ncols, nrows, nlays), got {density.shape}")
    ncols, nrows, nlays = density.shape

    west = np.zeros((nrows, nlays, 2), dtype=density.dtype)
    east = np.zeros_like(west)
    south = np.zeros((ncols, nlays, 2), dtype=density.dtype)
    north = np.zeros_like(south)
    west[..., -1] = density[0]
    east[..., -1] = density[-1]
    south[..., -1] = density[:, 0]
    north[..., -1] = density[:, -1]
    return BoundaryConditions(west=west, east=east, south=south, north=north)


def horizontal_divergence(
    uhat: NDArray[np.floating],
    vhat: NDArray[np.floating],
    dx1: float,
    dx2: float,
) -> NDArray[np.float64]:
    """Maximum positive horizontal divergence in each layer, in s-1."""
    u = np.asarray(uhat, dtype=np.float64)
    v = np.asarray(vhat, dtype=np.float64)
    if u.ndim != 3 or v.ndim != 3:
        raise ValueError(f"uhat and vhat must be 3-D, got {u.shape} and {v.shape}")
    expected_v = (u.shape[0] - 1, u.shape[1] + 1, u.shape[2])
    if v.shape != expected_v:
        raise ValueError(f"vhat shape must be {expected_v} for uhat {u.shape}, got {v.shape}")
    if dx1 <= 0.0 or dx2 <= 0.0:
        raise ValueError("dx1 and dx2 must be positive")

    divergence = np.diff(u, axis=0) / dx1 + np.diff(v, axis=1) / dx2
    return np.maximum(divergence, 0.0).max(axis=(0, 1))


def read_finn_co(path: Path, bbox: tuple[float, float, float, float]) -> FinnCO:
    """Read FINNv1.5 each-fire CO for one day and crop it to ``bbox``.

    FINN's gas columns are moles/day.  Values in the text archive use Fortran
    ``D`` exponents, which Python's ``float`` does not accept directly.
    """
    west, east, south, north = bbox
    latitude: list[float] = []
    longitude: list[float] = []
    co: list[float] = []
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"LATI", "LONGI", "CO"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError(f"{path} does not have FINN columns {sorted(required)}")
        for row in rows:
            lat = _fortran_float(row["LATI"])
            lon = _fortran_float(row["LONGI"])
            if west <= lon <= east and south <= lat <= north:
                latitude.append(lat)
                longitude.append(lon)
                co.append(_fortran_float(row["CO"]))

    return FinnCO(
        latitude=np.asarray(latitude, dtype=np.float64),
        longitude=np.asarray(longitude, dtype=np.float64),
        moles_per_day=np.asarray(co, dtype=np.float64),
    )


def grid_finn_co(
    fires: FinnCO,
    latitude: NDArray[np.floating],
    longitude: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Put each 1 km FINN fire into its nearest native CONUS404 cell.

    The result is kg CO/day per grid cell.  This is conservative: no source
    mass is created or discarded while mapping points to cells.
    """
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if lat.shape != lon.shape or lat.ndim != 2:
        raise ValueError(f"latitude/longitude must be matching 2-D fields, got {lat.shape}")

    gridded = np.zeros(lat.shape, dtype=np.float64)
    for fire_lat, fire_lon, mass in zip(
        fires.latitude, fires.longitude, fires.kilograms_per_day, strict=True
    ):
        northing = lat - fire_lat
        easting = (lon - fire_lon) * np.cos(np.deg2rad(fire_lat))
        cell = int(np.argmin(northing**2 + easting**2))
        gridded.flat[cell] += mass
    return gridded


def emission_tendency(
    kilograms_per_day: NDArray[np.floating],
    ds: NDArray[np.floating],
    cell_area: float,
) -> NDArray[np.float64]:
    """Surface CO source in coupled units per second, including rho*J slot."""
    daily = np.asarray(kilograms_per_day, dtype=np.float64)
    thickness = np.asarray(ds, dtype=np.float64)
    if daily.ndim != 2 or thickness.ndim != 1:
        raise ValueError("kilograms_per_day must be 2-D and ds must be 1-D")
    if cell_area <= 0.0 or np.any(thickness <= 0.0):
        raise ValueError("cell_area and all layer thicknesses must be positive")

    tendency = np.zeros((*daily.shape, thickness.size, 2), dtype=np.float64)
    tendency[:, :, 0, 0] = daily / (SECONDS_PER_DAY * cell_area * thickness[0])
    return tendency


def tracer_mass_kg(
    tracer: NDArray[np.floating], ds: NDArray[np.floating], cell_area: float
) -> float:
    """Integrate a coupled tracer field over the domain."""
    field = np.asarray(tracer, dtype=np.float64)
    thickness = np.asarray(ds, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != thickness.size:
        raise ValueError(f"tracer shape {field.shape} is incompatible with ds {thickness.shape}")
    return float(np.einsum("ijl,l->", field, thickness) * cell_area)


def negative_tracer_mass_kg(
    tracer: NDArray[np.floating], ds: NDArray[np.floating], cell_area: float
) -> float:
    """Magnitude of tracer mass below zero; exactly zero is the positivity target."""
    return max(0.0, -tracer_mass_kg(np.minimum(np.asarray(tracer), 0.0), ds, cell_area))


def vertical_centroid_m(
    tracer: NDArray[np.floating],
    ds: NDArray[np.floating],
    zface: NDArray[np.floating],
) -> float:
    """Tracer-mass-weighted height above mean sea level."""
    field = np.asarray(tracer, dtype=np.float64)
    faces = np.asarray(zface, dtype=np.float64)
    thickness = np.asarray(ds, dtype=np.float64)
    if faces.shape != (*field.shape[:2], field.shape[2] + 1):
        raise ValueError(f"zface shape {faces.shape} does not bracket tracer {field.shape}")
    weight = field * thickness[None, None, :]
    total = float(weight.sum())
    if total <= 0.0:
        return float("nan")
    midpoint = 0.5 * (faces[..., :-1] + faces[..., 1:])
    return float(np.sum(weight * midpoint) / total)


def coarsen_cells(field: NDArray[np.floating], factor: int, *, extensive: bool) -> NDArray:
    """Coarsen cell-centred leading x/y axes by block sum or mean."""
    values = np.asarray(field)
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    if factor == 1:
        return values.copy()
    nx = values.shape[0] // factor
    ny = values.shape[1] // factor
    if nx < 1 or ny < 1:
        raise ValueError(f"factor {factor} is larger than field {values.shape[:2]}")
    trimmed = values[: nx * factor, : ny * factor]
    blocked = trimmed.reshape(nx, factor, ny, factor, *values.shape[2:])
    return blocked.sum(axis=(1, 3)) if extensive else blocked.mean(axis=(1, 3))


def coarsen_u(uhat: NDArray[np.floating], factor: int) -> NDArray:
    """Coarsen x-face wind, retaining coincident coarse-grid x faces."""
    u = np.asarray(uhat)
    if u.ndim < 2 or factor < 1:
        raise ValueError(f"uhat must have at least two axes and factor must be positive: {u.shape}")
    nx = (u.shape[0] - 1) // factor
    ny = u.shape[1] // factor
    if nx < 1 or ny < 1:
        raise ValueError(f"factor {factor} is larger than uhat {u.shape[:2]}")
    sampled = u[: nx * factor + 1 : factor, : ny * factor]
    return sampled.reshape(nx + 1, ny, factor, *u.shape[2:]).mean(axis=2)


def coarsen_v(vhat: NDArray[np.floating], factor: int) -> NDArray:
    """Coarsen y-face wind, retaining coincident coarse-grid y faces."""
    v = np.asarray(vhat)
    if v.ndim < 2 or factor < 1:
        raise ValueError(f"vhat must have at least two axes and factor must be positive: {v.shape}")
    nx = v.shape[0] // factor
    ny = (v.shape[1] - 1) // factor
    if nx < 1 or ny < 1:
        raise ValueError(f"factor {factor} is larger than vhat {v.shape[:2]}")
    sampled = v[: nx * factor, : ny * factor + 1 : factor]
    return sampled.reshape(nx, factor, ny + 1, *v.shape[2:]).mean(axis=1)


def _fortran_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))

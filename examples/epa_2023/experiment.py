"""Testable I/O and diagnostics for the EPA projected-2023 CONUS case.

Arrays use CMAQ/JAX order, ``(column, row, layer)``, after the I/O boundary.
The EPA files use Models-3 order, ``(time, layer, row, column)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from cmaq_jax.hadv import BoundaryConditions
from cmaq_jax.io_mcip import ioapi_datetime

if TYPE_CHECKING:
    from netCDF4 import Dataset

CO_MOLAR_MASS = 28.01e-3  # kg mol-1
DRY_AIR_MOLAR_MASS = 28.9647e-3  # kg mol-1


@dataclass(frozen=True)
class GridSignature:
    """I/O API horizontal grid fields that must agree between inputs."""

    ncols: int
    nrows: int
    xorig: float
    yorig: float
    xcell: float
    ycell: float
    gdtyp: int
    p_alp: float
    p_bet: float
    p_gam: float
    xcent: float
    ycent: float


@dataclass(frozen=True)
class HourlyCO:
    """Hourly gridded CO emission rates on the 12US1 grid."""

    times: tuple[datetime, ...]
    kilograms_per_second: NDArray[np.float64]  # (time, column, row)
    grid: GridSignature

    def integrated_mass(self, seconds_per_interval: int = 3600) -> NDArray[np.float64]:
        """Cellwise mass over all intervals bracketed by the time records."""
        if len(self.times) < 2:
            raise ValueError("at least two hourly emission records are required")
        return self.kilograms_per_second[:-1].sum(axis=0) * seconds_per_interval


def grid_signature(dataset: Dataset) -> GridSignature:
    """Read the projection/grid fields shared by I/O API inputs."""

    def attr(name: str) -> Any:
        try:
            return dataset.getncattr(name)
        except AttributeError as exc:
            raise KeyError(f"I/O API file is missing global attribute {name}") from exc

    return GridSignature(
        ncols=int(attr("NCOLS")),
        nrows=int(attr("NROWS")),
        xorig=float(attr("XORIG")),
        yorig=float(attr("YORIG")),
        xcell=float(attr("XCELL")),
        ycell=float(attr("YCELL")),
        gdtyp=int(attr("GDTYP")),
        p_alp=float(attr("P_ALP")),
        p_bet=float(attr("P_BET")),
        p_gam=float(attr("P_GAM")),
        xcent=float(attr("XCENT")),
        ycent=float(attr("YCENT")),
    )


def _times(dataset: Dataset) -> tuple[datetime, ...]:
    if "TFLAG" not in dataset.variables:
        raise KeyError("I/O API file has no TFLAG variable")
    flags = np.asarray(dataset.variables["TFLAG"][:, 0, :], dtype=np.int64)
    return tuple(ioapi_datetime(int(jdate), int(jtime)) for jdate, jtime in flags)


def read_hourly_co(path: Path) -> HourlyCO:
    """Read the merged ``CO`` field and convert mol s-1 to kg s-1."""
    try:
        from netCDF4 import Dataset as NetCDFDataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("reading EPA emissions needs the package's 'io' extra") from exc

    with NetCDFDataset(path) as dataset:
        if "CO" not in dataset.variables:
            available = ", ".join(sorted(name for name in dataset.variables if name != "TFLAG"))
            raise KeyError(f"CO is not in {path.name}; it holds: {available}")
        variable = dataset.variables["CO"]
        dimensions = tuple(variable.dimensions)
        expected = ("TSTEP", "LAY", "ROW", "COL")
        if dimensions != expected:
            raise ValueError(f"CO dimensions are {dimensions}, expected {expected}")
        units = str(getattr(variable, "units", "")).strip().lower().replace(" ", "")
        if units not in {"moles/s", "mole/s", "mol/s", "molespersecond"}:
            raise ValueError(f"CO units are {getattr(variable, 'units', None)!r}, expected mol s-1")

        raw = np.ma.filled(variable[:], 0.0).astype(np.float64, copy=False)
        if raw.ndim != 4:
            raise ValueError(f"CO must be four-dimensional, got {raw.shape}")
        if not np.all(np.isfinite(raw)):
            raise ValueError("CO emission rates contain non-finite values")
        minimum = float(raw.min())
        if minimum < 0.0:
            raise ValueError(f"CO emission rates must be nonnegative, minimum is {minimum}")

        # Sum any gridded vertical layers, then ROW,COL -> COL,ROW.
        kilograms_per_second = raw.sum(axis=1).transpose(0, 2, 1) * CO_MOLAR_MASS
        times = _times(dataset)
        signature = grid_signature(dataset)

    if kilograms_per_second.shape != (len(times), signature.ncols, signature.nrows):
        raise ValueError(
            f"CO shape {kilograms_per_second.shape} does not match "
            f"{len(times)} times on {signature.ncols}x{signature.nrows}"
        )
    return HourlyCO(times, kilograms_per_second, signature)


def read_latitude_longitude(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read ``LAT``/``LON`` from GRIDCRO2D in model order."""
    try:
        from netCDF4 import Dataset as NetCDFDataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading EPA grid geometry needs the package's 'io' extra") from exc

    with NetCDFDataset(path) as dataset:
        fields: list[NDArray[np.float64]] = []
        for name in ("LAT", "LON"):
            if name not in dataset.variables:
                raise KeyError(f"{name} is not in {path.name}")
            field = np.asarray(dataset.variables[name][0, 0], dtype=np.float64).T
            fields.append(field)
        signature = grid_signature(dataset)

    expected = (signature.ncols, signature.nrows)
    if fields[0].shape != expected or fields[1].shape != expected:
        raise ValueError(f"LAT/LON shapes must both be {expected}, got {[v.shape for v in fields]}")
    return fields[0], fields[1]


def surface_emission_tendency(
    kilograms_per_second: NDArray[np.floating],
    ds: NDArray[np.floating],
    cell_area: float,
) -> NDArray[np.float64]:
    """Lowest-layer CO source in coupled transport units per second."""
    rates = np.asarray(kilograms_per_second, dtype=np.float64)
    thickness = np.asarray(ds, dtype=np.float64)
    if rates.ndim != 2 or thickness.ndim != 1:
        raise ValueError("kilograms_per_second must be 2-D and ds must be 1-D")
    if cell_area <= 0.0 or np.any(thickness <= 0.0):
        raise ValueError("cell_area and all layer thicknesses must be positive")

    tendency = np.zeros((*rates.shape, thickness.size, 2), dtype=np.float64)
    tendency[:, :, 0, 0] = rates / (cell_area * thickness[0])
    return tendency


def clean_boundary_conditions(rhoj: NDArray[np.floating]) -> BoundaryConditions:
    """Zero CO enhancement at inflow, with meteorological rho*J on every edge."""
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
    expected_v = (u.shape[0] - 1, u.shape[1] + 1, u.shape[2])
    if u.ndim != 3 or v.shape != expected_v:
        raise ValueError(f"incompatible u/v shapes: {u.shape} and {v.shape}")
    divergence = np.diff(u, axis=0) / dx1 + np.diff(v, axis=1) / dx2
    return np.maximum(divergence, 0.0).max(axis=(0, 1))


def tracer_mass_kg(
    tracer: NDArray[np.floating], ds: NDArray[np.floating], cell_area: float
) -> float:
    """Integrate one coupled tracer field over the domain."""
    field = np.asarray(tracer, dtype=np.float64)
    thickness = np.asarray(ds, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != thickness.size:
        raise ValueError(f"tracer shape {field.shape} is incompatible with ds {thickness.shape}")
    return float(np.einsum("ijl,l->", field, thickness) * cell_area)


def negative_tracer_mass_kg(
    tracer: NDArray[np.floating], ds: NDArray[np.floating], cell_area: float
) -> float:
    """Magnitude of tracer mass below zero."""
    return max(0.0, -tracer_mass_kg(np.minimum(np.asarray(tracer), 0.0), ds, cell_area))


def coupled_co_to_ppbv(
    coupled_co: NDArray[np.floating], rhoj: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Convert coupled CO mass to dry-air mole-fraction enhancement in ppbv."""
    co = np.asarray(coupled_co, dtype=np.float64)
    dry_air = np.asarray(rhoj, dtype=np.float64)
    if co.shape != dry_air.shape or np.any(dry_air <= 0.0):
        raise ValueError("coupled CO and positive rhoj must have matching shapes")
    return co / dry_air * (DRY_AIR_MOLAR_MASS / CO_MOLAR_MASS) * 1.0e9


def vertical_centroid_m(
    tracer: NDArray[np.floating],
    ds: NDArray[np.floating],
    layer_height: NDArray[np.floating],
) -> float:
    """Tracer-mass-weighted height MSL using MCIP ``ZH`` layer centres."""
    field = np.asarray(tracer, dtype=np.float64)
    thickness = np.asarray(ds, dtype=np.float64)
    height = np.asarray(layer_height, dtype=np.float64)
    if field.shape != height.shape or field.shape[-1] != thickness.size:
        raise ValueError(f"tracer, height and ds are incompatible: {field.shape}, {height.shape}")
    weight = field * thickness[None, None, :]
    total = float(weight.sum())
    return float(np.sum(weight * height) / total) if total > 0.0 else float("nan")

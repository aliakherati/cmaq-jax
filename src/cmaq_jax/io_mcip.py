"""Read MCIP meteorology in Models-3 I/O API format.

This is the bridge from a real CMAQ run's input files to the arrays
:func:`cmaq_jax.api.advect_step` expects. It replaces CMAQ's
``CENTRALIZED_IO_MODULE`` for the handful of fields advection needs:

===============  ==================  ==========================================
Variable         File                Used for
===============  ==================  ==========================================
``UWINDC``       ``MET_DOT_3D``      x face velocity, C-staggered (preferred)
``VWINDC``       ``MET_DOT_3D``      y face velocity, C-staggered (preferred)
``UHAT_JD``      ``MET_DOT_3D``      x contravariant velocity * Jacobian * rho
``VHAT_JD``      ``MET_DOT_3D``      y likewise -- the pre-MCIPv3.5 fallback
``DENSA_J``      ``MET_CRO_3D``      rho*J, the density advection is held to
``JACOBM``       ``MET_CRO_3D``      Jacobian, for coupling concentrations
``ZF``           ``MET_CRO_3D``      layer face heights, diagnostics only
``MSFX2``        ``GRID_CRO_2D``     map scale factor squared, for coupling
===============  ==================  ==========================================

Three things about the format matter enough to state plainly, because each is
a silent-wrong-answer trap rather than an error:

**Dimension order.** I/O API stores ``(TSTEP, LAY, ROW, COL)``. The model works
in ``(COL, ROW, LAY)``, so every field is transposed on the way in. Getting this
wrong on a square domain produces a transposed wind field that still runs.

**False dot points.** ``MET_DOT_3D`` is dimensioned ``(NCOLS+1, NROWS+1)``, but
C-staggered winds do not fill it. MCIP says so directly (``ctmproc.f90:878``):
the arrays "are all set to the dot-point dimensions to accommodate the false dot
points in the Arakawa-C staggered grid that are output in Models-3 I/O API DOT
files". ``UWINDC`` lives on west-east faces, so its meaningful extent is
``(NCOLS+1, NROWS)`` and the last **row** is false; ``VWINDC`` is
``(NCOLS, NROWS+1)`` and the last **column** is false. See
``init_ctm.f90:1330-1346``, where MCIP declares exactly these two shapes and
pads only for the M3IO writer. Keeping the false points would feed advection a
row of meaningless velocities along one edge.

**Time interpolation.** Meteorology is hourly; advection runs on a sync step of
minutes. Every field is linearly interpolated between the bracketing records,
which is what ``INTERP3``/``interpolate_var`` does inside CMAQ.

Only the *start*-of-step density is returned. CMAQ reads a second, end-of-step
``DENSA_J`` (``zadvppmwrf.F:313``) but uses it only through the ``FBLN`` blend
at line 372 -- and ``FBLN`` is hard-set to 1.0 at line 249, which zeroes that
term. :meth:`MetReader.density` will read any time you ask for, so re-enabling
the blend needs no change here.

``netCDF4`` is used directly rather than through ``xarray``. I/O API files carry
no CF time coordinate -- time lives in the ``TFLAG`` variable as packed
``YYYYDDD``/``HHMMSS`` integers -- so xarray's decoding has to be switched off
and nothing it offers beyond that is used here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cmaq_jax.config import DType, GridConfig, PPMConstants, sigma_layer_thickness
from cmaq_jax.velocity import face_velocity_from_flux

if TYPE_CHECKING:  # pragma: no cover - import cost, and netCDF4 is an extra
    from netCDF4 import Dataset

__all__ = [
    "MetFiles",
    "MetReader",
    "ioapi_datetime",
    "open_met",
    "read_sigma_faces",
]


def ioapi_datetime(jdate: int, jtime: int) -> datetime:
    """Convert an I/O API ``(YYYYDDD, HHMMSS)`` pair to a datetime.

    Both are packed decimal integers, not offsets: ``2018182`` is the 182nd day
    of 2018, and ``133000`` is 13:30:00.
    """
    year, day_of_year = divmod(int(jdate), 1000)
    if not 1 <= day_of_year <= 366:
        raise ValueError(f"jdate {jdate} has day-of-year {day_of_year}, outside 1..366")
    hours, rest = divmod(int(jtime), 10000)
    minutes, seconds = divmod(rest, 100)
    return datetime(year, 1, 1) + timedelta(
        days=day_of_year - 1, hours=hours, minutes=minutes, seconds=seconds
    )


def _global_attr(dataset: Dataset, name: str) -> Any:
    try:
        return dataset.getncattr(name)
    except AttributeError as exc:
        raise KeyError(
            f"{name!r} is missing from the file's global attributes; "
            f"this does not look like a Models-3 I/O API file"
        ) from exc


def read_sigma_faces(dataset: Dataset) -> NDArray[np.float64]:
    """Sigma-level faces from the ``VGLVLS`` header attribute.

    ``NLAYS + 1`` values running from 1.0 at the ground to 0.0 at the model top,
    which is CMAQ's ``X3FACE_GD``. Feed to
    :func:`cmaq_jax.config.sigma_layer_thickness` for the layer thicknesses.
    """
    faces = np.asarray(_global_attr(dataset, "VGLVLS"), dtype=np.float64)
    nlays = int(_global_attr(dataset, "NLAYS"))
    if faces.size != nlays + 1:
        raise ValueError(
            f"VGLVLS has {faces.size} entries but NLAYS is {nlays}; expected {nlays + 1}"
        )
    if np.any(np.diff(faces) >= 0.0):
        raise ValueError(
            "VGLVLS must decrease monotonically from the ground upward, got "
            f"{faces[:4]}...; a non-monotone column would give a negative layer thickness"
        )
    return faces


def _file_times(dataset: Dataset) -> list[datetime]:
    """Timestamps of every record, read from ``TFLAG``."""
    if "TFLAG" not in dataset.variables:
        raise KeyError("the file has no TFLAG variable, so its records cannot be dated")
    # TFLAG is (TSTEP, VAR, 2); every variable carries the same stamp, so
    # variable 0 speaks for the record.
    flags = np.asarray(dataset.variables["TFLAG"][:, 0, :], dtype=np.int64)
    return [ioapi_datetime(int(date), int(time)) for date, time in flags]


def _variable(dataset: Dataset, name: str, path: Path) -> Any:
    if name not in dataset.variables:
        available = ", ".join(sorted(v for v in dataset.variables if v != "TFLAG"))
        raise KeyError(f"{name!r} is not in {path.name}; it holds: {available}")
    return dataset.variables[name]


def _interpolate_in_time(
    dataset: Dataset, name: str, when: datetime, path: Path
) -> NDArray[np.float64]:
    """One record of ``name`` at ``when``, linearly interpolated, ``(LAY, ROW, COL)``.

    Requests outside the file's span are clamped to the nearest end rather than
    extrapolated. CMAQ does the same (``hcontvel.F:221-235``, the ``REVERT``
    branch): it warns and reuses the last available step, on the grounds that a
    linear extrapolation of meteorology is worse than a stale field.
    """
    variable = _variable(dataset, name, path)
    times = _file_times(dataset)
    if when <= times[0]:
        return np.asarray(variable[0], dtype=np.float64)
    if when >= times[-1]:
        return np.asarray(variable[-1], dtype=np.float64)

    after = next(i for i, stamp in enumerate(times) if stamp >= when)
    before = after - 1
    span = (times[after] - times[before]).total_seconds()
    weight = (when - times[before]).total_seconds() / span
    lower = np.asarray(variable[before], dtype=np.float64)
    upper = np.asarray(variable[after], dtype=np.float64)
    return lower + weight * (upper - lower)


def _to_model_order(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(LAY, ROW, COL)`` as stored, to ``(COL, ROW, LAY)`` as used."""
    if field.ndim != 3:
        raise ValueError(f"expected a 3-D (LAY, ROW, COL) field, got shape {field.shape}")
    return np.transpose(field, (2, 1, 0))


@dataclass(frozen=True)
class MetFiles:
    """Paths to the MCIP output a run needs.

    ``grid_cro_2d`` is optional: it carries ``MSFX2``, which advection itself
    never uses -- it belongs to the coupling step -- so a reader without it is
    still complete for transport.
    """

    met_cro_3d: Path
    met_dot_3d: Path
    grid_cro_2d: Path | None = None

    def __post_init__(self) -> None:
        for label in ("met_cro_3d", "met_dot_3d", "grid_cro_2d"):
            value = getattr(self, label)
            if value is None:
                continue
            path = Path(value)
            object.__setattr__(self, label, path)
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")


class MetReader:
    """Meteorology for one domain, at any time the files cover.

    Built by :func:`open_met`, which manages the underlying file handles. The
    reader holds them open: a sync step needs several fields at several times,
    and reopening per read would dominate the cost of a short step.
    """

    def __init__(self, files: MetFiles, cro: Dataset, dot: Dataset, grid: Dataset | None) -> None:
        self._files = files
        self._cro = cro
        self._dot = dot
        self._grid = grid

        self.ncols = int(_global_attr(cro, "NCOLS"))
        self.nrows = int(_global_attr(cro, "NROWS"))
        self.nlays = int(_global_attr(cro, "NLAYS"))
        self.dx1 = float(_global_attr(cro, "XCELL"))
        self.dx2 = float(_global_attr(cro, "YCELL"))
        self.sigma_faces = read_sigma_faces(cro)

        self._check_dot_grid()

    def _check_dot_grid(self) -> None:
        """The dot file must be one cell larger in each direction.

        Worth failing on rather than trusting, because a mismatched pair of
        files produces a wind field that is merely offset -- it still has a
        plausible shape and still runs.
        """
        dot_cols = int(_global_attr(self._dot, "NCOLS"))
        dot_rows = int(_global_attr(self._dot, "NROWS"))
        if (dot_cols, dot_rows) != (self.ncols + 1, self.nrows + 1):
            raise ValueError(
                f"{self._files.met_dot_3d.name} is {dot_cols}x{dot_rows}, but the cross file "
                f"is {self.ncols}x{self.nrows}, so the dot file should be "
                f"{self.ncols + 1}x{self.nrows + 1}; these are not the same domain"
            )

    @property
    def times(self) -> list[datetime]:
        """Record timestamps of the cross-point file."""
        return _file_times(self._cro)

    @property
    def ds(self) -> NDArray[np.float64]:
        """Sigma layer thicknesses, ``(nlays,)``."""
        return sigma_layer_thickness(self.sigma_faces)

    def grid_config(
        self, nspc_adv: int, *, dtype: DType = "float64", ppm: PPMConstants | None = None
    ) -> GridConfig:
        """A :class:`~cmaq_jax.config.GridConfig` matching these files.

        ``nspc_adv`` counts the rho*J slot, so a run transporting ``n`` species
        passes ``n + 1``.
        """
        return GridConfig(
            ncols=self.ncols,
            nrows=self.nrows,
            ds=self.ds,
            dx1=self.dx1,
            dx2=self.dx2,
            nspc_adv=nspc_adv,
            dtype=dtype,
            ppm=ppm if ppm is not None else PPMConstants(),
        )

    def cross(self, name: str, when: datetime) -> NDArray[np.float64]:
        """A cross-point field at ``when``, as ``(ncols, nrows, nlays)``."""
        raw = _interpolate_in_time(self._cro, name, when, self._files.met_cro_3d)
        return _to_model_order(raw)

    def density(self, when: datetime) -> NDArray[np.float64]:
        """``DENSA_J`` -- rho*J, the density advection is held to."""
        return self.cross("DENSA_J", when)

    def jacobian(self, when: datetime) -> NDArray[np.float64]:
        """``JACOBM`` -- the Jacobian at layer middles, for coupling."""
        return self.cross("JACOBM", when)

    def layer_face_height(self, when: datetime) -> NDArray[np.float64]:
        """``ZF`` -- layer face heights in metres."""
        return self.cross("ZF", when)

    @property
    def has_c_staggered_wind(self) -> bool:
        """Whether the dot file carries ``UWINDC``.

        The switch CMAQ makes once at startup (``hcontvel.F:168-173``,
        ``CSTAGUV``). True for anything MCIP v3.5 or later, i.e. Fall 2009
        onward; the fallback exists for older archives.
        """
        return "UWINDC" in self._dot.variables

    def face_velocities(self, when: datetime) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(uhat, vhat)`` at ``when``, ready for :class:`~cmaq_jax.api.Meteorology`.

        Shapes are ``(ncols+1, nrows, nlays)`` and ``(ncols, nrows+1, nlays)``,
        with the false dot points dropped -- see the module docstring.

        On the C-staggered path the winds are returned unchanged, which is what
        ``hcontvel.F`` does: it reads ``UWINDC`` and returns immediately, with
        no density division. Only the older ``UHAT_JD`` path divides by the
        face-interpolated density, and taking it for a modern file would divide
        out a density that was never multiplied in.
        """
        if self.has_c_staggered_wind:
            u = self._dot_field("UWINDC", when)[:, : self.nrows, :]
            v = self._dot_field("VWINDC", when)[: self.ncols, :, :]
            return u, v

        rhoj = jnp.asarray(self.density(when))
        u_jd = jnp.asarray(self._dot_field("UHAT_JD", when)[:, : self.nrows, :])
        v_jd = jnp.asarray(self._dot_field("VHAT_JD", when)[: self.ncols, :, :])
        u = np.asarray(face_velocity_from_flux(u_jd, rhoj, axis=0), dtype=np.float64)
        v = np.asarray(face_velocity_from_flux(v_jd, rhoj, axis=1), dtype=np.float64)
        return u, v

    def _dot_field(self, name: str, when: datetime) -> NDArray[np.float64]:
        """A dot-point field at ``when``, ``(ncols+1, nrows+1, nlays)``, uncropped."""
        raw = _interpolate_in_time(self._dot, name, when, self._files.met_dot_3d)
        return _to_model_order(raw)

    def map_scale_factor_squared(self) -> NDArray[np.float64]:
        """``MSFX2`` from the time-independent 2-D grid file, ``(ncols, nrows)``.

        Used by the coupling step, not by advection.
        """
        if self._grid is None:
            raise ValueError("MSFX2 needs grid_cro_2d, which was not given to MetFiles")
        variable = _variable(self._grid, "MSFX2", self._files.grid_cro_2d or Path("grid_cro_2d"))
        # Time-independent, but still written with a leading record dimension.
        field = np.asarray(variable[0], dtype=np.float64)
        return np.transpose(np.squeeze(field))


@contextmanager
def open_met(files: MetFiles) -> Iterator[MetReader]:
    """Open a set of MCIP files and yield a reader, closing them on exit.

    ``netCDF4`` comes from the ``io`` extra: ``pip install cmaq-jax[io]``.
    Nothing else in the package imports it, so a run on synthetic fields needs
    no netCDF stack at all.
    """
    try:
        from netCDF4 import Dataset as NetCDFDataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "reading MCIP meteorology needs netCDF4; install with: pip install 'cmaq-jax[io]'"
        ) from exc

    with ExitStack() as stack:
        cro = stack.enter_context(NetCDFDataset(str(files.met_cro_3d)))
        dot = stack.enter_context(NetCDFDataset(str(files.met_dot_3d)))
        grid = (
            stack.enter_context(NetCDFDataset(str(files.grid_cro_2d)))
            if files.grid_cro_2d is not None
            else None
        )
        yield MetReader(files, cro, dot, grid)

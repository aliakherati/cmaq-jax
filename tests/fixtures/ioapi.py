"""Write synthetic Models-3 I/O API files, for testing the MCIP reader.

No CMAQ benchmark meteorology is available here -- it is a separate download
(``DOCS/CMAQ_Data.md``) -- so :mod:`cmaq_jax.io_mcip` is exercised against files
built to the format instead. That tests everything about the reader except
whether real MCIP output matches the format, which is the one claim a synthetic
fixture cannot make and which the A3.5 notes record as still open.

The layout follows the I/O API convention MCIP writes:

* variables dimensioned ``(TSTEP, LAY, ROW, COL)``, ``float32``;
* a ``TFLAG(TSTEP, VAR, DATE-TIME)`` integer variable holding
  ``YYYYDDD``/``HHMMSS`` per record and per variable;
* the header carried in global attributes, including ``VGLVLS`` (the sigma
  faces) and ``VAR-LIST`` (the variable names, space-padded to 16 characters).

Fields are deterministic functions of position and time, so a test can predict
what a read should return rather than merely checking it is self-consistent.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["IOAPI_NAME_WIDTH", "MetFixture", "to_ioapi_date", "write_ioapi", "write_met_fixture"]

IOAPI_NAME_WIDTH = 16
"""Width of a variable name in ``VAR-LIST``. I/O API pads to 16 characters."""


def to_ioapi_date(when: datetime) -> tuple[int, int]:
    """A datetime as the I/O API ``(YYYYDDD, HHMMSS)`` pair."""
    jdate = when.year * 1000 + when.timetuple().tm_yday
    jtime = when.hour * 10000 + when.minute * 100 + when.second
    return jdate, jtime


def write_ioapi(
    path: Path,
    fields: dict[str, NDArray[np.floating]],
    *,
    times: list[datetime],
    ncols: int,
    nrows: int,
    nlays: int,
    sigma_faces: NDArray[np.floating],
    xcell: float = 12000.0,
    ycell: float = 12000.0,
) -> Path:
    """Write ``fields`` as one I/O API file.

    Each entry of ``fields`` is ``(ntimes, nlays, nrows, ncols)`` -- storage
    order, not model order -- so a test states the array exactly as the file
    holds it and the reader is responsible for the transpose.
    """
    from netCDF4 import Dataset  # noqa: PLC0415  (optional 'io' extra)

    ntimes = len(times)
    for name, array in fields.items():
        expected = (ntimes, nlays, nrows, ncols)
        if array.shape != expected:
            raise ValueError(f"{name} has shape {array.shape}, expected {expected}")

    with Dataset(path, "w", format="NETCDF3_CLASSIC") as nc:
        nc.createDimension("TSTEP", None)
        nc.createDimension("DATE-TIME", 2)
        nc.createDimension("LAY", nlays)
        nc.createDimension("VAR", len(fields))
        nc.createDimension("ROW", nrows)
        nc.createDimension("COL", ncols)

        tflag = nc.createVariable("TFLAG", "i4", ("TSTEP", "VAR", "DATE-TIME"))
        tflag.units = "<YYYYDDD,HHMMSS>"
        tflag.long_name = "TFLAG".ljust(IOAPI_NAME_WIDTH)
        stamps = np.array([to_ioapi_date(when) for when in times], dtype=np.int32)
        # Every variable carries the same stamp on a given record.
        tflag[:] = np.repeat(stamps[:, None, :], len(fields), axis=1)

        for name, array in fields.items():
            variable = nc.createVariable(name, "f4", ("TSTEP", "LAY", "ROW", "COL"))
            variable.units = "unknown"
            variable.long_name = name.ljust(IOAPI_NAME_WIDTH)
            variable.var_desc = name.ljust(80)
            variable[:] = array.astype(np.float32)

        first_date, first_time = to_ioapi_date(times[0])
        step = times[1] - times[0] if ntimes > 1 else timedelta(hours=1)
        hours, rest = divmod(int(step.total_seconds()), 3600)
        minutes, seconds = divmod(rest, 60)

        nc.IOAPI_VERSION = "N/A"
        nc.EXEC_ID = "cmaq-jax test fixture".ljust(80)
        nc.FTYPE = np.int32(1)
        nc.SDATE = np.int32(first_date)
        nc.STIME = np.int32(first_time)
        nc.TSTEP = np.int32(hours * 10000 + minutes * 100 + seconds)
        nc.NTHIK = np.int32(1)
        nc.NCOLS = np.int32(ncols)
        nc.NROWS = np.int32(nrows)
        nc.NLAYS = np.int32(nlays)
        nc.NVARS = np.int32(len(fields))
        nc.GDTYP = np.int32(2)
        nc.XCELL = np.float64(xcell)
        nc.YCELL = np.float64(ycell)
        nc.VGTYP = np.int32(7)
        nc.VGTOP = np.float32(5000.0)
        nc.VGLVLS = np.asarray(sigma_faces, dtype=np.float32)
        nc.GDNAM = "CMAQ_JAX_TEST".ljust(IOAPI_NAME_WIDTH)
        nc.UPNAM = "write_ioapi".ljust(IOAPI_NAME_WIDTH)
        setattr(nc, "VAR-LIST", "".join(name.ljust(IOAPI_NAME_WIDTH) for name in fields))
        nc.FILEDESC = "synthetic MCIP-like output".ljust(80)
        nc.HISTORY = ""

    return path


class MetFixture:
    """A matched cross/dot/grid file trio, plus the fields that went in.

    Holding the inputs alongside the paths is the point: a test asserts that
    what comes back out of the reader is what went in, transposed and cropped,
    which is exactly the claim worth making about a reader.
    """

    def __init__(
        self,
        directory: Path,
        *,
        ncols: int = 6,
        nrows: int = 5,
        nlays: int = 4,
        ntimes: int = 3,
        c_staggered: bool = True,
        start: datetime | None = None,
        step: timedelta = timedelta(hours=1),
    ) -> None:
        self.ncols, self.nrows, self.nlays = ncols, nrows, nlays
        self.start = start if start is not None else datetime(2018, 7, 1, 0, 0, 0)
        self.times = [self.start + i * step for i in range(ntimes)]
        self.c_staggered = c_staggered
        # Stretched to put the thin layers aloft, as a real column has.
        self.sigma_faces = np.linspace(1.0, 0.0, nlays + 1) ** 0.625

        rng = np.random.default_rng(20260827)
        cross_shape = (ntimes, nlays, nrows, ncols)
        dot_shape = (ntimes, nlays, nrows + 1, ncols + 1)

        # Time-varying so interpolation has something to interpolate: record i
        # is the base field scaled by (1 + i), making the expected value at any
        # time a closed form a test can state.
        base_density = 1.5 + 0.4 * rng.random(cross_shape[1:])
        self.density = np.stack([(1.0 + i) * base_density for i in range(ntimes)])
        self.jacobian = np.stack([(1.0 + i) * (200.0 + base_density) for i in range(ntimes)])
        self.layer_height = np.stack(
            [np.broadcast_to(np.arange(1, nlays + 1)[:, None, None] * 100.0, cross_shape[1:])]
            * ntimes
        ).astype(np.float64)

        base_u = rng.normal(0.0, 5.0, dot_shape[1:])
        base_v = rng.normal(0.0, 5.0, dot_shape[1:])
        self.u_dot = np.stack([(1.0 + i) * base_u for i in range(ntimes)])
        self.v_dot = np.stack([(1.0 + i) * base_v for i in range(ntimes)])

        wind_names = ("UWINDC", "VWINDC") if c_staggered else ("UHAT_JD", "VHAT_JD")
        self.met_cro_3d = write_ioapi(
            directory / "METCRO3D.nc",
            {
                "DENSA_J": self.density,
                "JACOBM": self.jacobian,
                "ZF": self.layer_height,
            },
            times=self.times,
            ncols=ncols,
            nrows=nrows,
            nlays=nlays,
            sigma_faces=self.sigma_faces,
        )
        self.met_dot_3d = write_ioapi(
            directory / "METDOT3D.nc",
            {wind_names[0]: self.u_dot, wind_names[1]: self.v_dot},
            times=self.times,
            ncols=ncols + 1,
            nrows=nrows + 1,
            nlays=nlays,
            sigma_faces=self.sigma_faces,
        )

        self.msfx2 = 1.0 + 0.01 * rng.random((1, 1, nrows, ncols))
        self.grid_cro_2d = write_ioapi(
            directory / "GRIDCRO2D.nc",
            {"MSFX2": self.msfx2},
            times=self.times[:1],
            ncols=ncols,
            nrows=nrows,
            nlays=1,
            sigma_faces=self.sigma_faces,
        )


def write_met_fixture(directory: Path, **kwargs: object) -> MetFixture:
    """Build a :class:`MetFixture` in ``directory``."""
    return MetFixture(directory, **kwargs)  # type: ignore[arg-type]

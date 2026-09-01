from pathlib import Path

import numpy as np
import pytest
from examples.epa_2023.experiment import (
    CO_MOLAR_MASS,
    read_hourly_co,
    surface_emission_tendency,
)

netCDF4 = pytest.importorskip("netCDF4")


def _write_emissions(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("TSTEP", 3)
        dataset.createDimension("LAY", 1)
        dataset.createDimension("ROW", 2)
        dataset.createDimension("COL", 3)
        dataset.createDimension("VAR", 1)
        dataset.createDimension("DATE-TIME", 2)
        for name, value in {
            "NCOLS": 3,
            "NROWS": 2,
            "XORIG": -2_556_000.0,
            "YORIG": -1_728_000.0,
            "XCELL": 12_000.0,
            "YCELL": 12_000.0,
            "GDTYP": 2,
            "P_ALP": 33.0,
            "P_BET": 45.0,
            "P_GAM": -97.0,
            "XCENT": -97.0,
            "YCENT": 40.0,
        }.items():
            dataset.setncattr(name, value)

        flags = dataset.createVariable("TFLAG", "i4", ("TSTEP", "VAR", "DATE-TIME"))
        flags[:, 0, 0] = 2016197
        flags[:, 0, 1] = [0, 10000, 20000]
        co = dataset.createVariable("CO", "f4", ("TSTEP", "LAY", "ROW", "COL"))
        co.units = "moles/s"
        co[:] = np.arange(18, dtype=np.float32).reshape(3, 1, 2, 3)


def test_read_hourly_co_transposes_and_converts_units(tmp_path: Path) -> None:
    path = tmp_path / "emissions.nc"
    _write_emissions(path)

    emissions = read_hourly_co(path)

    assert emissions.kilograms_per_second.shape == (3, 3, 2)
    assert emissions.times[1].hour == 1
    np.testing.assert_allclose(
        emissions.kilograms_per_second[0],
        np.arange(6, dtype=np.float64).reshape(2, 3).T * CO_MOLAR_MASS,
    )


def test_surface_tendency_integrates_to_input_rate() -> None:
    rates = np.arange(1, 7, dtype=np.float64).reshape(3, 2)
    ds = np.array([0.1, 0.3, 0.6])
    area = 144.0e6

    tendency = surface_emission_tendency(rates, ds, area)

    assert tendency.shape == (3, 2, 3, 2)
    assert np.all(tendency[..., -1] == 0.0)
    np.testing.assert_allclose(
        np.einsum("ijl,l->", tendency[..., 0], ds) * area,
        rates.sum(),
    )

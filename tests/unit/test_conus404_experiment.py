"""Unit checks for the real-field CONUS404 experiment's conversions."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from examples.conus404.experiment import (
    CO_MOLAR_MASS,
    DRY_AIR_MOLAR_MASS,
    GRAVITY,
    FinnCO,
    clean_boundary_conditions,
    coarsen_cells,
    coarsen_u,
    coarsen_v,
    coupled_co_to_ppbv,
    emission_tendency,
    grid_finn_co,
    horizontal_divergence,
    negative_tracer_mass_kg,
    read_finn_co,
    rhoj_from_dry_air_mass,
    tracer_mass_kg,
    vertical_centroid_m,
)


def test_mcip_density_reduction_and_clean_boundaries() -> None:
    dry_mass = np.array([[80_000.0, 81_000.0], [82_000.0, 83_000.0]])
    map_factor = np.array([[1.0, 1.1], [1.2, 1.3]])
    rhoj = rhoj_from_dry_air_mass(dry_mass, map_factor, nlays=3)
    expected = dry_mass / (GRAVITY * map_factor**2)
    np.testing.assert_allclose(rhoj, np.broadcast_to(expected[..., None], rhoj.shape))

    boundaries = clean_boundary_conditions(rhoj)
    assert boundaries.west.shape == (2, 3, 2)
    assert boundaries.south.shape == (2, 3, 2)
    np.testing.assert_array_equal(boundaries.west[..., 0], 0.0)
    np.testing.assert_array_equal(boundaries.west[..., 1], rhoj[0])
    np.testing.assert_array_equal(boundaries.east[..., 1], rhoj[-1])
    np.testing.assert_array_equal(boundaries.south[..., 1], rhoj[:, 0])
    np.testing.assert_array_equal(boundaries.north[..., 1], rhoj[:, -1])


def test_horizontal_divergence_reports_only_the_positive_maximum() -> None:
    u = np.zeros((4, 2, 2))
    v = np.zeros((3, 3, 2))
    u[:, :, 0] = np.arange(4)[:, None] * 2.0
    u[:, :, 1] = -np.arange(4)[:, None]
    v[:, :, 0] = np.arange(3)[None, :] * 3.0
    v[:, :, 1] = np.arange(3)[None, :] * 4.0
    result = horizontal_divergence(u, v, dx1=2.0, dx2=4.0)
    np.testing.assert_allclose(result, [1.75, 0.5])


def test_finn_reader_handles_fortran_exponents_and_gridding(tmp_path: Path) -> None:
    path = tmp_path / "finn.txt.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("DAY,TIME,LATI,LONGI,CO\n")
        stream.write("207,0,3.500D+01,-1.200D+02,1.000D+02\n")
        stream.write("207,0,3.600D+01,-1.190D+02,2.000D+02\n")
        stream.write("207,0,4.500D+01,-1.000D+02,9.000D+02\n")

    fires = read_finn_co(path, (-125.0, -115.0, 32.0, 42.0))
    np.testing.assert_array_equal(fires.moles_per_day, [100.0, 200.0])
    latitude = np.array([[35.0, 36.0], [35.0, 36.0]])
    longitude = np.array([[-120.0, -120.0], [-119.0, -119.0]])
    gridded = grid_finn_co(fires, latitude, longitude)
    np.testing.assert_allclose(gridded.sum(), 300.0 * CO_MOLAR_MASS)
    assert np.count_nonzero(gridded) == 2


def test_emission_tendency_integrates_back_to_daily_mass() -> None:
    daily = np.array([[10.0, 20.0], [30.0, 40.0]])
    ds = np.array([0.1, 0.2, 0.7])
    tendency = emission_tendency(daily, ds, cell_area=16.0e6)
    emitted = tendency[..., 0] * 86_400.0
    assert tracer_mass_kg(emitted, ds, 16.0e6) == daily.sum()
    np.testing.assert_array_equal(tendency[..., 1], 0.0)


def test_coupled_co_converts_to_dry_air_ppbv() -> None:
    rhoj = np.array([[8_000.0, 9_000.0]])
    expected_ppbv = np.array([[1.0, 250.0]])
    mass_mixing_ratio = expected_ppbv * 1.0e-9 * CO_MOLAR_MASS / DRY_AIR_MOLAR_MASS
    coupled_co = mass_mixing_ratio * rhoj
    np.testing.assert_allclose(coupled_co_to_ppbv(coupled_co, rhoj), expected_ppbv)


def test_mass_positivity_and_vertical_centroid_diagnostics() -> None:
    tracer = np.zeros((1, 1, 2))
    tracer[0, 0] = [-2.0, 3.0]
    ds = np.array([0.25, 0.75])
    assert tracer_mass_kg(tracer, ds, 4.0) == 7.0
    assert negative_tracer_mass_kg(tracer, ds, 4.0) == 2.0

    positive = np.array([[[1.0, 1.0]]])
    zface = np.array([[[0.0, 100.0, 300.0]]])
    expected = (0.25 * 50.0 + 0.75 * 200.0) / (0.25 + 0.75)
    assert vertical_centroid_m(positive, ds, zface) == expected


def test_coarsening_preserves_extensive_mass_and_face_staggering() -> None:
    cells = np.arange(24).reshape(4, 6)
    coarse_sum = coarsen_cells(cells, 2, extensive=True)
    coarse_mean = coarsen_cells(cells, 2, extensive=False)
    assert coarse_sum.sum() == cells.sum()
    np.testing.assert_allclose(coarse_mean * 4, coarse_sum)

    u = np.arange(5)[:, None, None] + 10.0 * np.arange(4)[None, :, None]
    coarse_u = coarsen_u(u, 2)
    assert coarse_u.shape == (3, 2, 1)
    np.testing.assert_allclose(coarse_u[..., 0], [[5.0, 25.0], [7.0, 27.0], [9.0, 29.0]])

    v = 10.0 * np.arange(4)[:, None, None] + np.arange(5)[None, :, None]
    coarse_v = coarsen_v(v, 2)
    assert coarse_v.shape == (2, 3, 1)
    np.testing.assert_allclose(coarse_v[..., 0], [[5.0, 7.0, 9.0], [25.0, 27.0, 29.0]])


def test_finn_mass_property_uses_co_molar_mass() -> None:
    fires = FinnCO(
        latitude=np.array([35.0]),
        longitude=np.array([-120.0]),
        moles_per_day=np.array([2.0]),
    )
    np.testing.assert_allclose(fires.kilograms_per_day, [2.0 * CO_MOLAR_MASS])

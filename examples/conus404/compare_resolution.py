#!/usr/bin/env python3
"""Compare final column CO mass from native and coarsened transport runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.fine) as fine_file, np.load(args.coarse) as coarse_file:
        fine = np.asarray(fine_file["column_mass_kg"], dtype=np.float64)
        coarse = np.asarray(coarse_file["column_mass_kg"], dtype=np.float64)
        fine_dx = float(fine_file["dx"])
        coarse_dx = float(coarse_file["dx"])

    factor = round(coarse_dx / fine_dx)
    if factor < 1 or not np.isclose(coarse_dx, factor * fine_dx):
        raise ValueError(f"coarse dx {coarse_dx} is not an integer multiple of fine dx {fine_dx}")
    nx, ny = coarse.shape
    trimmed = fine[: nx * factor, : ny * factor]
    aggregated = trimmed.reshape(nx, factor, ny, factor).sum(axis=(1, 3))

    reference_mass = float(aggregated.sum())
    coarse_mass = float(coarse.sum())
    normalized_l1 = float(np.abs(coarse - aggregated).sum() / reference_mass)

    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    def centre(field: np.ndarray) -> tuple[float, float]:
        total = float(field.sum())
        return float(np.sum(field * x) / total), float(np.sum(field * y) / total)

    fine_centre = centre(aggregated)
    coarse_centre = centre(coarse)
    displacement_km = (
        np.hypot(coarse_centre[0] - fine_centre[0], coarse_centre[1] - fine_centre[1])
        * coarse_dx
        / 1000.0
    )

    print(f"fine aggregated mass: {reference_mass:.6g} kg")
    print(f"coarse mass:          {coarse_mass:.6g} kg")
    print(f"coarse/fine mass:     {coarse_mass / reference_mass:.8f}")
    print(f"normalized L1 field difference: {normalized_l1:.6f}")
    print(f"centre-of-mass displacement:    {displacement_km:.3f} km")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

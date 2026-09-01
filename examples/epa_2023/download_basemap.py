#!/usr/bin/env python3
"""Download map assets used by the full-CONUS presentation GIF."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_BASEMAP = HERE / "data" / "world.topo.200407.3x5400x2700.jpg"
DEFAULT_BOUNDARIES = HERE / "data" / "north_america_boundaries.npz"
SOURCE = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74393/"
    "world.topo.200407.3x5400x2700.jpg"
)
SHA256 = "85a24885616097b5e06532b7d8e30fa5d749a782e70ea55971dd8539a0899206"
BOUNDARY_SOURCES = {
    "coast": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "v5.1.2/geojson/ne_50m_coastline.geojson"
    ),
    "countries": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "v5.1.2/geojson/ne_50m_admin_0_boundary_lines_land.geojson"
    ),
    "states": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "v5.1.2/geojson/ne_50m_admin_1_states_provinces_lines.geojson"
    ),
}
NORTH_AMERICA_BOX = (-170.0, -20.0, 5.0, 75.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_basemap(output: Path) -> None:
    """Download *output* atomically and verify the published asset bytes."""
    if output.exists():
        if _sha256(output) != SHA256:
            raise ValueError(f"existing basemap has the wrong SHA-256: {output}")
        print(f"already present: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    print(f"downloading {SOURCE}")
    with urllib.request.urlopen(SOURCE) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    actual = _sha256(partial)
    if actual != SHA256:
        raise ValueError(f"downloaded basemap SHA-256 {actual} != {SHA256}")
    partial.replace(output)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def _rings(geometry: dict) -> list:
    kind = geometry["type"]
    coordinates = geometry["coordinates"]
    if kind == "LineString":
        return [coordinates]
    if kind in {"MultiLineString", "Polygon"}:
        return coordinates
    if kind == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    return []


def download_boundaries(output: Path) -> None:
    """Download complete North American coast and administrative lines."""
    if output.exists():
        print(f"already present: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    west, east, south, north = NORTH_AMERICA_BOX
    saved: dict[str, np.ndarray] = {}
    for name, url in BOUNDARY_SOURCES.items():
        print(f"downloading {name}")
        with urllib.request.urlopen(url, timeout=180) as response:
            payload = json.load(response)
        count = 0
        for feature in payload["features"]:
            for ring in _rings(feature["geometry"]):
                line = np.asarray(ring, dtype=np.float64)[:, :2]
                inside = (
                    (line[:, 0] > west)
                    & (line[:, 0] < east)
                    & (line[:, 1] > south)
                    & (line[:, 1] < north)
                )
                if inside.any():
                    saved[f"{name}_{count}"] = np.where(inside[:, None], line, np.nan)
                    count += 1
    np.savez_compressed(output, **saved)
    print(f"wrote {output} ({output.stat().st_size / 1000:.0f} kB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basemap-output", type=Path, default=DEFAULT_BASEMAP)
    parser.add_argument("--boundaries-output", type=Path, default=DEFAULT_BOUNDARIES)
    args = parser.parse_args()
    download_basemap(args.basemap_output)
    download_boundaries(args.boundaries_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

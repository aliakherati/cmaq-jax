#!/usr/bin/env python3
"""Fetch coast and state lines used by the CONUS404 figures."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {
    "coast": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/ne_50m_coastline.geojson"
    ),
    "states": (
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
        "master/data/geojson/us-states.json"
    ),
}
BOX = (-134.0, -110.0, 26.0, 47.0)


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


def main() -> int:
    output = DATA / "boundaries.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"already present: {output}")
        return 0

    west, east, south, north = BOX
    saved: dict[str, np.ndarray] = {}
    for name, url in SOURCES.items():
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

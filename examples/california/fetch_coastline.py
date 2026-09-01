#!/usr/bin/env python3
"""Fetch a coastline and state outlines, cropped to the domain.

    python examples/california/fetch_coastline.py

Purely for legibility: without a coastline, a pollution plume over the Pacific
looks the same as one over the Sierra. Writes a small ``.npz`` of line segments
so the animation has no runtime network dependency.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {
    "coast": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson",
    "states": "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
}
BOX = (-134.0, -110.0, 26.0, 47.0)  # west, east, south, north


def rings(geometry: dict) -> list[list[list[float]]]:
    kind, coords = geometry["type"], geometry["coordinates"]
    if kind == "LineString":
        return [coords]
    if kind == "MultiLineString":
        return coords
    if kind == "Polygon":
        return coords
    if kind == "MultiPolygon":
        return [ring for polygon in coords for ring in polygon]
    return []


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "coastline.npz"
    if out.exists():
        print(f"{out.name}: already present")
        return 0

    west, east, south, north = BOX
    segments: dict[str, list[np.ndarray]] = {}
    for name, url in SOURCES.items():
        print(f"  {name}: fetching…", end="", flush=True)
        with urllib.request.urlopen(url, timeout=180) as response:
            payload = json.load(response)
        kept = []
        for feature in payload["features"]:
            for ring in rings(feature["geometry"]):
                line = np.asarray(ring, dtype=np.float64)[:, :2]
                inside = (
                    (line[:, 0] > west)
                    & (line[:, 0] < east)
                    & (line[:, 1] > south)
                    & (line[:, 1] < north)
                )
                if inside.any():
                    # Keep the whole ring but blank the far-away vertices, so
                    # lines entering the box are not cut off mid-stroke.
                    line = np.where(inside[:, None], line, np.nan)
                    kept.append(line)
        segments[name] = kept
        print(f" {len(kept)} lines")

    np.savez_compressed(
        out,
        **{
            f"{name}_{index}": line
            for name, lines in segments.items()
            for index, line in enumerate(lines)
        },
    )
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

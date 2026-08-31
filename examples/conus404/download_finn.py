#!/usr/bin/env python3
"""Download one day of public FINNv1.5 each-fire emissions.

Example:

    .venv/bin/python examples/conus404/download_finn.py --date 2018-07-26

FINNv1.5 is used here because NCAR still exposes its daily 1 km each-fire text
files without authentication.  The current FINNv2.5 page and units are linked
from the experiment README; this script never relabels the older inventory as
the newer release.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from datetime import date
from pathlib import Path

BASE = "https://www.acom.ucar.edu/acresp/MODELING/finn_emis_txt"
DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date(2018, 7, 26))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    day = args.date
    day_of_year = day.timetuple().tm_yday
    name = f"GLOB_MOZ4_{day.year}{day_of_year:03d}.txt.gz"
    url = f"{BASE}/FINNv1_{day.year}/{name}"
    output = args.output_dir / name
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"already present: {output} ({output.stat().st_size / 1e6:.1f} MB)")
        return 0

    temporary = output.with_suffix(output.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=600) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(output)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

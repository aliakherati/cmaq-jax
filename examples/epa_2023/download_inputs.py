#!/usr/bin/env python3
"""Download one EPA 2016v3 projected-2023 CONUS transport case.

EPA's ``2023gf`` case is an analytic 2023 emissions scenario evaluated with
2016 meteorology.  This downloader deliberately keeps both dates visible: the
default is projected-2023 emissions for the July 15, 2016 meteorological day.

The four public files total about 12.1 GB.  Existing complete files are never
downloaded again, and a failed transfer remains under a ``.part`` suffix.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import date
from pathlib import Path

BUCKET = "s3://2016v3platform/2016v3platform"
MCIP_PREFIX = f"{BUCKET}/MCIP/WRFv3.8_12US1_2016_35aL"
EMISSIONS_PREFIX = (
    f"{BUCKET}/2023gf_emissions/smoke_out/2023gf_16j/12US1/"
    "cmaq_cb6ae7/merged_withbeis_withrwc"
)
DEFAULT_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_MET_DATE = date(2016, 7, 15)


def input_objects(met_date: date) -> dict[str, str]:
    """Local names and public S3 objects for one meteorological day."""
    stamp = met_date.strftime("%y%m%d")
    yyyymmdd = met_date.strftime("%Y%m%d")
    return {
        f"METCRO3D.12US1.35L.{stamp}": f"{MCIP_PREFIX}/METCRO3D.12US1.35L.{stamp}",
        f"METDOT3D.12US1.35L.{stamp}": f"{MCIP_PREFIX}/METDOT3D.12US1.35L.{stamp}",
        f"GRIDCRO2D.12US1.35L.{stamp}": f"{MCIP_PREFIX}/GRIDCRO2D.12US1.35L.{stamp}",
        (
            f"emis_mole_all_{yyyymmdd}_12US1_withbeis_withrwc_2023gf_16j.ncf"
        ): (
            f"{EMISSIONS_PREFIX}/"
            f"emis_mole_all_{yyyymmdd}_12US1_withbeis_withrwc_2023gf_16j.ncf"
        ),
    }


def _download(source: str, target: Path) -> None:
    if target.exists():
        print(f"already present: {target} ({target.stat().st_size / 1e9:.2f} GB)")
        return
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading {source}")
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            "--no-sign-request",
            "--region",
            "us-east-1",
            source,
            str(partial),
        ],
        check=True,
    )
    partial.replace(target)
    print(f"wrote {target} ({target.stat().st_size / 1e9:.2f} GB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--met-date",
        type=date.fromisoformat,
        default=DEFAULT_MET_DATE,
        help="2016 meteorological/profile day used by EPA's projected-2023 case",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--only",
        choices=("all", "meteorology", "emissions"),
        default="all",
        help="download the entire case or only one input class",
    )
    args = parser.parse_args()

    if args.met_date.year != 2016:
        parser.error("EPA's 2023gf platform is indexed by its 2016 meteorological dates")
    if shutil.which("aws") is None:
        parser.error("the AWS CLI is required (the public bucket needs no credentials)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = input_objects(args.met_date)
    for name, source in objects.items():
        is_emissions = name.startswith("emis_")
        if args.only == "meteorology" and is_emissions:
            continue
        if args.only == "emissions" and not is_emissions:
            continue
        _download(source, args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

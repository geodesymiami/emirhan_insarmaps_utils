#!/usr/bin/env python3
"""
Subset a SARvey-style CSV (ogr2ogr AS_XY: columns X=lon, Y=lat, plus DYYYYMMDD) by AOI.

AOI format matches convert_bbox.py: lat_min:lat_max,lon_min:lon_max (S:N,W:E), or
POLYGON((lon lat,...)), or GoogleEarth-style points.

Default output (no -o): ``<input_stem>_subset.csv`` in the input directory, or
``<input_stem>_TAG.csv`` if a third positional TAG is given.

Negative-latitude AOI: use ``--`` before the bbox token if the shell would
swallow it, e.g. ``subset_csv.py in.csv -- -23.393:-23.097,-68.356:-68.175``

Run with ``--help`` for examples.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


def _input_to_bounds_local(input_str: str) -> tuple[float, float, float, float]:
    """Fallback when minsar is not on PYTHONPATH (same logic as convert_bbox._input_to_bounds)."""

    def _parse_bbox_string(s: str):
        s = s.strip()
        if "," not in s or ":" not in s:
            return None
        try:
            lat_part, lon_part = s.split(",", 1)
            lat_min, lat_max = map(float, lat_part.split(":"))
            lon_min, lon_max = map(float, lon_part.split(":"))
            return (min(lat_min, lat_max), max(lat_min, lat_max), min(lon_min, lon_max), max(lon_min, lon_max))
        except (ValueError, AttributeError):
            return None

    s = input_str.strip()
    if not s:
        raise ValueError("Empty AOI")

    if s.upper().startswith("POLYGON"):
        modified = s.removeprefix("POLYGON((").removesuffix("))")
        points = modified.split(",")
        longs, lats = [], []
        for point in points:
            parts = point.split()
            if len(parts) >= 2:
                longs.append(float(parts[0]))
                lats.append(float(parts[1]))
        if not longs or not lats:
            raise ValueError("POLYGON has no valid coordinates")
        return (min(lats), max(lats), min(longs), max(longs))

    bbox = _parse_bbox_string(s)
    if bbox is not None:
        return bbox

    try:
        points = s.split()
        longs, lats = [], []
        for point in points:
            parts = point.split(",")
            if len(parts) >= 2:
                longs.append(float(parts[0]))
                lats.append(float(parts[1]))
        if longs and lats:
            return (min(lats), max(lats), min(longs), max(longs))
    except (ValueError, AttributeError):
        pass

    raise ValueError(
        "Cannot parse AOI. Use POLYGON((...)), or lat_min:lat_max,lon_min:lon_max (S:N,W:E), or GoogleEarth points."
    )


def _get_input_to_bounds():
    try:
        from minsar.utils.convert_bbox import _input_to_bounds

        return _input_to_bounds
    except ImportError:
        return _input_to_bounds_local


def _looks_like_sn_we_bbox(s: str) -> bool:
    if not s or "," not in s or ":" not in s:
        return False
    t = s.strip()
    if t.upper().startswith("POLYGON"):
        return False
    return bool(re.match(r"^-?\d", t))


def _negative_sn_we_bbox_token(a: str) -> bool:
    if not _looks_like_sn_we_bbox(a):
        return False
    return len(a) > 1 and a[0] == "-" and (a[1].isdigit() or a[1] == ".")


def _fix_argv_negative_bbox(argv: list[str]) -> list[str]:
    """Insert -- before a S:N,W:E token whose latitude starts with '-' (convert_bbox-style)."""
    if "--" in argv:
        return argv
    for i, a in enumerate(argv):
        if _negative_sn_we_bbox_token(a):
            return argv[:i] + ["--"] + argv[i:]
    return argv


def subset_dataframe(
    df: pd.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    lon_col: str,
    lat_col: str,
) -> pd.DataFrame:
    lon = df[lon_col]
    lat = df[lat_col]
    mask = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
    return df.loc[mask].copy()


def _default_output_path(input_path: Path, tag: str | None) -> Path:
    """Same directory as input: <stem>_TAG.csv or <stem>_subset.csv."""
    stem = input_path.stem
    parent = input_path.parent
    if tag is not None:
        safe = re.sub(r'[\\/:*?"<>|]', "", tag).strip()
        if not safe:
            safe = "out"
        return parent / f"{stem}_{safe}.csv"
    return parent / f"{stem}_subset.csv"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _fix_argv_negative_bbox(argv)

    parser = argparse.ArgumentParser(
        description="Subset SARvey CSV by AOI (same formats as convert_bbox.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output file (if -o is not used):
  • With optional TAG:  <input_stem>_TAG.csv next to the input (e.g. sarvey.csv + TAG Porsche -> sarvey_Porsche.csv)
  • Without TAG:        <input_stem>_subset.csv (e.g. sarvey_subset.csv)

Examples:
  subset_csv.py sarvey.csv 0.645:0.778,-77.97:-77.77
      -> writes sarvey_subset.csv (same directory as sarvey.csv)

  subset_csv.py sarvey.csv 0.645:0.778,-77.97:-77.77 Porsche
      -> writes sarvey_Porsche.csv

  subset_csv.py sarvey.csv 0.645:0.778,-77.97:-77.77 -o /tmp/out.csv
      -> writes /tmp/out.csv

  subset_csv.py sarvey.csv -- -23.393:-23.097,-68.356:-68.175
      -> writes sarvey_subset.csv (negative latitude AOI; -- before AOI)

  subset_csv.py sarvey.csv "POLYGON((-77.97 0.645,-77.77 0.645,-77.77 0.778,-77.97 0.778,-77.97 0.645))"
      -> writes sarvey_subset.csv
""",
    )
    parser.add_argument("input_csv", help="Input SARvey CSV (X=lon, Y=lat, DYYYYMMDD columns)")
    parser.add_argument("aoi", help="AOI: lat_min:lat_max,lon_min:lon_max or POLYGON WKT or GoogleEarth points")
    parser.add_argument(
        "output_tag",
        nargs="?",
        default=None,
        metavar="TAG",
        help="Optional suffix for output filename: <input_stem>_TAG.csv (omit for <input_stem>_subset.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Explicit output path (overrides TAG / default naming)",
    )
    parser.add_argument("--lon-col", default="X", help="Longitude column name (default: X)")
    parser.add_argument("--lat-col", default="Y", help="Latitude column name (default: Y)")
    args = parser.parse_args(argv)

    try:
        min_lat, max_lat, min_lon, max_lon = _get_input_to_bounds()(args.aoi)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    path = Path(os.path.expandvars(os.path.expanduser(args.input_csv))).resolve()
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    for col in (args.lon_col, args.lat_col):
        if col not in df.columns:
            print(f"Error: column '{col}' not in CSV. Columns: {list(df.columns)}", file=sys.stderr)
            return 1

    out = subset_dataframe(df, min_lat, max_lat, min_lon, max_lon, args.lon_col, args.lat_col)
    n = len(out)
    print(f"AOI bounds: lat [{min_lat}, {max_lat}], lon [{min_lon}, {max_lon}] — kept {n} / {len(df)} rows", file=sys.stderr)

    if args.output:
        out_path = Path(os.path.expandvars(os.path.expanduser(args.output))).resolve()
    else:
        out_path = _default_output_path(path, args.output_tag)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    main()

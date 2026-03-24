#!/usr/bin/env python3
"""
Download daily 2 m temperature for a point (lat/lon) and date range.

Default provider: Open-Meteo Historical Weather API (archive-api.open-meteo.com).
Optional: Copernicus CDS ERA5 (--provider cds) requires ~/.cdsapirc and optional
packages cdsapi and netCDF4.

Displacement CSV convention (same as fit/remove thermal tools): **latitude is Y,
longitude is X** — use columns `lat`/`latitude` for Y and `lon`/`longitude` for X,
or Insarmaps-style `Y`/`X`.

Output: plain text, one line per day — ``YYYY-MM-DD  <temperature_C>`` (whitespace-separated, °C).
If ``--out`` is omitted, writes ``temp.txt`` in the current working directory.

Implementation: helper_thermal.py (shared with fit/remove thermal tools).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THERE = Path(__file__).resolve().parent
if str(_THERE) not in sys.path:
    sys.path.insert(0, str(_THERE))

from helper_thermal import (
    THERMAL_CLI_EXAMPLE_CSV,
    THERMAL_CLI_EXAMPLE_TEMP,
    date_range_from_disp_cols,
    download_cds_daily,
    download_open_meteo,
    lat_lon_from_first_row,
    load_displacement_csv,
    parse_iso_date,
    temperature_file_covers_range,
    unique_lat_lon,
    write_temperature_file,
)

EXAMPLES = f"""Thermal pipeline step 1: infer lat/lon (Y/X) and date range from D* columns.
Writes {THERMAL_CLI_EXAMPLE_TEMP} in the current directory unless you pass --out.
Next: fit_thermal_expansion.py --temperature {THERMAL_CLI_EXAMPLE_TEMP} (or remove_thermal_expansion.py for all-in-one).

  %(prog)s {THERMAL_CLI_EXAMPLE_CSV}

  %(prog)s --from-csv {THERMAL_CLI_EXAMPLE_CSV}

Without a displacement CSV (explicit point and dates; same default output name):

  %(prog)s --lalo -4.12367,145.02666 --start 2017-09-23 --end 2025-10-08
  %(prog)s --lat -4.12367 --lon 145.02666 --start 2017-09-23 --end 2025-10-08

CDS ERA5 instead of Open-Meteo (requires cdsapi, netCDF4, ~/.cdsapirc):

  %(prog)s --lalo -4.12,145.03 --start 2020-01-01 --end 2020-12-31 --provider cds
"""


def parse_lalo(value: str) -> tuple[float, float]:
    """Parse 'lat,lon' as Y,X (latitude, longitude), e.g. -4.12367,145.02666."""
    s = value.strip()
    if "," not in s:
        raise argparse.ArgumentTypeError(
            "--lalo expects LAT,LON with a comma (e.g. -4.12367,145.02666)"
        )
    parts = [p.strip() for p in s.split(",", 1)]
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        raise argparse.ArgumentTypeError(
            "--lalo expects exactly two numbers: latitude,longitude"
        )
    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--lalo values must be numeric: {value!r}"
        ) from e
    return lat, lon


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download daily mean 2 m temperature (°C) for a point. "
        "For CSV input: lat/latitude = Y (northing), lon/longitude = X (easting).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    p.add_argument(
        "csv",
        nargs="?",
        default=None,
        metavar="CSV",
        help="Displacement CSV: lat (Y) and lon (X) from first row for weather query; date range from D* columns "
        "(same as --from-csv)",
    )
    p.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Latitude in degrees (Y / northing); use with --lon and --start/--end",
    )
    p.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Longitude in degrees (X / easting); use with --lat and --start/--end",
    )
    p.add_argument(
        "--lalo",
        type=parse_lalo,
        default=None,
        metavar="LAT,LON",
        help="Latitude then longitude as Y,X, comma-separated (e.g. -4.12367,145.02666); "
        "use with --start and --end",
    )
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument(
        "--from-csv",
        type=str,
        default=None,
        metavar="CSV",
        help="Displacement CSV: lat=Y, lon=X from first row; infer date range from D* columns (alternative to positional CSV)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output text path (one line per day: YYYY-MM-DD  temperature_C). "
            f"Default: {THERMAL_CLI_EXAMPLE_TEMP} in the current directory."
        ),
    )
    p.add_argument(
        "--provider",
        choices=("open-meteo", "cds"),
        default="open-meteo",
        help="Temperature backend (default: open-meteo)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output exists and covers the range",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout seconds (open-meteo only)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.csv and args.from_csv:
        print(
            "Error: use either a positional CSV or --from-csv, not both.",
            file=sys.stderr,
        )
        return 1
    csv_path = args.csv or args.from_csv

    if csv_path:
        if args.lalo is not None or args.lat is not None or args.lon is not None:
            print(
                "Error: do not combine a displacement CSV with --lalo or --lat/--lon.",
                file=sys.stderr,
            )
            return 1
        if args.start is not None or args.end is not None:
            print(
                "Error: do not use --start/--end with a displacement CSV; dates come from D* columns.",
                file=sys.stderr,
            )
            return 1
        tab = load_displacement_csv(Path(csv_path))
        pts = unique_lat_lon(tab)
        if len(pts) > 1:
            print(
                f"Note: {len(pts)} distinct locations in CSV; using lat/lon from first row.",
                file=sys.stderr,
            )
        lat, lon = lat_lon_from_first_row(tab)
        start_s, end_s = date_range_from_disp_cols(tab.date_cols)
        start = parse_iso_date(start_s)
        end = parse_iso_date(end_s)
    elif args.lalo is not None:
        lat, lon = args.lalo
        if args.lat is not None or args.lon is not None:
            print("Error: use either --lalo or --lat and --lon, not both.", file=sys.stderr)
            return 1
        if not args.start or not args.end:
            print(
                "Error: --lalo requires --start and --end (YYYY-MM-DD).",
                file=sys.stderr,
            )
            return 1
        start, end = parse_iso_date(args.start), parse_iso_date(args.end)
    else:
        if args.lat is None or args.lon is None:
            print(
                "Error: provide --lalo LAT,LON, or --lat and --lon, or a displacement CSV.",
                file=sys.stderr,
            )
            return 1
        if not args.start or not args.end:
            print(
                "Error: --lat/--lon require --start and --end, or use a displacement CSV.",
                file=sys.stderr,
            )
            return 1
        lat, lon = float(args.lat), float(args.lon)
        start, end = parse_iso_date(args.start), parse_iso_date(args.end)

    if end < start:
        print("Error: end date before start date", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else Path(THERMAL_CLI_EXAMPLE_TEMP)

    if (
        out_path.is_file()
        and not args.force
        and temperature_file_covers_range(out_path, start, end)
    ):
        print(f"Using existing file (covers range): {out_path}")
        return 0

    if args.provider == "open-meteo":
        rows = download_open_meteo(lat, lon, start, end, timeout=args.timeout)
    else:
        rows = download_cds_daily(lat, lon, start, end)

    if not rows:
        print("Error: no temperature rows retrieved", file=sys.stderr)
        return 1

    write_temperature_file(out_path, rows)
    print(f"Wrote {len(rows)} days -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Remove thermal expansion contribution from displacement CSV (same fit/apply as fit_thermal_expansion).

Either supply --temperature, or let this script download daily temperature via the same
backends as download_temperature.py (Open-Meteo default, optional CDS).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

_THERE = Path(__file__).resolve().parent
if str(_THERE) not in sys.path:
    sys.path.insert(0, str(_THERE))

from helper_thermal import (
    THERMAL_CLI_EXAMPLE_CSV,
    THERMAL_CLI_EXAMPLE_TEMP,
    apply_thermal_correction,
    date_range_from_disp_cols,
    download_cds_daily,
    download_open_meteo,
    fit_thermal_coefficients,
    insert_column_after,
    lat_lon_from_first_row,
    load_displacement_csv,
    load_temperature_file,
    parse_iso_date,
    t_mean_for_overlap,
    unique_lat_lon,
    write_csv,
    write_temperature_file,
)

_EX_STEM = Path(THERMAL_CLI_EXAMPLE_CSV).stem
_EX_CORR = f"{_EX_STEM}_corrected.csv"
_EX_FIT_OUT = f"{_EX_STEM}_thermal_fit.csv"

HELP_DESCRIPTION = """\
Download temperature (optional), fit a thermal coefficient vs temperature anomaly, and write
a displacement CSV with corrected DYYYYMMDD values (mm).

Steps (in order):
  1. Load the input CSV (lat/lon as Y/X, dem, dem_error, D* displacement columns).
  2. Build daily temperature vs date: read --temperature if set; otherwise download
     (Open-Meteo or --provider cds) for the first row's lat/lon and the D* date span.
  3. Obtain mm/°C coefficients: joint regression of displacements vs T anomaly using
     h = dem_error - dem (and optional --with-intercept), or copy thermal_exp_coeff from
     the input with --use-existing-coeff.
  4. Apply correction: subtract coeff * (T(date) - mean_T from the fit) from each D* cell.
  5. Write output CSV (default <input_stem>_corrected.csv) with updated D* and thermal_exp_coeff.

Same logic as running download_temperature.py + fit_thermal_expansion.py separately, except
fit_thermal_expansion only writes *_thermal_fit.csv; this script always writes the corrected file.
"""

EXAMPLES = f"""All-in-one: downloads temperature (Open-Meteo by default), fits, writes thermally corrected D* columns.
Default output: <input_stem>_corrected.csv (e.g. {_EX_CORR} for {THERMAL_CLI_EXAMPLE_CSV}).
Step-by-step instead: download_temperature.py, then fit_thermal_expansion.py.

  %(prog)s {THERMAL_CLI_EXAMPLE_CSV}

With temperature file from download_temperature.py (e.g. temp.txt):

  %(prog)s {THERMAL_CLI_EXAMPLE_CSV} --temperature {THERMAL_CLI_EXAMPLE_TEMP}

Reuse coefficients already in the CSV (column thermal_exp_coeff), e.g. after fit_thermal_expansion.py:

  %(prog)s {_EX_FIT_OUT} --temperature {THERMAL_CLI_EXAMPLE_TEMP} --use-existing-coeff
"""


def _parse_date(s: str) -> date:
    return parse_iso_date(s)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=HELP_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    p.add_argument("input_csv", help="Displacement CSV with DYYYYMMDD columns (mm)")
    p.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output CSV path (optional; default: same directory as input, <stem>_corrected.csv)",
    )
    p.add_argument(
        "--temperature",
        default=None,
        metavar="FILE",
        help="Pre-downloaded temperature text file; if omitted, download for the CSV lat/lon and dates",
    )
    p.add_argument(
        "--provider",
        choices=("open-meteo", "cds"),
        default="open-meteo",
        help="When downloading temperature (default: open-meteo)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout when downloading temperature (open-meteo)",
    )
    p.add_argument(
        "--temp-out",
        default=None,
        metavar="FILE",
        help="Save downloaded temperature text file to this path (when not using --temperature)",
    )
    p.add_argument(
        "--use-existing-coeff",
        action="store_true",
        help="Use thermal_exp_coeff column from input CSV (skip regression)",
    )
    p.add_argument(
        "--dem-key",
        default="dem",
        help="DEM column name (default: dem)",
    )
    p.add_argument(
        "--dem-error-key",
        default="dem_error",
        help="DEM error column name (default: dem_error)",
    )
    p.add_argument(
        "--with-intercept",
        action="store_true",
        help="Include intercept when fitting (ignored with --use-existing-coeff)",
    )
    p.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original D* in DYYYYMMDD_raw columns",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inp = Path(args.input_csv)
    out = Path(args.out) if args.out else inp.with_name(inp.stem + "_corrected.csv")

    table = load_displacement_csv(inp)
    pts = unique_lat_lon(table)
    if len(pts) > 1:
        print(
            f"Note: {len(pts)} distinct locations in CSV; using lat/lon from first row for temperature download.",
            file=sys.stderr,
        )
    lat, lon = lat_lon_from_first_row(table)

    if args.temperature:
        t_path = Path(args.temperature)
        t_by_date = load_temperature_file(t_path)
    else:
        start_s, end_s = date_range_from_disp_cols(table.date_cols)
        start, end = _parse_date(start_s), _parse_date(end_s)
        tmp: Path | None = None
        if args.temp_out:
            t_path = Path(args.temp_out)
        else:
            fd, name = tempfile.mkstemp(suffix="_temp.txt")
            os.close(fd)
            tmp = Path(name)
            t_path = tmp
        try:
            if args.provider == "open-meteo":
                rows = download_open_meteo(
                    lat, lon, start, end, timeout=args.timeout
                )
            else:
                rows = download_cds_daily(lat, lon, start, end)
            if not rows:
                print("Error: no temperature data downloaded", file=sys.stderr)
                return 1
            write_temperature_file(t_path, rows)
            t_by_date = load_temperature_file(t_path)
        finally:
            if tmp and tmp.exists() and not args.temp_out:
                tmp.unlink(missing_ok=True)

    if args.use_existing_coeff:
        if "thermal_exp_coeff" not in table.fieldnames:
            print(
                "Error: --use-existing-coeff requires thermal_exp_coeff column",
                file=sys.stderr,
            )
            return 1
        coeff_arr = np.array(
            [
                float(str(row["thermal_exp_coeff"]).strip())
                for row in table.rows
            ],
            dtype=float,
        )
        try:
            t_mean = t_mean_for_overlap(table, t_by_date)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        try:
            fit = fit_thermal_coefficients(
                table,
                t_by_date,
                dem_key=args.dem_key,
                dem_err_key=args.dem_error_key,
                with_intercept=args.with_intercept,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        coeff_arr = fit.coeff_per_row
        t_mean = fit.t_mean

    fieldnames = list(table.fieldnames)
    if "thermal_exp_coeff" not in fieldnames:
        fieldnames = insert_column_after(
            fieldnames, "thermal_exp_coeff", args.dem_error_key
        )
    if args.keep_original:
        for dc in table.date_cols:
            raw = f"{dc}_raw"
            if raw not in fieldnames:
                fieldnames.append(raw)

    corrected = apply_thermal_correction(
        table, t_by_date, coeff_arr, t_mean
    )

    out_rows: list[dict[str, str]] = []
    for ri, row in enumerate(corrected):
        r = dict(row)
        r["thermal_exp_coeff"] = f"{float(coeff_arr[ri]):.8g}"
        if args.keep_original:
            orig = table.rows[ri]
            for dc in table.date_cols:
                r[f"{dc}_raw"] = orig.get(dc, "")
        out_rows.append(r)

    write_csv(out, fieldnames, out_rows)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

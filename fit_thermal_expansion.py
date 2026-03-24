#!/usr/bin/env python3
"""
Fit thermal expansion coefficient(s) and write displacement CSV with corrected D* columns.

Uses helper_thermal (this directory): joint model for multiple rows, scalar per row for
one row. Displacement values are millimeters (mm); temperature text file from download_temperature.py.
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
    apply_thermal_correction,
    fit_thermal_coefficients,
    insert_column_after,
    load_displacement_csv,
    load_temperature_file,
    write_csv,
)

EXAMPLES = f"""Thermal pipeline step 2: needs daily temperature file from download_temperature.py (e.g. temp.txt).
Writes <input_stem>_thermal_fit.csv next to the input unless you pass -o/--out.
Alternative: remove_thermal_expansion.py downloads temperature and writes *_corrected.csv in one run.

  %(prog)s {THERMAL_CLI_EXAMPLE_CSV} --temperature {THERMAL_CLI_EXAMPLE_TEMP}

  %(prog)s {THERMAL_CLI_EXAMPLE_CSV} --temperature {THERMAL_CLI_EXAMPLE_TEMP} --keep-original

  %(prog)s {THERMAL_CLI_EXAMPLE_CSV} --temperature {THERMAL_CLI_EXAMPLE_TEMP} --dem-key dem --dem-error-key dem_error
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit thermal expansion vs temperature anomaly and correct D* columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    p.add_argument("input_csv", help="Displacement CSV with DYYYYMMDD columns (mm)")
    p.add_argument(
        "--temperature",
        required=True,
        metavar="FILE",
        help="Temperature text file from download_temperature.py (YYYY-MM-DD  °C per line)",
    )
    p.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output CSV (default: input with _thermal_fit suffix)",
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
        help="Include intercept in regression (default: no intercept)",
    )
    p.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original D* values in DYYYYMMDD_raw columns",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inp = Path(args.input_csv)
    out = Path(args.out) if args.out else inp.with_name(inp.stem + "_thermal_fit.csv")

    table = load_displacement_csv(inp)
    t_by_date = load_temperature_file(Path(args.temperature))

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

    fieldnames = list(table.fieldnames)
    fieldnames = insert_column_after(
        fieldnames, "thermal_exp_coeff", args.dem_error_key
    )
    if "thermal_exp_coeff" not in fieldnames:
        fieldnames.append("thermal_exp_coeff")

    if args.keep_original:
        for dc in table.date_cols:
            raw = f"{dc}_raw"
            if raw not in fieldnames:
                fieldnames.append(raw)

    corrected = apply_thermal_correction(
        table,
        t_by_date,
        fit.coeff_per_row,
        fit.t_mean,
    )

    out_rows: list[dict[str, str]] = []
    for ri, row in enumerate(corrected):
        r = dict(row)
        r["thermal_exp_coeff"] = f"{float(fit.coeff_per_row[ri]):.8g}"
        if args.keep_original:
            orig = table.rows[ri]
            for dc in table.date_cols:
                r[f"{dc}_raw"] = orig.get(dc, "")
        out_rows.append(r)

    write_csv(out, fieldnames, out_rows)
    print(f"Wrote {out} ({len(out_rows)} rows, {len(fit.used_iso_dates)} dates used)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Plot daily temperature from the whitespace text format produced by download_temperature.py.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_THERE = Path(__file__).resolve().parent
if str(_THERE) not in sys.path:
    sys.path.insert(0, str(_THERE))

from helper_thermal import THERMAL_CLI_EXAMPLE_TEMP, load_temperature_file

EXAMPLES = f"""examples:
  # Default input {THERMAL_CLI_EXAMPLE_TEMP} in the current directory
  %(prog)s

  %(prog)s {THERMAL_CLI_EXAMPLE_TEMP}
  %(prog)s my_series.txt
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot daily 2 m temperature (°C) from a text file (YYYY-MM-DD  value per line).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    p.add_argument(
        "input_file",
        nargs="?",
        default=THERMAL_CLI_EXAMPLE_TEMP,
        help=f"Temperature text file (default: {THERMAL_CLI_EXAMPLE_TEMP})",
    )
    p.add_argument(
        "-o",
        "--out",
        metavar="PATH",
        help="Save figure to this path (PNG or other format matplotlib supports)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (use with --out)",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Plot title (default: derived from filename)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Error: matplotlib is required. Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return 1
    path = Path(args.input_file)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        t_by_date = load_temperature_file(path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    dates_sorted = sorted(t_by_date.keys())
    xs = [datetime.strptime(d, "%Y-%m-%d") for d in dates_sorted]
    ys = [t_by_date[d] for d in dates_sorted]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, color="steelblue", linewidth=0.8)
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    title = args.title if args.title is not None else f"Daily temperature — {path.name}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.no_show and not args.out:
        plt.close(fig)
        print(
            "Error: use --out when passing --no-show (no display requested).",
            file=sys.stderr,
        )
        return 1

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Wrote {args.out}")

    if not args.no_show:
        plt.show()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Library code for thermal expansion correction and daily temperature download.

Daily temperature series are read/written as plain text (e.g. ``temp.txt``): one line per day,
``YYYY-MM-DD`` (or ``YYYYMMDD``) and °C separated by whitespace.

Not a CLI — use download_temperature.py, fit_thermal_expansion.py, remove_thermal_expansion.py, plot_temperature.py.
"""

from __future__ import annotations

import calendar
import csv
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import requests

# Basenames for thermal CLI --help examples (not default paths).
THERMAL_CLI_EXAMPLE_CSV = (
    "TSX_036_20170923_20251008_N2598W08016_N2576W08016_N2576W08011_N2598W08011_subset.csv"
)
THERMAL_CLI_EXAMPLE_TEMP = "temp.txt"

# --- Displacement CSV / regression (formerly thermal_expansion.py) ---

# DYYYYMMDD
_DISP_COL_RE = re.compile(r"^D(\d{8})$")


@dataclass
class DisplacementTable:
    """Parsed displacement CSV."""

    fieldnames: list[str]
    rows: list[dict[str, str]]
    date_cols: list[str]
    lat_key: str
    lon_key: str


@dataclass
class ThermalFitResult:
    """Output of fit_thermal_coefficients."""

    coeff_per_row: np.ndarray
    beta: np.ndarray
    used_iso_dates: list[str]
    t_mean: float


def find_displacement_columns(fieldnames: Iterable[str]) -> list[str]:
    cols = [c for c in fieldnames if _DISP_COL_RE.match(str(c).strip())]
    return sorted(cols)


def parse_date_from_disp_col(name: str) -> str:
    """Return YYYYMMDD from DYYYYMMDD."""
    m = _DISP_COL_RE.match(name.strip())
    if not m:
        raise ValueError(f"Not a displacement column: {name}")
    return m.group(1)


def disp_col_to_iso(dcol: str) -> str:
    """D20180315 -> 2018-03-15."""
    ymd = parse_date_from_disp_col(dcol)
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def _field_by_alias(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    """Return the actual header string for the first alias present (case-insensitive)."""
    lower_to_orig: dict[str, str] = {}
    for f in fieldnames:
        k = str(f).strip().lower()
        if k not in lower_to_orig:
            lower_to_orig[k] = f
    for a in aliases:
        if a.lower() in lower_to_orig:
            return lower_to_orig[a.lower()]
    return None


def resolve_lat_lon_keys(fieldnames: list[str]) -> tuple[str, str]:
    """
    Latitude (Y) and longitude (X) column names.

    Tries, in order: latitude/lat (and longitude/lon), then Y_corr/Y and X_corr/X
    for Insarmaps-style exports where lon is X and lat is Y.
    """
    lat_key = _field_by_alias(
        fieldnames, ("latitude", "lat", "y_corr", "y")
    )
    lon_key = _field_by_alias(
        fieldnames, ("longitude", "lon", "x_corr", "x")
    )
    if lat_key and lon_key:
        return lat_key, lon_key
    raise ValueError(
        "CSV must have latitude/longitude columns: "
        "'lat'/'lon' or 'latitude'/'longitude', or Insarmaps-style 'Y'/'X' "
        "(lat=Y, lon=X) or 'Y_corr'/'X_corr'."
    )


def load_displacement_csv(path: Path) -> DisplacementTable:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Empty or headerless CSV: {path}")
        date_cols = find_displacement_columns(fieldnames)
        if not date_cols:
            raise ValueError(
                f"No displacement columns matching DYYYYMMDD in {path}"
            )
        lat_key, lon_key = resolve_lat_lon_keys(fieldnames)
        rows = [dict(r) for r in reader]
    return DisplacementTable(
        fieldnames=fieldnames,
        rows=rows,
        date_cols=date_cols,
        lat_key=lat_key,
        lon_key=lon_key,
    )


def load_temperature_file(path: Path) -> dict[str, float]:
    """Map ISO date YYYY-MM-DD (or YYYYMMDD) -> temperature (°C).

    Text format: one row per day, ``<date>  <temperature>`` (whitespace-separated).
    Blank lines and ``#`` comments are skipped.
    """
    path = Path(path)
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            ds, ts = parts[0], parts[1]
            try:
                iso = parse_iso_date(ds).isoformat()
                out[iso] = float(ts)
            except ValueError:
                continue
    if not out:
        raise ValueError(f"No temperature rows read from {path}")
    return out


def row_elevation(row: dict[str, str], dem_key: str, dem_err_key: str) -> float:
    if "elevation" in row and row.get("elevation", "").strip() != "":
        return float(row["elevation"])
    return float(row[dem_err_key]) - float(row[dem_key])


def _float_cell(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing numeric value for column {key!r}")
    return float(str(v).strip())


def fit_thermal_coefficients(
    table: DisplacementTable,
    t_by_date: dict[str, float],
    dem_key: str = "dem",
    dem_err_key: str = "dem_error",
    with_intercept: bool = False,
) -> ThermalFitResult:
    """
    coeff_per_row: shape (n_rows,); units mm/°C for the T' model.

    Multi-row: disp_{r,j} = (beta0 + beta1*h_r) * T'_j (no intercept), or
    with_intercept: y = alpha + beta0*T' + beta1*h*T'.

    Single row: disp = coeff * T' or disp = alpha + coeff * T'.
    """
    iso_dates = [disp_col_to_iso(c) for c in table.date_cols]
    col_indices = [
        i for i, d in enumerate(iso_dates) if d in t_by_date
    ]
    if len(col_indices) < 2:
        raise ValueError("Not enough overlapping dates for regression")

    t_vec = np.array(
        [t_by_date[iso_dates[i]] for i in col_indices], dtype=np.float64
    )
    t_mean = float(np.mean(t_vec))
    t_prime = t_vec - t_mean

    n_rows = len(table.rows)
    h = np.zeros(n_rows, dtype=np.float64)
    for i, row in enumerate(table.rows):
        h[i] = row_elevation(row, dem_key, dem_err_key)

    n_dates = len(col_indices)
    y = np.zeros(n_rows * n_dates, dtype=np.float64)
    for ri, row in enumerate(table.rows):
        for j, ci in enumerate(col_indices):
            dcol = table.date_cols[ci]
            idx = ri * n_dates + j
            y[idx] = _float_cell(row, dcol)

    used_iso = [iso_dates[i] for i in col_indices]

    if n_rows == 1:
        tp = t_prime
        dvec = y
        if with_intercept:
            X = np.column_stack([np.ones_like(tp), tp])
            beta, _, _, _ = np.linalg.lstsq(X, dvec, rcond=None)
            coeff = np.array([float(beta[1])])
        else:
            denom = float(np.dot(tp, tp))
            if denom <= 0:
                raise ValueError("Temperature anomaly is zero; cannot fit slope")
            coeff = np.array([float(np.dot(dvec, tp) / denom)])
            beta = coeff
        return ThermalFitResult(
            coeff_per_row=coeff,
            beta=beta,
            used_iso_dates=used_iso,
            t_mean=t_mean,
        )

    t_stack = np.tile(t_prime, n_rows)
    h_stack = np.repeat(h, n_dates) * t_stack
    if with_intercept:
        X = np.column_stack([np.ones(len(t_stack)), t_stack, h_stack])
    else:
        X = np.column_stack([t_stack, h_stack])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    beta = np.asarray(beta, dtype=np.float64)
    if with_intercept:
        coeff = beta[1] + beta[2] * h
    else:
        coeff = beta[0] + beta[1] * h
    return ThermalFitResult(
        coeff_per_row=coeff,
        beta=beta,
        used_iso_dates=used_iso,
        t_mean=t_mean,
    )


def apply_thermal_correction(
    table: DisplacementTable,
    t_by_date: dict[str, float],
    coeff_per_row: np.ndarray,
    t_mean: float,
) -> list[dict[str, str]]:
    """Subtract coeff_i * (T(date) - t_mean) from each D* cell (same t_mean as fit)."""
    out_rows: list[dict[str, str]] = []
    for ri, row in enumerate(table.rows):
        new_r = dict(row)
        c = float(coeff_per_row[ri])
        for dcol in table.date_cols:
            iso = disp_col_to_iso(dcol)
            if iso not in t_by_date:
                continue
            t_prime = t_by_date[iso] - t_mean
            raw = _float_cell(row, dcol)
            new_r[dcol] = f"{raw - c * t_prime:.8g}"
        out_rows.append(new_r)
    return out_rows


def unique_lat_lon(table: DisplacementTable) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for row in table.rows:
        lat = float(str(row[table.lat_key]).strip())
        lon = float(str(row[table.lon_key]).strip())
        key = (round(lat, 6), round(lon, 6))
        if key not in seen:
            seen.add(key)
            out.append((lat, lon))
    return out


def lat_lon_from_first_row(table: DisplacementTable) -> tuple[float, float]:
    """Latitude and longitude from the first data row (e.g. one weather query for multi-point CSVs)."""
    if not table.rows:
        raise ValueError("CSV has no data rows")
    row = table.rows[0]
    lat = float(str(row[table.lat_key]).strip())
    lon = float(str(row[table.lon_key]).strip())
    return lat, lon


def date_range_from_disp_cols(date_cols: Sequence[str]) -> tuple[str, str]:
    """Return (start_iso, end_iso) from DYYYYMMDD columns."""
    isos = sorted(disp_col_to_iso(c) for c in date_cols)
    return isos[0], isos[-1]


def t_mean_for_overlap(
    table: DisplacementTable, t_by_date: dict[str, float]
) -> float:
    """Mean temperature over displacement dates that exist in t_by_date (matches fit)."""
    isos = [
        disp_col_to_iso(c)
        for c in table.date_cols
        if disp_col_to_iso(c) in t_by_date
    ]
    if len(isos) < 2:
        raise ValueError(
            "Need at least 2 dates with temperature data overlapping displacement dates"
        )
    return float(np.mean([t_by_date[d] for d in isos]))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def insert_column_after(
    fieldnames: list[str], new_col: str, after: str | None
) -> list[str]:
    if new_col in fieldnames:
        return list(fieldnames)
    fn = list(fieldnames)
    if after and after in fn:
        i = fn.index(after) + 1
        fn.insert(i, new_col)
    else:
        fn.append(new_col)
    return fn


# --- Daily temperature download (Open-Meteo / CDS) ---

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def parse_iso_date(s: str) -> date:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date (use YYYY-MM-DD or YYYYMMDD): {s}")


def download_open_meteo(
    lat: float, lon: float, start: date, end: date, timeout: int = 60
) -> list[tuple[str, float]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    temps = daily.get("temperature_2m_mean") or []
    out: list[tuple[str, float]] = []
    for t, v in zip(times, temps):
        if v is None:
            continue
        out.append((str(t), float(v)))
    return out


def _cds_monthly_nc(
    lat: float,
    lon: float,
    y: int,
    m: int,
    out_nc: Path,
) -> None:
    try:
        import cdsapi
    except ImportError as e:
        raise RuntimeError(
            "CDS provider requires the cdsapi package (pip install cdsapi)"
        ) from e

    north, south = min(lat + 0.15, 90.0), max(lat - 0.15, -90.0)
    west, east = lon - 0.15, lon + 0.15
    if west > east:
        west, east = east, west
    last_day = calendar.monthrange(y, m)[1]
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": "2m_temperature",
            "year": str(y),
            "month": f"{m:02d}",
            "day": [f"{d:02d}" for d in range(1, last_day + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [north, west, south, east],
        },
        str(out_nc),
    )


def _nc_to_daily_means_celsius(nc_path: Path) -> dict[str, float]:
    from netCDF4 import Dataset, num2date

    ds = Dataset(str(nc_path), "r")
    try:
        tvar = ds.variables.get("t2m") or ds.variables.get("2t")
        if tvar is None:
            for k, v in ds.variables.items():
                if "temperature" in k.lower() or k == "t2m":
                    tvar = v
                    break
        if tvar is None:
            raise ValueError("No 2 m temperature variable in NetCDF")
        arr = tvar[:]
        units = getattr(tvar, "units", "K")
        if "K" in str(units).upper() or float(np.nanmean(arr)) > 100:
            arr = arr - 273.15
        time_var = ds.variables.get("time")
        if time_var is None:
            raise ValueError("No time coordinate in NetCDF")

        times = num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
        )
        if arr.ndim == 3:
            arr = np.mean(arr, axis=(1, 2))
        elif arr.ndim == 1:
            pass
        else:
            arr = np.mean(arr.reshape(arr.shape[0], -1), axis=1)
        daily: dict[str, list[float]] = {}
        for ti, val in zip(times, arr):
            d = ti.date() if hasattr(ti, "date") else ti
            if hasattr(d, "isoformat"):
                key = d.isoformat()
            else:
                key = str(d)[:10]
            daily.setdefault(key, []).append(float(val))
        return {k: sum(v) / len(v) for k, v in daily.items()}
    finally:
        ds.close()


def download_cds_daily(
    lat: float, lon: float, start: date, end: date
) -> list[tuple[str, float]]:
    merged: dict[str, float] = {}
    cur_y, cur_m = start.year, start.month
    end_key = (end.year, end.month)
    while (cur_y, cur_m) <= end_key:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tpath = Path(tmp.name)
        try:
            _cds_monthly_nc(lat, lon, cur_y, cur_m, tpath)
            month_daily = _nc_to_daily_means_celsius(tpath)
            merged.update(month_daily)
        finally:
            if tpath.exists():
                tpath.unlink(missing_ok=True)
        if cur_m == 12:
            cur_y += 1
            cur_m = 1
        else:
            cur_m += 1

    out: list[tuple[str, float]] = []
    d = start
    while d <= end:
        key = d.isoformat()
        if key in merged:
            out.append((key, merged[key]))
        d = date.fromordinal(d.toordinal() + 1)
    return out


def write_temperature_file(path: Path, rows: list[tuple[str, float]]) -> None:
    """Write daily temperatures as whitespace-separated ``YYYY-MM-DD  <°C>`` (aligned)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda x: x[0])
    str_temps = [f"{t:.6g}" for _, t in sorted_rows]
    w = max((len(s) for s in str_temps), default=1)
    with path.open("w", encoding="utf-8") as f:
        for (d, _), ts in zip(sorted_rows, str_temps):
            f.write(f"{d}  {ts:>{w}}\n")


def temperature_file_covers_range(path: Path, start: date, end: date) -> bool:
    if not path.is_file():
        return False
    dates: list[date] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                dates.append(parse_iso_date(parts[0]))
            except ValueError:
                continue
    if not dates:
        return False
    return min(dates) <= start and max(dates) >= end

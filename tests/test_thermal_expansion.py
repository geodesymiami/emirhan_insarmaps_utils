#!/usr/bin/env python3
"""Unit tests for helper_thermal (library) and download_temperature CLI wiring."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plot_temperature

from helper_thermal import (
    THERMAL_CLI_EXAMPLE_TEMP,
    DisplacementTable,
    apply_thermal_correction,
    disp_col_to_iso,
    download_open_meteo,
    fit_thermal_coefficients,
    lat_lon_from_first_row,
    load_temperature_file,
    parse_iso_date,
    resolve_lat_lon_keys,
    t_mean_for_overlap,
    write_temperature_file,
)


class TestPlotTemperatureCli(unittest.TestCase):
    def test_default_input_file_is_temp_txt(self):
        a = plot_temperature.parse_args([])
        self.assertEqual(a.input_file, THERMAL_CLI_EXAMPLE_TEMP)


class TestThermalExpansion(unittest.TestCase):
    def test_resolve_lat_lon_keys_yx(self):
        lat_k, lon_k = resolve_lat_lon_keys(["X", "Y", "D20180101"])
        self.assertEqual(lat_k, "Y")
        self.assertEqual(lon_k, "X")

    def test_resolve_lat_lon_keys_corr(self):
        lat_k, lon_k = resolve_lat_lon_keys(["X_corr", "Y_corr", "D20180101"])
        self.assertEqual(lat_k, "Y_corr")
        self.assertEqual(lon_k, "X_corr")

    def test_lat_lon_from_first_row(self):
        table = DisplacementTable(
            fieldnames=["X", "Y", "dem", "dem_error", "D20180101"],
            rows=[
                {"X": "145.0", "Y": "-4.0", "dem": "0", "dem_error": "0", "D20180101": "0"},
                {"X": "145.1", "Y": "-4.1", "dem": "0", "dem_error": "0", "D20180101": "1"},
            ],
            date_cols=["D20180101"],
            lat_key="Y",
            lon_key="X",
        )
        lat, lon = lat_lon_from_first_row(table)
        self.assertAlmostEqual(lat, -4.0)
        self.assertAlmostEqual(lon, 145.0)

    def test_disp_col_to_iso(self):
        self.assertEqual(disp_col_to_iso("D20180315"), "2018-03-15")

    def test_single_row_fit_no_intercept(self):
        # T' from mean([10, 20]) = 15 -> T' = [-5, 5]; disp = 2 * T'
        table = DisplacementTable(
            fieldnames=["lat", "lon", "dem", "dem_error", "D20180101", "D20180102"],
            rows=[
                {
                    "lat": "1",
                    "lon": "2",
                    "dem": "100",
                    "dem_error": "100",
                    "D20180101": "-10",
                    "D20180102": "10",
                }
            ],
            date_cols=["D20180101", "D20180102"],
            lat_key="lat",
            lon_key="lon",
        )
        t_by_date = {"2018-01-01": 10.0, "2018-01-02": 20.0}
        fit = fit_thermal_coefficients(table, t_by_date)
        self.assertEqual(len(fit.coeff_per_row), 1)
        self.assertAlmostEqual(fit.coeff_per_row[0], 2.0, places=5)

    def test_joint_model_two_rows(self):
        # beta0=0.5, beta1=0.01; h=[0, 100] -> coeff [0.5, 1.5]
        # T' = [2, -2] (mean of [0,4] is 2)
        table = DisplacementTable(
            fieldnames=[
                "lat",
                "lon",
                "dem",
                "dem_error",
                "D20180101",
                "D20180102",
            ],
            rows=[
                {
                    "lat": "1",
                    "lon": "2",
                    "dem": "0",
                    "dem_error": "0",
                    "D20180101": "-1.0",
                    "D20180102": "1.0",
                },
                {
                    "lat": "1",
                    "lon": "2",
                    "dem": "0",
                    "dem_error": "100",
                    "D20180101": "-3.0",
                    "D20180102": "3.0",
                },
            ],
            date_cols=["D20180101", "D20180102"],
            lat_key="lat",
            lon_key="lon",
        )
        t_by_date = {"2018-01-01": 0.0, "2018-01-02": 4.0}
        fit = fit_thermal_coefficients(table, t_by_date)
        self.assertAlmostEqual(fit.coeff_per_row[0], 0.5, places=5)
        self.assertAlmostEqual(fit.coeff_per_row[1], 1.5, places=5)

    def test_t_mean_for_overlap(self):
        table = DisplacementTable(
            fieldnames=["lat", "lon", "dem", "dem_error", "D20180101", "D20180102"],
            rows=[{"lat": "0", "lon": "0", "dem": "0", "dem_error": "0", "D20180101": "0", "D20180102": "0"}],
            date_cols=["D20180101", "D20180102"],
            lat_key="lat",
            lon_key="lon",
        )
        t_by_date = {"2018-01-01": 0.0, "2018-01-02": 4.0}
        self.assertAlmostEqual(t_mean_for_overlap(table, t_by_date), 2.0)

    def test_apply_correction_matches_fit(self):
        table = DisplacementTable(
            fieldnames=["lat", "lon", "dem", "dem_error", "D20180101", "D20180102"],
            rows=[
                {
                    "lat": "1",
                    "lon": "2",
                    "dem": "100",
                    "dem_error": "100",
                    "D20180101": "-10",
                    "D20180102": "10",
                }
            ],
            date_cols=["D20180101", "D20180102"],
            lat_key="lat",
            lon_key="lon",
        )
        t_by_date = {"2018-01-01": 10.0, "2018-01-02": 20.0}
        fit = fit_thermal_coefficients(table, t_by_date)
        corr = apply_thermal_correction(
            table, t_by_date, fit.coeff_per_row, fit.t_mean
        )
        self.assertAlmostEqual(float(corr[0]["D20180101"]), 0.0, places=5)
        self.assertAlmostEqual(float(corr[0]["D20180102"]), 0.0, places=5)

    def test_load_temperature_file(self):
        raw = "# header comment\n2018-01-01    10.5\n2018-01-02      11\n"
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(raw)
            name = f.name
        try:
            m = load_temperature_file(Path(name))
            self.assertEqual(m["2018-01-01"], 10.5)
            self.assertEqual(m["2018-01-02"], 11.0)
        finally:
            Path(name).unlink(missing_ok=True)

    def test_write_temperature_file_roundtrip(self):
        import tempfile

        rows = [("2018-01-02", 3.0), ("2018-01-01", 2.5)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "temp.txt"
            write_temperature_file(p, rows)
            m = load_temperature_file(p)
            self.assertEqual(m["2018-01-01"], 2.5)
            self.assertEqual(m["2018-01-02"], 3.0)

    def test_parse_iso_date(self):
        self.assertEqual(parse_iso_date("2018-01-01").isoformat(), "2018-01-01")
        self.assertEqual(parse_iso_date("20180101").isoformat(), "2018-01-01")

    @patch("helper_thermal.requests.get")
    def test_download_open_meteo_mock(self, mock_get):
        mock_get.return_value.json.return_value = {
            "daily": {
                "time": ["2018-01-01", "2018-01-02"],
                "temperature_2m_mean": [10.0, 11.0],
            }
        }
        mock_get.return_value.raise_for_status = lambda: None
        from datetime import date

        rows = download_open_meteo(0.0, 0.0, date(2018, 1, 1), date(2018, 1, 2))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], 10.0)


if __name__ == "__main__":
    unittest.main()

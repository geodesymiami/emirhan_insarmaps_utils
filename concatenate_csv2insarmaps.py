#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import re
import subprocess
import json
import shutil
import platform
from datetime import date

import sys

import pickle

try:
    import h5py
except ImportError:
    h5py = None


ssara = os.getenv("SSARAHOME")
if ssara:
    sys.path.insert(0, ssara)
else:
    print("[WARN] SSARAHOME is not set; password_config import may fail.")

import password_config as password



REQUIRED_COLS = {"X", "Y"}

def _corner_token(lat: float, lon: float) -> str:
    # 0.01° quantization like sarvey corner-string style
    lat_i = int(round(lat * 100.0))
    lon_i = int(round(abs(lon) * 100.0))
    return f"N{lat_i:04d}W{lon_i:05d}"

def _polygon_corners_string(min_lat, max_lat, min_lon, max_lon) -> str:
    # NW, SW, SE, NE ordering
    nw = _corner_token(max_lat, min_lon)
    sw = _corner_token(min_lat, min_lon)
    se = _corner_token(min_lat, max_lon)
    ne = _corner_token(max_lat, max_lon)
    return f"{nw}_{sw}_{se}_{ne}"

def extract_metadata_from_inputs(inputs_dir: Path):
    """Minimal slcStack.h5 metadata reader using h5py only."""
    attrs, errors = {}, []
    inputs_dir = Path(inputs_dir) if inputs_dir else None
    if not inputs_dir:
        return attrs, errors

    slc = inputs_dir / "slcStack.h5"
    if not slc.exists():
        errors.append(f"slcStack.h5 not found at: {slc}")
        return attrs, errors

    if h5py is None:
        errors.append("h5py not installed; cannot read slcStack.h5")
        return attrs, errors

    try:
        with h5py.File(slc, "r") as f:
            def _get(key, default=None):
                v = f.attrs.get(key, default)
                if isinstance(v, bytes):
                    return v.decode("utf-8", "ignore")
                return v

            mission = _get("mission") or _get("MISSION")
            platform = _get("platform") or _get("PLATFORM") or mission
            beam_mode = _get("beam_mode") or _get("BEAM_MODE")
            rel_orbit = _get("relative_orbit") or _get("RELATIVE_ORBIT") or _get("orbit") or _get("ORBIT")
            orbit_dir = _get("orbit_direction") or _get("ORBIT_DIRECTION") or _get("direction") or _get("DIRECTION")

            if rel_orbit is not None:
                try:
                    rel_orbit = int(rel_orbit)
                except Exception:
                    pass

            if mission is not None:
                attrs["mission"] = mission
                attrs["MISSION"] = mission
            if platform is not None:
                attrs["platform"] = platform
                attrs["PLATFORM"] = platform
            if beam_mode is not None:
                attrs["beam_mode"] = beam_mode
                attrs["BEAM_MODE"] = beam_mode
            if rel_orbit is not None:
                attrs["relative_orbit"] = rel_orbit
                attrs["RELATIVE_ORBIT"] = rel_orbit
            if orbit_dir is not None:
                attrs["orbit_direction"] = str(orbit_dir)
                attrs["ORBIT_DIRECTION"] = str(orbit_dir)

    except Exception as e:
        errors.append(f"Failed reading {slc}: {e}")

    return attrs, errors

def merge_into_metadata_pickle(json_dir: Path, attrs: dict):
    """Patch JSON/metadata.pickle before upload (mission/platform/orbit/direction)."""
    pkl = Path(json_dir) / "metadata.pickle"
    if not pkl.exists():
        print(f"[WARN] metadata.pickle not found in: {json_dir}")
        return

    with open(pkl, "rb") as f:
        meta = pickle.load(f)

    # overwrite provided fields
    for k in ["mission","platform","beam_mode","relative_orbit","orbit_direction","direction"]:
        if k in attrs and attrs[k] is not None:
            meta[k] = attrs[k]
    for k in ["MISSION","PLATFORM","BEAM_MODE","RELATIVE_ORBIT","ORBIT_DIRECTION","DIRECTION"]:
        if k in attrs and attrs[k] is not None:
            meta[k] = attrs[k]

    # derive A/D keys
    od = meta.get("orbit_direction") or meta.get("ORBIT_DIRECTION") or meta.get("direction") or meta.get("DIRECTION")
    if od:
        od_u = str(od).upper()
        if "DESC" in od_u:
            meta["direction_long"] = "DESCENDING"
            meta["direction_short"] = "D"
        elif "ASC" in od_u:
            meta["direction_long"] = "ASCENDING"
            meta["direction_short"] = "A"

    with open(pkl, "wb") as f:
        pickle.dump(meta, f)

def generate_dataset_name_from_csv(csv_path: Path, platform: str, orbit: str) -> str:
    """Sarvey-like dataset name: PLATFORM_ORBIT_START_END_<corner-string>."""
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    date_cols = sorted([c for c in cols if re.match(r"^D20\d{6}$", c)])
    if not date_cols:
        raise ValueError(f"No D20YYYYMMDD columns found in: {csv_path}")

    start_date = date_cols[0][1:]
    end_date = date_cols[-1][1:]

    xy = pd.read_csv(csv_path, usecols=["X", "Y"])
    min_lon, max_lon = float(xy["X"].min()), float(xy["X"].max())
    min_lat, max_lat = float(xy["Y"].min()), float(xy["Y"].max())

    corners = _polygon_corners_string(min_lat, max_lat, min_lon, max_lon)
    return f"{platform}_{orbit}_{start_date}_{end_date}_{corners}"


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV files and ingest into Insarmaps.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing CSVs to concatenate.")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Directory to write outputs.")
    parser.add_argument("--drop-duplicates", action="store_true", help="Drop duplicates based on (X, Y).")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix for final CSV filename (default: input folder name).")
    parser.add_argument("--insarmaps-host", type=str, default=os.environ.get("INSARMAPS_HOST", "insarmaps.miami.edu"), help="Insarmaps host")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Insarmaps upload step")
    parser.add_argument("--inputs-dir", type=str, default=None, help="Optional SARvey inputs/ directory containing slcStack.h5, geometryRadar.h5 for metadata.")

    return parser.parse_args()

def load_csvs(csv_files):
    #dataframes=dfs
    dfs = []
    for f in tqdm(csv_files, desc="Reading CSVs"):
        df = pd.read_csv(f)
        if not REQUIRED_COLS.issubset(df.columns):
            raise ValueError(f"Missing required columns in {f.name}")
        dfs.append(df)
    return dfs

def get_shared_date_columns(dfs):
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)
    return sorted([col for col in common_cols if re.match(r"^D20\d{6}$", col)])

def clean_and_concatenate(dfs, date_cols, drop_duplicates):
    core_cols = ["X", "Y"] + [c for c in dfs[0].columns if not c.startswith("D20") and c not in ["X", "Y"]]
    dfs = [df[core_cols + date_cols] for df in dfs]
    df = pd.concat(dfs, ignore_index=True)

    if drop_duplicates:
        # make duplicate resolution deterministic if point_id exists
        if "point_id" in df.columns:
            df = df.sort_values(by=["X", "Y", "point_id"], kind="mergesort")

        before = len(df)
        df.drop_duplicates(subset=["X", "Y"], inplace=True)
        print(f"[INFO] Dropped {before - len(df)} duplicate points based on (X, Y).")

    valid_obs = df[date_cols].notna().sum(axis=1)
    df = df[valid_obs >= 2].copy()
    print(f"[INFO] Kept {len(df)} rows with ≥2 valid observations.")

    valid_obs = df[date_cols].notna().sum(axis=1)
    df = df[valid_obs >= 5].copy()
    print(f"[INFO] Kept {len(df)} rows with ≥5 time series entries.")

    flat_std = df[date_cols].std(axis=1)
    df = df[~(flat_std.isna() | (flat_std == 0))].copy()
    print(f"[INFO] Removed rows with flat or constant time series.")

    return df

def extract_platform_orbit_from_filenames(csv_files):
    """
    Extract the platform and orbit from a list of filenames.
    Assuming filenames follow the convention: <platform>_<orbit>_YYYYMMDD_...
    """
    platforms = set()
    orbits = set()

    for csv_file in csv_files:
        match = re.match(r"([A-Z]+)_(\d{3})_", Path(csv_file).name)
        if match:
            platform, orbit = match.groups()
            platforms.add(platform)
            orbits.add(orbit)
        else:
            raise ValueError(f"Filename format not recognized: {csv_file}")

    if len(platforms) > 1 or len(orbits) > 1:
        raise ValueError(f"Inconsistent platform or orbit across files:\n  Platforms: {platforms}\n  Orbits: {orbits}")

    return platforms.pop(), orbits.pop()

def generate_filename_from_csv(df, date_cols, csv_files, suffix="data"):
    time_cols = sorted([col[1:] for col in date_cols])
    start_date = time_cols[0]
    end_date = time_cols[-1]

    min_lat, max_lat = df["Y"].min(), df["Y"].max()
    min_lon, max_lon = df["X"].min(), df["X"].max()

    lat1 = f"N{int(min_lat * 10000):05d}"
    lat2 = f"N{int(max_lat * 10000):05d}"
    lon1 = f"W{abs(int(max_lon * 10000)):06d}"
    lon2 = f"W{abs(int(min_lon * 10000)):06d}"

    platform, orbit = extract_platform_orbit_from_filenames(csv_files)
    return f"{platform}_{orbit}_{start_date}_{end_date}_{lat1}_{lat2}_{lon1}_{lon2}_{suffix}.csv"

def run_command(cmd, cwd=None):
    print("[Running]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def ingest_to_insarmaps(csv_path, output_dir, insarmaps_host, skip_upload, inputs_dir=None):
    json_dir = Path(output_dir) / "JSON"
    json_dir.mkdir(parents=True, exist_ok=True)
    mbtiles_path = json_dir / csv_path.with_suffix(".mbtiles").name

    run_command(["hdfeos5_or_csv_2json_mbtiles.py", str(csv_path), str(json_dir)])

    metadata = {}
    if inputs_dir:
        metadata, errs = extract_metadata_from_inputs(inputs_dir)
        for e in errs:
            print(f"[WARN] {e}")


    #Force mission/direction keys into JSON/metadata.pickle (Insarmaps title needs)
    merge_into_metadata_pickle(json_dir, metadata)

    if not skip_upload:
        run_command([
            "json_mbtiles2insarmaps.py",
            "--num-workers", "3",
            "-u", password.docker_insaruser,
            "-p", password.docker_insarpass,
            "--host", insarmaps_host,
            "-P", password.docker_databasepass,
            "-U", password.docker_databaseuser,
            "--json_folder", str(json_dir),
            "--mbtiles_file", str(mbtiles_path)
        ])

    #print Insarmaps URL
    #df = pd.read_csv(csv_path)
    #lat = df["Y"].mean()
    #lon = df["X"].mean()
    #dataset_name = csv_path.stem
    #print(f"[INFO] View in Insarmaps: http://{insarmaps_host}/start/{lat:.4f}/{lon:.4f}/11.0?startDataset={dataset_name}")
    # Load metadata (optional, for mission/orbit/direction normalization)

    #dataset_name = csv_path.stem
    #url = generate_insarmaps_url(insarmaps_host, dataset_name, metadata, geocorr=False)
    #print(f"[INFO] View in Insarmaps: {url}")
    df0 = pd.read_csv(csv_path, usecols=["X","Y"])
    lat = df0["Y"].mean()
    lon = df0["X"].mean()
    dataset_name = csv_path.stem
    print(f"[INFO] View in Insarmaps: http://{insarmaps_host}/start/{lat:.4f}/{lon:.4f}/11.0?startDataset={dataset_name}")



def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print("[WARN] No CSV files found.")
        return

    dfs = load_csvs(csv_files)
    date_cols = get_shared_date_columns(dfs)
    print(f"[INFO] Using {len(date_cols)} shared time series columns.")

    df = clean_and_concatenate(dfs, date_cols, args.drop_duplicates)

    #suffix = args.suffix or input_dir.name
    #output_filename = generate_filename_from_csv(df, date_cols, csv_files, suffix)
    #final_path = output_dir / output_filename

    #df.to_csv(final_path, index=False)
    #print(f"[INFO] Final CSV saved to: {final_path}")

    #ingest_to_insarmaps(final_path, output_dir, args.insarmaps_host, args.skip_upload)

    suffix = args.suffix or input_dir.name

    #write a temporary CSV first
    tmp_csv = output_dir / f"{input_dir.name}_concat_tmp.csv"
    df.to_csv(tmp_csv, index=False)

    #compute final dataset name from the CSV (dates + corners)
    inputs_dir = Path(args.inputs_dir).resolve() if args.inputs_dir else None
    
    platform0, orbit0 = extract_platform_orbit_from_filenames(csv_files)

    # If inputs_dir is provided and readable, prefer its mission/orbit
    if inputs_dir:
        md, errs = extract_metadata_from_inputs(inputs_dir)
        for e in errs:
            print(f"[WARN] {e}")

        if md.get("mission"):
            platform0 = str(md["mission"]).upper()
        if md.get("relative_orbit") is not None:
            try:
                orbit0 = f"{int(md['relative_orbit']):03d}"
            except Exception:
                pass

    dataset_name = generate_dataset_name_from_csv(tmp_csv, platform=platform0, orbit=orbit0)


    #append suffix (optional)
    dataset_stem = f"{dataset_name}_{suffix}" if suffix else dataset_name
    final_path = output_dir / f"{dataset_stem}.csv"

    shutil.move(str(tmp_csv), str(final_path))
    print(f"[INFO] Final CSV saved to: {final_path}")

    #ingest (it will also patch metadata + print URL)
    ingest_to_insarmaps(final_path, output_dir, args.insarmaps_host, args.skip_upload, inputs_dir=inputs_dir)


if __name__ == "__main__":
    main()


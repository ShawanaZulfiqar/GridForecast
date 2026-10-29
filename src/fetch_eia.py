"""
Pull hourly electricity demand (D) and the grid operator's day-ahead
demand forecast (DF) from the EIA API, one month at a time.

Saves one Parquet file per region per month:
    data/raw/eia/<REGION>/<YYYY-MM>.parquet

Usage (run from the project folder):
    python src/fetch_eia.py                      # all regions, full history
    python src/fetch_eia.py --start 2024-01      # quick test: 2024 onward
    python src/fetch_eia.py --region MISO
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "regions.yaml"
RAW_DIR = ROOT / "data" / "raw" / "eia"
URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
PAGE_SIZE = 5000          # max rows EIA returns per request
REFRESH_MONTHS = 2        # always re-pull the latest months (EIA revises recent data)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_with_retry(params, tries=4):
    """Call the EIA API, retrying with a growing wait if it fails."""
    for attempt in range(tries):
        try:
            r = requests.get(URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"    HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"    Request error: {e}")
        time.sleep(2 ** attempt)
    raise RuntimeError("EIA request failed after several tries")


def fetch_month(api_key, eia_code, month_start):
    """Return all D and DF rows for one region and one month (UTC)."""
    last_hour = month_start + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1)
    rows, offset = [], 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",             # 'hourly' gives UTC timestamps
            "data[0]": "value",
            "facets[respondent][]": eia_code,
            "facets[type][]": ["D", "DF"],
            "start": month_start.strftime("%Y-%m-%dT%H"),
            "end": last_hour.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_SIZE,
        }
        payload = get_with_retry(params)["response"]
        batch = payload.get("data", [])
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(payload.get("total", 0)):
            return rows


def to_wide(rows, region):
    """Turn EIA's long format into one row per hour: demand_mw, forecast_mw."""
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    wide = (
        df.drop_duplicates(["period", "type"])
        .pivot(index="period", columns="type", values="value")  # keeps blank hours so we can flag them
        .rename(columns={"D": "demand_mw", "DF": "forecast_mw"})
        .reset_index()
        .rename(columns={"period": "timestamp_utc"})
    )
    for col in ("demand_mw", "forecast_mw"):
        if col not in wide.columns:
            wide[col] = pd.NA
    wide.insert(1, "region", region)
    return wide[["timestamp_utc", "region", "demand_mw", "forecast_mw"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", help="Only this region, e.g. MISO")
    parser.add_argument("--start", help="First month to pull, e.g. 2024-01")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise SystemExit("EIA_API_KEY not found. Add it to the .env file.")

    config = load_config()
    regions = config["regions"]
    if args.region:
        regions = {args.region: regions[args.region]}

    now = pd.Timestamp.now(tz="UTC")
    start = args.start or config["history_start"]
    months = pd.date_range(pd.Timestamp(f"{start}-01", tz="UTC"), now, freq="MS")
    refresh_from = months[-REFRESH_MONTHS] if len(months) >= REFRESH_MONTHS else months[0]

    for region, info in regions.items():
        out_dir = RAW_DIR / region
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{region}: {len(months)} months")

        for month in months:
            out_file = out_dir / f"{month:%Y-%m}.parquet"
            if out_file.exists() and month < refresh_from:
                continue  # already have this complete month

            rows = fetch_month(api_key, info["eia_code"], month)
            if not rows:
                print(f"  {month:%Y-%m}: no data")
                continue

            wide = to_wide(rows, region)
            wide.to_parquet(out_file, index=False)
            print(f"  {month:%Y-%m}: {len(wide)} hours saved")
            time.sleep(0.3)  # be polite to the API

    print("\nDone.")


if __name__ == "__main__":
    main()
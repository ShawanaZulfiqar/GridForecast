"""
Pull hourly historical weather for each region's cities from Open-Meteo
(free, no API key). One request per city per year.

Saves one Parquet file per region per year:
    data/raw/weather/<REGION>/<YYYY>.parquet
Columns: timestamp_utc, region, city, temp_f, humidity_pct

Usage (run from the project folder):
    python src/fetch_weather.py                  # all regions, full history
    python src/fetch_weather.py --start 2024     # quick test: 2024 onward

Weather data by Open-Meteo.com (CC BY 4.0) - credit it on the dashboard.
"""
import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "regions.yaml"
RAW_DIR = ROOT / "data" / "raw" / "weather"
URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_LAG_DAYS = 6      # the archive runs a few days behind today


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_with_retry(params, tries=5):
    """Call Open-Meteo, waiting longer after each failure (incl. rate limits)."""
    for attempt in range(tries):
        try:
            r = requests.get(URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"    HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code == 429:          # too many requests
                time.sleep(60)
                continue
        except requests.RequestException as e:
            print(f"    Request error: {e}")
        time.sleep(2 ** attempt)
    raise RuntimeError("Open-Meteo request failed after several tries")


def fetch_city(city, start_date, end_date):
    """Hourly temperature and humidity for one city, timestamps in UTC."""
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "GMT",                    # UTC, to match the EIA data
    }
    hourly = get_with_retry(params)["hourly"]
    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(hourly["time"], utc=True),
        "temp_f": hourly["temperature_2m"],
        "humidity_pct": hourly["relative_humidity_2m"],
    })
    df.insert(1, "city", city["name"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", help="Only this region, e.g. MISO")
    parser.add_argument("--start", type=int, help="First year to pull, e.g. 2024")
    args = parser.parse_args()

    config = load_config()
    regions = config["regions"]
    if args.region:
        regions = {args.region: regions[args.region]}

    history_start = date.fromisoformat(f"{config['history_start']}-01")
    last_day = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    first_year = args.start or history_start.year
    years = range(first_year, last_day.year + 1)

    for region, info in regions.items():
        out_dir = RAW_DIR / region
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{region}: {len(info['weather_cities'])} cities, years {years[0]}-{years[-1]}")

        for year in years:
            out_file = out_dir / f"{year}.parquet"
            if out_file.exists() and year < last_day.year:
                continue  # complete past year already saved

            start_date = max(date(year, 1, 1), history_start)
            end_date = min(date(year, 12, 31), last_day)

            frames = []
            for city in info["weather_cities"]:
                frames.append(fetch_city(city, start_date, end_date))
                time.sleep(1)  # stay well inside the free limits

            df = pd.concat(frames, ignore_index=True)
            df.insert(1, "region", region)
            df.to_parquet(out_file, index=False)
            print(f"  {year}: {len(df)} rows saved ({start_date} to {end_date})")

    print("\nDone.")


if __name__ == "__main__":
    main()
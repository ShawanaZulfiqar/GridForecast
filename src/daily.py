"""
Week 3: the daily live run. GitHub Actions runs this every morning.

For each region it:
  1. refreshes demand (EIA) and weather history (Open-Meteo archive)
  2. downloads the latest weather forecast (Open-Meteo forecast API)
  3. retrains the boosting model on all history
  4. forecasts every hour of tomorrow (local time) and LOCKS it in a file
     that is never overwritten - so the scoreboard stays honest
  5. scores every past forecast against actual demand, MISO's own
     forecast, and the same-hour-last-week baseline

Files written:
    data/raw/weather_recent/<REGION>/<run date>.parquet   weather forecast used
    data/forecasts/<REGION>/<target date>.parquet          locked forecasts
    data/scoreboard/<REGION>.csv                           daily live scores

Usage (run from the project folder):
    python src/daily.py                 # full run
    python src/daily.py --skip-fetch    # use data already downloaded
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

from build_db import clean_demand, load_config, read_folder, weighted_weather
from model import GBM_FEATURES, build_features, mape

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FORECAST_DIR = ROOT / "data" / "forecasts"
SCORE_DIR = ROOT / "data" / "scoreboard"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def make_gbm():
    """Same settings as the backtest in model.py."""
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                         min_samples_leaf=50, categorical_features=["hour", "dow", "month"],
                                         random_state=42)


# ---------------------------------------------------------------- 1-2. fresh data
def refresh_history():
    """Re-run the fetch scripts for recent months only."""
    two_months_ago = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=1)).strftime("%Y-%m")
    this_year = str(pd.Timestamp.now(tz="UTC").year)
    for cmd in (["src/fetch_eia.py", "--start", two_months_ago],
                ["src/fetch_weather.py", "--start", this_year]):
        subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def fetch_recent_weather(region, cities):
    """Past 7 days + next 3 days of hourly weather for each city."""
    frames = []
    for city in cities:
        params = {
            "latitude": city["lat"], "longitude": city["lon"],
            "hourly": "temperature_2m,relative_humidity_2m",
            "temperature_unit": "fahrenheit", "timezone": "GMT",
            "past_days": 7, "forecast_days": 3,
        }
        for attempt in range(4):
            r = requests.get(FORECAST_URL, params=params, timeout=60)
            if r.status_code == 200:
                break
            print(f"    Open-Meteo HTTP {r.status_code}, retrying")
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        hourly = r.json()["hourly"]
        frames.append(pd.DataFrame({
            "timestamp_utc": pd.to_datetime(hourly["time"], utc=True),
            "region": region, "city": city["name"],
            "temp_f": hourly["temperature_2m"],
            "humidity_pct": hourly["relative_humidity_2m"],
        }))
        time.sleep(1)

    out_dir = RAW_DIR / "weather_recent" / region
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out_dir / f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.parquet", index=False)


# ---------------------------------------------------------------- 3-4. forecast
def load_inputs(region, info):
    """Demand history (outliers removed) and weather (archive first, then latest forecast)."""
    demand = clean_demand(read_folder(RAW_DIR / "eia" / region), region).set_index("timestamp_utc")
    demand.loc[demand["is_outlier"] == 1, "demand_mw"] = float("nan")

    archive = weighted_weather(read_folder(RAW_DIR / "weather" / region), info["weather_cities"])
    recent_files = sorted((RAW_DIR / "weather_recent" / region).glob("*.parquet"))
    if not recent_files:
        raise SystemExit(f"{region}: no weather forecast file - run without --skip-fetch")
    recent = weighted_weather(pd.read_parquet(recent_files[-1]), info["weather_cities"])

    weather = (archive.set_index("timestamp_utc")[["temp_f", "humidity_pct"]]
               .combine_first(recent.set_index("timestamp_utc")[["temp_f", "humidity_pct"]]))
    return demand, weather


def tomorrow_hours(local_tz):
    """Every hour of tomorrow in local time (23 or 25 hours on clock-change days), as UTC."""
    start = (pd.Timestamp.now(tz=local_tz) + pd.Timedelta(days=1)).normalize()
    hours = pd.date_range(start, start + pd.Timedelta(days=1), freq="h", inclusive="left")
    return start.date(), hours.tz_convert("UTC")


def forecast_tomorrow(region, info, demand, weather):
    local_tz = info["display_timezone"]
    target_date, target_hours = tomorrow_hours(local_tz)
    out_file = FORECAST_DIR / region / f"{target_date}.parquet"
    if out_file.exists():
        print(f"  Forecast for {target_date} already issued - keeping it locked")
        return

    index = pd.date_range(demand.index.min(), target_hours[-1], freq="h")
    frame = pd.DataFrame(index=index)
    frame["demand_mw"] = demand["demand_mw"]
    frame[["temp_f", "humidity_pct"]] = weather[["temp_f", "humidity_pct"]]
    f = build_features(frame, local_tz)

    train = f.join(frame["demand_mw"]).dropna(
        subset=["demand_mw", "temp_f", "temp_24h_mean", "lag_2d", "lag_1w"])
    train = train[train.index < target_hours[0]]
    X_tomorrow = f.loc[target_hours]
    if X_tomorrow["temp_f"].isna().any():
        raise SystemExit(f"{region}: weather forecast missing for some of tomorrow's hours")

    gbm = make_gbm().fit(train[GBM_FEATURES], train["demand_mw"])
    out = pd.DataFrame({
        "timestamp_utc": target_hours,
        "region": region,
        "target_date": str(target_date),
        "issued_at_utc": pd.Timestamp.now(tz="UTC"),
        "model_mw": gbm.predict(X_tomorrow[GBM_FEATURES]).round(0),
        "temp_f_used": X_tomorrow["temp_f"].values,
    })
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False)

    peak = out.loc[out["model_mw"].idxmax()]
    print(f"  Forecast for {target_date}: peak {peak['model_mw'] / 1000:.1f} GW "
          f"at {peak['timestamp_utc'].tz_convert(local_tz):%H:%M} local "
          f"(trained on {len(train):,} hours)")


# ---------------------------------------------------------------- 5. scoreboard
def score(region, info, demand):
    """Compare every locked forecast whose day has fully happened."""
    local_tz = info["display_timezone"]
    actual = demand["demand_mw"]
    rows = []
    for path in sorted((FORECAST_DIR / region).glob("*.parquet")):
        fc = pd.read_parquet(path).set_index("timestamp_utc")
        fc["actual_mw"] = actual.reindex(fc.index)
        fc["operator_mw"] = demand["forecast_mw"].reindex(fc.index)
        fc["baseline_mw"] = actual.reindex(fc.index - pd.Timedelta(hours=168)).values
        done = fc.dropna(subset=["actual_mw", "operator_mw", "baseline_mw"])
        if len(done) < 0.9 * len(fc):
            continue  # day not over yet, or data not published yet

        def peak_hour(col):
            return done[col].idxmax().tz_convert(local_tz).hour

        rows.append({
            "target_date": fc["target_date"].iloc[0],
            "hours": len(done),
            "model_mape": round(mape(done["actual_mw"], done["model_mw"]), 2),
            "operator_mape": round(mape(done["actual_mw"], done["operator_mw"]), 2),
            "baseline_mape": round(mape(done["actual_mw"], done["baseline_mw"]), 2),
            "actual_peak_gw": round(done["actual_mw"].max() / 1000, 1),
            "actual_peak_hour": peak_hour("actual_mw"),
            "model_peak_hour": peak_hour("model_mw"),
            "operator_peak_hour": peak_hour("operator_mw"),
        })

    if not rows:
        print("  Scoreboard: no completed days yet")
        return
    board = pd.DataFrame(rows)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    board.to_csv(SCORE_DIR / f"{region}.csv", index=False)
    print(f"  Scoreboard ({len(board)} live days): model {board['model_mape'].mean():.2f}% | "
          f"operator {board['operator_mape'].mean():.2f}% | baseline {board['baseline_mape'].mean():.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", help="Only this region, e.g. MISO")
    parser.add_argument("--skip-fetch", action="store_true", help="Use data already downloaded")
    args = parser.parse_args()

    config = load_config()
    regions = config["regions"]
    if args.region:
        regions = {args.region: regions[args.region]}

    if not args.skip_fetch:
        refresh_history()

    for region, info in regions.items():
        print(f"\n{region}")
        if not args.skip_fetch:
            fetch_recent_weather(region, info["weather_cities"])
        demand, weather = load_inputs(region, info)
        forecast_tomorrow(region, info, demand, weather)
        score(region, info, demand)

    print("\nDaily run complete.")


if __name__ == "__main__":
    main()
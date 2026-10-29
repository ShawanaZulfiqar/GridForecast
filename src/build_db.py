"""
Combine the raw Parquet files into one SQLite database and run data checks.

Creates data/grid.db with three tables:
    demand        hourly demand + operator forecast, with is_missing / is_outlier flags
    weather_city  hourly weather for each city
    weather       population-weighted hourly weather per region

Problems are flagged, never silently dropped.

Usage (run from the project folder):
    python src/build_db.py
"""
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "regions.yaml"
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "grid.db"
OUTLIER_LIMIT = 0.40      # flag hours more than 40% away from the surrounding week's median


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def read_folder(folder):
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def clean_demand(df, region):
    """One row per hour with no gaps; flag missing hours and outliers."""
    df = df.drop_duplicates("timestamp_utc", keep="last").set_index("timestamp_utc").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_index)
    df.index.name = "timestamp_utc"
    df["region"] = region

    demand = df["demand_mw"].astype(float)
    weekly_median = demand.rolling(168, center=True, min_periods=24).median()
    df["is_missing"] = demand.isna().astype(int)
    df["is_outlier"] = ((demand - weekly_median).abs() / weekly_median > OUTLIER_LIMIT).astype(int)
    return df.reset_index()


def weighted_weather(wx, cities):
    """Average the cities' weather each hour, weighted by population."""
    weights = {c["name"]: c["population"] for c in cities}
    wx = wx.dropna(subset=["temp_f", "humidity_pct"]).copy()
    wx["w"] = wx["city"].map(weights)
    wx["wt"] = wx["temp_f"] * wx["w"]
    wx["wh"] = wx["humidity_pct"] * wx["w"]
    agg = wx.groupby(["timestamp_utc", "region"])[["wt", "wh", "w"]].sum().reset_index()
    agg["temp_f"] = (agg["wt"] / agg["w"]).round(2)
    agg["humidity_pct"] = (agg["wh"] / agg["w"]).round(1)
    return agg[["timestamp_utc", "region", "temp_f", "humidity_pct"]]


def to_text_time(df):
    """SQLite has no date type; store UTC times as sortable text."""
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main():
    config = load_config()
    demand_frames, city_frames, weighted_frames = [], [], []

    print("Data quality report")
    print("-------------------")
    for region, info in config["regions"].items():
        demand_raw = read_folder(RAW_DIR / "eia" / region)
        weather_raw = read_folder(RAW_DIR / "weather" / region)
        if demand_raw.empty or weather_raw.empty:
            print(f"{region}: missing raw files - run the fetch scripts first")
            continue

        demand = clean_demand(demand_raw, region)
        weighted = weighted_weather(weather_raw, info["weather_cities"])

        print(f"\n{region}")
        print(f"  Hours covered:        {demand['timestamp_utc'].min():%Y-%m-%d} to {demand['timestamp_utc'].max():%Y-%m-%d} ({len(demand):,} hours)")
        print(f"  Missing demand hours: {demand['is_missing'].sum():,}")
        print(f"  Missing forecast hrs: {demand['forecast_mw'].isna().sum():,}")
        print(f"  Flagged outliers:     {demand['is_outlier'].sum():,}")
        print(f"  Weather hours:        {len(weighted):,}")

        demand_frames.append(demand[["timestamp_utc", "region", "demand_mw", "forecast_mw", "is_missing", "is_outlier"]])
        city_frames.append(weather_raw)
        weighted_frames.append(weighted)

    if not demand_frames:
        raise SystemExit("Nothing to load.")

    with sqlite3.connect(DB_PATH) as con:
        to_text_time(pd.concat(demand_frames)).to_sql("demand", con, if_exists="replace", index=False)
        to_text_time(pd.concat(city_frames)).to_sql("weather_city", con, if_exists="replace", index=False)
        to_text_time(pd.concat(weighted_frames)).to_sql("weather", con, if_exists="replace", index=False)
        for table in ("demand", "weather_city", "weather"):
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table} ON {table} (region, timestamp_utc)")

    print(f"\nSaved {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
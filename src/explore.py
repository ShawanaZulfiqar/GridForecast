"""
First look at the data: what drives demand, and how good is the operator's forecast?

Reads data/grid.db (run build_db.py first), saves charts to reports/,
and prints the first scoreboard: operator forecast vs a naive baseline.

Usage (run from the project folder):
    python src/explore.py                 # MISO
    python src/explore.py --region MISO
"""
import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "regions.yaml"
DB_PATH = ROOT / "data" / "grid.db"
REPORTS = ROOT / "reports"

QUERY = """
SELECT d.timestamp_utc, d.demand_mw, d.forecast_mw, d.is_missing, d.is_outlier,
       w.temp_f, w.humidity_pct
FROM demand d
LEFT JOIN weather w
       ON w.region = d.region AND w.timestamp_utc = d.timestamp_utc
WHERE d.region = ?
ORDER BY d.timestamp_utc
"""


def mape(actual, predicted):
    """Mean absolute percentage error, in percent."""
    return ((actual - predicted).abs() / actual).mean() * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="MISO")
    args = parser.parse_args()
    region = args.region

    with open(CONFIG_PATH) as f:
        local_tz = yaml.safe_load(f)["regions"][region]["display_timezone"]

    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(QUERY, con, params=(region,))

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    df["baseline_mw"] = df["demand_mw"].shift(168)        # same hour last week
    df["local_time"] = df.index.tz_convert(local_tz)
    df["hour"] = df["local_time"].dt.hour
    df["month"] = df["local_time"].dt.month
    df["year"] = df["local_time"].dt.year

    clean = df[(df["is_missing"] == 0) & (df["is_outlier"] == 0)]
    REPORTS.mkdir(exist_ok=True)

    # 1. Demand vs temperature (daily averages) - expect a U shape
    daily = clean.resample("D")[["demand_mw", "temp_f"]].mean().dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(daily["temp_f"], daily["demand_mw"] / 1000, s=6, alpha=0.4)
    ax.set_xlabel("Population-weighted temperature (°F)")
    ax.set_ylabel("Average daily demand (GW)")
    ax.set_title(f"{region}: demand rises in both cold and hot weather")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{region}_demand_vs_temp.png", dpi=150)
    plt.close(fig)

    # 2. Typical daily shape: average demand by local hour, summer vs winter
    season = clean["month"].map(lambda m: "Summer (Jun-Aug)" if m in (6, 7, 8)
                                else "Winter (Dec-Feb)" if m in (12, 1, 2) else None)
    by_hour = clean.assign(season=season).dropna(subset=["season"]) \
        .groupby(["season", "hour"])["demand_mw"].mean().unstack(0) / 1000
    fig, ax = plt.subplots(figsize=(8, 5))
    by_hour.plot(ax=ax, marker="o", markersize=3)
    ax.set_xlabel(f"Hour of day ({local_tz})")
    ax.set_ylabel("Average demand (GW)")
    ax.set_title(f"{region}: when demand peaks during the day")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(REPORTS / f"{region}_demand_by_hour.png", dpi=150)
    plt.close(fig)

    # 3. First scoreboard: operator forecast vs "same hour last week"
    scored = clean.dropna(subset=["demand_mw", "forecast_mw", "baseline_mw"])
    table = scored.groupby("year").apply(
        lambda g: pd.Series({
            "hours": len(g),
            "operator_mape_%": round(mape(g["demand_mw"], g["forecast_mw"]), 2),
            "baseline_mape_%": round(mape(g["demand_mw"], g["baseline_mw"]), 2),
        }),
        include_groups=False,
    )
    table["hours"] = table["hours"].astype(int)

    print(f"\n{region} data: {clean.index.min():%Y-%m-%d} to {clean.index.max():%Y-%m-%d}, "
          f"{len(clean):,} clean hours")
    print(f"Peak demand: {clean['demand_mw'].max() / 1000:,.1f} GW "
          f"on {clean['demand_mw'].idxmax().tz_convert(local_tz):%Y-%m-%d %H:%M}")
    print("\nForecast error by year (lower is better)")
    print(table.to_string())
    print(f"\nAll years: operator {mape(scored['demand_mw'], scored['forecast_mw']):.2f}%  |  "
          f"baseline {mape(scored['demand_mw'], scored['baseline_mw']):.2f}%")
    print(f"\nCharts saved in {REPORTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
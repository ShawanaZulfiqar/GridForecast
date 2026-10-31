"""
Week 2: build forecasting features, train models, and backtest them.

Walk-forward backtest: for each test year, train only on the years before it,
then predict that whole year - as if we had been running live.

Models compared on exactly the same hours:
    baseline  - demand at the same hour last week
    linear    - ridge regression on weather + calendar + recent demand
    boosting  - gradient boosting (scikit-learn's HistGradientBoosting,
                the same technique as LightGBM, no extra installs) on the same information
    operator  - the grid operator's own day-ahead forecast (the benchmark)

Usage (run from the project folder):
    python src/model.py                  # MISO
    python src/model.py --region MISO
"""
import argparse
import sqlite3
from pathlib import Path

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "regions.yaml"
DB_PATH = ROOT / "data" / "grid.db"
REPORTS = ROOT / "reports"
PRED_DIR = ROOT / "data" / "predictions"

QUERY = """
SELECT d.timestamp_utc, d.demand_mw, d.forecast_mw, d.is_missing, d.is_outlier,
       w.temp_f, w.humidity_pct
FROM demand d
LEFT JOIN weather w
       ON w.region = d.region AND w.timestamp_utc = d.timestamp_utc
WHERE d.region = ?
ORDER BY d.timestamp_utc
"""

GBM_FEATURES = ["hour", "dow", "month", "doy", "holiday", "temp_f", "humidity",
                "temp_24h_mean", "lag_2d", "lag_1w"]


# ---------------------------------------------------------------- data + features
def load_region(region):
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(QUERY, con, params=(region,))
    if df.empty:
        raise SystemExit(f"No {region} data in grid.db - run `python src/build_db.py` first.")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    # a flagged outlier should not be used as a "recent demand" input either
    df.loc[df["is_outlier"] == 1, "demand_mw"] = np.nan
    return df


def build_features(df, local_tz):
    """Everything the model may know when forecasting a day ahead.

    Recent-demand features look back at least 48 hours, so they are already
    known when tomorrow's forecast is made.
    """
    local = df.index.tz_convert(local_tz)
    us_holidays = holidays.US(years=range(local.year.min(), local.year.max() + 2))

    f = pd.DataFrame(index=df.index)
    f["hour"] = local.hour
    f["dow"] = local.day_of_week                    # 0 = Monday
    f["month"] = local.month
    f["doy"] = local.day_of_year
    f["holiday"] = [int(d in us_holidays) for d in local.date]
    f["local_date"] = local.date
    f["temp_f"] = df["temp_f"]
    f["humidity"] = df["humidity_pct"]
    f["temp_24h_mean"] = df["temp_f"].rolling(24, min_periods=12).mean()   # buildings warm/cool slowly
    f["hdd"] = (55 - df["temp_f"]).clip(lower=0)  # heating degrees
    f["cdd"] = (df["temp_f"] - 65).clip(lower=0)  # cooling degrees
    f["lag_2d"] = df["demand_mw"].shift(48)        # same hour two days ago
    f["lag_1w"] = df["demand_mw"].shift(168)       # same hour last week
    return f


def linear_design(f):
    """Ridge needs numbers it can add up: one-hot calendar + degree terms."""
    X = pd.concat([
        pd.get_dummies(f["hour"], prefix="h", dtype=float),
        pd.get_dummies(f["dow"], prefix="d", dtype=float),
        pd.get_dummies(f["month"], prefix="m", dtype=float),
        f[["holiday", "hdd", "cdd", "humidity", "temp_24h_mean", "lag_2d", "lag_1w"]],
    ], axis=1)
    X["hdd2"] = f["hdd"] ** 2
    X["cdd2"] = f["cdd"] ** 2
    for h in range(24):                            # air conditioning matters more in the afternoon
        X[f"cdd_h{h}"] = f["cdd"] * (f["hour"] == h)
    return X


# ---------------------------------------------------------------- scoring
def mape(actual, predicted):
    return float(((actual - predicted).abs() / actual).mean() * 100)


def peak_hour_hits(test, col):
    """Share of days where the predicted peak hour is within 1 hour of the real one."""
    days = test.groupby("local_date").filter(lambda g: len(g) >= 23)
    real = days.groupby("local_date")["demand_mw"].idxmax()
    pred = days.groupby("local_date")[col].idxmax()
    gap = (real - pred).abs() / pd.Timedelta(hours=1)
    return float((gap <= 1).mean() * 100)


# ---------------------------------------------------------------- backtest
def backtest(df, f):
    data = f.join(df[["demand_mw", "forecast_mw"]])
    data["baseline"] = data["lag_1w"]
    data["year"] = pd.DatetimeIndex(data.index).year
    usable = data.dropna(subset=["demand_mw", "temp_f", "temp_24h_mean", "lag_2d", "lag_1w"])

    years = sorted(usable["year"].unique())
    test_years = years[2:]                         # always train on at least two years
    results = []

    for year in test_years:
        train = usable[usable["year"] < year]
        test = usable[usable["year"] == year].dropna(subset=["forecast_mw"]).copy()

        cols = linear_design(pd.concat([train, test]))   # same columns for both
        ridge = Ridge(alpha=1.0).fit(cols.loc[train.index], train["demand_mw"])
        test["linear"] = ridge.predict(cols.loc[test.index])

        gbm = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                            min_samples_leaf=50, categorical_features=["hour", "dow", "month"],
                                            random_state=42)
        gbm.fit(train[GBM_FEATURES], train["demand_mw"])
        test["boosting"] = gbm.predict(test[GBM_FEATURES])
        test = test.rename(columns={"forecast_mw": "operator"})
        results.append(test)
        print(f"  {year}: trained on {len(train):,} hours, tested on {len(test):,}")

    # how much worse the latest model gets when each input is scrambled
    sample = test.sample(min(5000, len(test)), random_state=42)
    perm = permutation_importance(gbm, sample[GBM_FEATURES], sample["demand_mw"],
                                  n_repeats=3, random_state=42)
    importance = pd.Series(perm.importances_mean, index=GBM_FEATURES).sort_values(ascending=False)
    return pd.concat(results), importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="MISO")
    region = parser.parse_args().region

    with open(CONFIG_PATH) as fh:
        local_tz = yaml.safe_load(fh)["regions"][region]["display_timezone"]

    print(f"{region}: building features and running the walk-forward backtest")
    df = load_region(region)
    f = build_features(df, local_tz)
    preds, importance = backtest(df, f)

    models = ["baseline", "linear", "boosting", "operator"]
    by_year = preds.groupby("year").apply(
        lambda g: pd.Series({m: round(mape(g["demand_mw"], g[m]), 2) for m in models}),
        include_groups=False,
    )
    overall = {m: round(mape(preds["demand_mw"], preds[m]), 2) for m in models}
    peaks = {m: round(peak_hour_hits(preds, m), 1) for m in models}

    print("\nError by test year - MAPE % (lower is better)")
    print(by_year.to_string())
    print("\nAll test years")
    print(pd.DataFrame({"MAPE %": overall, "peak hour within 1h, % of days": peaks}).to_string())
    print("\nWhat the boosting model relies on most (higher = more important)")
    print(importance.head(8).round(3).to_string())

    gate = "PASSED" if overall["boosting"] < overall["baseline"] else "NOT passed"
    print(f"\nGo-live gate (beat the baseline): {gate}")
    print("Note: the backtest uses actual weather. Live forecasts use forecast weather,")
    print("so expect live error to be somewhat higher.")

    REPORTS.mkdir(exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(REPORTS / f"{region}_backtest_by_year.csv")
    preds[["demand_mw"] + models[:-1] + ["operator"]].to_parquet(PRED_DIR / f"{region}_backtest.parquet")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        ax.plot(by_year.index, by_year[m], marker="o", label=m)
    ax.set_xlabel("Test year")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"{region}: forecast error by year (walk-forward backtest)")
    ax.set_xticks(by_year.index)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / f"{region}_backtest_mape.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved reports/{region}_backtest_by_year.csv and reports/{region}_backtest_mape.png")


if __name__ == "__main__":
    main()
"""
Grid Forecast dashboard (Streamlit).

Reads the files the daily GitHub run saves in data/ - no database or API keys
needed - so it can be hosted for free on Streamlit Community Cloud.

Run on your Mac (from the project folder):
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_db import clean_demand, load_config, read_folder, weighted_weather  # noqa: E402

DATA = ROOT / "data"
REPO_URL = "https://github.com/ShawanaZulfiqar/GridForecast"

COLORS = {"actual": "#1f2937", "model": "#2563eb", "operator": "#ea580c",
          "baseline": "#9ca3af", "temp": "#dc2626"}
LABELS = {"model": "My model", "operator": "Grid operator", "baseline": "Same hour last week"}

st.set_page_config(page_title="Grid Forecast", page_icon="⚡", layout="wide")


# ---------------------------------------------------------------- data loading
@st.cache_data(ttl=3600)
def get_config():
    return load_config()


@st.cache_data(ttl=3600)
def load_region(region, cities):
    """Hourly demand, operator forecast and population-weighted weather, indexed in UTC."""
    raw = read_folder(DATA / "raw" / "eia" / region)
    if raw.empty:
        return pd.DataFrame()
    demand = clean_demand(raw, region).set_index("timestamp_utc")

    cols = ["temp_f", "humidity_pct"]
    weather = pd.DataFrame(columns=cols)
    archive = read_folder(DATA / "raw" / "weather" / region)
    if not archive.empty:
        weather = weighted_weather(archive, cities).set_index("timestamp_utc")[cols]
    for path in sorted((DATA / "raw" / "weather_recent" / region).glob("*.parquet"), reverse=True):
        recent = weighted_weather(pd.read_parquet(path), cities).set_index("timestamp_utc")[cols]
        weather = recent if weather.empty else weather.combine_first(recent)

    df = demand.join(weather, how="left")
    df.loc[df["is_outlier"] == 1, "demand_mw"] = float("nan")
    return df


@st.cache_data(ttl=3600)
def load_forecasts(region):
    files = sorted((DATA / "forecasts" / region).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@st.cache_data(ttl=3600)
def load_scoreboard(region):
    path = DATA / "scoreboard" / f"{region}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_backtest(region):
    path = DATA / "predictions" / f"{region}_backtest.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def mape(actual, predicted):
    return float(((actual - predicted).abs() / actual).mean() * 100)


def hour_label(h):
    return pd.Timestamp(2000, 1, 1, int(h)).strftime("%-I %p")


def style(fig, height=420, y_title="Demand (GW)"):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
                      yaxis_title=y_title, plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


# ---------------------------------------------------------------- sidebar
config = get_config()
available = [r for r in config["regions"] if any((DATA / "raw" / "eia" / r).glob("*.parquet"))]
if not available:
    st.error("No data found. Run the fetch scripts or the daily pipeline first.")
    st.stop()

with st.sidebar:
    st.title("⚡ Grid Forecast")
    st.caption("Tomorrow's hourly electricity demand for six U.S. power grids, "
               "scored every day against each grid operator's own forecast.")
    region = st.selectbox("Grid region", available, format_func=lambda r: f"{r}: {config['regions'][r]['name']}")
    info = config["regions"][region]
    tz = info["display_timezone"]
    st.caption(f"Times shown in {tz.replace('_', ' ')}.")
    st.divider()
    st.caption(f"Demand and operator forecasts: [U.S. EIA Hourly Electric Grid Monitor]"
               f"(https://www.eia.gov/electricity/gridmonitor/)  \n"
               f"Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0)  \n"
               f"Code: [GitHub]({REPO_URL})")

df = load_region(region, info["weather_cities"])
forecasts = load_forecasts(region)
board = load_scoreboard(region)
backtest = load_backtest(region)

local = df.copy()
local.index = local.index.tz_convert(tz)
actual = local["demand_mw"].dropna()

# ---------------------------------------------------------------- header + KPIs
st.header(info["name"])
st.caption(f"Actual demand data through {actual.index.max():%b %-d, %Y %-I %p} · "
           f"{len(actual):,} hours since {actual.index.min() + pd.Timedelta(days=1):%b %Y}")

latest = pd.DataFrame()
if not forecasts.empty:
    latest = forecasts[forecasts["target_date"] == forecasts["target_date"].max()].copy()
    latest["local"] = pd.to_datetime(latest["timestamp_utc"], utc=True).dt.tz_convert(tz)

k1, k2, k3, k4 = st.columns(4)
if not latest.empty:
    peak = latest.loc[latest["model_mw"].idxmax()]
    k1.metric(f"Forecast peak · {pd.Timestamp(peak['target_date']):%a %b %-d}, {peak['local']:%-I %p}",
              f"{peak['model_mw'] / 1000:,.1f} GW")
else:
    k1.metric("Forecast peak", "—")
k2.metric("Live days scored", f"{len(board)}")
if not board.empty:
    k3.metric("My model · live error", f"{board['model_mape'].mean():.2f}%",
              f"{board['model_mape'].mean() - board['operator_mape'].mean():+.2f} pts vs operator",
              delta_color="inverse")
    k4.metric("Grid operator · live error", f"{board['operator_mape'].mean():.2f}%")
elif not backtest.empty:
    k3.metric("My model · backtest error", f"{mape(backtest['demand_mw'], backtest['boosting']):.2f}%")
    k4.metric("Grid operator · backtest error", f"{mape(backtest['demand_mw'], backtest['operator']):.2f}%")

tab_tomorrow, tab_score, tab_drivers, tab_history, tab_about = st.tabs(
    ["Tomorrow", "Scoreboard", "What drives demand", "History", "About"])

# ---------------------------------------------------------------- Tomorrow
with tab_tomorrow:
    if latest.empty:
        st.info("No forecast issued yet for this region. The daily run issues one every morning.")
    else:
        target = pd.Timestamp(latest["target_date"].iloc[0])
        today = pd.Timestamp.now(tz=tz).normalize().tz_localize(None)
        if target <= today:
            st.warning(f"The newest forecast is for {target:%b %-d}. Today's run may not have finished yet.")

        start = latest["local"].min() - pd.Timedelta(days=3)
        recent = local[local.index >= start]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent["demand_mw"] / 1000, name="Actual demand",
                                 line=dict(color=COLORS["actual"], width=2)))
        fig.add_trace(go.Scatter(x=recent.index, y=recent["forecast_mw"] / 1000, name="Grid operator forecast",
                                 line=dict(color=COLORS["operator"], width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=latest["local"], y=latest["model_mw"] / 1000, name="My forecast",
                                 line=dict(color=COLORS["model"], width=3)))
        fig.add_vrect(x0=latest["local"].min(), x1=latest["local"].max(),
                      fillcolor=COLORS["model"], opacity=0.06, line_width=0,
                      annotation_text=f"{target:%a %b %-d}", annotation_position="top left")
        fig = style(fig)
        fig.update_xaxes(range=[start, latest["local"].max()])
        st.plotly_chart(fig, width="stretch")

        low = latest.loc[latest["model_mw"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Peak:** {peak['model_mw'] / 1000:,.1f} GW at {peak['local']:%-I %p}")
        c2.markdown(f"**Lowest:** {low['model_mw'] / 1000:,.1f} GW at {low['local']:%-I %p}")
        c3.markdown(f"**Forecast temperature:** {latest['temp_f_used'].min():.0f}–"
                    f"{latest['temp_f_used'].max():.0f} °F (population-weighted)")
        st.caption(f"Forecast issued {pd.Timestamp(latest['issued_at_utc'].iloc[0]).tz_convert(tz):%b %-d, %-I:%M %p} "
                   "and locked: it is never changed after it is issued.")

# ---------------------------------------------------------------- Scoreboard
with tab_score:
    st.subheader("Live scoreboard")
    st.caption("Each forecast is saved before the day happens, then compared with actual demand. "
               "Error = mean absolute percentage error (MAPE); lower is better.")
    if board.empty:
        st.info("No completed days yet. The first live day is scored the morning after it ends.")
    else:
        b = board.copy()
        b["target_date"] = pd.to_datetime(b["target_date"])
        fig = go.Figure()
        for key in ["model", "operator", "baseline"]:
            fig.add_trace(go.Scatter(x=b["target_date"], y=b[f"{key}_mape"], name=LABELS[key],
                                     mode="lines+markers", line=dict(color=COLORS[key])))
        fig = style(fig, 360, "Daily error (%)")
        fig.update_xaxes(tickformat="%b %-d", dtick=86400000)
        st.plotly_chart(fig, width="stretch")

        wins = int((b["model_mape"] < b["operator_mape"]).sum())
        peak_model = (b["model_peak_hour"] - b["actual_peak_hour"]).abs().le(1).mean() * 100
        peak_op = (b["operator_peak_hour"] - b["actual_peak_hour"]).abs().le(1).mean() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Days my model beat the operator", f"{wins} of {len(b)}")
        c2.metric("Peak hour within 1 h · my model", f"{peak_model:.0f}%")
        c3.metric("Peak hour within 1 h · operator", f"{peak_op:.0f}%")

        show = b.sort_values("target_date", ascending=False).rename(columns={
            "target_date": "Date", "model_mape": "My model %", "operator_mape": "Operator %",
            "baseline_mape": "Last week %", "actual_peak_gw": "Peak GW", "actual_peak_hour": "Actual peak hour",
            "model_peak_hour": "My peak hour", "operator_peak_hour": "Operator peak hour"}).drop(columns="hours")
        show["Date"] = show["Date"].dt.strftime("%a %b %-d")
        st.dataframe(show, hide_index=True, width="stretch")

    st.subheader("All regions at a glance")
    rows = []
    for r in available:
        rb, bt = load_scoreboard(r), load_backtest(r)
        rows.append({
            "Region": r,
            "Live days": len(rb),
            "Live · my model %": round(rb["model_mape"].mean(), 2) if not rb.empty else None,
            "Live · operator %": round(rb["operator_mape"].mean(), 2) if not rb.empty else None,
            "Backtest · my model %": round(mape(bt["demand_mw"], bt["boosting"]), 2) if not bt.empty else None,
            "Backtest · operator %": round(mape(bt["demand_mw"], bt["operator"]), 2) if not bt.empty else None,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Backtest: replaying past years")
    if backtest.empty:
        st.info(f"No backtest saved for {region}. Run `python src/model.py --region {region}` and push.")
    else:
        st.caption("Each year is predicted by a model trained only on earlier years. The backtest uses the "
                   "weather that actually happened, while the operator had to use weather forecasts, so it "
                   "flatters my model. The live scoreboard above is the fair comparison.")
        bt = backtest.copy()
        bt["year"] = pd.DatetimeIndex(bt.index).year          # UTC year, same as model.py
        cols = {"boosting": "My model", "linear": "Linear regression", "operator": "Grid operator",
                "baseline": "Same hour last week"}
        by_year = bt.groupby("year").apply(
            lambda g: pd.Series({name: round(mape(g["demand_mw"], g[c]), 2) for c, name in cols.items()}),
            include_groups=False)
        fig = go.Figure()
        for c, name in cols.items():
            color = {"boosting": COLORS["model"], "linear": "#93c5fd",
                     "operator": COLORS["operator"], "baseline": COLORS["baseline"]}[c]
            fig.add_trace(go.Bar(x=by_year.index.astype(str), y=by_year[name], name=name, marker_color=color))
        fig.update_layout(barmode="group")
        st.plotly_chart(style(fig, 360, "Error (%)"), width="stretch")

# ---------------------------------------------------------------- What drives demand
with tab_drivers:
    daily = local[["demand_mw", "temp_f"]].resample("D").mean().dropna()
    if daily.empty:
        st.info("Weather data is needed for this view.")
    else:
        daily["Day type"] = ["Weekend" if d >= 5 else "Weekday" for d in daily.index.day_of_week]
        st.subheader("Temperature: demand rises in both cold and hot weather")
        fig = go.Figure()
        for kind, color in [("Weekday", COLORS["model"]), ("Weekend", COLORS["operator"])]:
            part = daily[daily["Day type"] == kind]
            fig.add_trace(go.Scatter(x=part["temp_f"], y=part["demand_mw"] / 1000, mode="markers", name=kind,
                                     marker=dict(color=color, size=5, opacity=0.45),
                                     customdata=part.index.strftime("%b %-d, %Y"),
                                     hovertemplate="%{customdata}<br>%{x:.0f} °F · %{y:.1f} GW<extra></extra>"))
        fig = style(fig, 420, "Average daily demand (GW)")
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="Population-weighted daily temperature (°F)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Each dot is one day. Weekends sit lower at the same temperature, "
                   "which is why the model uses day of week as well as weather.")

    st.subheader("Time of day: when demand peaks")
    seasons = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    prof = actual.to_frame()
    prof["season"] = prof.index.month.map(seasons)
    prof["hour"] = prof.index.hour
    by_hour = prof.groupby(["season", "hour"])["demand_mw"].mean().unstack(0) / 1000
    fig = go.Figure()
    for s, color in [("Summer", "#dc2626"), ("Winter", "#2563eb"), ("Spring", "#16a34a"), ("Fall", "#ca8a04")]:
        if s in by_hour:
            fig.add_trace(go.Scatter(x=[hour_label(h) for h in by_hour.index], y=by_hour[s], name=s,
                                     mode="lines+markers", line=dict(color=color)))
    st.plotly_chart(style(fig, 380, "Average demand (GW)"), width="stretch")

    st.subheader("Year by year")
    yearly = actual.resample("YE").agg(["mean", "max", "count"])
    yearly = yearly[yearly["count"] > 24 * 300]          # full years only
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly.index.year.astype(str), y=yearly["mean"] / 1000, name="Average",
                         marker_color=COLORS["model"]))
    fig.add_trace(go.Scatter(x=yearly.index.year.astype(str), y=yearly["max"] / 1000, name="Peak hour",
                             mode="lines+markers", line=dict(color=COLORS["operator"])))
    st.plotly_chart(style(fig, 340), width="stretch")

# ---------------------------------------------------------------- History
with tab_history:
    first, last = actual.index.min().date(), actual.index.max().date()
    c1, c2 = st.columns([2, 1])
    picked = c1.date_input("Date range", value=(max(first, last - pd.Timedelta(days=30)), last),
                           min_value=first, max_value=last)
    step = c2.radio("Show", ["Hourly", "Daily", "Monthly"], horizontal=True)
    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        start_d, end_d = picked
        mask = (local.index.date >= start_d) & (local.index.date <= end_d)
        part = local.loc[mask, ["demand_mw", "forecast_mw", "temp_f", "humidity_pct"]]
        if step != "Hourly":
            part = part.resample({"Daily": "D", "Monthly": "MS"}[step]).mean()

        show_temp = st.checkbox("Show temperature", value=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=part.index, y=part["demand_mw"] / 1000, name="Actual demand",
                                 line=dict(color=COLORS["actual"])), secondary_y=False)
        fig.add_trace(go.Scatter(x=part.index, y=part["forecast_mw"] / 1000, name="Grid operator forecast",
                                 line=dict(color=COLORS["operator"], dash="dot")), secondary_y=False)
        if show_temp:
            fig.add_trace(go.Scatter(x=part.index, y=part["temp_f"], name="Temperature (°F)",
                                     line=dict(color=COLORS["temp"], width=1), opacity=0.6), secondary_y=True)
        fig = style(fig, 460)
        fig.update_yaxes(title_text="Temperature (°F)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, width="stretch")

        out = part.copy()
        out.index = out.index.tz_localize(None)
        out.index.name = f"time_{tz.split('/')[-1].lower()}"
        out = out.rename(columns={"demand_mw": "actual_demand_mw", "forecast_mw": "operator_forecast_mw",
                                  "temp_f": "weighted_temp_f", "humidity_pct": "weighted_humidity_pct"}).round(1)
        st.download_button("Download this data (CSV)", out.to_csv().encode(),
                           file_name=f"{region}_{start_d}_{end_d}_{step.lower()}.csv", mime="text/csv")
        st.caption("Source: U.S. EIA (Form EIA-930) and Open-Meteo. Hours flagged as outliers are blank.")
    else:
        st.info("Pick a start and an end date.")

# ---------------------------------------------------------------- About
with tab_about:
    st.markdown(f"""
**What this is.** Every morning an automated job (GitHub Actions) downloads the latest electricity demand
from the U.S. Energy Information Administration and the weather forecast from Open-Meteo, retrains a
model for each grid, and predicts every hour of tomorrow. Each forecast is saved before the day
happens and never edited, then scored against actual demand and the grid operator's own day-ahead forecast.

**The model.** Gradient boosting (scikit-learn) on population-weighted temperature and humidity across
each grid's largest cities, hour of day, day of week, season, U.S. holidays, and demand from two days
and one week earlier. One model per grid, retrained daily on all history since January 2019.

**How it is judged.**
- *Same hour last week* is the floor: a model must beat it to be worth using.
- *The grid operator's forecast* is the professional benchmark.
- *Backtest* results replay past years with the weather that actually happened, so they flatter the
  model; only the live scoreboard is a fair comparison.

**Limitations.** Grid regions are large, so a few cities' weather is a simplification. The model has no
solar or cloud-cover input, which matters most in California. EIA sometimes revises recent data.

**Built by** Shawana Zulfiqar, MS Business Analytics, University of Michigan-Dearborn. [Code on GitHub]({REPO_URL}).
""")
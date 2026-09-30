from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
INTERIM_PATH = ROOT / "data" / "interim" / "city_day_clean.csv"
RAW_PATH = ROOT / "data" / "raw" / "city_day.csv"
KEEPALIVE_INTERVAL_SECONDS = 10 * 60


@st.cache_resource
def start_render_keepalive() -> bool:
    """Ping this Render service periodically for a bounded time after startup.

    Render sets RENDER_EXTERNAL_URL automatically. The background thread is
    only a best-effort free-tier workaround; Render can still restart or stop
    a free instance independently.
    """
    service_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not service_url:
        return False

    try:
        requested_days = int(os.environ.get("KEEPALIVE_DAYS", "10"))
    except ValueError:
        requested_days = 10
    keepalive_days = min(10, max(5, requested_days))

    def ping_until_expiry() -> None:
        deadline = time.monotonic() + keepalive_days * 24 * 60 * 60
        while time.monotonic() < deadline:
            time.sleep(KEEPALIVE_INTERVAL_SECONDS)
            if time.monotonic() >= deadline:
                break
            try:
                request = Request(service_url, headers={"User-Agent": "RenderKeepalive/1.0"})
                with urlopen(request, timeout=20) as response:
                    response.read(1)
            except (OSError, URLError) as exc:
                logging.warning("Render keep-alive request failed: %s", exc)

    threading.Thread(
        target=ping_until_expiry,
        name="render-keepalive",
        daemon=True,
    ).start()
    logging.info("Render keep-alive enabled for up to %s days", keepalive_days)
    return True


start_render_keepalive()


@st.cache_data
def load_dataset() -> pd.DataFrame:
    candidate = INTERIM_PATH if INTERIM_PATH.exists() else RAW_PATH
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found: {candidate}")

    df = pd.read_csv(candidate)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "City" in df.columns:
        df["City"] = df["City"].astype(str)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data
def get_city_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "City" not in df.columns or "AQI" not in df.columns:
        return pd.DataFrame(columns=["City", "Average AQI", "Max AQI", "Min AQI"])
    summary = (
        df.groupby("City", dropna=False)["AQI"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(columns={"mean": "Average AQI", "max": "Max AQI", "min": "Min AQI"})
    )
    return summary.sort_values("Average AQI", ascending=False).reset_index(drop=True)


@st.cache_data
def get_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns or "AQI" not in df.columns:
        return pd.DataFrame(columns=["Month", "Average AQI"])
    monthly = df.copy()
    monthly["Month"] = monthly["Date"].dt.month
    return (
        monthly.groupby("Month", as_index=False)["AQI"]
        .mean()
        .rename(columns={"AQI": "Average AQI"})
        .sort_values("Month")
    )


st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

try:
    df = load_dataset()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.title("Clean Air Climate Resilience Dashboard")
st.caption("Interactive AQI analysis for Indian cities")

city_list = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []
selected_cities = st.multiselect(
    "Choose cities",
    city_list,
    default=city_list[:5] if city_list else [],
)

if selected_cities:
    filtered_df = df[df["City"].isin(selected_cities)].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.info("No data available for the selected filters.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Cities", filtered_df["City"].nunique() if "City" in filtered_df.columns else 0)
col2.metric("Average AQI", round(float(filtered_df["AQI"].mean()), 2) if "AQI" in filtered_df.columns else 0.0)
col3.metric("Max AQI", int(filtered_df["AQI"].max()) if "AQI" in filtered_df.columns else 0)

st.subheader("City comparison")
city_summary = get_city_summary(filtered_df)
if not city_summary.empty:
    st.dataframe(city_summary, use_container_width=True)

monthly = get_monthly_trends(filtered_df)
if not monthly.empty:
    st.subheader("Monthly AQI trend")
    st.line_chart(monthly.set_index("Month")["Average AQI"], use_container_width=True)

st.subheader("AQI by city")
city_bar = city_summary[["City", "Average AQI"]].set_index("City") if not city_summary.empty else pd.DataFrame()
if not city_bar.empty:
    st.bar_chart(city_bar, use_container_width=True)

st.subheader("Pollutant relationship with AQI")
pollutants = [
    c for c in filtered_df.columns
    if c != "AQI" and c != "Date" and c != "City" and pd.api.types.is_numeric_dtype(filtered_df[c])
]
selected_pollutant = st.selectbox("Select pollutant for scatter plot", pollutants, index=0 if pollutants else None)
if selected_pollutant and "AQI" in filtered_df.columns:
    scatter_df = filtered_df[[selected_pollutant, "AQI"]].dropna()
    st.scatter_chart(scatter_df, x=selected_pollutant, y="AQI", use_container_width=True)

st.subheader("Latest AQI snapshot")
latest = filtered_df.sort_values("Date").tail(15) if "Date" in filtered_df.columns else filtered_df.head(15)
if not latest.empty:
    latest_display = latest[["Date", "City", "AQI"]] if "Date" in latest.columns else latest[["City", "AQI"]]
    st.dataframe(latest_display, use_container_width=True)

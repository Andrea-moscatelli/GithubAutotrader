import sqlite3
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import os
import json
from datetime import datetime, timedelta

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
DB_PATH = "data/db/italian_stocks.db"
SETTINGS_FILE = os.path.expanduser("~/.config/ticker_viewer/settings.json")
INTERVAL_OPTIONS = {
    "All": None,
    "Last 5 years": 5,
    "Last 2 years": 2,
    "Last 1 year": 1,
    "Last 6 months": 0.5,
    "Last 1 month": 1/12,
}

# ------------------------------------------------------
# LOAD & SAVE SETTINGS
# ------------------------------------------------------
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

# ------------------------------------------------------
# DB HELPERS
# ------------------------------------------------------
@st.cache_data(ttl=300)
def get_tables(db_path):
    conn = sqlite3.connect(db_path)
    rows = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    conn.close()
    return rows

def extract_tickers(table_list):
    tickers = set()
    for t in table_list:
        if t.startswith("t_") and "_MI_" in t:
            parts = t.split("_")
            tickers.add(parts[1])
    return sorted(list(tickers))

def table_for(ticker, freq):
    return f"t_{ticker}_MI_{freq}"

@st.cache_data(ttl=300)
def load_price_data(db_path, table, years_back):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY datetime ASC", conn)
    conn.close()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    if years_back is not None:
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=years_back*365), tz='UTC')
        df = df[df["datetime"] >= cutoff]
    return df

# ------------------------------------------------------
# MARKET HOURS FILTER (collapse nights/weekends)
# ------------------------------------------------------
def collapse_market_hours_local(df, market_start="09:00", market_end="17:30"):
    """
    Collassa notti e weekend, mantiene vuoti intra-day.
    Converte datetime in Europe/Rome per filtraggio e tooltip.
    """
    df = df.copy()

    # Assicuriamoci che datetime sia UTC-aware
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Crea colonna locale
    df["datetime_local"] = df["datetime"].dt.tz_convert("Europe/Rome")

    # Filtra solo giorni feriali (lun-ven)
    df = df[df["datetime_local"].dt.weekday < 5]

    # Filtra solo ore di mercato (09:00–17:30 locali)
    start_time = pd.to_datetime(market_start).time()
    end_time = pd.to_datetime(market_end).time()
    df = df[(df["datetime_local"].dt.time >= start_time) &
            (df["datetime_local"].dt.time <= end_time)]

    # Crea asse X continuo
    df = df.reset_index(drop=True)
    df["x_continuous"] = range(len(df))

    return df


# ------------------------------------------------------
# INDICATORS
# ------------------------------------------------------
def add_sma(df, periods):
    for p in periods:
        df[f"SMA_{p}"] = df["close"].rolling(p).mean()
    return df

def add_ema(df, periods):
    for p in periods:
        df[f"EMA_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df

def add_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    df["MACD_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["MACD_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = df["MACD_fast"] - df["MACD_slow"]
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df

# ------------------------------------------------------
# UI SETUP
# ------------------------------------------------------
st.set_page_config(page_title="Ticker DB Viewer", layout="wide")
st.title("📊 Ticker DB Viewer — Full Indicators Edition")

settings = load_settings()
tables = get_tables(DB_PATH)
all_tickers = extract_tickers(tables)

# ------------------------------------------------------
# SIDEBAR — CONTROLS
# ------------------------------------------------------
st.sidebar.header("🔎 Search & Controls")
search = st.sidebar.text_input("Search ticker", value=settings.get("search", ""))
suggestions = [t for t in all_tickers if search.upper() in t.upper()] or all_tickers
selected_tickers = st.sidebar.multiselect(
    "Select tickers", suggestions,
    default=settings.get("tickers", suggestions[:1] if suggestions else [])
)
if not selected_tickers:
    st.stop()

frequency = st.sidebar.radio("Frequency", ["2m", "1h"], index=["2m", "1h"].index(settings.get("freq", "2m")))
interval_label = st.sidebar.selectbox("Interval", list(INTERVAL_OPTIONS.keys()), index=list(INTERVAL_OPTIONS.keys()).index(settings.get("interval", "All")))
years_back = INTERVAL_OPTIONS[interval_label]

# Chart type
chart_type = st.sidebar.radio("Chart type", ["Candlestick", "Line"], index=["Candlestick", "Line"].index(settings.get("chart_type", "Candlestick")))

# Indicators
st.sidebar.subheader("📐 Indicators (overlay)")
use_sma = st.sidebar.checkbox("SMA", value=settings.get("use_sma", False))
use_ema = st.sidebar.checkbox("EMA", value=settings.get("use_ema", False))
use_rsi = st.sidebar.checkbox("RSI", value=settings.get("use_rsi", False))
use_macd = st.sidebar.checkbox("MACD", value=settings.get("use_macd", False))

# SMA/EMA parsing sicuro
sma_periods = []
if use_sma:
    for x in st.sidebar.text_input("SMA periods", value=settings.get("sma","20,50")).split(","):
        x = x.strip()
        if x.isdigit(): sma_periods.append(int(x))

ema_periods = []
if use_ema:
    for x in st.sidebar.text_input("EMA periods", value=settings.get("ema","20")).split(","):
        x = x.strip()
        if x.isdigit(): ema_periods.append(int(x))

rsi_period = st.sidebar.number_input("RSI period", min_value=5, max_value=50, value=settings.get("rsi_period",14)) if use_rsi else 14
fast = st.sidebar.number_input("MACD fast", min_value=5, max_value=50, value=settings.get("macd_fast",12)) if use_macd else 12
slow = st.sidebar.number_input("MACD slow", min_value=5, max_value=100, value=settings.get("macd_slow",26)) if use_macd else 26
signal = st.sidebar.number_input("MACD signal", min_value=3, max_value=30, value=settings.get("macd_signal",9)) if use_macd else 9

# Save settings
save_settings({
    "search": search,
    "tickers": selected_tickers,
    "freq": frequency,
    "interval": interval_label,
    "chart_type": chart_type,
    "use_sma": use_sma,
    "use_ema": use_ema,
    "use_rsi": use_rsi,
    "use_macd": use_macd,
    "sma": ",".join(map(str,sma_periods)) if sma_periods else "",
    "ema": ",".join(map(str,ema_periods)) if ema_periods else "",
    "rsi_period": rsi_period,
    "macd_fast": fast,
    "macd_slow": slow,
    "macd_signal": signal,
})

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
dfs = {}
for tk in selected_tickers:
    table = table_for(tk, frequency)
    if table not in tables: continue
    df = load_price_data(DB_PATH, table, years_back)
    if df.empty: continue

    df = collapse_market_hours_local(df)

    if use_sma: df = add_sma(df, sma_periods)
    if use_ema: df = add_ema(df, ema_periods)
    if use_rsi: df = add_rsi(df, rsi_period)
    if use_macd: df = add_macd(df, fast, slow, signal)

    dfs[tk] = df

if not dfs:
    st.error("No data available.")
    st.stop()

# ------------------------------------------------------
# PLOT
# ------------------------------------------------------
st.subheader(f"📈 Graph — {', '.join(selected_tickers)}")
primary = selected_tickers[0]
df_main = dfs[primary]

fig = go.Figure()

# Candlestick or Line
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df_main["x_continuous"],
        open=df_main["open"], high=df_main["high"],
        low=df_main["low"], close=df_main["close"],
        text=df_main["datetime_local"],  # tooltip con ora locale
        hovertemplate="%{text}<br>Open: %{open}<br>High: %{high}<br>Low: %{low}<br>Close: %{close}"
    ))
else:
    fig.add_trace(go.Scatter(
        x=df_main["x_continuous"], y=df_main["close"],
        mode="lines",
        text=df_main["datetime_local"],
        hovertemplate="%{text}<br>Close: %{y}"
    ))

# SMA/EMA overlays
for p in sma_periods:
    fig.add_trace(go.Scatter(x=df_main["x_continuous"], y=df_main[f"SMA_{p}"], mode="lines", name=f"SMA {p}"))
for p in ema_periods:
    fig.add_trace(go.Scatter(x=df_main["x_continuous"], y=df_main[f"EMA_{p}"], mode="lines", name=f"EMA {p}"))

# RSI
if use_rsi:
    fig.add_trace(go.Scatter(x=df_main["x_continuous"], y=df_main[f"RSI_{rsi_period}"], mode="lines", name=f"RSI {rsi_period}", opacity=0.6))

# MACD
if use_macd:
    fig.add_trace(go.Scatter(x=df_main["x_continuous"], y=df_main["MACD"], mode="lines", name="MACD", opacity=0.6))
    fig.add_trace(go.Scatter(x=df_main["x_continuous"], y=df_main["MACD_signal"], mode="lines", name="MACD signal", opacity=0.6))
    fig.add_trace(go.Bar(x=df_main["x_continuous"], y=df_main["MACD_hist"], name="MACD hist", opacity=0.4))

# Zoom su entrambi gli assi
fig.update_layout(dragmode="zoom")
fig.update_xaxes(fixedrange=False)
fig.update_yaxes(fixedrange=False)

st.plotly_chart(fig, width='stretch')

# Raw data
with st.expander("Raw data"):
    for tk, df in dfs.items():
        st.write(f"### {tk}")
        st.dataframe(df)

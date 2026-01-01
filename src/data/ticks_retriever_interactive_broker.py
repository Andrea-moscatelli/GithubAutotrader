import os
import re
import sqlite3
import shutil
import time
from datetime import datetime, timedelta, timezone, date

import pandas as pd
import pytz
from ib_insync import (
    IB,
    Stock,
    util
)

# =========================
# CONFIG
# =========================

DB_NAME = "italian_stocks_ib.db"
TICKERS_FILE = "milan_tickers.csv"
INTERVAL = "30m"
BAR_SIZE = "30 mins"
DAYS_TO_RETRIEVE = 365 * 10
NOT_FOUND_FILE = "not_found_tickers_30m_ib.txt"

IB_HOST = "127.0.0.1"
IB_PORT = 4002 # paper # 4001 = Live
# IB_PORT = 7496 # Live
IB_CLIENT_ID = 15

RESET = os.environ.get("RESET", "FALSE").upper()

ROME_TZ = pytz.timezone("Europe/Rome")

# =========================
# DB UTILS
# =========================

def table_name_for(ticker: str, interval: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", ticker)
    return f"t_{safe}_{interval}"

def ensure_db(data_folder: str) -> bool:
    if RESET == "TRUE" and os.path.exists(data_folder):
        shutil.rmtree(data_folder)
        print("⚠️ DB resettato")

    os.makedirs(data_folder, exist_ok=True)
    db_path = os.path.join(data_folder, DB_NAME)
    is_new = not os.path.exists(db_path)

    conn = sqlite3.connect(db_path)
    conn.close()
    return is_new

def create_table_if_not_exists(ticker: str, data_folder: str):
    table = table_name_for(ticker, INTERVAL)
    conn = sqlite3.connect(os.path.join(data_folder, DB_NAME))
    c = conn.cursor()

    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            datetime TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    """)

    c.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table}_dt
        ON {table}(datetime)
    """)

    conn.commit()
    conn.close()

def get_last_timestamp(ticker: str, data_folder: str):
    table = table_name_for(ticker, INTERVAL)
    conn = sqlite3.connect(os.path.join(data_folder, DB_NAME))
    c = conn.cursor()
    try:
        c.execute(f"SELECT MAX(datetime) FROM {table}")
        v = c.fetchone()[0]
    except sqlite3.OperationalError:
        v = None
    conn.close()

    if not v:
        return None
    return pd.to_datetime(v, utc=True)

def add_to_db(df: pd.DataFrame, ticker: str, data_folder: str):
    if df.empty:
        return

    create_table_if_not_exists(ticker, data_folder)
    table = table_name_for(ticker, INTERVAL)

    rows = []
    for _, r in df.iterrows():
        rows.append((
            r["datetime"].isoformat(),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            int(r["volume"])
        ))

    conn = sqlite3.connect(os.path.join(data_folder, DB_NAME))
    c = conn.cursor()
    c.executemany(
        f"""INSERT OR IGNORE INTO {table}
            (datetime, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?)""",
        rows
    )
    conn.commit()
    conn.close()

# =========================
# FILE UTILS
# =========================

def load_tickers_file(data_folder: str) -> list:
    path = os.path.join(data_folder, TICKERS_FILE)
    if not os.path.exists(path):
        return []
    return pd.read_csv(path)["symbol"].dropna().unique().tolist()

def load_not_found(data_folder: str) -> set:
    path = os.path.join(data_folder, NOT_FOUND_FILE)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(x.strip() for x in f if x.strip())

def save_not_found(data_folder: str, s: set):
    path = os.path.join(data_folder, NOT_FOUND_FILE)
    with open(path, "w") as f:
        for x in sorted(s):
            f.write(x + "\n")

# =========================
# IB UTILS
# =========================

def connect_ib() -> IB:
    ib = IB()
    ib.connect(
        IB_HOST,
        IB_PORT,
        clientId=IB_CLIENT_ID,
        timeout=10
    )
    return ib

def ib_contract_from_ticker(ticker: str) -> Stock:
    symbol = ticker.replace(".MI", "")
    return Stock(
        symbol=symbol,
        exchange="BVME",
        currency="EUR",
        primaryExchange="BVME"
    )

# =========================
# DOWNLOAD CORE
# =========================

def download_ib_30m(
    ib: IB,
    ticker: str,
    start: datetime,
    end: datetime
) -> pd.DataFrame | None:

    contract = ib_contract_from_ticker(ticker)
    ib.qualifyContracts(contract)

    all_parts = []
    cur_end = end

    while cur_end > start:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=cur_end,
            durationStr="6 M",
            barSizeSetting=BAR_SIZE,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1
        )

        if not bars:
            break

        df = util.df(bars)
        if df.empty:
            break

        all_parts.append(df)
        cur_end = df["date"].min() - timedelta(minutes=30)

        time.sleep(1)  # rate limit safety

    if not all_parts:
        return None

    out = pd.concat(all_parts).drop_duplicates(subset=["date"])
    out = out.sort_values("date")

    out["datetime"] = (
        out["date"]
        .dt.tz_localize(ROME_TZ, ambiguous="infer")
        .dt.tz_convert(timezone.utc)
    )

    return pd.DataFrame({
        "datetime": out["datetime"],
        "open": out["open"],
        "high": out["high"],
        "low": out["low"],
        "close": out["close"],
        "volume": out["volume"]
    })

# =========================
# UPSERT LOGIC
# =========================

def upsert_ticker(
    ib: IB,
    ticker: str,
    is_first_run: bool,
    data_folder: str
) -> bool:

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_TO_RETRIEVE)

    last_ts = None
    if not is_first_run:
        last_ts = get_last_timestamp(ticker, data_folder)
        if last_ts:
            start = last_ts + timedelta(minutes=30)

    df = download_ib_30m(ib, ticker, start, end)

    if df is None or df.empty:
        if not last_ts or (end - last_ts).days > DAYS_TO_RETRIEVE:
            return False
        return True

    add_to_db(df, ticker, data_folder)
    print(f"✅ {ticker}: {len(df)} barre 30m")
    return True

# =========================
# MAIN
# =========================

def main(data_folder: str):
    is_first_run = ensure_db(data_folder)
    tickers = load_tickers_file(data_folder)

    if not tickers:
        print("⚠️ Nessun ticker trovato")
        return

    not_found = load_not_found(data_folder)
    ib = connect_ib()

    try:
        for t in (x for x in tickers if x not in not_found):
            try:
                ok = upsert_ticker(ib, t, is_first_run, data_folder)
                if not ok:
                    not_found.add(t)
            except Exception as e:
                print(f"❌ {t}: {e}")
                not_found.add(t)
    finally:
        ib.disconnect()

    save_not_found(data_folder, not_found)
    print("🏁 Aggiornamento completato")

# =========================

if __name__ == "__main__":
    main("../../data/db")

import os
import re
import sqlite3
import shutil
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz
from ib_insync import IB, Stock, util

# =========================
# CONFIG
# =========================

DB_NAME = "market_data.db"
TICKERS_FILE = "milan_tickers_for_IB.csv"

INTERVAL_MINS = 30
INTERVAL = f"{INTERVAL_MINS}m"
BAR_SIZE = f"{INTERVAL_MINS} mins"

DAYS_TO_RETRIEVE = 365 * 10

IB_HOST = "127.0.0.1"
IB_PORT = 4002          # 4001 live / 4002 paper
IB_CLIENT_ID = 15

RESET = os.environ.get("RESET", "FALSE").upper()

ROME_TZ = pytz.timezone("Europe/Rome")

# =========================
# DB HELPERS
# =========================

def db_path(data_folder: str) -> str:
    return os.path.join(data_folder, DB_NAME)

def bars_table(conId: int, interval: str) -> str:
    return f"bars_{interval}_{conId}"

# =========================
# DB INIT
# =========================

def ensure_db(data_folder: str) -> bool:
    """Crea DB e tabelle necessarie"""
    if RESET == "TRUE" and os.path.exists(data_folder):
        shutil.rmtree(data_folder)
        print("⚠️ DB resettato")

    os.makedirs(data_folder, exist_ok=True)
    path = db_path(data_folder)
    is_new = not os.path.exists(path)

    conn = sqlite3.connect(path)
    c = conn.cursor()

    # strumenti IB
    c.execute("""
        CREATE TABLE IF NOT EXISTS instruments (
            conId INTEGER PRIMARY KEY,
            symbol TEXT,
            exchange TEXT,
            primaryExchange TEXT,
            currency TEXT,
            secType TEXT,
            localSymbol TEXT,
            UNIQUE(symbol, primaryExchange, secType, currency)
        )
    """)

    # stato ticker
    c.execute("""
        CREATE TABLE IF NOT EXISTS tickers_status (
            conId INTEGER PRIMARY KEY,
            symbol TEXT,
            exchange TEXT,
            status TEXT,
            last_checked TEXT,
            note TEXT
        )
    """)

    # stato run
    c.execute("""
        CREATE TABLE IF NOT EXISTS run_state (
            id INTEGER PRIMARY KEY,
            last_conId INTEGER
        )
    """)

    conn.commit()
    conn.close()
    return is_new

# =========================
# INSTRUMENTS
# =========================

def upsert_instrument(contract, data_folder: str):
    conn = sqlite3.connect(db_path(data_folder))
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO instruments
        (conId, symbol, exchange, primaryExchange, currency, secType, localSymbol)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        contract.conId,
        contract.symbol,
        contract.exchange,
        contract.primaryExchange,
        contract.currency,
        contract.secType,
        contract.localSymbol
    ))
    conn.commit()
    conn.close()

# =========================
# BARS TABLE
# =========================

def ensure_bars_table(conId: int, interval:str, data_folder: str):
    table = bars_table(conId=conId, interval=interval)
    conn = sqlite3.connect(db_path(data_folder))
    c = conn.cursor()

    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            datetime TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (datetime)
        )
    """)

    c.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table}_dt
        ON {table} (datetime)
    """)

    conn.commit()
    conn.close()

def last_bar_timestamp(conId: int, interval: str, data_folder: str):
    table = bars_table(conId=conId, interval=interval)
    conn = sqlite3.connect(db_path(data_folder))
    c = conn.cursor()
    try:
        c.execute(f"""
            SELECT MAX(datetime)
            FROM {table}
            WHERE conId = ?
        """, (conId,))
        v = c.fetchone()[0]
    except sqlite3.OperationalError:
        v = None
    conn.close()
    return pd.to_datetime(v, utc=True) if v else None


def insert_bars(df: pd.DataFrame, conId: int, interval: str, data_folder: str):
    if df.empty:
        return

    ensure_bars_table(conId, interval, data_folder)
    table = bars_table(conId=conId, interval=interval)

    rows = [
        (
            r["datetime"].isoformat(),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            int(r["volume"])
        )
        for _, r in df.iterrows()
    ]

    conn = sqlite3.connect(db_path(data_folder))
    c = conn.cursor()
    c.executemany(
        f"""INSERT OR IGNORE INTO {table}
            (datetime, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)""",
        rows
    )
    conn.commit()
    conn.close()

# =========================
# TICKER STATUS
# =========================

def set_ticker_status(conId, symbol, exchange, status, note, data_folder):
    conn = sqlite3.connect(db_path(data_folder))
    c = conn.cursor()
    c.execute("""
        INSERT INTO tickers_status
        (conId, symbol, exchange, status, last_checked, note)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conId) DO UPDATE SET
            status=excluded.status,
            last_checked=excluded.last_checked,
            note=excluded.note
    """, (
        conId, symbol, exchange,
        status, datetime.utcnow().isoformat(), note
    ))
    conn.commit()
    conn.close()

def invalid_conIds(data_folder: str) -> set[int]:
    conn = sqlite3.connect(db_path(data_folder))
    c = conn.cursor()
    c.execute("SELECT conId FROM tickers_status WHERE status='INVALID'")
    out = {r[0] for r in c.fetchall()}
    conn.close()
    return out

# =========================
# RUN STATE
# =========================

def last_processed_conId(data_folder: str):
    # TODO rivedere logica per ripartire da dove si è arrivati
    conn = sqlite3.connect(db_path(data_folder))
    c = conn.cursor()
    c.execute("SELECT last_conId FROM run_state WHERE id=1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def set_last_processed_conId(conId: int, data_folder: str):
    conn = sqlite3.connect(db_path(data_folder))
    c = conn.cursor()
    c.execute("""
        INSERT INTO run_state (id, last_conId)
        VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET last_conId=excluded.last_conId
    """, (conId,))
    conn.commit()
    conn.close()

# =========================
# IB
# =========================

def connect_ib() -> IB:
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
    return ib

def contract_from_row(row) -> Stock:
    return Stock(
        symbol=row["symbol"],
        exchange=row["exchange"],
        currency=row["currency"],
        primaryExchange=row["primaryExchange"]
    )

def error_handler_factory(data_folder):
    def handler(reqId, errorCode, errorString, contract):
        if errorCode == 200 and contract and contract.conId:
            set_ticker_status(
                contract.conId,
                contract.symbol,
                contract.exchange,
                "INVALID",
                "IB Error 200",
                data_folder
            )
    return handler

# =========================
# DOWNLOAD
# =========================

def download_history(
        ib: IB,
        contract,
        start: datetime,
        end: datetime
) -> pd.DataFrame | None:

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
        cur_end = df["date"].min() - timedelta(minutes=INTERVAL_MINS)
        time.sleep(1)  # rate limit

    if not all_parts:
        return None

    df = pd.concat(all_parts).drop_duplicates("date").sort_values("date")
    df["datetime"] = (
        df["date"]
        .dt.tz_localize(ROME_TZ, ambiguous="infer")
        .dt.tz_convert(timezone.utc)
    )

    return df[["datetime", "open", "high", "low", "close", "volume"]]

# =========================
# MAIN LOGIC
# =========================

def main(data_folder: str):
    is_first_run = ensure_db(data_folder)

    tickers = pd.read_csv(os.path.join(data_folder, "..", "tickers", TICKERS_FILE)).to_dict("records")
    if not tickers:
        print("⚠️ Nessun ticker")
        return

    ib = connect_ib()
    ib.errorEvent += error_handler_factory(data_folder)

    invalid = invalid_conIds(data_folder)
    last_conId = last_processed_conId(data_folder)

    try:
        for row in tickers:
            contract = contract_from_row(row)
            ib.qualifyContracts(contract)

            conId = contract.conId
            if conId in invalid:
                continue

            upsert_instrument(contract, data_folder)

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=DAYS_TO_RETRIEVE)

            if not is_first_run:
                last_ts = last_bar_timestamp(
                    conId=conId, interval=INTERVAL, data_folder=data_folder
                )
                if last_ts:
                    start = last_ts + timedelta(minutes=INTERVAL_MINS)

            try:
                df = download_history(ib, contract, start, end)
            except Exception as e:
                print(f"⚠️ {contract.symbol}: {e}")
                continue

            if df is not None and not df.empty:
                insert_bars(df=df, conId=conId, interval=INTERVAL, data_folder=data_folder)
                set_ticker_status(
                    conId, contract.symbol, contract.exchange,
                    "VALID", f"{len(df)} bars", data_folder
                )
                print(f"✅ {contract.symbol}: {len(df)} bars")

            set_last_processed_conId(conId, data_folder)

    finally:
        ib.disconnect()

    print("🏁 Completato")

# =========================
if __name__ == "__main__":
    main("../../data/db")

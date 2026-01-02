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

DB_NAME = "italian_stocks_ib.db"
TICKERS_FILE = "milan_tickers.csv"
INTERVAL_MINS = 30
INTERVAL = f'{INTERVAL_MINS}m'
BAR_SIZE = f'{INTERVAL_MINS} mins'
DAYS_TO_RETRIEVE = 365 * 10

IB_HOST = "127.0.0.1"
IB_PORT = 4002  # Paper Trading # 4001 = Live
IB_CLIENT_ID = 15
RESET = os.environ.get("RESET", "FALSE").upper()

ROME_TZ = pytz.timezone("Europe/Rome")

# =========================
# DB UTILS
# =========================

def table_name_for(ticker: str, interval: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", ticker)
    return f"t_{safe}_BVME_{interval}"

def ensure_db(data_folder: str) -> bool:
    """Crea DB e tabelle necessarie"""
    if RESET == "TRUE" and os.path.exists(data_folder):
        shutil.rmtree(data_folder)
        print("⚠️ DB resettato")

    os.makedirs(data_folder, exist_ok=True)
    db_path = os.path.join(data_folder, DB_NAME)
    is_new = not os.path.exists(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Tabella ticker validi/invalidi
    c.execute("""
        CREATE TABLE IF NOT EXISTS tickers_status (
            symbol TEXT PRIMARY KEY,
            status TEXT,
            last_checked TEXT,
            note TEXT
        )
    """)
    # Tabella per salvare lo stato del run
    c.execute("""
        CREATE TABLE IF NOT EXISTS run_state (
            id INTEGER PRIMARY KEY,
            last_ticker TEXT
        )
    """)
    conn.commit()
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
# TICKER STATUS UTILS
# =========================

def set_ticker_status(data_folder: str, ticker: str, status: str, note: str = None):
    conn = sqlite3.connect(os.path.join(data_folder, DB_NAME))
    c = conn.cursor()
    c.execute("""
        INSERT INTO tickers_status (symbol, status, last_checked, note)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            status=excluded.status,
            last_checked=excluded.last_checked,
            note=excluded.note
    """, (ticker, status, datetime.utcnow().isoformat(), note))
    conn.commit()
    conn.close()

def get_invalid_tickers(data_folder: str):
    conn = sqlite3.connect(os.path.join(data_folder, DB_NAME))
    c = conn.cursor()
    c.execute("SELECT symbol FROM tickers_status WHERE status='INVALID'")
    rows = [r[0] for r in c.fetchall()]
    conn.close()
    return set(rows)

# =========================
# RUN STATE UTILS
# =========================

def load_tickers_file(data_folder: str) -> list:
    path = os.path.join(data_folder, TICKERS_FILE)
    if not os.path.exists(path):
        return []
    return pd.read_csv(path)["symbol"].dropna().unique().tolist()

def get_last_processed_ticker(data_folder: str) -> str | None:
    conn = sqlite3.connect(os.path.join(data_folder, DB_NAME))
    c = conn.cursor()
    c.execute("SELECT last_ticker FROM run_state WHERE id=1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def set_last_processed_ticker(data_folder: str, ticker: str):
    conn = sqlite3.connect(os.path.join(data_folder, DB_NAME))
    c = conn.cursor()
    c.execute("""
        INSERT INTO run_state (id, last_ticker)
        VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET last_ticker=excluded.last_ticker
    """, (ticker,))
    conn.commit()
    conn.close()

# =========================
# IB UTILS
# =========================

def connect_ib() -> IB:
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
    return ib

def ib_contract_from_ticker(ticker: str) -> Stock:
    # symbol = ticker.replace(".MI", "")
    symbol = ticker
    return Stock(symbol=symbol, exchange="BVME", currency="EUR", primaryExchange="BVME")

# =========================
# DOWNLOAD CORE
# =========================

def invalid_ticker_identifier(data_folder):
    def error_handler(reqId, errorCode, errorString, contract):
        # errore 200 = Unknown contract
        if errorCode == 200:
            set_ticker_status(data_folder, contract.symbol, "INVALID", "Error 200")
    return error_handler

def download_ib_data(ib: IB, ticker: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    contract = ib_contract_from_ticker(ticker)

    ib.qualifyContracts(contract)
    # try:
    #     ib.qualifyContracts(contract)
    # except Exception as e:
    #     if "200" in str(e):
    #         raise ValueError("INVALID_CONTRACT") from e
    #     else:
    #         raise

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

    out = pd.concat(all_parts).drop_duplicates(subset=["date"])
    out = out.sort_values("date")
    out["datetime"] = out["date"].dt.tz_localize(ROME_TZ, ambiguous="infer").dt.tz_convert(timezone.utc)
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

def upsert_ticker(ib: IB, ticker: str, is_first_run: bool, data_folder: str) -> bool:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_TO_RETRIEVE)

    if not is_first_run:
        last_ts = get_last_timestamp(ticker, data_folder)
        if last_ts:
            start = last_ts + timedelta(minutes=INTERVAL_MINS)

    try:
        df = download_ib_data(ib, ticker, start, end)
    # except ValueError as ve:
    #     if str(ve) == "INVALID_CONTRACT":
    #         set_ticker_status(data_folder, ticker, "INVALID", "Error 200")
    #         print(f"❌ {ticker}: ticker non valido / delisted")
    #         return False
    #     else:
    #         raise
    except Exception as e:
        print(f"⚠️ {ticker}: errore temporaneo {e}")
        return True  # retry next run

    if df is None or df.empty:
        return True

    add_to_db(df, ticker, data_folder)
    set_ticker_status(data_folder, ticker, "VALID", f"{len(df)} barre {INTERVAL_MINS}m")
    print(f"✅ {ticker}: {len(df)} barre {INTERVAL_MINS}m")
    return True

# =========================
# MAIN
# =========================

def main(data_folder: str):
    is_first_run = ensure_db(data_folder)
    tickers = load_tickers_file(data_folder)
    tickers = sorted(tickers)
    tickers = [t.replace(".MI", "") for t in tickers]

    if not tickers:
        print("⚠️ Nessun ticker trovato")
        return

    ib = connect_ib()
    ib.errorEvent += invalid_ticker_identifier(data_folder)
    # valid_tickers = get_valid_tickers(data_folder)
    invalid_tickers = get_invalid_tickers(data_folder)

    # Recupera ticker da cui ripartire
    last_ticker = get_last_processed_ticker(data_folder)
    if last_ticker and last_ticker in tickers:
        last_idx = tickers.index(last_ticker)
        tickers = tickers[last_idx+1:] + tickers[:last_idx+1]  # rotazione per ripartire dal successivo
    else:
        tickers = tickers  # parte dall'inizio


    try:
        for t in tickers:
            if t in invalid_tickers:
                continue
            # if t in valid_tickers and not is_first_run:
            #     continue
            try:
                upsert_ticker(ib, t, is_first_run, data_folder)
            except Exception as e:
                print(f"❌ {t}: errore imprevisto {e}")
            finally:
                # aggiorna ticker ultimo processato
                set_last_processed_ticker(data_folder, t)
    finally:
        ib.disconnect()

    print("🏁 Aggiornamento completato")

# =========================
if __name__ == "__main__":
    main("../../data/db")

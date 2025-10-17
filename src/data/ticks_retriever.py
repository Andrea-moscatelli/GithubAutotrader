import yfinance as yf
import pandas as pd
import sqlite3
import datetime
import os

DB_PATH = "data/italian_stocks.db"


def get_italian_tickers():
    """
    Restituisce la lista dei principali titoli di Borsa Italiana in formato Yahoo Finance.
    """
    return [
        "ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI", "LUX.MI",
        "STM.MI", "ATL.MI", "TEN.MI", "MONC.MI", "REC.MI"
    ]


def ensure_db():
    """Crea il database se non esiste e ritorna True se era nuovo."""
    is_new = not os.path.exists(DB_PATH)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            ticker TEXT,
            datetime TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, datetime)
        )
    """)
    conn.commit()
    conn.close()
    return is_new


def get_last_timestamp(ticker: str):
    """Restituisce l'ultimo timestamp disponibile nel DB per un ticker."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(datetime) FROM stock_data WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()[0]
    conn.close()
    if result:
        return datetime.datetime.fromisoformat(result)
    return None


def download_data(ticker: str, start: datetime.datetime, end: datetime.datetime):
    """Scarica i dati a 5 minuti per il ticker nel range indicato."""
    print(f"📥 Scarico dati per {ticker} da {start.date()} a {end.date()}...")
    df = yf.download(
        ticker,
        interval="5m",
        start=start,
        end=end,
        progress=False
    )
    if df.empty:
        print(f"⚠️ Nessun dato per {ticker}")
        return None

    df.reset_index(inplace=True)
    df.rename(columns={
        "Datetime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    df["ticker"] = ticker
    df["datetime"] = df["datetime"].astype(str)
    return df


def save_to_db(df: pd.DataFrame):
    """Salva i dati nel database."""
    if df is None or df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("stock_data", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def update_ticker_data(ticker: str, is_first_run: bool):
    """Aggiorna o inizializza i dati per un singolo ticker."""
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=59)  # ultimi 59 giorni

    if not is_first_run:
        last_ts = get_last_timestamp(ticker)
        if last_ts and last_ts < end - datetime.timedelta(minutes=5):
            start = last_ts
        else:
            print(f"✅ {ticker}: dati già aggiornati")
            return

    df = download_data(ticker, start, end)
    save_to_db(df)
    print(f"✅ {ticker}: aggiornato con {len(df) if df is not None else 0} righe.")


def main():
    is_first_run = ensure_db()
    tickers = get_italian_tickers()

    print("🚀 Avvio aggiornamento dati azioni italiane")
    if is_first_run:
        print("📂 Database nuovo: scarico ultimi 6 mesi completi...")
    else:
        print("📈 Database esistente: aggiorno solo nuovi dati...")

    for t in tickers:
        try:
            update_ticker_data(t, is_first_run)
        except Exception as e:
            print(f"❌ Errore con {t}: {e}")

    print("🏁 Aggiornamento completato.")


if __name__ == "__main__":
    main()

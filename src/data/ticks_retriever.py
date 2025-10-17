# python
import yfinance as yf
import pandas as pd
import sqlite3
import datetime
import os
import re

DB_PATH = "data/italian_stocks.db"


def get_italian_tickers():
    return [
        "ENEL.MI"
        # , "ENI.MI", "ISP.MI", "UCG.MI", "LUX.MI",
        # "STM.MI", "ATL.MI", "TEN.MI", "MONC.MI", "REC.MI"
    ]


def table_name_for(ticker: str) -> str:
    """Genera un nome tabella sicuro da un ticker (es: ENEL.MI -> t_ENEL_MI)."""
    safe = re.sub(r'[^0-9A-Za-z_]', '_', ticker)
    return f"t_{safe}"


def ensure_db():
    """Crea la cartella del DB se non esiste. Restituisce True se il DB è nuovo."""
    is_new = not os.path.exists(DB_PATH)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    # apri e chiudi per creare il file se mancante
    conn = sqlite3.connect(DB_PATH)
    conn.close()
    return is_new


def create_table_for_ticker(ticker: str):
    """Crea la tabella specifica per il ticker se non esiste."""
    table = table_name_for(ticker)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            datetime TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    """)
    conn.commit()
    conn.close()


def get_last_timestamp(ticker: str):
    """Restituisce l'ultimo timestamp disponibile nella tabella del ticker o None."""
    table = table_name_for(ticker)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT MAX(datetime) FROM {table}")
        result = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        # tabella inesistente
        result = None
    conn.close()
    if result:
        return datetime.datetime.fromisoformat(result)
    return None


import pandas as pd


def normalize_yf_df(df, default_ticker=None):
    """
    Normalizza un DataFrame restituito da yf.download() in formato coerente:
    colonne ['datetime', 'open', 'high', 'low', 'close', 'volume'].

    Gestisce automaticamente:
    - colonne singole (es. 'Open', 'High', ...)
    - colonne MultiIndex (es. ('ENEL.MI','Open'))
    - colonne flat con suffissi (es. 'Open_ENEL.MI' o 'ENI.MI_Open')
    """
    df = df.copy()

    # Assicura che l'indice sia datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # === Caso 1: colonne MultiIndex ===
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        fields_set = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        # Struttura: (field, ticker)
        if len(fields_set & lvl0) > 0:
            long = df.stack(level=1).reset_index()
            long = long.rename(columns={"level_0": "datetime", "level_1": "ticker"})
        # Struttura: (ticker, field)
        else:
            long = df.stack(level=0).reset_index()
            long = long.rename(columns={"level_0": "datetime", "level_1": "ticker"})

        # Rinomina colonne in formato uniforme
        col_map = {
            "Adj Close": "adj_close",
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        long = long.rename(columns={k: v for k, v in col_map.items() if k in long.columns})

        # Mantieni solo colonne rilevanti
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in long.columns:
                long[c] = None

        return long[["datetime", "open", "high", "low", "close", "volume"]]

    # === Caso 2: colonne single-level ===
    df2 = df.reset_index()
    if df2.columns[0] != "datetime":
        df2 = df2.rename(columns={df2.columns[0]: "datetime"})

    cols = df2.columns.tolist()
    fields_set = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

    # Caso: singolo ticker (colonne semplici)
    if any(c in fields_set for c in cols):
        ticker = default_ticker or "SINGLE"
        out = df2[["datetime"]].copy()
        # out["ticker"] = ticker
        out["open"] = df2.get("Open")
        out["high"] = df2.get("High")
        out["low"] = df2.get("Low")
        out["close"] = df2.get("Close")
        out["volume"] = df2.get("Volume")
        return out[["datetime", "open", "high", "low", "close", "volume"]]

    # Caso: colonne flat con suffissi field_ticker o ticker_field
    suf_map = {}
    for col in cols:
        if col == "datetime":
            continue
        parts = col.rsplit("/", 1)
        if len(parts) == 2:
            a, b = parts
            if a in fields_set:
                field, ticker = a, b
            elif b in fields_set:
                ticker, field = a, b
            else:
                continue
            suf_map.setdefault(ticker, {})[field] = col

    out_rows = []
    for _, row in df2.iterrows():
        for t, fmap in suf_map.items():
            out_rows.append({
                "datetime": row["datetime"],
                # "ticker": t,
                "open": row.get(fmap.get("Open")),
                "high": row.get(fmap.get("High")),
                "low": row.get(fmap.get("Low")),
                "close": row.get(fmap.get("Close")),
                "volume": row.get(fmap.get("Volume")),
            })

    out = pd.DataFrame(out_rows)
    return out[["datetime", "open", "high", "low", "close", "volume"]]


def download_data(ticker: str, start: datetime.datetime, end: datetime.datetime):
    """Scarica i dati a 5 minuti per il ticker nel range indicato."""
    print(f"📥 Scarico dati per {ticker} da {start} a {end}...")
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

    df_clean = normalize_yf_df(df, default_ticker=ticker)

    # assicurarsi che 'datetime' sia datetime e poi stringa ISO (compatibile con sqlite)
    try:
        df_clean["datetime"] = pd.to_datetime(df_clean["datetime"], errors="coerce")
    except Exception:
        pass
    df_clean["datetime"] = df_clean["datetime"].apply(
        lambda x: x.isoformat() if (hasattr(x, "isoformat")) and pd.notna(x) else str(x)
    )

    return df_clean[["datetime", "open", "high", "low", "close", "volume"]]


def save_to_db(df: pd.DataFrame, ticker: str):
    """Salva i dati nella tabella specifica del ticker usando INSERT OR IGNORE.
    Arrotonda i valori float a 3 decimali."""
    if df is None or df.empty:
        return
    create_table_for_ticker(ticker)
    table = table_name_for(ticker)

    def to_iso(val):
        if pd.isna(val):
            return None
        if isinstance(val, (pd.Timestamp, datetime.datetime)):
            return val.isoformat()
        return str(val)

    def to_float3(v):
        if pd.isna(v) or v is None:
            return None
        try:
            return round(float(v), 3)
        except Exception:
            return None

    rows = []
    for _, row in df.iterrows():
        dt = to_iso(row.get("datetime"))
        open_v = to_float3(row.get("open"))
        high_v = to_float3(row.get("high"))
        low_v = to_float3(row.get("low"))
        close_v = to_float3(row.get("close"))
        vol = None if pd.isna(row.get("volume")) else int(row.get("volume"))
        rows.append((dt, open_v, high_v, low_v, close_v, vol))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany(
        f"INSERT OR IGNORE INTO {table} (datetime, open, high, low, close, volume) VALUES (?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany(
        f"INSERT OR IGNORE INTO {table} (datetime, open, high, low, close, volume) VALUES (?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()


def update_ticker_data(ticker: str, is_first_run: bool):
    """Aggiorna o inizializza i dati per un singolo ticker (tabella separata)."""
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=59)  # ultimi 59 giorni

    if not is_first_run:
        last_ts = get_last_timestamp(ticker)
        if not last_ts:
            print(f"⚠️ {ticker}: nessun dato trovato, scarico ultimi 59 giorni mesi completi.")
            pass
        elif last_ts and last_ts < end - datetime.timedelta(minutes=5):
            # iniziare da un minuto dopo l'ultimo timestamp per evitare duplicati
            start = last_ts + datetime.timedelta(minutes=5)
        else:
            print(f"✅ {ticker}: dati già aggiornati")
            return

    df = download_data(ticker, start, end)
    save_to_db(df, ticker)
    print(f"✅ {ticker}: processate {len(df) if df is not None else 0} righe.")


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

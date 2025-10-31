import shutil

import yfinance as yf
import sqlite3
from datetime import datetime, timezone, date, timedelta
import re

import os
import pandas as pd
from playwright.sync_api import sync_playwright

# DB_PATH = "db/italian_stocks.db"
DB_NAME = "italian_stocks.db"
MILAN_TICKERS_FILE_NAME = "milan_tickers.csv"
BASE_URL = "https://live.euronext.com/en/markets/milan/equities/list?page={}"
TICKS_INTERVAL = "5m"
DAYS_TO_RETRIEVE = 59

NOT_WORKING_TICKERS = f"not_found_tickers_for_interval_{TICKS_INTERVAL}.txt"


def load_not_found_tickers(db_folder):
    try:
        with open(os.path.join(db_folder, NOT_WORKING_TICKERS), "r") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


# funzione per salvare il set su file
def save_not_found_tickers(db_folder, tickers_set):
    with open(os.path.join(db_folder, NOT_WORKING_TICKERS), "w") as f:
        for ticker in tickers_set:
            f.write(f"{ticker}\n")


# def get_italian_tickers(headless=True, delay=1.0):
#     """
#     Usa Selenium per navigare tra le pagine del listino A-Z di Borsa Italiana.
#     Cicla su 'page' fino a quando la pagina corrente è uguale alla precedente.
#     Restituisce una lista di ticker con suffisso '.MI'.
#     """
#     base_url = "https://www.borsaitaliana.it/borsa/azioni/listino-a-z.html?page={}&lang=it"
#     tickers = set()
#     prev_page_tickers = set()
#
#     # Configura Chrome headless
#     options = Options()
#     if headless:
#         options.add_argument("--headless=new")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#
#     driver = webdriver.Chrome(options=options)
#
#     page = 1
#     while True:
#         url = base_url.format(page)
#         driver.get(url)
#         time.sleep(delay)
#
#         html = driver.page_source
#         soup = BeautifulSoup(html, "html.parser")
#         table = soup.find("table")
#         if not table:
#             print(f"⚠️ Nessuna tabella trovata a pagina {page}")
#             break
#
#         current_page_tickers = set()
#         for tr in table.find_all("tr"):
#             cols = tr.find_all("td")
#             if not cols:
#                 continue
#             a = cols[0].find("a")
#             if a and a.text.strip():
#                 t = a.text.strip().upper()
#                 if not t.endswith(".MI"):
#                     t += ".MI"
#                 current_page_tickers.add(t)
#
#         # Se la pagina è identica alla precedente, abbiamo finito
#         if current_page_tickers == prev_page_tickers or not current_page_tickers:
#             print(f"✅ Fine raggiunta a pagina {page}.")
#             break
#
#         tickers.update(current_page_tickers)
#         prev_page_tickers = current_page_tickers
#         page += 1
#
#     driver.quit()
#     return sorted(tickers)


# if __name__ == "__main__":
#     all_tickers = fetch_borsa_italiana_tickers_selenium()
#     print(f"Totale tickers trovati: {len(all_tickers)}")
#     print(all_tickers[:20])


# def get_italian_tickers():
#     prendile da qui https://live.euronext.com/en/markets/milan/equities/list
#     return [
#         # "ENEL.MI"
#         # "1AXP.MI",
#         # "2ADBE.MI"
#         "BMPS.MI"
#         # , "ENI.MI", "ISP.MI", "UCG.MI", "LUX.MI",
#         # "STM.MI", "ATL.MI", "TEN.MI", "MONC.MI", "REC.MI"
#     ]


def scrape_milan_stocks(db_folder):
    os.makedirs(db_folder, exist_ok=True)
    output_csv = os.path.join(db_folder, MILAN_TICKERS_FILE_NAME)

    all_tickers = set()
    prev_tickers = set()
    data = []
    page_num = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        while True:
            url = BASE_URL.format(page_num)
            print(f"Caricamento pagina: {url}")
            page.goto(url, wait_until="networkidle", timeout=20000)
            # page.wait_for_selector("table#stocks-data-table-es")
            page.wait_for_selector("table#stocks-data-table-es", state="attached")
            rows = page.query_selector_all("table#stocks-data-table-es > tbody > tr")

            current_tickers = set()
            for r in rows:
                name = r.query_selector(".stocks-name a")
                isin = r.query_selector(".stocks-isin")
                symbol = r.query_selector(".stocks-symbol")
                market = r.query_selector(".stocks-market div")
                last_price = r.query_selector(".stocks-lastPrice .pd_last_price_es")
                pct_change = r.query_selector(".stocks-precentDayChange .pd_percent span")
                last_trade_time = r.query_selector(".stocks-lastTradeTime div")

                ticker = symbol.inner_text().strip() if symbol else None
                if ticker:
                    current_tickers.add(ticker)
                    data.append({
                        "name": name.inner_text().strip() if name else None,
                        "isin": isin.inner_text().strip() if isin else None,
                        "symbol": ticker,
                        "market": market.inner_text().strip() if market else None,
                        "last_price": last_price.inner_text().strip() if last_price else None,
                        "percent_change": pct_change.inner_text().strip() if pct_change else None,
                        "last_trade_time": last_trade_time.inner_text().strip() if last_trade_time else None
                    })
                    print(f"Trovato titolo: {ticker} - {data[-1]['name']} - {data[-1]['isin']}")

            # Se non ci sono nuovi ticker, termina
            if not current_tickers or current_tickers == prev_tickers:
                print(f"✅ Fine raggiunta a pagina {page_num}.")
                break

            all_tickers.update(current_tickers)
            prev_tickers = current_tickers
            page_num += 1
            break

        browser.close()

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Trovati {len(df)} titoli. Salvato in '{output_csv}'.")

    tickers = []
    for sym in df["symbol"].dropna().unique():
        if not sym.endswith(".MI"):
            sym += ".MI"
        tickers.append(sym)
    return tickers


def table_name_for(ticker: str) -> str:
    """Genera un nome tabella sicuro da un ticker (es: ENEL.MI -> t_ENEL_MI)."""
    safe = re.sub(r'[^0-9A-Za-z_]', '_', ticker)
    return f"t_{safe}"


def ensure_db(db_folder):
    """Crea la cartella del DB se non esiste. Restituisce True se il DB è nuovo."""
    # rimuovi il file DB_PATH
    if os.environ.get("MARKET", "FALSE") == "TRUE" and os.path.exists(db_folder):
        shutil.rmtree(db_folder)
        os.makedirs(db_folder)
        print("!!!Dati app resettati!!!")

    db_file_path = os.path.join(db_folder, DB_NAME)
    is_new = not os.path.exists(db_file_path)
    os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
    conn = sqlite3.connect(db_file_path)
    conn.close()
    return is_new


def create_table_for_ticker_if_not_exists(ticker: str, db_folder):
    """Crea la tabella specifica per il ticker se non esiste."""
    table = table_name_for(ticker)
    db_file_path = os.path.join(db_folder, DB_NAME)
    conn = sqlite3.connect(db_file_path)
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


def get_last_timestamp(ticker: str, db_folder):
    """Restituisce l'ultimo timestamp disponibile nella tabella del ticker o None."""
    table = table_name_for(ticker)
    db_file_path = os.path.join(db_folder, DB_NAME)
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT MAX(datetime) FROM {table}")
        result = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        result = None
    conn.close()
    if not result:
        return None
    ts = pd.to_datetime(result, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


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
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # === Caso 1: colonne MultiIndex ===
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        fields_set = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if len(fields_set & lvl0) > 0:
            long = df.stack(level=1, future_stack=True).reset_index()
            long = long.rename(columns={"level_0": "datetime", "level_1": "ticker"})
        else:
            long = df.stack(level=0).reset_index()
            long = long.rename(columns={"level_0": "datetime", "level_1": "ticker"})
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

    if any(c in fields_set for c in cols):
        out = df2[["datetime"]].copy()
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
                "open": row.get(fmap.get("Open")),
                "high": row.get(fmap.get("High")),
                "low": row.get(fmap.get("Low")),
                "close": row.get(fmap.get("Close")),
                "volume": row.get(fmap.get("Volume")),
            })

    out = pd.DataFrame(out_rows)
    return out[["datetime", "open", "high", "low", "close", "volume"]]


def download_data(ticker: str, start: datetime, end: datetime):
    print(f"📥 Scarico dati per {ticker} da {start} a {end}...")
    df = yf.download(
        ticker,
        interval=TICKS_INTERVAL,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        print(f"⚠️ Nessun dato per {ticker}")
        return None

    df_clean = normalize_yf_df(df, default_ticker=ticker)

    try:
        df_clean["datetime"] = pd.to_datetime(df_clean["datetime"], errors="coerce")
    except Exception:
        pass

    # convert to ISO strings (or None)
    df_clean["datetime"] = df_clean["datetime"].apply(
        lambda x: x.isoformat() if (hasattr(x, "isoformat")) and pd.notna(x) else None
    )

    return df_clean[["datetime", "open", "high", "low", "close", "volume"]]


def add_to_db(df: pd.DataFrame, ticker: str, db_folder):

    if df is None or df.empty:
        return
    create_table_for_ticker_if_not_exists(ticker, db_folder)
    table = table_name_for(ticker)

    def to_iso(val):
        if pd.isna(val):
            return None
        if isinstance(val, str):
            return val
        if isinstance(val, (pd.Timestamp, datetime)):
            return val.isoformat()
        try:
            # last resort: attempt to parse then isoformat
            parsed = pd.to_datetime(val, errors="coerce")
            return parsed.isoformat() if pd.notna(parsed) else None
        except Exception:
            return None

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
        if not dt:
            # skip rows without valid datetime (primary key)
            continue
        open_v = to_float3(row.get("open"))
        high_v = to_float3(row.get("high"))
        low_v = to_float3(row.get("low"))
        close_v = to_float3(row.get("close"))
        vol = None if pd.isna(row.get("volume")) else int(row.get("volume"))
        rows.append((dt, open_v, high_v, low_v, close_v, vol))

    if not rows:
        return

    db_file_path = os.path.join(db_folder, DB_NAME)
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.executemany(
        f"INSERT OR IGNORE INTO {table} (datetime, open, high, low, close, volume) VALUES (?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()


def upsert_ticker_data(ticker: str, is_first_run: bool, db_folder):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_TO_RETRIEVE)

    last_ts = None
    if not is_first_run:
        last_ts = get_last_timestamp(ticker, db_folder)
        if not last_ts:
            print(f"⚠️ {ticker}: nessun dato trovato, scarico ultimi 59 giorni completi.")
        elif last_ts and last_ts < end - timedelta(minutes=5):
            start = last_ts + timedelta(minutes=5)
        else:
            print(f"✅ {ticker}: dati già aggiornati")
            return

    df = download_data(ticker, start, end)
    if (df is None or df.empty) and not last_ts: #TODO oppure il last_ts è più vecchio di X giorni? i ticks che hanno come ultimo valore un prezzo di un anno fa devo smettere di cercare di reperirlo
        return False

    add_to_db(df, ticker, db_folder)
    print(f"✅ {ticker}: processate {len(df) if df is not None else 0} righe.")
    return True


def main(db_folder):
    is_first_run = ensure_db(db_folder)

    tickers_file = os.path.join(db_folder, MILAN_TICKERS_FILE_NAME)
    if date.today().day == 1 or not os.path.exists(tickers_file): # primo del mese, riscarica i tickers
        tickers = scrape_milan_stocks(db_folder)
        # tickers = tickers[:20]
        save_tickers_file(db_folder, tickers)
    else:
        tickers = load_tickers_file(db_folder)
    if not tickers:
        print("⚠️ Nessun dato trovato.")
        return

    # tickers = get_italian_tickers()

    print("🚀 Avvio aggiornamento dati azioni italiane")
    if is_first_run:
        print("📂 Database nuovo: scarico ultimi 6 mesi completi...")
    else:
        print("📈 Database esistente: aggiorno solo nuovi dati...")

    not_found_tickers = load_not_found_tickers(db_folder)
    for t in (t for t in tickers if t not in not_found_tickers):
        try:
            if not upsert_ticker_data(t, is_first_run, db_folder):
                not_found_tickers.add(t)
        except Exception as e:
            print(f"❌ Errore con {t}: {e}")

    save_not_found_tickers(db_folder, not_found_tickers)

    print("🏁 Aggiornamento completato.")


def load_tickers_file(db_folder: str) -> list:
    tickers_file = os.path.join(db_folder, MILAN_TICKERS_FILE_NAME)
    df_tickers = pd.read_csv(tickers_file)
    tickers = df_tickers["symbol"].dropna().unique().tolist()
    return tickers


def save_tickers_file(db_folder: str, tickers: list):
    tickers_file = os.path.join(db_folder, MILAN_TICKERS_FILE_NAME)
    df_tickers = pd.DataFrame({"symbol": tickers})
    df_tickers.to_csv(tickers_file, index=False, encoding="utf-8-sig")
    print(f"✅ Salvati {len(tickers)} tickers in '{tickers_file}'")


if __name__ == "__main__":
    main("db")

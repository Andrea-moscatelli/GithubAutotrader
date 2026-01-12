import pandas as pd
import sqlite3
from ib_insync import IB, Stock

# =========================
# CONFIG
# =========================

CSV_PATH = "euronext_milan_raw.csv"
DB_PATH = "contracts.db"

IB_HOST = "127.0.0.1"
IB_PORT = 4002   # paper
IB_CLIENT_ID = 50

VALID_MARKETS = {
    "Euronext Milan": {
        "exchange": "BVME",
        "primaryExchange": "BVME",
        "currency": "EUR"
    },
    "Euronext Growth Milan": {
        "exchange": "BVME",
        "primaryExchange": "BVME",
        "currency": "EUR"
    },
    "EuroTLX": {
        "exchange": "ETLX",
        "primaryExchange": "ETLX",
        "currency": "EUR"
    }
}

# =========================
# DB
# =========================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            conId INTEGER PRIMARY KEY,
            symbol TEXT,
            exchange TEXT,
            primaryExchange TEXT,
            currency TEXT,
            isin TEXT,
            description TEXT,
            market TEXT,
            source_symbol TEXT
        )
    """)

    conn.commit()
    conn.close()

# =========================
# IB
# =========================

def connect_ib():
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
    return ib

def discover_contract(ib: IB, symbol, exchange, primaryExchange, currency):
    contract = Stock(
        symbol=symbol,
        exchange=exchange,
        primaryExchange=primaryExchange,
        currency=currency
    )

    cds = ib.reqContractDetails(contract)
    if not cds:
        return None

    # prendiamo il primo (di solito unico)
    cd = cds[0]
    return cd.contract, cd.longName

# =========================
# MAIN LOGIC
# =========================

def main():
    init_db()

    df = pd.read_csv(CSV_PATH, sep=";")
    df = df[df["Market"].isin(VALID_MARKETS.keys())]
    df = df.drop_duplicates(subset=["ISIN", "Market"])

    ib = connect_ib()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        for _, row in df.iterrows():
            isin = row["ISIN"]
            symbol = row["Symbol"]
            market = row["Market"]

            cfg = VALID_MARKETS[market]

            print(f"🔍 {symbol} ({market})")

            try:
                result = discover_contract(
                    ib,
                    symbol=symbol,
                    exchange=cfg["exchange"],
                    primaryExchange=cfg["primaryExchange"],
                    currency=cfg["currency"]
                )
            except Exception as e:
                print(f"❌ errore IB: {e}")
                continue

            if not result:
                print("⚠️ contratto non trovato")
                continue

            contract, description = result

            c.execute("""
                INSERT OR IGNORE INTO contracts (
                    isin,
                    symbol,
                    exchange,
                    primaryExchange,
                    currency,
                    conId,
                    localSymbol
                    description,
                    secType, 
                    market,
                    source_symbol
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                isin,
                contract.symbol,
                contract.exchange,
                contract.primaryExchange,
                contract.currency,
                contract.conId,
                contract.localSymbol,
                description,
                contract.secType,
                market,
                symbol
            ))

            conn.commit()
            print(f"✅ salvato conId={contract.conId}")

    finally:
        conn.close()
        ib.disconnect()

    print("🏁 Discovery completata")

# =========================
if __name__ == "__main__":
    main()

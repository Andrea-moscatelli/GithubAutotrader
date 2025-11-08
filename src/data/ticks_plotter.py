import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go

DB_PATH = "../data/binance_data.db"  # percorso del tuo DB


def get_symbols_with_counts(conn):
    """Restituisce la lista delle tabelle che cominciano con 't_', rimuovendo il prefisso,
    insieme al numero di righe presenti in ciascuna tabella."""
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 't_%';"
    rows = conn.execute(query).fetchall()

    symbols_info = []
    for row in rows:
        table_name = row[0]
        # rimuove il prefisso 't_'
        symbol = table_name.removeprefix("t_")
        # conta il numero di righe
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        symbols_info.append((symbol, table_name, count))
    return symbols_info


def load_data(conn, table_name):
    """Carica i dati OHLCV per il simbolo scelto."""
    df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY datetime", conn, parse_dates=["datetime"])
    return df


def plot_candlestick(df, symbol, db_path):
    """Mostra il grafico candlestick interattivo con Plotly."""
    fig = go.Figure(data=[go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=symbol
    )])

    fig.update_layout(
        title=f"📈 {symbol} — Dati da SQLite",
        xaxis_title="Data / Ora",
        yaxis_title="Prezzo",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified",
    )
    # il tuo fig
    # fig = go.Figure(data=[go.Scatter(y=[1, 3, 2])])
    fig_path = db_path.replace("db/", "plots/").replace(".db", f"_{symbol}_candlestick.html")
    folder_path = os.path.dirname(fig_path)

    # crea le cartelle se non esistono
    os.makedirs(folder_path, exist_ok=True)
    fig.write_html(fig_path)


def main(db_path):
    conn = sqlite3.connect(db_path)
    symbols_info = get_symbols_with_counts(conn)

    if not symbols_info:
        print("⚠️ Nessuna tabella trovata nel database.")
        return

    print("\n📊 Simboli disponibili:")
    for i, (symbol, _, count) in enumerate(symbols_info, 1):
        print(f"{i}. {symbol} ({count} righe)")

    # Scelta dell’utente
    try:
        choice = int(input("\nSeleziona il numero del simbolo da visualizzare: "))
        _, table_name, _ = symbols_info[choice - 1]
    except (ValueError, IndexError):
        print("Scelta non valida.")
        return

    print(f"\nCaricamento dati per {table_name}...")
    df = load_data(conn, table_name)
    conn.close()

    if df.empty:
        print("⚠️ Nessun dato trovato per questo simbolo.")
        return

    plot_candlestick(df, table_name, db_path)

if __name__ == "__main__":
    db_path = "../../data/db/italian_stocks.db"
    main(db_path)

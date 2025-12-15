import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # =========================
    # PARAMETRI GENERALI
    # =========================
    DB_PATH = "../../data/db/italian_stocks.db"

    INITIAL_CAPITAL = 100_000   # capitale allocato PER strategia

    ROLLING_WINDOW = 20
    ZSCORE_THRESHOLD = 2.0

    # =========================
    # COMMISSIONI
    # =========================
    COMMISSION_TYPE = "fixed"
    COMMISSION_VALUE = 8.0

    # COMMISSION_TYPE = "percent"
    # COMMISSION_VALUE = 0.0019   # 0.19%

    # RECUPERO TABELLE
    def get_all_tables(db_path: str) -> list[str]:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            AND name LIKE 't_%'
        """)

        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    # BACKTEST SU UNA TABELLA
    def backtest_zscore_on_table(
            db_path: str,
            table_name: str,
    ) -> dict:

        conn = sqlite3.connect(db_path)

        df = pd.read_sql(
            f"""
            SELECT datetime, close
            FROM {table_name}
            ORDER BY datetime
            """,
            conn,
            parse_dates=["datetime"],
        )

        conn.close()

        if len(df) < ROLLING_WINDOW + 5:
            return {
                "table": table_name,
                "pnl": 0.0,
                "equity": INITIAL_CAPITAL,
                "trades": 0,
            }

        df.set_index("datetime", inplace=True)

        # =========================
        # Z-SCORE
        # =========================
        df["mean"] = df["close"].rolling(ROLLING_WINDOW).mean()
        df["std"] = df["close"].rolling(ROLLING_WINDOW).std()
        df["zscore"] = (df["close"] - df["mean"]) / df["std"]

        # =========================
        # SEGNALI
        # =========================
        df["position"] = 0
        df.loc[df["zscore"] < -ZSCORE_THRESHOLD, "position"] = 1
        df.loc[df["zscore"] > ZSCORE_THRESHOLD, "position"] = -1

        df["position"] = (
            df["position"]
            .replace(0, np.nan)
            .ffill()
            .fillna(0)
        )

        # =========================
        # TRADE COUNT
        # =========================
        df["trades"] = df["position"].diff().abs().fillna(0)

        # =========================
        # COMMISSIONI
        # =========================
        df["commission"] = 0.0

        if COMMISSION_TYPE == "fixed":
            df.loc[df["trades"] > 0, "commission"] = (
                    df["trades"] * COMMISSION_VALUE
            )

        elif COMMISSION_TYPE == "percent":
            df.loc[df["trades"] > 0, "commission"] = (
                    df["trades"] * INITIAL_CAPITAL * COMMISSION_VALUE
            )

        # =========================
        # RETURNS
        # =========================
        df["returns"] = df["close"].pct_change()

        df["strategy_returns"] = (
                df["position"].shift(1) * df["returns"]
                - df["commission"] / INITIAL_CAPITAL
        )

        df["equity"] = (
                (1 + df["strategy_returns"])
                .cumprod()
                * INITIAL_CAPITAL
        )

        pnl = df["equity"].iloc[-1] - INITIAL_CAPITAL

        return {
            "table": table_name,
            "pnl": pnl,
            "equity": df["equity"].iloc[-1],
            "trades": int(df["trades"].sum()),
        }


    # BACKTEST SU TUTTE LE TABELLE
    tables = get_all_tables(DB_PATH)

    results = []

    for table in tables:
        res = backtest_zscore_on_table(DB_PATH, table)
        # stampa il pnl per ogni tabella
        print(f"Tabella: {table} | PnL: {res['pnl']:,.2f} € | Trades: {res['trades']}")
        results.append(res)

    results_df = pd.DataFrame(results)


    # RISULTATI PER TABELLA
    results_df = results_df.sort_values("pnl", ascending=False)

    print("\n===== PNL PER TABELLA =====")
    print(results_df[["table", "pnl", "trades"]])


    # PNL GLOBALE
    total_pnl = results_df["pnl"].sum()
    total_equity = INITIAL_CAPITAL * len(results_df) + total_pnl

    print("\n===== RISULTATO GLOBALE =====")
    print(f"Strategie testate: {len(results_df)}")
    print(f"PnL totale: {total_pnl:,.2f} €")
    print(f"Equity finale aggregata: {total_equity:,.2f} €")

    # ISTOGRAMMA PNL
    plt.figure(figsize=(10, 6))
    plt.hist(results_df["pnl"], bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribuzione del PnL per Tabella")
    plt.xlabel("PnL (€)")
    plt.ylabel("Frequenza")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # stampa la percentuale di successo delle strategie
    winning_strategies = results_df[results_df["pnl"] > 0].shape[0]
    total_strategies = results_df.shape[0]
    success_rate = (winning_strategies / total_strategies) * 100
    print(f"\nPercentuale di strategie vincenti: {success_rate:.2f}% ({winning_strategies} su {total_strategies})")


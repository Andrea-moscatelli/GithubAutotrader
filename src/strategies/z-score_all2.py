import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import skew

DB_PATH = "italian_stocks.db"

INITIAL_CAPITAL = 100_000

ROLLING_WINDOW = 20
ZSCORE_THRESHOLD = 2.0

VALIDATION_RATIO = 0.6  # 60% validation / 40% test
TOP_N = 10

COMMISSION_TYPE = "percent"
COMMISSION_VALUE = 0.0019

VALIDATION_RESULTS_FILE = "validation_results.csv"


def get_all_tables(db_path):
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 't_%'",
        conn
    )["name"].tolist()
    conn.close()
    return tables


def load_data(db_path, table):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        f"SELECT datetime, close FROM {table} ORDER BY datetime",
        conn,
        parse_dates=["datetime"]
    )
    conn.close()
    df.set_index("datetime", inplace=True)
    return df


def split_validation_test(df, ratio):
    split = int(len(df) * ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def run_backtest(df):
    df = df.copy()

    df["mean"] = df["close"].rolling(ROLLING_WINDOW).mean()
    df["std"] = df["close"].rolling(ROLLING_WINDOW).std()
    df["zscore"] = (df["close"] - df["mean"]) / df["std"]

    df["position"] = 0
    df.loc[df["zscore"] < -ZSCORE_THRESHOLD, "position"] = 1
    df.loc[df["zscore"] > ZSCORE_THRESHOLD, "position"] = -1
    df["position"] = df["position"].replace(0, np.nan).ffill().fillna(0)

    # calcolo differenza posizione
    df["trades"] = df["position"].diff().fillna(0)

    # valore negoziato = |Δposition| * INITIAL_CAPITAL => ad ogni trade investo INITIAL_CAPITAL
    df["traded_value"] = df["trades"].abs() * INITIAL_CAPITAL

    # commissione
    if COMMISSION_TYPE == "percent":
        df["commission"] = df["traded_value"] * COMMISSION_VALUE
    elif COMMISSION_TYPE == "fixed":
        df["commission"] = df["traded_value"].apply(lambda x: COMMISSION_VALUE if x > 0 else 0)
    else:
        df["commission"] = 0.0

    # df["trades"] = df["position"].diff().abs().fillna(0)
    #
    # df["commission"] = 0.0
    # if COMMISSION_TYPE == "percent":
    #     df.loc[df["trades"] > 0, "commission"] = (
    #             df["trades"] * INITIAL_CAPITAL * COMMISSION_VALUE
    #     )

    # df["returns"] = df["close"].pct_change()
    # df["strategy_returns"] = (
    #         df["position"].shift(1) * df["returns"]
    #         - df["commission"] / INITIAL_CAPITAL
    # )
    #
    # df["equity"] = (1 + df["strategy_returns"]).cumprod() * INITIAL_CAPITAL

    df["returns"] = df["close"].pct_change()

    # moltiplico per INITIAL_CAPITAL per ottenere guadagno in € reale
    df["strategy_returns"] = df["position"].shift(1) * (df["returns"] * INITIAL_CAPITAL) - df["commission"]

    # equity cumulata
    df["equity"] = INITIAL_CAPITAL + df["strategy_returns"].cumsum()

    return df


def annualization_factor(interval):
    if interval == "1d":
        return 252
    if interval == "1h":
        return 252 * 8
    if interval == "2m":
        return 252 * 240
    raise ValueError("Interval not supported")


def extract_interval_from_table(table_name: str) -> str:
    """
    Estrae l'interval dal nome tabella.
    Esempi:
    - t_ENI_2m -> '2m'
    - t_FCA_1h -> '1h'
    """
    try:
        interval = table_name.split("_")[-1]
        if interval not in {"1d", "1h", "2m"}:
            raise ValueError(f"Interval per tabella {table_name} non riconosciuto: {interval}")
        return interval
    except IndexError:
        raise ValueError(f"Nome tabella non valido: {table_name}")


def compute_metrics(df, interval):
    pnl = df["equity"].iloc[-1] - INITIAL_CAPITAL

    factor = annualization_factor(interval)

    # ritorno percentuale sul capitale investito
    df["strategy_returns_pct"] = df["strategy_returns"] / INITIAL_CAPITAL

    sharpe = (
                     df["strategy_returns_pct"].mean() / df["strategy_returns_pct"].std()
             ) * np.sqrt(factor)

    drawdown = (df["equity"] / df["equity"].cummax() - 1).min()

    win_rate = (df["strategy_returns"] > 0).mean()

    skewness = skew(df["strategy_returns_pct"].dropna())

    turnover = df["trades"].mean()

    volatility = df["returns"].std() * np.sqrt(factor)

    rolling_pnl_3w = df["strategy_returns"].rolling(15).sum().mean()

    return {
        "pnl": pnl,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "win_rate": win_rate,
        "skewness": skewness,
        "turnover": turnover,
        "volatility": volatility,
        "pnl_rolling_3w": rolling_pnl_3w,
    }


def run_validation():
    tables = get_all_tables(DB_PATH)
    results = []

    for table in tables:
        df = load_data(DB_PATH, table)
        if len(df) < 100:
            continue

        val, _ = split_validation_test(df, VALIDATION_RATIO)
        bt = run_backtest(val)
        interval = extract_interval_from_table(table)
        metrics = compute_metrics(bt, interval)

        metrics["table"] = table
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(VALIDATION_RESULTS_FILE, index=False)
    return results_df


def load_validation_results():
    return pd.read_csv(VALIDATION_RESULTS_FILE)


def select_top_n(df, n):
    return (
        df.sort_values("sharpe", ascending=False)
        .head(n)["table"]
        .tolist()
    )


def run_test_on_selected(tables):
    results = []

    for table in tables:
        df = load_data(DB_PATH, table)
        _, test = split_validation_test(df, VALIDATION_RATIO)

        bt = run_backtest(test)
        pnl = bt["equity"].iloc[-1] - INITIAL_CAPITAL

        results.append({
            "table": table,
            "test_pnl": pnl
        })

    return pd.DataFrame(results)


# 1. Validation (una volta)
validation_df = run_validation()

# 2. In futuro puoi fare solo:
validation_df = load_validation_results()

# 3. Selezione
top_tables = select_top_n(validation_df, TOP_N)

# 4. Test
test_results = run_test_on_selected(top_tables)

print(test_results)
print("PnL totale test:", test_results["test_pnl"].sum())

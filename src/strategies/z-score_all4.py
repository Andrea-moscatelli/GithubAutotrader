import sqlite3
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import math
from matplotlib.widgets import Button

from src.strategies.WalkForwardInspector import WalkForwardInspector

DB_PATH = "../../data/db/italian_stocks.db"
INITIAL_CAPITAL = 100_000  # capitale allocato PER AZIONE

ROLLING_WINDOW = 20
ZSCORE_THRESHOLD = 2.0

VALIDATION_RATIO = 0.6  # 60% validation / 40% test
# TOP_N = 10

# =========================
# CONDIZIONI DI USCITA (COMBINATE IN 'OR' SE NON SONO None)
# =========================
EXIT_ZCORE_ZERO = True
EXIT_MA_AT_ENTRY = True
STOP_LOSS_PCT = 0.05  # 0.05  # 5%
STOP_LOSS_TIME_BARS = None  # 100
TRAILING_STOP_PCT = None  # 0.03  # 3%
# =========================

COMMISSION_TYPE = "percent"  # "percent" o "fixed"
COMMISSION_VALUE = 0.002  # 0.2% commissioni
# COMMISSION_TYPE = "fixed"  # "percent" o "fixed"
# COMMISSION_VALUE = 0  # no commissioni

VALIDATION_RESULTS_FILE = "validation_results.csv"

def inspect_walk_forward(wf_results):
    for i, wf in wf_results.iterrows():
        print(f"\n=== WALK-FORWARD TEST WINDOW {i} when start bar is {wf_results['start_bar'][i]} ===")

        if wf["selected_tables"] is None or len(wf["selected_tables"]) == 0:
            print("Nessuna tabella selezionata in questa finestra.")
            continue

        test_bts = wf["test_backtests"]
        selected_bt = {k: v for k, v in test_bts.items() if k in wf["selected_tables"]}
        oracle_bt = {k: v for k, v in test_bts.items() if k in wf["oracle_tables"]}

        print("Selected portfolio")
        browse_portfolio(selected_bt, "SELECTED PORTFOLIO")

        print("Oracle portfolio")
        browse_portfolio(oracle_bt, "ORACLE PORTFOLIO")


# =========================
# UTILITIES
# =========================
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


def extract_interval_from_table(table_name: str) -> str:
    interval = table_name.split("_")[-1]
    if interval not in {"1d", "1h", "2m"}:
        raise ValueError(f"Interval non riconosciuto: {interval}")
    return interval


def annualization_factor(interval: str) -> int:
    trading_days = 252
    hours_per_day = 8
    if interval.endswith("d"):
        return trading_days
    if interval.endswith("h"):
        hours = int(interval.replace("h", ""))
        return trading_days * (hours_per_day // hours)
    if interval.endswith("m"):
        minutes = int(interval.replace("m", ""))
        bars_per_day = (hours_per_day * 60) // minutes
        return trading_days * bars_per_day
    raise ValueError(f"Interval non supportato: {interval}")


# =========================
# BACKTEST ENGINE
# =========================

def run_backtest_zscore(
        df,
        rolling_window,
        zscore_threshold,
        initial_capital,
        commission_type="percent",
        commission_value=0.0,
        max_trades=None,
        exit_zscore_zero=True,
        exit_ma_at_entry=True,
        stop_loss_pct=None,
        time_stop_bars=None,
        trailing_stop_pct=None,
):
    df = df.copy()

    # =========================
    # Z-SCORE
    # =========================
    df["mean"] = df["close"].rolling(rolling_window).mean()
    df["std"] = df["close"].rolling(rolling_window).std()

    df["zscore"] = np.where(
        df["std"] > 0,
        (df["close"] - df["mean"]) / df["std"],
        0.0,
    )

    # =========================
    # INIZIALIZZAZIONE
    # =========================
    df["position"] = 0
    df["entry_price"] = np.nan
    df["entry_bar"] = np.nan
    df["trail_price"] = np.nan
    df["trades"] = 0.0

    position = 0
    entry_price = None
    entry_bar = None
    trail_price = None
    trade_count = 0

    # =========================
    # LOOP PRINCIPALE
    # =========================
    for i in range(rolling_window + 1, len(df)):
        price = df.iloc[i]["close"]
        z = df.iloc[i]["zscore"]
        mean = df.iloc[i]["mean"]

        # =====================
        # ENTRY
        # =====================
        if position == 0:
            if z < -zscore_threshold:
                position = 1  # Entering LONG
                entry_price = price
                entry_bar = i
                trail_price = price
                trade_count += 1

            elif z > zscore_threshold:
                position = -1  # Entering SHORT
                entry_price = price
                entry_bar = i
                trail_price = price
                trade_count += 1

        # =====================
        # EXIT
        # =====================
        else:
            holding_bars = i - entry_bar

            if position == 1:  # I'm currently LONG
                trail_price = max(trail_price, price)
                pnl_pct = (price - entry_price) / entry_price
                trail_hit = (
                        trailing_stop_pct is not None
                        and pnl_pct > 0
                        and price <= trail_price * (1 - trailing_stop_pct)
                )
            else:  # I'm currently SHORT
                trail_price = min(trail_price, price)
                pnl_pct = (entry_price - price) / entry_price
                trail_hit = (
                        trailing_stop_pct is not None
                        and pnl_pct > 0
                        and price >= trail_price * (1 + trailing_stop_pct)
                )

            exit_signal = False

            # 1️⃣ z-score = 0
            if exit_zscore_zero and (
                    (position == 1 and z >= 0)
                    or (position == -1 and z <= 0)
            ):
                exit_signal = True

            # 2️⃣ MA raggiunge prezzo di ingresso
            if exit_ma_at_entry and (
                    (position == 1 and mean <= entry_price)
                    or (position == -1 and mean >= entry_price)
            ):
                exit_signal = True

            # 3️⃣ stop loss %
            if stop_loss_pct is not None and pnl_pct <= -stop_loss_pct:
                exit_signal = True

            # 4️⃣ stop temporale
            if time_stop_bars is not None and holding_bars >= time_stop_bars:
                exit_signal = True

            # 5️⃣ trailing stop
            if trail_hit:
                exit_signal = True

            if exit_signal:
                position = 0
                entry_price = None
                entry_bar = None
                trail_price = None
                trade_count += 1

        # =====================
        # SALVATAGGIO STATO
        # =====================
        df.at[df.index[i], "position"] = position
        df.at[df.index[i], "entry_price"] = entry_price if position != 0 else np.nan
        df.at[df.index[i], "entry_bar"] = entry_bar if position != 0 else np.nan
        df.at[df.index[i], "trail_price"] = trail_price if position != 0 else np.nan

        if max_trades is not None and trade_count >= max_trades:
            break

    # =========================
    # TRADES & COMMISSIONI
    # =========================
    df["trades"] = df["position"].diff().fillna(0).abs()  # può essere 0/1/2
    df["traded_value"] = df["trades"] * initial_capital  # può essere 0/initial_capital/2*initial_capital

    if commission_type == "percent":
        df["commission"] = df["traded_value"] * commission_value
    elif commission_type == "fixed":
        df["commission"] = np.where(df["traded_value"] > 0, commission_value, 0.0)
    else:
        df["commission"] = 0.0

    # =========================
    # PERFORMANCE
    # =========================
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["strategy_returns"] = (
            df["position"].shift(1).fillna(0.0)
            * df["returns"]
            * initial_capital
            - df["commission"]
    )

    df["equity"] = initial_capital + df["strategy_returns"].cumsum()

    return df


def extract_trades(df):
    trades = []

    prev_pos = 0
    entry_equity = None

    for i in range(len(df)):
        curr_pos = df["position"].iloc[i]
        curr_equity = df["equity"].iloc[i]

        # ENTRY
        if prev_pos == 0 and curr_pos != 0:
            entry_equity = curr_equity

        # EXIT (pos → 0 OR sign flip)
        elif prev_pos != 0 and (
                curr_pos == 0 or np.sign(prev_pos) != np.sign(curr_pos)
        ):
            trades.append(curr_equity - entry_equity)

            # se flip, nuova entry nello stesso bar
            if curr_pos != 0:
                entry_equity = curr_equity
            else:
                entry_equity = None

        prev_pos = curr_pos

    return np.array(trades)


def sharpe_ratio(returns):
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret == 0:
        return 0.0

    return mean_ret / std_ret


def sortino_ratio(returns):
    downside = returns[returns < 0]

    if len(downside) == 0:
        return np.inf

    downside_std = downside.std()

    if downside_std == 0:
        return np.inf

    return returns.mean() / downside_std


# =========================
# METRICHE
# =========================
def compute_metrics(df, initial_capital, interval):
    trades_pnl = extract_trades(df)

    pnl = df["equity"].iloc[-1] - initial_capital
    max_dd = (df["equity"] / df["equity"].cummax() - 1).min()

    n_trades = len(trades_pnl)

    win_trades = trades_pnl[trades_pnl > 0]
    loss_trades = trades_pnl[trades_pnl < 0]

    win_rate = len(win_trades) / n_trades if n_trades > 0 else 0.0

    avg_win = win_trades.mean() if len(win_trades) > 0 else 0.0
    avg_loss = loss_trades.mean() if len(loss_trades) > 0 else 0.0

    profit_factor = (
        win_trades.sum() / abs(loss_trades.sum())
        if len(loss_trades) > 0
        else np.inf
    )

    expectancy = (
        win_rate * avg_win + (1 - win_rate) * avg_loss
        if n_trades > 0
        else 0.0
    )

    # exposure
    exposure = (df["position"] != 0).mean()  # percentuale di tempo in posizione - ignora warning

    # pnl ultimi 200 bar
    pnl_last_200 = df.tail(200)["strategy_returns"].sum()

    returns = df["strategy_returns"] / initial_capital

    factor = annualization_factor(interval)
    sharpe = sharpe_ratio(returns) * math.sqrt(factor)
    sortino = sortino_ratio(returns) * math.sqrt(factor)

    return {
        "pnl": pnl,
        "max_drawdown": max_dd,
        "num_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "exposure": exposure,
        "pnl_last_200_bars": pnl_last_200,
        "sharpe": sharpe,
        "sortino": sortino,
        # Sul mercato italiano, realisticamente:
        # Metrica	            Buona
        # Sharpe	            0.5 – 1.2
        # Sortino	            0.8 – 2.0
        # Sortino > Sharpe	    ✅ tipico mean reversion
        #
        # Se:
        # Sharpe basso
        # Sortino buono
        # 👉 stai facendo mean reversion vera, non trend following mascherato.
    }



def load_validation_results():
    return pd.read_csv(VALIDATION_RESULTS_FILE)


# def run_test_all_tables(db_path, validation_ratio, max_trades=None):
#     tables = get_all_tables(db_path)
#     results = []
#
#     for idx, table in enumerate(tables):
#         print(f"Test table {idx + 1}/{len(tables)}: {table}")
#         df = load_data(db_path, table)
#         if len(df) < 100:
#             continue
#
#         _, test = split_validation_test(df, validation_ratio)
#         bt = run_backtest(test, max_trades)
#
#         pnl = bt["equity"].iloc[-1] - INITIAL_CAPITAL
#
#         results.append({
#             "table": table,
#             "test_pnl": pnl
#         })
#
#     return pd.DataFrame(results)


def portfolio_stats(df_to_consider, tables):
    """
    Calcola statistiche di portafoglio dato un insieme di tabelle.
    Ogni tabella investe INITIAL_CAPITAL.
    """
    df = df_to_consider[df_to_consider["table"].isin(tables)]

    return {
        "df": df,
        "num_assets": len(df),
        "total_pnl": df["test_pnl"].sum(),
        "avg_pnl": df["test_pnl"].mean(),
        "positive_assets_ratio": (df["test_pnl"] > 0).mean()
    }


def efficiency_score(selected_stats, oracle_stats):
    """
    Calcola l'efficienza della selezione ex-ante rispetto all'oracle.
    1.0 → selezione perfetta (impossibile nella realtà)
    0.6 → molto buona
    0.4 → discreta
    < 0.3 → segnali deboli / ranking rumoroso
    < 0 → selezione dannosa
    """
    oracle_pnl = oracle_stats["total_pnl"]
    selected_pnl = selected_stats["total_pnl"]

    if oracle_pnl == 0:
        return np.nan

    return selected_pnl / oracle_pnl


def filter_by_thresholds(df, rules):
    mask = pd.Series(True, index=df.index)

    for metric, rule in rules.items():
        if rule["direction"] == ">":
            mask &= df[metric] > rule["threshold"]
        elif rule["direction"] == "<":
            mask &= df[metric] < rule["threshold"]
        else:
            raise ValueError(f"Direzione non valida per {metric}")

    return df[mask].copy()


def add_weighted_score(df, m_priority_map):
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    for metric, priority in m_priority_map.items():
        values = df[metric]

        # z-score cross-sectional
        z = (values - values.mean()) / values.std()

        # metriche dove "meno è meglio"
        if metric in {"volatility"}:
            z = -z

        score += priority * z

    df["score"] = score
    return df


def get_selected_ranked_tickers(validation_df, filter_rules, metric_priority_map, top_n=None):
    # 1. filtro hard
    filtered = filter_by_thresholds(validation_df, filter_rules)

    if filtered.empty:
        print("Nessun ticker soddisfa i criteri")
        return filtered

    # 2. ranking pesato
    ranked = add_weighted_score(filtered, metric_priority_map)

    ranked = ranked.sort_values("score", ascending=False)

    if top_n is not None:
        ranked = ranked.head(top_n)

    return ranked


def run_walk_forward(
        db_path,
        validation_bars,
        test_bars,
        top_n,
):
    tables = [t for t in get_all_tables(db_path) if "1h" in t]

    # TODO: limitazione di test
    # tables = tables[: 20]  # per testare più velocemente
    # tables = ["t_1ADS_MI_1h"]

    print("Loading data...")
    data = {t: load_data(db_path, t) for t in tables}

    start = 0
    wf_results = []

    # limite massimo raggiungibile (ticker più lungo)
    max_len = max(len(df) for df in data.values())

    while start + validation_bars + test_bars <= max_len:
        print(f"\n=== WALK-FORWARD WINDOW starting at bar {start} ===")

        validation_metrics = []
        test_results = []

        # =========================
        # SELEZIONE TICKER VALIDI
        # =========================
        valid_tables = [
            t for t, df in data.items()
            if len(df) >= start + validation_bars + test_bars
        ]

        # if len(valid_tables) < top_n:
        #     print("Not enough valid tickers for this window, stopping WF.")
        #     break

        print(f"Valid tickers in this window: {len(valid_tables)}")

        # =========================
        # VALIDATION
        # =========================
        for idx, table in enumerate(valid_tables):
            df = data[table]

            if idx % 50 == 0:
                print(f"Validating table: {table} ({idx + 1}/{len(valid_tables)})")

            val = df.iloc[start : start + validation_bars]

            bt_val = run_backtest_zscore(
                val,
                rolling_window=ROLLING_WINDOW,
                zscore_threshold=ZSCORE_THRESHOLD,
                initial_capital=INITIAL_CAPITAL,
                commission_type=COMMISSION_TYPE,
                commission_value=COMMISSION_VALUE,
                exit_zscore_zero=EXIT_ZCORE_ZERO,
                exit_ma_at_entry=EXIT_MA_AT_ENTRY,
                stop_loss_pct=STOP_LOSS_PCT,
                time_stop_bars=STOP_LOSS_TIME_BARS,
                trailing_stop_pct=TRAILING_STOP_PCT,
            )

            interval = extract_interval_from_table(table)
            m = compute_metrics(bt_val, INITIAL_CAPITAL, interval)
            m["table"] = table
            validation_metrics.append(m)

        val_df = pd.DataFrame(validation_metrics)

        # =========================
        # SELEZIONE EX-ANTE
        # =========================
        selected = get_selected_ranked_tickers(
            validation_df=val_df,
            filter_rules=FILTER_RULES,
            metric_priority_map=METRIC_PRIORITY_MAP,
            top_n=top_n,
        )

        selected_tables = selected["table"].tolist()
        print(f"Selected {len(selected_tables)} tables for testing.")

        # =========================
        # TEST
        # =========================

        bt_test_dict = {}
        for idx_table, table in enumerate(valid_tables):
            if idx_table % 50 == 0:
                print(f"Testing table: {table} ({idx_table + 1}/{len(valid_tables)})")

            df = data[table]

            test = df.iloc[
                start + validation_bars :
                start + validation_bars + test_bars
            ]

            bt_test = run_backtest_zscore(
                test,
                rolling_window=ROLLING_WINDOW,
                zscore_threshold=ZSCORE_THRESHOLD,
                initial_capital=INITIAL_CAPITAL,
                commission_type=COMMISSION_TYPE,
                commission_value=COMMISSION_VALUE,
                exit_zscore_zero=EXIT_ZCORE_ZERO,
                exit_ma_at_entry=EXIT_MA_AT_ENTRY,
                stop_loss_pct=STOP_LOSS_PCT,
                time_stop_bars=STOP_LOSS_TIME_BARS,
                trailing_stop_pct=TRAILING_STOP_PCT,
            )

            pnl = bt_test["equity"].iloc[-1] - INITIAL_CAPITAL
            test_results.append({"table": table, "test_pnl": pnl})
            # if table in selected_tables:
            bt_test_dict[table] = bt_test

        test_df = pd.DataFrame(test_results)

        # =========================
        # PORTAFOGLI
        # =========================
        selected_port = portfolio_stats(test_df, selected_tables)

        oracle_tables = (
            test_df
            .sort_values("test_pnl", ascending=False)["table"]
            .tolist()
        )

        oracle_tables = oracle_tables[:len(selected_tables)]

        oracle_port = portfolio_stats(test_df, oracle_tables)

        eff = efficiency_score(selected_port, oracle_port)

        wf_results.append({
            "start_bar": start,
            "num_assets": len(valid_tables),
            "selected_pnl": selected_port["total_pnl"],
            "oracle_pnl": oracle_port["total_pnl"],
            "efficiency": eff,
            "selected_tables": selected_tables,
            "oracle_tables": oracle_tables,
            # "selected_test_history": {
            #     table: bt_test_dict[table]
            #     for table in selected_tables
            # },
            # "oracle_test_history": {
            #     table: bt_test_dict[table]
            #     for table in oracle_tables
            # },
            "test_backtests": {
                table: bt_test_dict[table]
                for table in set(selected_tables + oracle_tables)
            },
        })

        print(f"******** Selected portfolio PnL: {selected_port['total_pnl']:.2f}")
        print(f"******** Oracle portfolio PnL:   {oracle_port['total_pnl']:.2f}")
        print(f"******** Efficiency ratio:        {eff:.2f}")

        # advance di un periodo di test
        start += test_bars

        # TODO limitazione test
        # break

    return pd.DataFrame(wf_results)



def summarize_walk_forward(wf_df):
    return {
        "periods": len(wf_df),
        "total_selected_pnl": wf_df["selected_pnl"].sum(),
        "avg_selected_pnl": wf_df["selected_pnl"].mean(),
        "total_oracle_pnl": wf_df["oracle_pnl"].sum(),
        "avg_oracle_pnl": wf_df["oracle_pnl"].mean(),
        "avg_efficiency": wf_df["efficiency"].mean(),
        "positive_periods_ratio": (wf_df["selected_pnl"] > 0).mean(),
    }





FILTER_RULES = {
    "expectancy": {"direction": ">", "threshold": 0},
    "profit_factor": {"direction": ">", "threshold": 1.2},
    "max_drawdown": {"direction": ">", "threshold": -0.25},
    "exposure": {"direction": ">", "threshold": 0.05},
}

METRIC_PRIORITY_MAP = {
    "expectancy": 3.0,
    "profit_factor": 2.0,
    "sharpe": 1.5,
    "sortino": 1.5,
}


# =========================
# WALK-FORWARD PARAMS
# =========================
VALIDATION_BARS = 252 * 8      # 1 anno ~ hourly
TEST_BARS = 21 * 8             # 1 mese ~ hourly


wf_df = run_walk_forward(
    db_path=DB_PATH,
    validation_bars=VALIDATION_BARS,
    test_bars=TEST_BARS,
    top_n=None,
)

summary = summarize_walk_forward(wf_df)

print("\n=== WALK-FORWARD SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v}")



WalkForwardInspector(
    wf_df,
    zscore_threshold=ZSCORE_THRESHOLD,
    wf_idx=0
)
# inspect_walk_forward(wf_df)


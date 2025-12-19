import sqlite3
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import math

DB_PATH = "../../data/db/italian_stocks.db"
INITIAL_CAPITAL = 100_000

ROLLING_WINDOW = 20
ZSCORE_THRESHOLD = 2.0

VALIDATION_RATIO = 0.6  # 60% validation / 40% test
TOP_N = 10

COMMISSION_TYPE = "percent"  # "percent" o "fixed"
COMMISSION_VALUE = 0.002  # 0.2% commissioni
# COMMISSION_TYPE = "fixed"  # "percent" o "fixed"
# COMMISSION_VALUE = 0  # no commissioni

VALIDATION_RESULTS_FILE = "validation_results.csv"


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


def split_validation_test(df, ratio):
    split = int(len(df) * ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


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


def bars_in_4_weeks(interval: str) -> int:
    """
    Restituisce il numero di barre che corrispondono a 4 settimane di trading
    considerando l'interval.
    """
    trading_days_per_week = 5

    if interval.endswith("d"):
        bars_per_day = 1
    elif interval.endswith("h"):
        hours_per_day = 8
        bars_per_day = hours_per_day // int(interval.replace("h", ""))
    elif interval.endswith("m"):
        minutes_per_day = 8 * 60
        bars_per_day = minutes_per_day // int(interval.replace("m", ""))
    else:
        raise ValueError(f"Interval non supportato: {interval}")

    return bars_per_day * trading_days_per_week * 4  # 4 settimane


# =========================
# BACKTEST ENGINE
# =========================
def run_backtest(df):
    df = df.copy()

    # Z-score
    df["mean"] = df["close"].rolling(ROLLING_WINDOW).mean()
    df["std"] = df["close"].rolling(ROLLING_WINDOW).std()
    # df.loc[df["std"] == 0, "std"] = np.nan
    # df["zscore"] = (df["close"] - df["mean"]) / df["std"]

    df["zscore"] = np.where(
        df["std"] > 0,
        (df["close"] - df["mean"]) / df["std"],
        0
    )

    # Posizione
    df["position"] = 0
    df.loc[df["zscore"] < -ZSCORE_THRESHOLD, "position"] = 1 # LONG
    df.loc[df["zscore"] > ZSCORE_THRESHOLD, "position"] = -1 # SHORT
    df["position"] = df["position"].replace(0, np.nan).ffill().fillna(0)

    # Differenza posizione -> numero ordini
    df["trades"] = df["position"].diff().fillna(0)
    # valore negoziato = |Δposition| * INITIAL_CAPITAL => ad ogni trade investo INITIAL_CAPITAL
    df["traded_value"] = df["trades"].abs() * INITIAL_CAPITAL

    # Commissioni
    if COMMISSION_TYPE == "percent":
        df["commission"] = df["traded_value"] * COMMISSION_VALUE
    elif COMMISSION_TYPE == "fixed":
        df["commission"] = df["traded_value"].apply(lambda x: COMMISSION_VALUE if x > 0 else 0)
    else:
        df["commission"] = 0.0

    # Ritorni giornalieri
    df["returns"] = df["close"].pct_change()

    # Guadagno/Perdita in € reale
    df["strategy_returns"] = df["position"].shift(1) * (df["returns"] * INITIAL_CAPITAL) - df["commission"]

    # Pulizia dati
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Equity cumulata
    df["equity"] = INITIAL_CAPITAL + df["strategy_returns"].cumsum()

    return df


# =========================
# METRICHE
# =========================
def compute_metrics(df, interval):
    """
   Calcola tutte le metriche principali di performance della strategia.

   Metriche calcolate:
    - pnl: perdita massima dal picco all’abbassamento più basso. Misura il rischio di ribasso e l’ampiezza dei drawdown.
    - sharpe: misura del rapporto rischio/ritorno. Indica quanto rendimento extra si ottiene per unità di rischio (volatilità).
    - max_drawdown: misura del rapporto rischio/ritorno. Indica quanto rendimento extra si ottiene per unità di rischio (volatilità).
    - win_rate: profitto o perdita totale in euro nel periodo considerato. Indica quanto la strategia ha guadagnato o perso.
    - skewness: asimmetria della distribuzione dei ritorni. Negativo → rischio di code lunghe verso perdite estreme; positivo → code verso guadagni.
    - turnover: Numero medio di trade o variazione di posizione. Indica quanto la strategia scambia spesso e impatta costi di commissione.
    - volatility: Deviazione standard dei ritorni annualizzata. Misura quanto sono oscillanti i prezzi (rischio di mercato).
    - pnl_4w: Profitto o perdita totale negli ultimi 4 settimane.
    - total_trades: numero totale di trades effettuati
   """
    pnl = df["equity"].iloc[-1] - INITIAL_CAPITAL
    factor = annualization_factor(interval)

    # percentuali sul capitale investito
    df["strategy_returns_pct"] = df["strategy_returns"] / INITIAL_CAPITAL

    sharpe = (
                 df["strategy_returns_pct"].mean() / df["strategy_returns_pct"].std()
                 if df["strategy_returns_pct"].std() > 0 else 0.0
             ) * np.sqrt(factor)

    drawdown = (df["equity"] / df["equity"].cummax() - 1).min()
    win_rate = (df["strategy_returns"] > 0).mean()
    skewness = skew(df["strategy_returns_pct"].dropna())
    turnover = df["trades"].abs().mean()
    total_trades = df["trades"].abs().sum()
    volatility = df["returns"].std() * np.sqrt(factor)

    n_bars = bars_in_4_weeks(interval)
    pnl_4w = df.tail(n_bars)["strategy_returns"].sum()
    pnl_last_200_bars = df.tail(200)["strategy_returns"].sum()

    return {
        "pnl": pnl,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "win_rate": win_rate,
        "skewness": skewness,
        "turnover": turnover,
        "total_trades": total_trades,
        "volatility": volatility,
        "pnl_4w": pnl_4w,
        "pnl_last_200_bars": pnl_last_200_bars
    }


# =========================
# VALIDATION
# =========================
def run_validation():
    tables = get_all_tables(DB_PATH)
    results = []

    for idx, table in enumerate(tables):
        print(f"Validating table {idx + 1}/{len(tables)}: {table}")
        df_table = load_data(DB_PATH, table)
        if len(df_table) < 100:
            continue
        val, _ = split_validation_test(df_table, VALIDATION_RATIO)
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


# def select_top_n(df, n):
#     return df.sort_values("sharpe", ascending=False).head(n)["table"].tolist()

def select_top_n_with_zscore_and_weights(df, top_n, weights):
    """
    Ritorna le top N tabelle basate su un ranking multi-metrica avanzato.

    Parametri:
    - df: DataFrame con tutte le metriche di validation per ogni tabella
    - top_n: numero di tabelle da selezionare
    - weights: dizionario dei pesi per ciascuna metrica
        Esempio:
        weights = {
            "sharpe": 0.3,
            "pnl": 0.25,
            "max_drawdown": 0.15,
            "win_rate": 0.1,
            "skewness": 0.05,
            "volatility": 0.05,
            "turnover": 0.05,
            "pnl_4w": 0.05
        }

    Restituisce:
    - lista delle top N tabelle
    """
    df = df.copy()

    # Default weights se non forniti
    # if weights is None:
    #     weights = {
    #         "sharpe": 0.3,
    #         "total_trades": 0.2,
    #         "pnl_4w": 0.1,
    #         "win_rate": 0.1,
    #         "max_drawdown": 0.1,
    #         "volatility": 0.1,
    #         "skewness": 0.05,
    #         "turnover": 0.05,
    #         "pnl": 0.0,
    #     }

    # Lista di metriche da considerare
    metrics = list(weights.keys())

    # Normalizzazione z-score
    for m in metrics:
        if m not in df.columns:
            continue
        # Invertiamo il segno per metriche dove minore è meglio
        if m in ["max_drawdown", "volatility", "turnover"]:
            df[f"{m}_z"] = -(df[m] - df[m].mean()) / df[m].std()
        else:
            df[f"{m}_z"] = (df[m] - df[m].mean()) / df[m].std()

    # Score combinato
    df["score"] = sum(df[f"{m}_z"] * weights.get(m, 0) for m in metrics if f"{m}_z" in df.columns)

    # Selezione top N
    top_tables = df.sort_values("score", ascending=False).head(top_n)["table"].tolist()
    return top_tables


# =========================
# TEST
# =========================
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


def run_test_all_tables(db_path, validation_ratio):
    tables = get_all_tables(db_path)
    results = []

    for idx, table in enumerate(tables):
        print(f"Test table {idx + 1}/{len(tables)}: {table}")
        df = load_data(db_path, table)
        if len(df) < 100:
            continue

        _, test = split_validation_test(df, validation_ratio)
        bt = run_backtest(test)

        pnl = bt["equity"].iloc[-1] - INITIAL_CAPITAL

        results.append({
            "table": table,
            "test_pnl": pnl
        })

    return pd.DataFrame(results)


def portfolio_pnl_from_selection(test_results_df, selected_tables):
    portfolio = test_results_df[
        test_results_df["table"].isin(selected_tables)
    ]

    portfolio_df = portfolio.sort_values("test_pnl", ascending=False).filter(items=["table", "test_pnl"]).reset_index(
        drop=True)
    return {"df": portfolio_df,
            "total_pnl": portfolio_df["test_pnl"].sum(),
            "avg_pnl": portfolio_df["test_pnl"].mean()
            }


def get_real_top_n_test(test_results_df, n):
    top_df = test_results_df.sort_values("test_pnl", ascending=False).head(n).reset_index(drop=True)

    return {"df": top_df,
            "total_pnl": top_df["test_pnl"].sum(),
            "avg_pnl": top_df["test_pnl"].mean()
            }


def attach_validation_metrics(real_top_n_df, validation_df):
    return real_top_n_df.merge(
        validation_df,
        on="table",
        how="left",
        suffixes=("_test", "_validation")
    )


def portfolio_stats(test_results_df, tables):
    """
    Calcola statistiche di portafoglio dato un insieme di tabelle.
    Ogni tabella investe INITIAL_CAPITAL.
    """
    df = test_results_df[test_results_df["table"].isin(tables)]

    return {
        "num_assets": len(df),
        "total_pnl": df["test_pnl"].sum(),
        "avg_pnl": df["test_pnl"].mean(),
        "positive_ratio": (df["test_pnl"] > 0).mean()
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


def build_validation_roc_dataset(validation_df):
    """
    Costruisce il dataset per ROC usando solo il validation set.
    target = 1 se pnl > 0
    target = 0 altrimenti
    """
    df = validation_df.copy()

    # Target binario: azione profittevole nel validation
    df["positive"] = (df["pnl"] > 0).astype(int)

    return df


def plot_roc_for_metric_validation(df, metric):
    """
    Plotta la ROC curve per una metrica (validation only),
    mostrando nel titolo la threshold ottimale.
    """
    scores = df[metric]
    y_true = df["positive"]

    # Rimuoviamo NaN
    mask = scores.notna()
    scores = scores[mask]
    y_true = y_true[mask]

    # # Alcune metriche vanno invertite (più basso è meglio)
    # if metric in ["max_drawdown", "volatility", "turnover"]:
    #     scores = -scores

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(
        fpr[best_idx],
        tpr[best_idx],
        marker="o",
        label="Optimal threshold"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC – {metric}\nOptimal threshold = {best_threshold:.4f}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "metric": metric,
        "auc": auc,
        "best_threshold": best_threshold,
        "tpr": tpr[best_idx],
        "fpr": fpr[best_idx]
    }


def roc_analysis_validation(validation_df, metrics):
    roc_df = build_validation_roc_dataset(validation_df)
    results = []

    for metric in metrics:
        res = plot_roc_for_metric_validation(roc_df, metric)
        results.append(res)

    return (
        pd.DataFrame(results)
        .sort_values("auc", ascending=False)
        .reset_index(drop=True)
    )


def plot_all_roc_validation(validation_df, metrics):
    """
    Plotta tutte le curve ROC delle metriche di validation in un unico grafico.
    Mostra AUC e threshold ottimale in legenda.
    """
    df = validation_df.copy()

    # Target binario: pnl positivo nel validation
    df["positive"] = (df["pnl"] > 0).astype(int)

    plt.figure(figsize=(8, 7))

    roc_summary = []

    for metric in metrics:
        scores = df[metric]
        y_true = df["positive"]

        # Rimozione NaN
        mask = scores.notna()
        scores = scores[mask]
        y_true = y_true[mask]

        if y_true.nunique() < 2:
            continue  # ROC non definibile

        # Metriche dove più basso è meglio → invertiamo
        if metric in ["max_drawdown", "volatility", "turnover"]:
            scores = -scores

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)

        # Youden's J
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]

        plt.plot(
            fpr,
            tpr,
            label=f"{metric} | AUC={auc:.2f} | thr={best_threshold:.3f}"
        )

        roc_summary.append({
            "metric": metric,
            "auc": auc,
            "best_threshold": best_threshold
        })

    # Diagonale random
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – Validation Metrics")
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(roc_summary).sort_values("auc", ascending=False)


def plot_roc_grid_validation(validation_df, metrics, cols=3):
    """
    Plotta una ROC per ogni metrica in una griglia di subplot
    all'interno di un'unica finestra.
    """
    df = validation_df.copy()
    df["positive"] = (df["pnl"] > 0).astype(int)

    n_metrics = len(metrics)
    rows = math.ceil(n_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        scores = df[metric]
        y_true = df["positive"]

        mask = scores.notna()
        scores = scores[mask]
        y_true = y_true[mask]

        if y_true.nunique() < 2:
            ax.set_title(f"{metric} (not enough classes)")
            continue

        # # metriche dove minore è meglio
        # if metric in ["max_drawdown", "volatility", "turnover"]:
        #     scores = -scores

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)

        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thr = thresholds[best_idx]

        ax.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.scatter(fpr[best_idx], tpr[best_idx], color="red", s=30)

        ax.set_title(f"{metric}\nthr={best_thr:.3f}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True)
        ax.legend(fontsize=9)

    # rimuove subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("ROC Curves – Validation Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def chose_tickers_by_validation_thresholds(validation_df, test_df, thresholds):
    """
    Seleziona i ticker da testare nel dataset di test,
    basandosi sulle soglie calcolate sul validation set.

    Parametri:
    - validation_df: pd.DataFrame con metriche di validation, deve avere colonna 'table'
    - thresholds: dict con le soglie, es.:
        {
            "drawdown": -0.239,
            "volatility": 0.51,
            "skewness": -0.177,
            "turnover": 0.043,
            "sharpe": 0.008
        }

    Restituisce:
    - lista di ticker selezionati
    """
    # 1️⃣ seleziona i ticker validi in validation
    df_filtered = validation_df.copy()

    if "drawdown" in thresholds:
        df_filtered = df_filtered[df_filtered["max_drawdown"] >= thresholds["drawdown"]]
    if "volatility" in thresholds:
        df_filtered = df_filtered[df_filtered["volatility"] <= thresholds["volatility"]]
    if "skewness" in thresholds:
        df_filtered = df_filtered[df_filtered["skewness"] >= thresholds["skewness"]]
    if "turnover" in thresholds:
        df_filtered = df_filtered[df_filtered["turnover"] <= thresholds["turnover"]]
    if "sharpe" in thresholds:
        df_filtered = df_filtered[df_filtered["sharpe"] >= thresholds["sharpe"]]

    # 2️⃣ prendi solo i ticker
    selected_tables = df_filtered["table"].unique()

    return selected_tables


def plot_metric_with_youden(validation_df, metric):
    """
    Grafico doppio per una metrica:
    - Sopra: Youden's J vs soglia
    - Sotto: istogramma valori della metrica per PnL positivo/negativo
    Titolo: AUC e soglia ottimale
    """
    df = validation_df.copy()
    df["positive"] = (df["pnl"] > 0).astype(int)

    scores = df[metric].values
    y_true = df["positive"].values

    # Rimuoviamo NaN
    mask = ~np.isnan(scores)
    scores = scores[mask]
    y_true = y_true[mask]

    if len(np.unique(y_true)) < 2:
        print(f"Non abbastanza classi per {metric}")
        return

    # invertiamo metriche dove più basso è meglio
    invert_metrics = ["max_drawdown", "volatility", "turnover"]
    if metric in invert_metrics:
        scores = -scores

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.5]})

    # Sopra: Youden's J
    ax1.plot(thresholds, j_scores, color='blue')
    ax1.axvline(best_thr, color='red', linestyle='--', label=f"Optimal thr={best_thr:.3f}")
    ax1.set_ylabel("Youden's J")
    ax1.legend()
    ax1.grid(True)

    # Sotto: istogramma valori separati per target
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]

    bins = np.histogram_bin_edges(scores, bins='auto')
    ax2.hist(pos_scores, bins=bins, alpha=0.6, label="PnL>0", color='green')
    ax2.hist(neg_scores, bins=bins, alpha=0.6, label="PnL<0", color='red')
    ax2.set_xlabel(metric)
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"{metric} – AUC={auc:.3f}, Optimal threshold={best_thr:.3f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return best_thr, auc


def plot_all_metrics_youden_stacked_cols(validation_df, metrics, cols=3):
    """
    Plotta tutte le metriche in un'unica finestra con più colonne.
    Ogni metrica ha due subplot impilati verticalmente:
    - sopra: Youden's J vs soglia
    - sotto: istogramma valori per PnL positivo/negativo
    """
    df = validation_df.copy()
    df["positive"] = (df["pnl"] > 0).astype(int)

    n_metrics = len(metrics)
    n_rows = math.ceil(n_metrics / cols) * 2  # due righe per metrica
    fig, axes = plt.subplots(n_rows, cols, figsize=(4 * cols, 1.5 * n_rows), constrained_layout=True)

    # Flatten axes per indicizzazione semplice
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(metrics):
        scores = df[metric].values
        y_true = df["positive"].values
        mask = ~np.isnan(scores)
        scores = scores[mask]
        y_true = y_true[mask]

        if len(np.unique(y_true)) < 2:
            continue

        # invertiamo metriche dove minore è meglio
        invert_metrics = ["max_drawdown", "volatility", "turnover"]
        scores_plot = -scores if metric in invert_metrics else scores

        # ROC + Youden
        fpr, tpr, thresholds = roc_curve(y_true, scores_plot)
        auc = roc_auc_score(y_true, scores_plot)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thr = thresholds[best_idx]

        # Calcolo posizione subplot: due righe verticali per metrica
        col = i % cols
        row = (i // cols) * 2

        ax_youden = axes[row * cols + col]
        ax_hist = axes[(row + 1) * cols + col]

        # Sopra: Youden
        ax_youden.plot(thresholds, j_scores, color='blue')
        ax_youden.axvline(best_thr, color='red', linestyle='--', label=f"Optimal thr={best_thr:.3f}")
        ax_youden.set_ylabel("Youden's J")
        ax_youden.set_title(f"{metric} – AUC={auc:.3f}")
        ax_youden.grid(True)
        ax_youden.legend(fontsize=8)

        # Sotto: istogramma
        pos_scores = scores[y_true == 1]
        neg_scores = scores[y_true == 0]
        bins = np.histogram_bin_edges(scores, bins='auto')
        ax_hist.hist(pos_scores, bins=bins, alpha=0.6, color='green', label='PnL>0')
        ax_hist.hist(neg_scores, bins=bins, alpha=0.6, color='red', label='PnL<0')
        ax_hist.set_xlabel(metric)
        ax_hist.set_ylabel("Count")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True)

        # Allineiamo assi x
        ax_youden.set_xlim(ax_hist.get_xlim())

    # Rimuove eventuali subplot vuoti
    for j in range(i * 2 + 2, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("ROC + Histogram Metrics – Validation Set", fontsize=16)
    plt.show()


def axis_ref(idx):
    """
    Restituisce 'x domain' / 'x2 domain' / 'x3 domain' ecc.
    Plotly NON accetta 'x1 domain'
    """
    return "x domain" if idx == 1 else f"x{idx} domain"


def plot_all_metrics_youden_interactive(validation_df, metrics, cols=3):
    """
    Plotta tutte le metriche in una singola finestra interattiva Plotly.

    Per ogni metrica:
    - sopra: curva di Youden’s J (TPR - FPR) vs soglia
    - sotto: istogrammi dei valori reali della metrica
        - verde  -> PnL > 0
        - rosso  -> PnL < 0

    Il titolo di ogni metrica contiene:
    - AUC
    - soglia ottimale (Youden)
    """

    df = validation_df.copy()
    df["positive"] = (df["pnl"] > 0).astype(int)

    # invert_metrics = {"max_drawdown", "volatility", "turnover"}
    invert_metrics = {}

    n_metrics = len(metrics)
    rows = math.ceil(n_metrics / cols) * 2

    fig = make_subplots(
        rows=rows,
        cols=cols,
        vertical_spacing=0.06,
        subplot_titles=[""] * (rows * cols)
    )

    def axis_ref(idx):
        return "x domain" if idx == 1 else f"x{idx} domain"

    for i, metric in enumerate(metrics):
        col = (i % cols) + 1
        base_row = (i // cols) * 2 + 1

        scores = df[metric].values
        y_true = df["positive"].values

        mask = ~np.isnan(scores)
        scores = scores[mask]
        y_true = y_true[mask]

        if np.unique(y_true).size < 2:
            continue

        # === ROC / Youden ===
        roc_scores = -scores if metric in invert_metrics else scores
        fpr, tpr, thresholds = roc_curve(y_true, roc_scores)
        auc = roc_auc_score(y_true, roc_scores)

        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thr = thresholds[best_idx]

        # --- Youden plot (sopra) ---
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=j_scores,
                mode="lines",
                name=f"{metric} Youden",
                showlegend=False
            ),
            row=base_row,
            col=col
        )

        fig.add_vline(
            x=best_thr,
            row=base_row,
            col=col,
            line_dash="dash",
            line_color="red"
        )

        # --- ISTOGRAMMI (valori REALI) ---
        scores_hist = df.loc[mask, metric].values
        pos_scores = scores_hist[y_true == 1]
        neg_scores = scores_hist[y_true == 0]

        fig.add_trace(
            go.Histogram(
                x=pos_scores,
                marker_color="green",
                opacity=0.6,
                name="PnL > 0",
                showlegend=(i == 0)
            ),
            row=base_row + 1,
            col=col
        )

        fig.add_trace(
            go.Histogram(
                x=neg_scores,
                marker_color="red",
                opacity=0.6,
                name="PnL < 0",
                showlegend=(i == 0)
            ),
            row=base_row + 1,
            col=col
        )

        # --- Titolo per metrica ---
        subplot_idx = (base_row - 1) * cols + col

        fig.add_annotation(
            text=f"<b>{metric}</b><br>AUC={auc:.3f} | thr={best_thr:.4f}",
            x=0.5,
            y=1.08,
            xref=axis_ref(subplot_idx),
            yref=axis_ref(subplot_idx).replace("x", "y"),
            showarrow=False,
            align="center",
            font=dict(size=12)
        )

    fig.update_layout(
        height=350 * rows,
        title_text="Youden + Histogram Analysis (Validation Set)",
        bargap=0.1,
        hovermode="closest"
    )

    fig.show()


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
    # df = df.copy()
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
        raise ValueError("Nessun ticker soddisfa i criteri")

    # 2. ranking pesato
    ranked = add_weighted_score(filtered, metric_priority_map)

    ranked = ranked.sort_values("score", ascending=False)

    if top_n is not None:
        ranked = ranked.head(top_n)

    return ranked


# =========================
# WORKFLOW
# =========================
# 1. Validation (una volta)
validation_df = run_validation()

# 1. Carico metriche di validation
validation_df = load_validation_results()



metrics_weights = {
    "sharpe": 0.3,
    "total_trades": 0.2,
    "pnl_4w": 0.1,
    "win_rate": 0.1,
    "max_drawdown": 0.1,
    "volatility": 0.1,
    "skewness": 0.05,
    "pnl_last_200_bars": 0.05,
    # "turnover": 0.05,
    # "pnl": 0.0,
}




# plot_roc_grid_validation(
#     validation_df=validation_df,
#     metrics=metrics_weights.keys()
# )

# plot_all_metrics_youden_stacked_cols(validation_df, metrics_weights.keys())
# plot_all_metrics_youden_interactive(
#     validation_df,
#     metrics_weights.keys(),
#     cols=3
# )

# for metric in metrics_weights.keys():
#     plot_metric_with_youden(validation_df, metric)

# exit()

# roc_results = roc_analysis_validation(
#     validation_df=validation_df,
#     metrics=metrics_weights.keys()
# )

# print("ROC RESULTS: \n", roc_results)

# 2. Selezione TOP N ex-ante (con ranking avanzato)
# top_n_selected = select_top_n_with_zscore_and_weights(
#     validation_df,
#     top_n=TOP_N,
#     weights=metrics_weights
# )

FILTER_RULES = {
    "sharpe": {
        "threshold": 0.0081,
        "direction": ">"
    },
    "pnl_4w": {
        "threshold": 0,
        "direction": ">"
    },
    "pnl_last_200_bars": {
        "threshold": 0,
        "direction": ">"
    },
    "win_rate": {
        "threshold": 0.5,
        "direction": ">"
    },
    "max_drawdown": {
        "threshold": -0.2,
        "direction": ">"
    },
    "skewness": {
        "threshold": -0.177,
        "direction": ">"
    },
    "volatility": {
        "threshold": 0.51,
        "direction": "<"
    }
}

METRIC_PRIORITY = {
    # "sharpe": 1.00,
    "pnl_last_200_bars": 0.99,
    "win_rate" : 0.80,
    # "pnl_4w": 0.50,
    # "max_drawdown": 0.76,
    # "skewness": 0.72,
    # "volatility": 0.62
}

top_selected_df = get_selected_ranked_tickers(
    validation_df=validation_df,
    filter_rules=FILTER_RULES,
    metric_priority_map=METRIC_PRIORITY,
    # top_n=10
)

selected_counter = len(top_selected_df)

top_selected_tables = top_selected_df["table"].tolist()

print(f"✅ Selezionate {selected_counter} azioni per il test.")

# 3. Test su TUTTE le azioni
test_results_all = run_test_all_tables(DB_PATH, VALIDATION_RATIO)

# 4. PnL portafoglio selezionato
selected_portfolio = portfolio_pnl_from_selection(
    test_results_all,
    top_selected_tables
)

# 5. Vere TOP N nel test
real_top_n_test = get_real_top_n_test(
    test_results_all,
    selected_counter
)

# 6. Metriche di validation delle vere TOP N
real_top_n_with_validation = attach_validation_metrics(
    real_top_n_test["df"],
    validation_df
)

print("📦 PORTAFOGLIO SELEZIONATO (ex-ante)")
# aggiungi l'informazione sul ranking che avevano
print(selected_portfolio["df"].merge(
    top_selected_df[["table"]].reset_index().rename(columns={"index": "rank"}),
    on="table",
    how="left"
).sort_values("test_pnl", ascending=False))
print(f"\n💰 PNL TOTALE PORTAFOGLIO SELEZIONATO: {selected_portfolio['total_pnl']:.2f} €")
print(f"💰 PNL MEDIO PER AZIONE SELEZIONATA: {selected_portfolio['avg_pnl']:.2f} €")

print(f"\n🥇 REALI TOP {selected_counter} NEL TEST (oracle)")
print(real_top_n_test["df"])
print(f"\n🥇 PNL TOTALE REALI TOP {selected_counter}: {real_top_n_test['total_pnl']:.2f} €")
print(f"🥇 PNL MEDIO REALI TOP {selected_counter}: {real_top_n_test['avg_pnl']:.2f} €")

print(f"\n🔍 METRICHE DI VALIDATION DELLE TOP {selected_counter} SELEZIONATE")
print(

)

print(f"\n🔍 METRICHE DI VALIDATION DELLE REALI TOP {selected_counter}")
print(
    real_top_n_with_validation[
        ["table", "test_pnl", "sharpe", "pnl", "max_drawdown",
         "skewness" ]
    ]
)

selected_portfolio_stats = portfolio_stats(
    test_results_df=test_results_all,
    tables=top_selected_tables
)

print("📦 PORTAFOGLIO TOP N SELEZIONATE (ex-ante)")
for k, v in selected_portfolio_stats.items():
    print(f"{k}: {v}")

real_top_n_tables = real_top_n_test["df"]["table"].tolist()

oracle_portfolio_stats = portfolio_stats(
    test_results_df=test_results_all,
    tables=real_top_n_tables
)

print("\n🥇 PORTAFOGLIO TOP N ORACLE (ex-post)")
for k, v in oracle_portfolio_stats.items():
    print(f"{k}: {v}")

# Efficiency score
eff_score = efficiency_score(
    selected_portfolio_stats,
    oracle_portfolio_stats
)

print("\n📊 EFFICIENCY SCORE")
print(f"Efficiency Score: {eff_score:.2f}")

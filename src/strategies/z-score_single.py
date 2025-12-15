import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETRI GENERALI
# =========================
DB_PATH = "italian_stocks.db"
TABLE_NAME = "ohlcv"
SYMBOL = "ENI"

INITIAL_CAPITAL = 100_000

# =========================
# PARAMETRI STRATEGIA
# =========================
ROLLING_WINDOW = 20
ZSCORE_THRESHOLD = 2.0

# =========================
# COMMISSIONI
# =========================
# COMMISSION_TYPE = "fixed"
# COMMISSION_VALUE = 8.0      # € per trade

COMMISSION_TYPE = "percent"
COMMISSION_VALUE = 0.0019     # 0.19%

#CARICAMENTO DATI

conn = sqlite3.connect(DB_PATH)

query = f"""
SELECT date, close
FROM {TABLE_NAME}
WHERE symbol = '{SYMBOL}'
ORDER BY date
"""

df = pd.read_sql(query, conn, parse_dates=["date"])
conn.close()

df.set_index("date", inplace=True)

#CALCOLO Z-SCORE

df["mean"] = df["close"].rolling(ROLLING_WINDOW).mean()
df["std"] = df["close"].rolling(ROLLING_WINDOW).std()
df["zscore"] = (df["close"] - df["mean"]) / df["std"]

#GENERAZIONE SEGNALI DI TRADING

df["position"] = 0

df.loc[df["zscore"] < -ZSCORE_THRESHOLD, "position"] = 1   # LONG
df.loc[df["zscore"] > ZSCORE_THRESHOLD, "position"] = -1  # SHORT

# Manteniamo la posizione finché non arriva un nuovo segnale
df["position"] = (
    df["position"]
    .replace(0, np.nan)
    .ffill()
    .fillna(0)
)


# CONTEGGIO TRADINGS
df["trades"] = df["position"].diff().abs()
df["trades"].fillna(0, inplace=True)


# COMMISSIONI
df["commission"] = 0.0

if COMMISSION_TYPE == "fixed":
    df.loc[df["trades"] > 0, "commission"] = (
            df["trades"] * COMMISSION_VALUE
    )

elif COMMISSION_TYPE == "percent":
    traded_capital = INITIAL_CAPITAL
    df.loc[df["trades"] > 0, "commission"] = (
            df["trades"] * traded_capital * COMMISSION_VALUE
    )


# RETURNS AND PNL
# Ritorni del mercato
df["returns"] = df["close"].pct_change()

# Ritorni lordi strategia (NO look-ahead)
df["strategy_returns_gross"] = (
        df["position"].shift(1) * df["returns"]
)

# Ritorni netti
df["strategy_returns_net"] = (
        df["strategy_returns_gross"]
        - df["commission"] / INITIAL_CAPITAL
)

# Equity curve
df["equity"] = (
        (1 + df["strategy_returns_net"])
        .cumprod()
        * INITIAL_CAPITAL
)

df["pnl"] = df["equity"] - INITIAL_CAPITAL



# METRICHE
total_return = df["equity"].iloc[-1] / INITIAL_CAPITAL - 1

max_drawdown = (
        df["equity"] / df["equity"].cummax() - 1
).min()

sharpe = (
                 df["strategy_returns_net"].mean() /
                 df["strategy_returns_net"].std()
         ) * np.sqrt(252)

print("===== PERFORMANCE =====")
print(f"Total Return: {total_return:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")


# PLOT EQUITY CURVE
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["equity"], label="Equity Curve")
plt.title(f"Z-Score Mean Reversion – {SYMBOL}")
plt.xlabel("Date")
plt.ylabel("Equity (€)")
plt.legend()
plt.grid()
plt.show()


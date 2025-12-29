import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def plot_price_and_zscore(ax_price, ax_z, df, title, zscore_threshold):
    ax_price.clear()
    ax_z.clear()

    # ===== PRICE =====
    ax_price.plot(df.index, df["close"], color="black", label="Close")

    # mean (rolling mean usata per lo z-score)
    if "mean" in df.columns:
        ax_price.plot(df.index, df["mean"], color="orange", linestyle="--", label="Mean")

    long_entries = df[(df["position"].shift(1) == 0) & (df["position"] == 1)]
    short_entries = df[(df["position"].shift(1) == 0) & (df["position"] == -1)]
    exits = df[(df["position"].shift(1) != 0) & (df["position"] == 0)]

    ax_price.scatter(
        long_entries.index, long_entries["close"],
        marker="^", color="green", label="Long"
    )
    ax_price.scatter(
        short_entries.index, short_entries["close"],
        marker="v", color="red", label="Short"
    )
    ax_price.scatter(
        exits.index, exits["close"],
        marker="x", color="blue", label="Exit"
    )

    ax_price.set_title(title)
    ax_price.legend()
    ax_price.grid(True)

    # 🔕 rimuove label asse X (rimane solo nello z-score)
    ax_price.tick_params(labelbottom=False)

    # ===== Z-SCORE =====
    ax_z.plot(df.index, df["zscore"], color="purple")
    ax_z.axhline(0, color="black")
    ax_z.axhline(zscore_threshold, color="red", linestyle="--")
    ax_z.axhline(-zscore_threshold, color="red", linestyle="--")

    ax_z.set_ylabel("Z-score")
    ax_z.grid(True)


class WalkForwardInspector:
    def __init__(self, wf_df, zscore_threshold, wf_idx=0):
        self.wf_df = wf_df
        self.zscore_threshold = zscore_threshold

        self.wf_idx = wf_idx
        self.pair_idx = 0

        self.fig = plt.figure(figsize=(14, 10))

        gs = self.fig.add_gridspec(
            4, 1,
            height_ratios=[3.2, 0.8, 3.2, 0.8],
            hspace=0.25   # compatta verticalmente
        )

        self.fig.subplots_adjust(top=0.93)

        self.ax_sel_price = self.fig.add_subplot(gs[0])
        self.ax_sel_z = self.fig.add_subplot(gs[1], sharex=self.ax_sel_price)

        self.ax_or_price = self.fig.add_subplot(gs[2])
        self.ax_or_z = self.fig.add_subplot(gs[3], sharex=self.ax_or_price)

        # =====================
        # BOTTONI
        # =====================
        ax_prev_pair = plt.axes([0.38, 0.01, 0.12, 0.045])
        ax_next_pair = plt.axes([0.52, 0.01, 0.12, 0.045])


        ax_prev_wf = plt.axes([0.01, 0.45, 0.04, 0.1])
        ax_next_wf = plt.axes([0.95, 0.45, 0.04, 0.1])

        self.btn_prev_pair = Button(ax_prev_pair, "⬅ Prev Pair")
        self.btn_next_pair = Button(ax_next_pair, "Next Pair ➡")

        self.btn_prev_wf = Button(ax_prev_wf, "⬆ Prev WF")
        self.btn_next_wf = Button(ax_next_wf, "⬇ Next WF")

        self.btn_prev_pair.on_clicked(self.prev_pair)
        self.btn_next_pair.on_clicked(self.next_pair)
        self.btn_prev_wf.on_clicked(self.prev_wf)
        self.btn_next_wf.on_clicked(self.next_wf)

        self.load_wf()
        self.update()

        plt.show()

    # =====================
    # CARICAMENTO WF
    # =====================
    def load_wf(self):
        self.wf = self.wf_df.iloc[self.wf_idx]

        self.selected = self.wf["selected_tables"]
        self.oracle = self.wf["oracle_tables"]
        self.backtests = self.wf["test_backtests"]

        self.max_pairs = min(len(self.selected), len(self.oracle))
        self.pair_idx = 0

    # =====================
    # UPDATE PLOT
    # =====================
    def update(self):
        if self.max_pairs == 0:
            self.fig.suptitle(
                f"WF {self.wf_idx} | start bar {self.wf['start_bar']} | NO PAIRS",
                fontsize=13, y=0.98
            )

            for ax in [
                self.ax_sel_price, self.ax_sel_z,
                self.ax_or_price, self.ax_or_z
            ]:
                ax.clear()

            self.ax_sel_price.text(
                0.5, 0.5, "No selected assets",
                ha="center", va="center", transform=self.ax_sel_price.transAxes
            )
            self.ax_or_price.text(
                0.5, 0.5, "No oracle assets",
                ha="center", va="center", transform=self.ax_or_price.transAxes
            )

            self.fig.canvas.draw_idle()
            return

        # --- tabelle correnti ---
        t_sel = self.selected[self.pair_idx]
        t_or = self.oracle[self.pair_idx]

        df_sel = self.backtests[t_sel]
        df_or = self.backtests[t_or]

        # --- PNL pair ---
        # pnl_sel_pair = compute_pnl(df_sel)
        # pnl_or_pair = compute_pnl(df_or)

        # --- PNL finestra WF (aggregato) ---
        wf_pnl_sel = 0.0
        wf_pnl_or = 0.0

        # for t in self.selected:
            # wf_pnl_sel += compute_pnl(self.backtests[t])

        # for t in self.oracle:
            # wf_pnl_or += compute_pnl(self.backtests[t])

        # --- titolo compatto ---
        self.fig.suptitle(
            f"WF {self.wf_idx} | start bar {self.wf['start_bar']} | "
            f"Pair {self.pair_idx + 1}/{self.max_pairs}",
            fontsize=13, y=0.98
        )

        # --- grafico SELECTED ---
        plot_price_and_zscore(
            self.ax_sel_price,
            self.ax_sel_z,
            df_sel,
            f"SELECTED [{self.pair_idx}]: {t_sel}",
            self.zscore_threshold
        )

        # --- grafico ORACLE ---
        plot_price_and_zscore(
            self.ax_or_price,
            self.ax_or_z,
            df_or,
            f"ORACLE [{self.pair_idx}]: {t_or}",
            self.zscore_threshold
        )

        # --- refresh ---
        self.fig.canvas.draw_idle()


    # =====================
    # CALLBACK BOTTONI
    # =====================
    def next_pair(self, event):
        if self.pair_idx < self.max_pairs - 1:
            self.pair_idx += 1
            self.update()

    def prev_pair(self, event):
        if self.pair_idx > 0:
            self.pair_idx -= 1
            self.update()

    def next_wf(self, event):
        if self.wf_idx < len(self.wf_df) - 1:
            self.wf_idx += 1
            self.load_wf()
            self.update()

    def prev_wf(self, event):
        if self.wf_idx > 0:
            self.wf_idx -= 1
            self.load_wf()
            self.update()


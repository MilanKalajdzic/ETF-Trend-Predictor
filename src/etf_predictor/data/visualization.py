"""Exploratory data analysis plots for the ETF dataset.

Reproduces and extends the paper's figures:
    - Figure 1 style: ETF open price behaviour over time
    - Figure 3 style: Cumulative Γ movement per ETF
    - Class balance bar chart
    - Missing data heatmap (pre-cleaning)
    - Indicator correlation heatmap
    - Sector exposure table plot

All methods return ``matplotlib.figure.Figure`` objects so they can
be saved or embedded in a Quarto report.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# European ETF sector weights (approximate, for visualization)
# Source: iShares / SPDR fund pages
SECTOR_WEIGHTS = {
    "IEUR": {
        "Financials": 17.2,
        "Industrials": 15.1,
        "Health Care": 13.8,
        "Cons. Disc.": 11.4,
        "Materials": 8.9,
    },
    "FEZ": {
        "Financials": 20.5,
        "Industrials": 14.3,
        "Technology": 10.2,
        "Cons. Disc.": 10.1,
        "Energy": 7.8,
    },
    "EUFN": {
        "Banks": 55.3,
        "Diversified Fin.": 18.6,
        "Insurance": 14.7,
        "Real Estate": 6.1,
        "Capital Markets": 5.3,
    },
    "IVV": {
        "Info. Tech": 27.7,
        "Health Care": 13.4,
        "Cons. Disc.": 12.0,
        "Communication": 11.2,
        "Financials": 10.9,
    },
}


class EDAVisualizer:
    """Generate EDA plots for the ETF dataset.

    Parameters
    ----------
    style : str
        Matplotlib style sheet to use. Defaults to ``"seaborn-v0_8-whitegrid"``.
    fig_size : tuple[float, float]
        Default figure size (width, height) in inches.
    dpi : int
        Figure DPI for saving.

    Examples
    --------
    >>> viz = EDAVisualizer()
    >>> fig = viz.plot_open_prices(raw_data)
    >>> fig.savefig("reports/figures/open_prices.png", dpi=150)

    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        fig_size: tuple[float, float] = (12, 5),
        dpi: int = 150,
    ) -> None:
        self.style = style
        self.fig_size = fig_size
        self.dpi = dpi
        try:
            plt.style.use(self.style)
        except OSError:
            logger.warning(
                "Style '%s' not found, using default.", self.style
            )

    # ------------------------------------------------------------------
    # Figure 1 equivalent — Open price behaviour
    # ------------------------------------------------------------------

    def plot_open_prices(
        self,
        raw_data: dict[str, pd.DataFrame],
        tickers: list[str] | None = None,
        normalize: bool = True,
    ) -> Figure:
        """Plot normalised (or raw) Open prices for all ETFs.

        Reproduces the style of Figure 2 from the paper.

        Parameters
        ----------
        raw_data : dict[str, pd.DataFrame]
            Mapping of ticker → raw OHLCV DataFrame.
        tickers : list[str] or None
            Subset of tickers to plot. ``None`` → plot all.
        normalize : bool
            If ``True``, normalise each series to start at 1.0 so
            that relative performance is visible on the same axis.

        Returns
        -------
        Figure

        """
        tickers = tickers or list(raw_data.keys())
        fig, ax = plt.subplots(figsize=self.fig_size)

        for ticker in tickers:
            series = raw_data[ticker]["Open"].dropna()
            if normalize:
                series = series / series.iloc[0]
            ax.plot(series.index, series.values, label=ticker, linewidth=1.2)

        ax.set_title("ETF Open Price Behaviour (normalised to 1.0)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalised Open Price" if normalize else "Open Price (USD)")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 3 equivalent — Cumulative Γ movement
    # ------------------------------------------------------------------

    def plot_cumulative_gamma(
        self,
        processed_data: dict[str, pd.DataFrame],
        tickers: list[str] | None = None,
        target_col: str = "Gamma",
    ) -> Figure:
        """Plot cumulative sum of Γ over time (paper Figure 3 style).

        A rising line means more UP days than DOWN days in that period.

        Parameters
        ----------
        processed_data : dict[str, pd.DataFrame]
            Mapping of ticker → processed DataFrame containing
            ``target_col``.
        tickers : list[str] or None
            Subset to plot.
        target_col : str
            Name of the target column.

        Returns
        -------
        Figure

        """
        tickers = tickers or list(processed_data.keys())
        fig, ax = plt.subplots(figsize=self.fig_size)

        for ticker in tickers:
            df = processed_data[ticker]
            if target_col not in df.columns:
                logger.warning(
                    "%s: target column '%s' not found, skipping.",
                    ticker,
                    target_col,
                )
                continue
            gamma_cum = df[target_col].cumsum().reset_index(drop=True)
            ax.plot(gamma_cum.index, gamma_cum.values, label=ticker, linewidth=1.2)

        ax.set_title("Cumulative Γ Movement per ETF")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Cumulative Γ")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Class balance
    # ------------------------------------------------------------------

    def plot_class_balance(
        self,
        processed_data: dict[str, pd.DataFrame],
        target_col: str = "Gamma",
    ) -> Figure:
        """Bar chart showing UP / DOWN class balance per ETF.

        Parameters
        ----------
        processed_data : dict[str, pd.DataFrame]
            Processed DataFrames with target column.
        target_col : str
            Name of the target column.

        Returns
        -------
        Figure

        """
        tickers = list(processed_data.keys())
        up_vals = []
        down_vals = []

        for ticker in tickers:
            df = processed_data[ticker]
            total = len(df)
            up_vals.append(100 * (df[target_col] == 1).sum() / total)
            down_vals.append(100 * (df[target_col] == -1).sum() / total)

        x = np.arange(len(tickers))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))
        bars_up = ax.bar(
            x - width / 2, up_vals, width, label="UP (+1)", color="#4C72B0"
        )
        bars_dn = ax.bar(
            x + width / 2, down_vals, width, label="DOWN (−1)", color="#DD8452"
        )

        ax.set_title("Class Balance per ETF")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers)
        ax.set_ylim(0, 70)
        ax.axhline(50, color="grey", linewidth=0.8, linestyle="--", label="50% line")
        ax.legend()

        for bar in list(bars_up) + list(bars_dn):
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Missing data heatmap
    # ------------------------------------------------------------------

    def plot_missing_data(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        max_cols: int = 80,
    ) -> Figure:
        """Heatmap of missing values across indicator columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame *before* cleaning (after indicator computation).
        ticker : str
            Ticker label for the title.
        max_cols : int
            Maximum number of columns to show (avoids overly wide plots).

        Returns
        -------
        Figure

        """
        nan_mask = df.isna().iloc[:, :max_cols]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.imshow(
            nan_mask.T.values,
            aspect="auto",
            cmap="RdYlGn_r",
            interpolation="none",
        )
        ax.set_title(
            f"Missing Data Heatmap — {ticker} "
            f"(first {max_cols} columns, green=present, red=NaN)"
        )
        ax.set_xlabel("Time (rows)")
        ax.set_ylabel("Features")
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Correlation heatmap (Selected(5) features)
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        cols: list[str] | None = None,
        ticker: str = "",
    ) -> Figure:
        """Pearson correlation heatmap for selected feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Processed DataFrame.
        cols : list[str] or None
            Columns to include. ``None`` → all numeric non-target cols
            (capped at 30 for readability).
        ticker : str
            Ticker label for the title.

        Returns
        -------
        Figure

        """
        if cols is None:
            numeric = df.select_dtypes(include="number").columns.tolist()
            cols = [c for c in numeric if c != "Gamma"][:30]

        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=6)
        ax.set_yticklabels(cols, fontsize=6)
        ax.set_title(f"Pearson Correlation — {ticker}")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Sector exposure
    # ------------------------------------------------------------------

    def plot_sector_exposure(
        self,
        sector_weights: dict[str, dict[str, float]] | None = None,
    ) -> Figure:
        """Horizontal bar chart of top sector weights per ETF.

        Parameters
        ----------
        sector_weights : dict or None
            Mapping of ticker → {sector: weight_%}.
            Defaults to the hardcoded approximate weights.

        Returns
        -------
        Figure

        """
        weights = sector_weights or SECTOR_WEIGHTS
        tickers = list(weights.keys())
        n = len(tickers)

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
        if n == 1:
            axes = [axes]

        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        for ax, ticker in zip(axes, tickers, strict=False):
            sectors = list(weights[ticker].keys())
            values = list(weights[ticker].values())
            bars = ax.barh(sectors, values, color=colors[: len(sectors)])
            ax.set_title(ticker, fontweight="bold")
            ax.set_xlabel("Weight (%)")
            ax.invert_yaxis()
            for bar, val in zip(bars, values, strict=True):
                ax.text(
                    val + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=8,
                )
            ax.set_xlim(0, max(values) * 1.25)

        fig.suptitle("Top Sector Exposure per ETF", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Convenience: save all figures
    # ------------------------------------------------------------------

    def save_all(
        self,
        figures: dict[str, Figure],
        output_dir: str = "reports/figures",
    ) -> None:
        """Save a dict of named figures to *output_dir*.

        Parameters
        ----------
        figures : dict[str, Figure]
            Mapping of filename stem → Figure.
        output_dir : str
            Directory to write PNG files.

        """
        from pathlib import Path  # noqa: PLC0415

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = out / f"{name}.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved figure: %s", path)
        plt.close("all")

"""
Equity-curve construction and visualisation for walk-forward backtests
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_predictor.models.walk_forward import WalkForwardResult

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def equity_curve(strategy_returns: pd.Series) -> pd.Series:
    """
    Compound a return series into equity curve.

    Parameters
    strategy_returns : pd.Series
        Daily simple returns. NaNs are treated as 0 (no position effect).

    Returns
    pd.Series
        Equity values; index preserved.
    """
    r = strategy_returns.fillna(0.0)
    return (1.0 + r).cumprod()


def buy_and_hold_curve(
    close_unscaled: pd.Series, index: pd.Index) -> pd.Series:
    """
    Buy-and-hold equity curve aligned with *index*.
    """
    close = close_unscaled.reindex(index).ffill()
    return close / close.iloc[0]


def summary_metrics(strategy_returns: pd.Series) -> dict[str, float]:
    """Return total/annualised/sharpe/drawdown stats for a return series."""
    r = strategy_returns.dropna()
    if len(r) == 0:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "n_days": 0,
        }
    eq = equity_curve(r)
    total = float(eq.iloc[-1] - 1.0)
    years = max(len(r) / TRADING_DAYS, 1e-9)
    ann_ret = float((eq.iloc[-1]) ** (1 / years) - 1.0)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan
    drawdown = float((eq / eq.cummax() - 1.0).min())
    hit = float((r > 0).mean())
    return {
        "total_return": total,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "hit_rate": hit,
        "n_days": int(len(r)),
    }


def build_comparison(
    results: Iterable[WalkForwardResult],
    close_unscaled: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned equity curves and a metrics table for several models.

    Parameters
    results : iterable of WalkForwardResult
        Each model's walk-forward output.
    close_unscaled : pd.Series
        Unscaled close price used to construct the buy-and-hold baseline.

    Returns
    (curves, metrics) : tuple of pd.DataFrame
        ``curves`` is indexed by date, one column per model plus
        ``"BuyAndHold"``. ``metrics`` has one row per model and the
        buy-and-hold baseline.
    """
    results = list(results)
    if not results:
        raise ValueError("Need at least one WalkForwardResult.")

    common_index = results[0].predictions.index
    for r in results[1:]:
        common_index = common_index.intersection(r.predictions.index)
    if len(common_index) == 0:
        raise ValueError(
            "WalkForwardResults have no overlapping test dates."
        )

    curves: dict[str, pd.Series] = {}
    metric_rows = []

    for res in results:
        ret = res.predictions.loc[common_index, "strategy_return"]
        # strategy_return at day t is earned over t -> t+1, while the
        # buy-and-hold curve at day t only includes returns up to t.
        # Shift by one day so every curve starts at 1.0 and both are
        # marked to market on the same day.
        curves[res.model_name] = equity_curve(ret.shift(1))
        row = {"model": res.model_name, **summary_metrics(ret)}
        metric_rows.append(row)

    bh = buy_and_hold_curve(close_unscaled, common_index)
    curves["BuyAndHold"] = bh
    bh_returns = bh.pct_change()
    metric_rows.append({"model": "BuyAndHold", **summary_metrics(bh_returns)})

    curves_df = pd.DataFrame(curves)
    metrics_df = pd.DataFrame(metric_rows).set_index("model")
    return curves_df, metrics_df


def plot_equity_curves(
    curves: pd.DataFrame,
    title: str = "Walk-forward equity curves",
    out_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (11.0, 6.0),
) -> plt.Figure:
    """Plot model equity curves on a single axes.

    Parameters
    curves : pd.DataFrame
        Output of :func:`build_comparison`.
    title : str
        Figure title.
    out_path : str or Path or None
        If given, the figure is saved (parent dirs are created).
    figsize : tuple
        Matplotlib figure size.

    Return
    matplotlib.figure.Figure
        The created figure (also displayed if ``out_path`` is ``None``).
    """
    fig, ax = plt.subplots(figsize=figsize)
    styles = {
        "BuyAndHold": {"linestyle": "--", "color": "grey", "alpha": 0.8},
        "MLP": {"linestyle": "-", "color": "C0"},
        "LSTM": {"linestyle": "-", "color": "C3"},
    }
    for col in curves.columns:
        s = styles.get(col, {})
        ax.plot(curves.index, curves[col], label=col, **s)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        logger.info("Saved equity plot: %s", out_path)
    return fig


def plot_fold_metric(
    fold_metrics: dict[str, pd.DataFrame],
    metric: str = "accuracy",
    out_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (10.0, 4.5),
) -> plt.Figure:
    """Bar chart of a per-fold metric across models.

    Parameters
    ----------
    fold_metrics : dict[model_name → pd.DataFrame]
        Each frame should match ``WalkForwardResult.fold_metrics``.
    metric : str
        Column to plot (e.g. ``"accuracy"``, ``"sharpe"``,
        ``"mean_return"``).
    out_path : str or Path or None
        If provided, save the figure here.
    """
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / max(len(fold_metrics), 1)
    folds = None
    for offset, (name, df) in enumerate(fold_metrics.items()):
        if metric not in df.columns:
            raise KeyError(
                f"Metric '{metric}' not in fold_metrics for {name}."
            )
        if folds is None:
            folds = df["fold"].values
        x = np.arange(len(df)) + offset * width
        ax.bar(x, df[metric].values, width=width, label=name)

    if folds is not None:
        ax.set_xticks(np.arange(len(folds)) + width * (len(fold_metrics) - 1) / 2)
        ax.set_xticklabels([str(f) for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    ax.set_title(f"Per-fold {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        logger.info("Saved fold metric plot: %s", out_path)
    return fig

"""Re-score every walk-forward run net of transaction costs.

Reads the saved walk-forward predictions, charges a one-way cost per
unit of notional traded (a long-to-short flip trades two units) and
recomputes the Sharpe ratio at several cost levels. Also reports how
often each strategy changes position and the cost at which its Sharpe
ratio would fall to zero.

Writes ``reports/results/cost_sensitivity.csv``.

Usage::

    PYTHONPATH=src python scripts/cost_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_predictor.models.equity import (
    TRADING_DAYS,
    apply_transaction_costs,
    summary_metrics,
)

RESULTS_DIR = Path("reports/results")
TICKERS = ["IEUR", "FEZ", "EUFN", "IVV"]
MODELS = ["MLP", "LSTM"]
FEATURE_SETS = {"all": "", "top10": "_top10"}
COSTS_BPS = [0, 1, 2, 5, 10]


def sharpe_net(signal: pd.Series, returns: pd.Series, cost_bps: float) -> float:
    """Sharpe ratio of *returns* after deducting *cost_bps* per unit traded."""
    net = apply_transaction_costs(signal, returns, cost_bps)
    return summary_metrics(net)["sharpe"]


def breakeven_bps(signal: pd.Series, returns: pd.Series, upper: float = 500.0) -> float:
    """One-way cost in bps at which the Sharpe ratio falls to zero."""
    if sharpe_net(signal, returns, 0.0) <= 0:
        return 0.0
    lo, hi = 0.0, upper
    for _ in range(50):
        mid = (lo + hi) / 2
        if sharpe_net(signal, returns, mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    """Score all saved runs and write the cost sensitivity table."""
    rows = []
    for ticker in TICKERS:
        for features, suffix in FEATURE_SETS.items():
            metrics_path = RESULTS_DIR / f"{ticker}_walkforward_metrics{suffix}.csv"
            if not metrics_path.exists():
                continue
            bh_sharpe = pd.read_csv(metrics_path, index_col="model").loc[
                "BuyAndHold", "sharpe"
            ]
            for model in MODELS:
                path = RESULTS_DIR / f"{ticker}_{model}_predictions{suffix}.csv"
                if not path.exists():
                    continue
                pred = pd.read_csv(path, index_col="Date", parse_dates=True)
                signal, returns = pred["signal"], pred["strategy_return"]
                flips = int((signal.diff().fillna(0) != 0).sum())
                row = {
                    "ticker": ticker,
                    "model": model,
                    "features": features,
                    "flips_per_year": flips / (len(pred) / TRADING_DAYS),
                    "long_share": float((signal == 1).mean()),
                }
                for bps in COSTS_BPS:
                    row[f"sharpe_{bps}bps"] = sharpe_net(signal, returns, bps)
                row["breakeven_bps"] = breakeven_bps(signal, returns)
                row["buy_and_hold_sharpe"] = bh_sharpe
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

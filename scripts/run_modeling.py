"""Train and backtest the MLP and LSTM models on every ETF.

Both models are evaluated on the processed dataset with expanding-window
walk-forward validation. Outputs:

    reports/figures/equity_<TICKER>.png            equity-curve comparison
    reports/figures/fold_accuracy_<TICKER>.png     per-fold accuracy bars
    reports/results/<TICKER>_walkforward_metrics.csv   summary metrics
    reports/results/<TICKER>_<model>_folds.csv         per-fold metrics
    reports/results/<TICKER>_<model>_predictions.csv   day-level signals

Usage
    PYTHONPATH=src python scripts/run_modeling.py
    # or
    make modeling
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from etf_predictor.data.loader import YahooFinanceLoader
from etf_predictor.data.pipeline import DataPipeline
from etf_predictor.models.equity import (
    build_comparison,
    plot_equity_curves,
    plot_fold_metric,
)
from etf_predictor.models.lstm_value import LSTMValueModel
from etf_predictor.models.mlp_signal import MLPSignalModel
from etf_predictor.models.walk_forward import WalkForwardValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_modeling")

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--initial-train", type=int, default=750,
        help="Rows in the first training window (~3y of trading days).",
    )
    p.add_argument(
        "--test-size", type=int, default=120,
        help="Test rows per fold (~6 months of trading days).",
    )
    p.add_argument(
        "--max-folds", type=int, default=None,
        help="Optional cap on the number of folds (useful for smoke tests).",
    )
    p.add_argument(
        "--mlp-epochs", type=int, default=30,
        help="Training epochs for the MLP per fold.",
    )
    p.add_argument(
        "--lstm-epochs", type=int, default=20,
        help="Training epochs for the LSTM per fold.",
    )
    p.add_argument(
        "--lstm-seq-len", type=int, default=20,
        help="LSTM look-back window length.",
    )
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Subset of tickers to run (defaults to all processed).",
    )
    p.add_argument(
        "--top-k-features", type=int, default=None,
        help=(
            "If set, train on only the top-K features from "
            "reports/results/<TICKER>_feature_importance.csv "
            "(plus the Gamma target). Mirrors the Selected(n) "
            "approach from Sagaceta-Mejía et al. (2024)."
        ),
    )
    p.add_argument(
        "--feature-suffix", type=str, default="",
        help=(
            "Suffix appended to every output filename (e.g. '_top10'). "
            "Use this when running with --top-k-features so the full-"
            "feature baseline outputs are not overwritten."
        ),
    )
    return p.parse_args()


def load_top_k_features(ticker: str, k: int) -> list[str]:
    """Return the top-K feature names from the per-ticker importance CSV."""
    path = RESULTS_DIR / f"{ticker}_feature_importance.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature-importance file not found for {ticker}: {path}. "
            "Run the statistical analysis (`make analysis`) first."
        )
    imp = pd.read_csv(path)
    if "feature" not in imp.columns:
        raise KeyError(
            f"{path} must have a 'feature' column; got {list(imp.columns)}."
        )
    return imp["feature"].head(k).tolist()


def select_features(
    df: pd.DataFrame, ticker: str, top_k: int, target_col: str = "Gamma",
) -> pd.DataFrame:
    """Slice *df* to the top-K importance features plus the target column."""
    top_features = load_top_k_features(ticker, top_k)
    missing = [f for f in top_features if f not in df.columns]
    if missing:
        raise KeyError(
            f"{ticker}: top-K features not in processed frame: {missing}"
        )
    cols = top_features + [target_col]
    logger.info(
        "[%s] training on top-%d features: %s",
        ticker, top_k, top_features,
    )
    return df[cols].copy()


def load_processed_or_run_pipeline(
    tickers_filter: list[str] | None,
) -> dict[str, pd.DataFrame]:
    """Load the scaled/cleaned per-ticker frames, building them if absent."""
    processed_dir = Path("data/processed")
    available = sorted(processed_dir.glob("*_processed.parquet"))
    if not available:
        logger.info("No processed parquet files — running DataPipeline.")
        datasets = DataPipeline().run()
    else:
        datasets = {}
        for path in available:
            ticker = path.stem.replace("_processed", "")
            datasets[ticker] = pd.read_parquet(path)
        logger.info("Loaded processed parquet for: %s", list(datasets.keys()))

    if tickers_filter:
        datasets = {t: d for t, d in datasets.items() if t in tickers_filter}
        if not datasets:
            raise ValueError(
                f"None of --tickers={tickers_filter} matched available tickers."
            )
    return datasets


def load_unscaled_close(ticker: str, index: pd.Index) -> pd.Series:
    """Return raw close series aligned with *index* (downloads/caches if needed)."""
    loader = YahooFinanceLoader(
        tickers=[ticker], include_benchmark=False,
    )
    raw = loader.load().get(ticker)
    if raw is None or raw.empty:
        raise RuntimeError(f"Could not load raw OHLCV for {ticker}.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()
    close.index = pd.to_datetime(close.index)
    aligned = close.reindex(index).ffill().bfill()
    if aligned.isna().any():
        raise RuntimeError(
            f"Unscaled close for {ticker} still has NaN after alignment."
        )
    return aligned.rename("Close")


def run_one(ticker: str, df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Run both walk-forward backtests for one ticker and save the results."""
    logger.info("===== %s =====", ticker)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "Gamma" not in df.columns:
        raise KeyError(f"{ticker}: processed frame is missing the 'Gamma' column.")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    close_unscaled = load_unscaled_close(ticker, df.index)

    if args.top_k_features is not None:
        df = select_features(df, ticker, args.top_k_features)

    suffix = args.feature_suffix
    title_extra = (
        f" — top-{args.top_k_features} features"
        if args.top_k_features is not None else ""
    )

    validator = WalkForwardValidator(
        initial_train_size=args.initial_train,
        test_size=args.test_size,
        max_folds=args.max_folds,
    )

    def mlp_factory() -> MLPSignalModel:
        return MLPSignalModel(epochs=args.mlp_epochs)

    def lstm_factory() -> LSTMValueModel:
        return LSTMValueModel(
            epochs=args.lstm_epochs,
            sequence_length=args.lstm_seq_len,
        )

    logger.info("[%s] Running MLP walk-forward …", ticker)
    mlp_result = validator.run_mlp(df, close_unscaled, model_factory=mlp_factory)
    logger.info("[%s] Running LSTM walk-forward …", ticker)
    lstm_result = validator.run_lstm(df, close_unscaled, model_factory=lstm_factory)

    curves, metrics = build_comparison(
        [mlp_result, lstm_result],
        close_unscaled=close_unscaled,
    )

    logger.info("[%s] summary metrics:\n%s", ticker, metrics.round(4).to_string())

    metrics.to_csv(RESULTS_DIR / f"{ticker}_walkforward_metrics{suffix}.csv")
    mlp_result.fold_metrics.to_csv(
        RESULTS_DIR / f"{ticker}_MLP_folds{suffix}.csv", index=False,
    )
    lstm_result.fold_metrics.to_csv(
        RESULTS_DIR / f"{ticker}_LSTM_folds{suffix}.csv", index=False,
    )
    mlp_result.predictions.to_csv(
        RESULTS_DIR / f"{ticker}_MLP_predictions{suffix}.csv",
    )
    lstm_result.predictions.to_csv(
        RESULTS_DIR / f"{ticker}_LSTM_predictions{suffix}.csv",
    )
    curves.to_csv(RESULTS_DIR / f"{ticker}_equity_curves{suffix}.csv")

    plot_equity_curves(
        curves,
        title=(
            f"{ticker}{title_extra} — walk-forward equity "
            "(MLP vs LSTM vs Buy & Hold)"
        ),
        out_path=FIGURES_DIR / f"equity_{ticker}{suffix}.png",
    )
    plot_fold_metric(
        {"MLP": mlp_result.fold_metrics, "LSTM": lstm_result.fold_metrics},
        metric="accuracy",
        out_path=FIGURES_DIR / f"fold_accuracy_{ticker}{suffix}.png",
    )


def main() -> None:
    """Run the walk-forward backtests for every processed ticker."""
    args = parse_args()
    datasets = load_processed_or_run_pipeline(args.tickers)
    for ticker, df in datasets.items():
        try:
            run_one(ticker, df, args)
        except Exception:  # noqa: BLE001
            logger.exception("[%s] modeling run failed — continuing.", ticker)


if __name__ == "__main__":
    main()

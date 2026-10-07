"""
pipeline.py
-----------
Orchestrates the full data preparation workflow.

Steps (matching CRISP-DM and the paper's Section 2):
    1. Load raw OHLCV data          (YahooFinanceLoader)
    2. Compute technical indicators  (TechnicalIndicators)
    3. Build the Γ target            (TargetBuilder)
    4. Normalize features            (ExpandingMinMaxScaler, causal)
    5. Clean NaN rows                (DataCleaner)

The output is a dict of clean, scaled, labelled DataFrames, one per
ticker, ready for feature selection and model training. Every feature
value at day t is computed from data up to and including day t, so the
output can be split into walk-forward folds without look-ahead leakage.

Run as a module to rebuild ``data/processed/``::

    PYTHONPATH=src python -m etf_predictor.data.pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from etf_predictor.data.indicators import TechnicalIndicators
from etf_predictor.data.loader import YahooFinanceLoader
from etf_predictor.data.preprocessing import DataCleaner, ExpandingMinMaxScaler
from etf_predictor.data.targets import TargetBuilder

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["IEUR", "FEZ", "EUFN"]
TARGET_COL = "Gamma"


class DataPipeline:
    """End-to-end data preparation pipeline for ETF trend prediction.

    This is the single entry point the modeling team should import.
    It wires together data loading, indicator computation, target
    construction, normalisation, and cleaning.

    Parameters
    ----------
    tickers : list[str]
        ETF ticker symbols to process.
    start : str
        Start date (``YYYY-MM-DD``, inclusive).
    end : str
        End date (``YYYY-MM-DD``, exclusive).
    cache_dir : str or Path
        Directory for raw data parquet cache.
    processed_dir : str or Path
        Directory where processed parquet files are saved.
    indicator_categories : list[str] or None
        pandas-ta categories to compute. ``None`` → all categories.
    cleaning_strategy : str
        ``"drop_rows"`` (default) or ``"fill_forward"``.
    horizon : int
        Look-ahead horizon for the target. Defaults to ``1``.
    include_benchmark : bool
        Whether to include IVV as a comparison benchmark.
    force_download : bool
        If ``True``, bypass the raw data cache.

    Examples
    --------
    >>> pipeline = DataPipeline(
    ...     tickers=["IEUR", "FEZ", "EUFN"],
    ...     start="2010-01-01",
    ...     end="2020-01-01",
    ... )
    >>> datasets = pipeline.run()
    >>> datasets["IEUR"].shape
    (2400, 217)  # approximate
    """

    def __init__(
        self,
        tickers: list[str] = DEFAULT_TICKERS,
        start: str = "2010-01-01",
        end: str = "2026-05-01",
        cache_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        indicator_categories: Optional[list[str]] = None,
        cleaning_strategy: str = "drop_rows",
        horizon: int = 1,
        include_benchmark: bool = True,
        force_download: bool = False,
    ) -> None:
        self.tickers = list(tickers)
        self.start = start
        self.end = end
        self.cache_dir = Path(cache_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.indicator_categories = indicator_categories
        self.cleaning_strategy = cleaning_strategy
        self.horizon = horizon
        self.include_benchmark = include_benchmark
        self.force_download = force_download

        # Instantiate pipeline components
        self._loader = YahooFinanceLoader(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            cache_dir=self.cache_dir,
            include_benchmark=self.include_benchmark,
        )
        # "Adj Close" is excluded from the features: Yahoo back-adjusts it
        # with dividends paid after each day, so its value at day t is
        # revised whenever a later dividend arrives (look-ahead).
        self._indicator_engine = TechnicalIndicators(
            categories=self.indicator_categories,
            exclude_cols=["Adj Close"],
        )
        self._target_builder = TargetBuilder(
            horizon=self.horizon,
            target_col=TARGET_COL,
        )
        self._cleaner = DataCleaner(
            strategy="fill_forward",
            exclude_cols=[TARGET_COL],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, save: bool = True) -> dict[str, pd.DataFrame]:
        """Execute the full pipeline for all tickers.

        Parameters
        ----------
        save : bool
            If ``True``, each processed DataFrame is saved as a parquet
            file under ``self.processed_dir``.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ticker → fully processed DataFrame, with all
            indicator columns, the ``"Gamma"`` target, and no NaNs.
        """
        logger.info("=== DataPipeline.run() started ===")

        # Step 1 – Load raw data
        raw_data = self._loader.load(force_download=self.force_download)

        processed: dict[str, pd.DataFrame] = {}
        for ticker, df_raw in raw_data.items():
            logger.info("--- Processing %s ---", ticker)
            df = self._process_single(ticker, df_raw)
            processed[ticker] = df
            if save:
                self._save(ticker, df)

        logger.info(
            "=== DataPipeline.run() complete. Tickers: %s ===",
            list(processed.keys()),
        )
        return processed

    def summary(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a summary table of shapes and class balance.

        Parameters
        ----------
        datasets : dict[str, pd.DataFrame]
            Output of ``run()``.

        Returns
        -------
        pd.DataFrame
            One row per ticker with columns:
            ``["rows", "features", "up_pct", "down_pct"]``.
        """
        records = []
        for ticker, df in datasets.items():
            n_features = df.shape[1] - 1  # exclude Gamma
            up_pct = 100 * (df[TARGET_COL] == 1).mean()
            down_pct = 100 * (df[TARGET_COL] == -1).mean()
            records.append(
                {
                    "ticker": ticker,
                    "rows": len(df),
                    "features": n_features,
                    "up_pct": round(up_pct, 2),
                    "down_pct": round(down_pct, 2),
                }
            )
        return pd.DataFrame(records).set_index("ticker")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_single(
        self, ticker: str, df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """Run all pipeline steps for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol (used for logging only).
        df_raw : pd.DataFrame
            Raw OHLCV DataFrame from the loader.

        Returns
        -------
        pd.DataFrame
            Fully processed DataFrame.
        """
        # Step 2 – Compute technical indicators
        df = self._indicator_engine.compute(df_raw)

        # Step 3 – Build target Γ (drops first `horizon` rows)
        df = self._target_builder.build(df)

        # Step 4 – Normalize features causally: each value is scaled with
        # the min/max observed up to that day, never the full sample, so
        # walk-forward folds cut from this frame see no future ranges.
        df = ExpandingMinMaxScaler(exclude_cols=[TARGET_COL]).transform(df)

        # Step 5 – Clean NaN rows
        df = self._cleaner.clean(df)

        logger.info(
            "%s — Final shape: %s", ticker, df.shape
        )
        return df

    def _save(self, ticker: str, df: pd.DataFrame) -> None:
        """Save processed DataFrame to parquet."""
        path = self.processed_dir / f"{ticker}_processed.parquet"
        df.to_parquet(path)
        logger.info("Saved processed data: %s", path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    pipeline = DataPipeline()
    print(pipeline.summary(pipeline.run(save=True)))

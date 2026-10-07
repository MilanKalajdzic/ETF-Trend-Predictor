"""Download and cache raw OHLCV data from Yahoo Finance."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["IEUR", "FEZ", "EUFN"]
BENCHMARK_TICKER = "IVV"  # S&P 500 ETF for developed-market comparison
OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class YahooFinanceLoader:
    """Download and cache raw daily OHLCV data from Yahoo Finance.

    Data is stored as parquet files under ``cache_dir`` so that
    subsequent runs do not re-download. Delete the cache directory
    to force a fresh download.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download (e.g. ``["IEUR", "FEZ", "EUFN"]``).
    start : str
        Start date in ``YYYY-MM-DD`` format (inclusive).
    end : str
        End date in ``YYYY-MM-DD`` format (exclusive).
    cache_dir : str or Path
        Directory where parquet files are stored.
        Defaults to ``data/raw`` relative to the working directory.
    include_benchmark : bool
        If ``True``, also download ``IVV`` as a developed-market
        benchmark for comparison. Defaults to ``True``.

    Examples
    --------
    >>> loader = YahooFinanceLoader(
    ...     tickers=["IEUR", "FEZ", "EUFN"],
    ...     start="2010-01-01",
    ...     end="2026-05-01",
    ... )
    >>> raw_data = loader.load()
    >>> raw_data["IEUR"].head()

    """

    def __init__(
        self,
        tickers: list[str] = DEFAULT_TICKERS,
        start: str = "2010-01-01",
        end: str = "2026-05-01",
        cache_dir: str | Path = "data/raw",
        include_benchmark: bool = True,
    ) -> None:
        self.tickers = list(tickers)
        if include_benchmark and BENCHMARK_TICKER not in self.tickers:
            self.tickers.append(BENCHMARK_TICKER)
        self.start = start
        self.end = end
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, force_download: bool = False) -> dict[str, pd.DataFrame]:
        """Load data for all tickers, using the cache when available.

        Parameters
        ----------
        force_download : bool
            If ``True``, ignore cached files and re-download.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ticker symbol → DataFrame with columns
            ``["Open", "High", "Low", "Close", "Adj Close", "Volume"]``
            and a ``DatetimeIndex``.

        """
        result: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            df = self._load_single(ticker, force_download=force_download)
            if df is not None:
                result[ticker] = df
        logger.info("Loaded data for tickers: %s", list(result.keys()))
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        """Return the parquet path for *ticker*."""
        return self.cache_dir / f"{ticker}_{self.start}_{self.end}.parquet"

    def _load_single(
        self, ticker: str, force_download: bool = False
    ) -> pd.DataFrame | None:
        """Load one ticker from cache or download it fresh.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol.
        force_download : bool
            Skip cache and download even if cached file exists.

        Returns
        -------
        pd.DataFrame or None
            OHLCV DataFrame or ``None`` if the download failed.

        """
        path = self._cache_path(ticker)
        if path.exists() and not force_download:
            logger.info("Loading %s from cache: %s", ticker, path)
            return pd.read_parquet(path)

        logger.info("Downloading %s from Yahoo Finance …", ticker)
        try:
            raw = yf.download(
                ticker,
                start=self.start,
                end=self.end,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download %s: %s", ticker, exc)
            return None

        if raw.empty:
            logger.warning("No data returned for %s.", ticker)
            return None

        # yfinance may return MultiIndex columns — flatten them
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()]

        df = raw[OHLCV_COLS].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        df.to_parquet(path)
        logger.info("Cached %s → %s", ticker, path)
        return df

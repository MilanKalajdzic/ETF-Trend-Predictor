"""Tests for etf_predictor.data.loader."""

from unittest.mock import patch

import pandas as pd
import pytest

from etf_predictor.data.loader import OHLCV_COLS, YahooFinanceLoader


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Minimal OHLCV DataFrame for testing."""
    dates = pd.date_range("2015-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [40.0, 41.0, 39.5, 42.0, 41.5],
            "High": [41.5, 42.0, 40.5, 43.0, 42.5],
            "Low": [39.5, 40.5, 38.5, 41.5, 40.5],
            "Close": [41.0, 39.5, 42.0, 41.5, 42.0],
            "Adj Close": [41.0, 39.5, 42.0, 41.5, 42.0],
            "Volume": [100000, 120000, 95000, 130000, 110000],
        },
        index=dates,
    )


class TestYahooFinanceLoader:
    def test_init_adds_benchmark(self):
        loader = YahooFinanceLoader(
            tickers=["IEUR"], include_benchmark=True
        )
        assert "IVV" in loader.tickers

    def test_init_no_benchmark(self):
        loader = YahooFinanceLoader(
            tickers=["IEUR"], include_benchmark=False
        )
        assert "IVV" not in loader.tickers

    def test_cache_path_format(self, tmp_path):
        loader = YahooFinanceLoader(
            tickers=["IEUR"],
            start="2010-01-01",
            end="2020-01-01",
            cache_dir=tmp_path,
        )
        path = loader._cache_path("IEUR")
        assert path.suffix == ".parquet"
        assert "IEUR" in path.name

    def test_load_from_cache(self, tmp_path, sample_ohlcv):
        loader = YahooFinanceLoader(
            tickers=["IEUR"],
            start="2010-01-01",
            end="2020-01-01",
            cache_dir=tmp_path,
            include_benchmark=False,
        )
        # Write a fake cache file
        cache_path = loader._cache_path("IEUR")
        sample_ohlcv.to_parquet(cache_path)

        result = loader.load(force_download=False)

        assert "IEUR" in result
        assert list(result["IEUR"].columns) == OHLCV_COLS

    @patch("etf_predictor.data.loader.yf.download")
    def test_load_downloads_when_no_cache(
        self, mock_download, tmp_path, sample_ohlcv
    ):
        mock_download.return_value = sample_ohlcv
        loader = YahooFinanceLoader(
            tickers=["IEUR"],
            start="2010-01-01",
            end="2020-01-01",
            cache_dir=tmp_path,
            include_benchmark=False,
        )
        result = loader.load()
        assert "IEUR" in result
        mock_download.assert_called_once()

    @patch("etf_predictor.data.loader.yf.download")
    def test_returns_none_on_empty_download(
        self, mock_download, tmp_path
    ):
        mock_download.return_value = pd.DataFrame()
        loader = YahooFinanceLoader(
            tickers=["FAKE"],
            cache_dir=tmp_path,
            include_benchmark=False,
        )
        result = loader.load()
        assert "FAKE" not in result

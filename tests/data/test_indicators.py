"""Tests for etf_predictor.data.indicators, focused on look-ahead safety."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas_ta")

from etf_predictor.data.indicators import (  # noqa: E402
    NONCAUSAL_PREFIXES,
    TechnicalIndicators,
)


def _synthetic_ohlcv(n: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(50_000, 500_000, n).astype(float)
    index = pd.bdate_range("2015-01-01", periods=n, name="Date")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=index,
    )


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _synthetic_ohlcv()


@pytest.fixture(scope="module")
def computed(ohlcv) -> pd.DataFrame:
    return TechnicalIndicators().compute(ohlcv)


class TestTechnicalIndicators:
    def test_output_index_matches_input(self, ohlcv, computed):
        # Ichimoku's forward projections must not add future-dated rows
        assert computed.index.equals(ohlcv.index)

    def test_column_names_unique(self, computed):
        assert not computed.columns.duplicated().any()

    def test_original_columns_kept(self, ohlcv, computed):
        pd.testing.assert_frame_equal(computed[ohlcv.columns], ohlcv)

    def test_noncausal_columns_dropped(self, computed):
        leaked = [c for c in computed.columns if str(c).startswith(NONCAUSAL_PREFIXES)]
        assert leaked == []

    def test_noncausal_columns_present_when_not_dropped(self, ohlcv):
        raw = TechnicalIndicators(drop_noncausal=False).compute(ohlcv)
        assert any(str(c).startswith("ICS_") for c in raw.columns)

    def test_all_remaining_indicators_are_causal(self, ohlcv):
        # Value at day t must not change when data after t is removed
        assert TechnicalIndicators().find_noncausal_columns(ohlcv) == []

    def test_probe_detects_lookahead(self, ohlcv):
        assert "ICS_26" in TechnicalIndicators(
            drop_noncausal=False
        ).find_noncausal_columns(ohlcv, cut_points=[500])

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            TechnicalIndicators(categories=["astrology"])

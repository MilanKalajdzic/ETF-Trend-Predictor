"""End-to-end look-ahead test for the processed dataset."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas_ta")

from etf_predictor.data.pipeline import TARGET_COL, DataPipeline  # noqa: E402
from tests.data.test_indicators import _synthetic_ohlcv  # noqa: E402


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> DataPipeline:
    d = tmp_path_factory.mktemp("data")
    return DataPipeline(
        tickers=["TEST"],
        include_benchmark=False,
        cache_dir=d / "raw",
        processed_dir=d / "processed",
    )


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return _synthetic_ohlcv(n=1200, seed=5)


@pytest.fixture(scope="module")
def processed(pipeline, raw) -> pd.DataFrame:
    return pipeline._process_single("TEST", raw)


class TestProcessedDataset:
    def test_handoff_contract(self, processed, raw):
        features = processed.drop(columns=TARGET_COL)
        assert processed.isna().sum().sum() == 0
        assert processed.index.isin(raw.index).all()
        assert processed[TARGET_COL].isin([1, -1]).all()
        assert features.min().min() >= 0.0
        assert features.max().max() <= 1.0

    def test_adjusted_close_excluded(self, processed):
        # Back-adjusted prices are revised with later dividends (look-ahead)
        assert "Adj Close" not in processed.columns

    @pytest.mark.parametrize("cut", [800, 1000])
    def test_no_lookahead_end_to_end(self, pipeline, raw, processed, cut):
        # Rebuilding the dataset without the data after `cut` must leave
        # every earlier row unchanged: no feature, scaling step or fill
        # may use future prices.
        part = pipeline._process_single("TEST", raw.iloc[: cut + 1])
        cols = [c for c in part.columns if c in processed.columns]
        rows = part.index.intersection(processed.index)
        assert len(rows) > 0
        a = processed.loc[rows, cols].to_numpy(dtype=float)
        b = part.loc[rows, cols].to_numpy(dtype=float)
        changed = ~np.isclose(a, b, rtol=1e-7, atol=1e-10, equal_nan=True)
        leaking = sorted({cols[j] for j in np.where(changed)[1]})
        assert leaking == []

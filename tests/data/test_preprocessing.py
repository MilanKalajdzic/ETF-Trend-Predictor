"""Tests for etf_predictor.data.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from etf_predictor.data.preprocessing import (
    DataCleaner,
    ExpandingMinMaxScaler,
    MinMaxScaler,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Gamma": [1, -1, 1, 1, -1],
        }
    )


@pytest.fixture()
def df_with_nans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [np.nan, np.nan, 3.0, 4.0, 5.0],
            "B": [np.nan, 20.0, 30.0, 40.0, 50.0],
            "Gamma": [1, -1, 1, 1, -1],
        }
    )


class TestMinMaxScaler:
    def test_scaled_range(self, sample_df):
        scaler = MinMaxScaler(exclude_cols=["Gamma"])
        scaled = scaler.fit_transform(sample_df)
        assert scaled["A"].min() == pytest.approx(0.0)
        assert scaled["A"].max() == pytest.approx(1.0)

    def test_excludes_target_column(self, sample_df):
        scaler = MinMaxScaler(exclude_cols=["Gamma"])
        scaled = scaler.fit_transform(sample_df)
        # Gamma should be unchanged
        pd.testing.assert_series_equal(scaled["Gamma"], sample_df["Gamma"])

    def test_transform_without_fit_raises(self, sample_df):
        scaler = MinMaxScaler()
        with pytest.raises(RuntimeError, match="fit()"):
            scaler.transform(sample_df)

    def test_constant_column_no_division_by_zero(self):
        df = pd.DataFrame({"A": [5.0, 5.0, 5.0], "B": [1.0, 2.0, 3.0]})
        scaler = MinMaxScaler()
        result = scaler.fit_transform(df)
        # Constant column should yield 0.0 (not raise, not NaN)
        assert not result["A"].isna().any()
        assert (result["A"] == 0.0).all()

    def test_fit_on_train_apply_to_test(self, sample_df):
        train = sample_df.iloc[:3]
        test = sample_df.iloc[3:]
        scaler = MinMaxScaler(exclude_cols=["Gamma"])
        scaler.fit(train)
        scaled_test = scaler.transform(test)
        # Values can exceed [0,1] since test goes beyond training range
        assert scaled_test["A"].max() > 1.0


class TestDataCleaner:
    def test_drops_nan_rows(self, df_with_nans):
        cleaner = DataCleaner(exclude_cols=["Gamma"])
        result = cleaner.clean(df_with_nans)
        assert result.isna().sum().sum() == 0

    def test_length_after_drop(self, df_with_nans):
        cleaner = DataCleaner(exclude_cols=["Gamma"])
        result = cleaner.clean(df_with_nans)
        # Rows 0 and 1 should be dropped (NaN in A or B)
        assert len(result) == 3

    def test_fill_forward_strategy(self, df_with_nans):
        cleaner = DataCleaner(strategy="fill_forward", exclude_cols=["Gamma"])
        result = cleaner.clean(df_with_nans)
        # Row 0 still has NaN in A after ffill (nothing to fill from)
        # Row 1 B=20 → ffill keeps 20, A still NaN → dropped
        assert not result.isna().any().any()

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            DataCleaner(strategy="magic")

    def test_nan_report_columns(self, df_with_nans):
        cleaner = DataCleaner()
        report = cleaner.nan_report(df_with_nans)
        assert "nan_count" in report.columns
        assert "nan_pct" in report.columns
        assert "A" in report.index


@pytest.fixture()
def random_walk_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "A": np.cumsum(rng.normal(size=200)),
            "B": np.cumsum(rng.normal(size=200)) * 100,
            "Gamma": rng.choice([1, -1], size=200),
        }
    )


class TestExpandingMinMaxScaler:
    @pytest.mark.parametrize("window", [None, 20])
    def test_no_lookahead(self, random_walk_df, window):
        # Scaling a prefix must give exactly the first rows of scaling the
        # full series: no value may depend on later data.
        scaler = ExpandingMinMaxScaler(window=window, exclude_cols=["Gamma"])
        full = scaler.transform(random_walk_df)
        for cut in (30, 100, 170):
            prefix = scaler.transform(random_walk_df.iloc[:cut])
            pd.testing.assert_frame_equal(prefix, full.iloc[:cut])

    def test_values_in_unit_interval(self, random_walk_df):
        scaled = ExpandingMinMaxScaler(exclude_cols=["Gamma"]).transform(
            random_walk_df
        )
        assert scaled[["A", "B"]].min().min() >= 0.0
        assert scaled[["A", "B"]].max().max() <= 1.0

    def test_excludes_target_column(self, random_walk_df):
        scaled = ExpandingMinMaxScaler(exclude_cols=["Gamma"]).transform(
            random_walk_df
        )
        pd.testing.assert_series_equal(scaled["Gamma"], random_walk_df["Gamma"])

    def test_constant_column_is_zero(self):
        df = pd.DataFrame({"A": [5.0, 5.0, 5.0]})
        scaled = ExpandingMinMaxScaler().transform(df)
        assert (scaled["A"] == 0.0).all()

    def test_nan_stays_nan(self):
        df = pd.DataFrame({"A": [np.nan, 1.0, 3.0, 2.0]})
        scaled = ExpandingMinMaxScaler().transform(df)
        assert np.isnan(scaled["A"].iloc[0])
        assert scaled["A"].iloc[1:].tolist() == [0.0, 1.0, 0.5]

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window"):
            ExpandingMinMaxScaler(window=1)

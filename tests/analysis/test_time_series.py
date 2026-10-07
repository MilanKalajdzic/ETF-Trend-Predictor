import numpy as np
import pandas as pd

from etf_predictor.analysis.time_series import TimeSeriesAnalyzer


def test_adf_test_returns_expected_keys() -> None:
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(size=200))

    analyzer = TimeSeriesAnalyzer(series)
    result = analyzer.adf_test()

    assert {"adf_statistic", "p_value", "used_lags", "n_obs"} == set(result.keys())


def test_forecast_returns_requested_number_of_steps() -> None:
    rng = np.random.default_rng(42)
    series = pd.Series(np.cumsum(rng.normal(size=200)))

    analyzer = TimeSeriesAnalyzer(series)
    analyzer.fit_arima(order=(1, 1, 1))
    forecast = analyzer.forecast(steps=5)

    assert len(forecast) == 5


def test_evaluate_holdout_returns_metrics() -> None:
    rng = np.random.default_rng(42)
    series = pd.Series(np.cumsum(rng.normal(size=250)))

    analyzer = TimeSeriesAnalyzer(series)
    result = analyzer.evaluate_holdout(order=(1, 1, 1), test_size=0.2)

    assert {"mae", "rmse", "train_size", "test_size"} == set(result.keys())
    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["train_size"] > 0
    assert result["test_size"] > 0

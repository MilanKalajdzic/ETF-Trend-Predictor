import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


class TimeSeriesAnalyzer:
    """Run basic time-series diagnostics and ARIMA forecasting."""

    def __init__(self, series: pd.Series) -> None:
        # Trading days have holiday gaps, so a date index has no frequency
        # and statsmodels cannot extend it when forecasting. Index by
        # trading-day position instead; observations stay evenly spaced in
        # trading time.
        self.series = series.dropna().reset_index(drop=True)
        self.model_fit = None

    def adf_test(self) -> dict[str, float]:
        """Run Augmented Dickey-Fuller test."""
        result = adfuller(self.series)

        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "used_lags": result[2],
            "n_obs": result[3],
        }

    def fit_arima(self, order: tuple[int, int, int] = (1, 1, 1)):
        """Fit ARIMA model and store fitted result."""
        model = ARIMA(self.series, order=order)
        self.model_fit = model.fit()
        return self.model_fit

    def forecast(self, steps: int = 5) -> pd.Series:
        """Forecast future values from fitted ARIMA model."""
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit_arima() first.")
        return self.model_fit.forecast(steps=steps)

    def evaluate_holdout(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        test_size: float = 0.2,
    ) -> dict[str, float]:
        """Evaluate ARIMA on a simple holdout split preserving time order."""
        split_idx = int(len(self.series) * (1 - test_size))

        train = self.series.iloc[:split_idx]
        test = self.series.iloc[split_idx:]

        model = ARIMA(train, order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=len(test))

        return {
            "mae": mean_absolute_error(test, forecast),
            "rmse": mean_squared_error(test, forecast) ** 0.5,
            "train_size": len(train),
            "test_size": len(test),
        }

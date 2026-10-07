from pathlib import Path
import pandas as pd

from etf_predictor.analysis.correlation_analysis import CorrelationAnalyzer
from etf_predictor.analysis.feature_analysis import FeatureAnalyzer
from etf_predictor.analysis.feature_importance import FeatureImportanceAnalyzer
from etf_predictor.data.loader import YahooFinanceLoader
from etf_predictor.data.pipeline import DataPipeline
from etf_predictor.analysis.time_series import TimeSeriesAnalyzer


def main() -> None:
    pipeline = DataPipeline()
    datasets = pipeline.run()

    # ADF and ARIMA need the raw price series. The processed "Close" column
    # is min-max scaled with an expanding window, a non-linear transform
    # that would distort both the stationarity test and the forecast errors.
    raw_data = YahooFinanceLoader(
        tickers=list(datasets),
        start=pipeline.start,
        end=pipeline.end,
        cache_dir=pipeline.cache_dir,
        include_benchmark=False,
    ).load()

    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df in datasets.items():
        print(f"\n===== {ticker} =====")

        X = df.drop(columns=["Gamma"])
        y = df["Gamma"]

        analyzer = FeatureAnalyzer(X)
        variance_series = analyzer.variance().sort_values(ascending=False)

        print("\nTop 10 variances:")
        print(variance_series.head(10))
        variance_series.to_csv(
            results_dir / f"{ticker}_variance.csv",
            header=["variance"],
        )

        corr_analyzer = CorrelationAnalyzer(X)
        high_corr = corr_analyzer.highly_correlated_pairs(threshold=0.95)

        print("\nHighly correlated pairs (> 0.95):")
        print(len(high_corr))

        if not high_corr.empty:
            print(high_corr.head(10))

        high_corr.to_csv(
            results_dir / f"{ticker}_high_corr.csv",
            index=False,
        )

        print("\nCorrelation matrix shape:")
        print(analyzer.correlation_matrix().shape)

        importance_analyzer = FeatureImportanceAnalyzer()
        top_features = importance_analyzer.top_features(X, y, top_n=10)

        print("\nTop 10 feature importances:")
        print(top_features)

        full_importance = importance_analyzer.fit_importance(X, y)

        print("\nRandom Forest test accuracy:")
        print(round(full_importance.attrs["accuracy"], 4))

        full_importance.to_csv(
            results_dir / f"{ticker}_feature_importance.csv",
            index=False,
        )

        summary_df = pd.DataFrame(
            {
                "ticker": [ticker],
                "n_features": [X.shape[1]],
                "n_high_corr_pairs": [len(high_corr)],
                "rf_test_accuracy": [full_importance.attrs["accuracy"]],
                "gamma_up_share": [(y == 1).mean()],
                "gamma_down_share": [(y == -1).mean()],
            }
        )
        summary_df.to_csv(
            results_dir / f"{ticker}_summary.csv",
            index=False,
        )

        print("\nTarget distribution:")
        print(y.value_counts(normalize=True))

        if ticker in raw_data:
            close = raw_data[ticker]["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            ts_analyzer = TimeSeriesAnalyzer(close.reindex(df.index))

            adf_results = ts_analyzer.adf_test()
            arima_eval = ts_analyzer.evaluate_holdout(order=(1, 1, 1))

            print("\nADF test results:")
            print(adf_results)

            print("\nARIMA(1,1,1) holdout evaluation:")
            print(arima_eval)

            arima_summary_df = pd.DataFrame(
                {
                    "ticker": [ticker],
                    "series": ["Close"],
                    "adf_statistic": [adf_results["adf_statistic"]],
                    "adf_p_value": [adf_results["p_value"]],
                    "used_lags": [adf_results["used_lags"]],
                    "n_obs": [adf_results["n_obs"]],
                    "mae": [arima_eval["mae"]],
                    "rmse": [arima_eval["rmse"]],
                    "train_size": [arima_eval["train_size"]],
                    "test_size": [arima_eval["test_size"]],
                }
            )

            arima_summary_df.to_csv(
                results_dir / f"{ticker}_arima_summary.csv",
                index=False,
            )
            
if __name__ == "__main__":
    main()
import pandas as pd

from etf_predictor.analysis.feature_analysis import FeatureAnalyzer


def test_variance_returns_series() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 1],
            "b": [1, 2, 3, 4],
            "c": [4, 3, 2, 1],
        }
    )

    analyzer = FeatureAnalyzer(df)
    result = analyzer.variance()

    assert isinstance(result, pd.Series)
    assert result["a"] == 0
    assert result["b"] > 0


def test_drop_low_variance_removes_constant_columns() -> None:
    df = pd.DataFrame(
        {
            "constant": [5, 5, 5, 5],
            "signal": [1, 2, 3, 4],
        }
    )

    analyzer = FeatureAnalyzer(df)
    filtered = analyzer.drop_low_variance(threshold=0.01)

    assert "constant" not in filtered.columns
    assert "signal" in filtered.columns

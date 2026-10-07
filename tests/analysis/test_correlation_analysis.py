import pandas as pd

from etf_predictor.analysis.correlation_analysis import CorrelationAnalyzer


def test_highly_correlated_pairs_detects_duplicate_features() -> None:
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],  # perfectly correlated with x1
            "x3": [5, 4, 3, 2, 1],
        }
    )

    analyzer = CorrelationAnalyzer(df)
    pairs = analyzer.highly_correlated_pairs(threshold=0.95)

    assert not pairs.empty
    assert {"feature_1", "feature_2", "correlation"}.issubset(pairs.columns)


def test_columns_to_drop_returns_list() -> None:
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
            "x3": [5, 7, 9, 11, 13],
        }
    )

    analyzer = CorrelationAnalyzer(df)
    cols = analyzer.columns_to_drop(threshold=0.95)

    assert isinstance(cols, list)
    assert len(cols) >= 1

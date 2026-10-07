import pandas as pd


class FeatureAnalyzer:
    """Analyze statistical properties of technical-indicator features."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def variance(self) -> pd.Series:
        """Return variance of each feature column."""
        return self.data.var()

    def correlation_matrix(self) -> pd.DataFrame:
        """Return feature correlation matrix."""
        return self.data.corr()

    def drop_low_variance(self, threshold: float = 0.01) -> pd.DataFrame:
        """Return only columns with variance above the threshold."""
        variances = self.data.var()
        selected = variances[variances > threshold].index
        return self.data[selected]

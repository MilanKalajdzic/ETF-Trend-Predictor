import numpy as np
import pandas as pd


class CorrelationAnalyzer:
    """Analyze pairwise feature correlations."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def correlation_matrix(self) -> pd.DataFrame:
        """Return absolute correlation matrix."""
        return self.data.corr().abs()

    def highly_correlated_pairs(self, threshold: float = 0.95) -> pd.DataFrame:
        """Return feature pairs with correlation above the threshold."""
        corr = self.correlation_matrix()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        rows = []
        for col in upper.columns:
            for row in upper.index:
                value = upper.loc[row, col]
                if pd.notna(value) and value > threshold:
                    rows.append(
                        {
                            "feature_1": row,
                            "feature_2": col,
                            "correlation": value,
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

        return pd.DataFrame(rows).sort_values(
            by="correlation",
            ascending=False,
        )

    def columns_to_drop(self, threshold: float = 0.95) -> list[str]:
        """Suggest columns to drop based on pairwise correlation."""
        corr = self.correlation_matrix()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        return [col for col in upper.columns if any(upper[col] > threshold)]

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class FeatureImportanceAnalyzer:
    """Estimate feature importance for Gamma classification."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size

    def _time_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train/test sets while preserving time order."""
        split_idx = int(len(X) * (1 - self.test_size))
        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()
        return X_train, X_test, y_train, y_test

    def fit_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit random forest and return sorted feature importances."""
        X_train, X_test, y_train, y_test = self._time_split(X, y)

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False)

        importance_df.attrs["accuracy"] = accuracy
        return importance_df

    def top_features(
        self, X: pd.DataFrame, y: pd.Series, top_n: int = 10
    ) -> pd.DataFrame:
        """Return top N most important features."""
        importance_df = self.fit_importance(X, y)
        return importance_df.head(top_n)

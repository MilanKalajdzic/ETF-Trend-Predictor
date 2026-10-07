"""Normalise and clean the feature data.

Data normalisation and cleaning as described in Sagaceta-Mejía et al.
(2024), Sections 2.5 and 2.6.

Three classes are provided:

- ``MinMaxScaler``          per-column min-max normalisation, fitted
  on a training split and applied to a test split.
- ``ExpandingMinMaxScaler`` causal min-max normalisation: each value is
  scaled with the min and max observed up to that day only, so a
  dataset scaled once can be split into walk-forward folds without
  leaking the future range of a feature into earlier rows. This is what
  the data pipeline uses.
- ``DataCleaner``           removes rows containing NaN values that
  arise from the warm-up period of technical indicators (e.g. SMA
  requires ``n`` prior rows before it can be computed).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MinMaxScaler:
    """Per-column min-max normalisation to the [0, 1] range.

    Implements the formula from the paper (Section 2.5)::

        scaled = (value[i] - min) / (max - min)

    The scaler is **fit on training data only** and then applied to
    both train and test splits, preventing data leakage — a point the
    paper does not make explicit but is critical in practice.

    Parameters
    ----------
    feature_range : tuple[float, float]
        Target range after scaling. Defaults to ``(0.0, 1.0)``.
    exclude_cols : list[str] or None
        Columns to leave unscaled (e.g. the target column ``"Gamma"``).

    Examples
    --------
    >>> scaler = MinMaxScaler(exclude_cols=["Gamma"])
    >>> scaler.fit(X_train)
    >>> X_train_scaled = scaler.transform(X_train)
    >>> X_test_scaled  = scaler.transform(X_test)

    """

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, 1.0),
        exclude_cols: list[str] | None = None,
    ) -> None:
        self.feature_range = feature_range
        self.exclude_cols = set(exclude_cols or [])
        self._min: pd.Series | None = None
        self._max: pd.Series | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> MinMaxScaler:
        """Compute per-column min and max from *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Training split (features only — should not include the
            target column, or pass it in ``exclude_cols``).

        Returns
        -------
        MinMaxScaler
            Returns ``self`` for method chaining.

        """
        df = df.loc[:, ~df.columns.duplicated()]
        cols = self._feature_cols(df)
        self._min = df[cols].min()
        self._max = df[cols].max()
        self._fitted = True
        logger.info("MinMaxScaler fitted on %d columns.", len(cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling to *df* using fitted statistics.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to scale.

        Returns
        -------
        pd.DataFrame
            Scaled DataFrame with the same index and columns.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.

        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        result = df.copy()
        result = result.loc[:, ~result.columns.duplicated()]

        lo, hi = self.feature_range

        # Only scale columns present in both the fitted stats and this df
        cols = [
            c for c in self._feature_cols(result)
            if c in self._min.index and c in self._max.index
        ]

        for col in cols:
            min_val = self._min[col]
            max_val = self._max[col]
            denom = max_val - min_val
            if denom == 0:
                result[col] = 0.0
            else:
                result[col] = (result[col] - min_val) / denom * (hi - lo) + lo

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* and transform it in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        pd.DataFrame
            Scaled training data.

        """
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return columns that should be scaled."""
        return [c for c in df.columns if c not in self.exclude_cols]


class ExpandingMinMaxScaler:
    """Causal (expanding-window) min-max normalisation to [0, 1].

    Each value is scaled with statistics known on that day::

        scaled_t = (x_t - min(x_1..x_t)) / (max(x_1..x_t) - min(x_1..x_t))

    or, with ``window`` set, over the trailing ``window`` observations.
    Unlike a full-sample min-max fit, the scaled value at day t never
    depends on data after t, so the output can be split into
    walk-forward folds without look-ahead leakage. Values stay in
    [0, 1]; a column with no range so far (constant) maps to 0.0.

    Parameters
    ----------
    window : int or None
        ``None`` (default) for an expanding window over all history,
        or a trailing window length in rows.
    exclude_cols : list[str] or None
        Columns to leave unscaled (e.g. the target column ``"Gamma"``).

    Examples
    --------
    >>> scaler = ExpandingMinMaxScaler(exclude_cols=["Gamma"])
    >>> df_scaled = scaler.transform(df)

    """

    def __init__(
        self,
        window: int | None = None,
        exclude_cols: list[str] | None = None,
    ) -> None:
        if window is not None and window < 2:
            raise ValueError(f"window must be >= 2 or None, got {window}")
        self.window = window
        self.exclude_cols = set(exclude_cols or [])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale every non-excluded column using only past and current values.

        Parameters
        ----------
        df : pd.DataFrame
            Time-ordered DataFrame (oldest row first).

        Returns
        -------
        pd.DataFrame
            Scaled copy of *df* with the same index and columns. NaNs
            stay NaN.

        """
        result = df.copy()
        result = result.loc[:, ~result.columns.duplicated()]
        cols = [c for c in result.columns if c not in self.exclude_cols]
        x = result[cols].astype(float)

        if self.window is None:
            lo = x.cummin()
            hi = x.cummax()
        else:
            lo = x.rolling(self.window, min_periods=1).min()
            hi = x.rolling(self.window, min_periods=1).max()

        denom = (hi - lo).replace(0, np.nan)
        scaled = (x - lo) / denom
        # No range yet (constant so far): 0.0 where the input is defined
        scaled = scaled.mask(denom.isna() & x.notna(), 0.0)
        result[cols] = scaled
        return result


class DataCleaner:
    """Remove rows with NaN values introduced by indicator warm-up.

    Technical indicators such as SMA(200) require 200 prior observations
    before yielding a value. The first *n* rows of each indicator column
    will therefore be NaN. This class drops all rows where *any*
    feature column contains NaN, matching the paper's approach.

    Parameters
    ----------
    strategy : {"drop_rows", "fill_forward"}
        How to handle NaN values.

        - ``"drop_rows"`` (default, matches paper): remove any row
          containing at least one NaN.
        - ``"fill_forward"``: forward-fill NaN values then drop
          remaining NaNs at the start of the series.
    exclude_cols : list[str] or None
        Columns to ignore when checking for NaNs (e.g. target column).

    Examples
    --------
    >>> cleaner = DataCleaner(strategy="drop_rows")
    >>> df_clean = cleaner.clean(df_with_indicators)

    """

    def __init__(
        self,
        strategy: str = "drop_rows",
        exclude_cols: list[str] | None = None,
    ) -> None:
        valid = {"drop_rows", "fill_forward"}
        if strategy not in valid:
            raise ValueError(
                f"strategy must be one of {valid}, got '{strategy}'"
            )
        self.strategy = strategy
        self.exclude_cols = set(exclude_cols or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop near-empty columns, then fill or drop rows with missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Feature frame to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned copy of *df*.

        """
        original_len = len(df)
        feature_cols = [c for c in df.columns if c not in self.exclude_cols]

        # Drop columns that are more than 95% NaN: sparse indicators
        # that would otherwise eliminate every row
        thresh = int(len(df) * 0.95)
        sparse_cols = [
            c for c in feature_cols
            if df[c].isna().sum() > thresh
        ]
        if sparse_cols:
            df = df.drop(columns=sparse_cols)
            feature_cols = [c for c in feature_cols if c not in sparse_cols]
            logger.info(
                "Dropped %d sparse columns (>95%% NaN): %s",
                len(sparse_cols),
                sparse_cols[:5],
            )

        if self.strategy == "fill_forward":
            result = df.copy()
            result[feature_cols] = result[feature_cols].ffill()
            result = result.dropna(subset=feature_cols)
        else:
            result = df.dropna(subset=feature_cols)

        removed = original_len - len(result)
        pct = 100 * removed / original_len if original_len else 0
        logger.info(
            "DataCleaner removed %d rows (%.1f%%). Remaining: %d",
            removed,
            pct,
            len(result),
        )
        return result.copy()

    def nan_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary of NaN counts per column.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (before cleaning).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["nan_count", "nan_pct"]``,
            sorted by ``nan_count`` descending, showing only columns
            that have at least one NaN.

        """
        nan_counts = df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        report = pd.DataFrame(
            {
                "nan_count": nan_counts,
                "nan_pct": 100 * nan_counts / len(df),
            }
        ).sort_values("nan_count", ascending=False)
        return report

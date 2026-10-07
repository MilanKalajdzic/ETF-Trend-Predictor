"""Build the binary classification target Γ(t).

Γ(t) follows the definition in Sagaceta-Mejía et al. (2024),
Section 2.4.

    Γ(t) =  1  if Open(t) − Open(t−1) > 0
            -1  otherwise
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "Gamma"
UP_LABEL = 1
DOWN_LABEL = -1


class TargetBuilder:
    """Build the binary trend-direction target from Open price series.

    The target Γ(t) is +1 when today's open is higher than yesterday's
    and −1 otherwise, matching the paper's classification setup.

    Parameters
    ----------
    horizon : int
        Number of periods ahead to look. ``1`` (default) reproduces the
        paper; larger values enable multi-step-ahead experiments.
    target_col : str
        Name of the output column. Defaults to ``"Gamma"``.

    Examples
    --------
    >>> builder = TargetBuilder()
    >>> df_with_target = builder.build(df)
    >>> df_with_target["Gamma"].value_counts()

    """

    def __init__(
        self,
        horizon: int = 1,
        target_col: str = TARGET_COL,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon
        self.target_col = target_col

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the Γ column to *df* and drop the first row.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that must contain an ``"Open"`` column.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with a new column ``self.target_col``.
            The first ``horizon`` rows are removed.

        Raises
        ------
        KeyError
            If ``"Open"`` is not a column in *df*.

        """
        if "Open" not in df.columns:
            raise KeyError("DataFrame must contain an 'Open' column.")

        result = df.copy()

        open_series = result["Open"]
        if isinstance(open_series, pd.DataFrame):
            open_series = open_series.iloc[:, 0]
        open_series = open_series.squeeze()

        delta = open_series.diff(self.horizon)
        result[self.target_col] = (
            (delta > 0).astype(int).replace({1: UP_LABEL, 0: DOWN_LABEL})
        )

        result = result.iloc[self.horizon:].copy()

        up_count = (result[self.target_col] == UP_LABEL).sum()
        down_count = (result[self.target_col] == DOWN_LABEL).sum()
        total = len(result)
        logger.info(
            "Target built — UP: %d (%.1f%%), DOWN: %d (%.1f%%)",
            up_count,
            100 * up_count / total,
            down_count,
            100 * down_count / total,
        )
        return result

    def class_balance(self, df: pd.DataFrame) -> pd.Series:
        """Return the proportion of each class in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains ``self.target_col``.

        Returns
        -------
        pd.Series
            Value counts normalised to fractions.

        """
        if self.target_col not in df.columns:
            raise KeyError(
                f"Column '{self.target_col}' not found. Run build() first."
            )
        return df[self.target_col].value_counts(normalize=True)

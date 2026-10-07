"""Compute the pandas-ta technical indicators used as features.

Wraps the pandas-ta library to compute the full set of technical
indicators used in Sagaceta-Mejía et al. (2024), Section 2.3.

pandas-ta is applied to the six base features sourced from
Yahoo Finance: Open, High, Low, Close, Adj Close, Volume.
This expands the feature set to ~250 daily features.

Look-ahead safety
-----------------
Some pandas-ta indicators use future prices with their default
settings. Their value at day t changes when data after t is removed,
which leaks the future into any model trained on them. These columns
are dropped (see ``NONCAUSAL_PREFIXES``), and indicator outputs dated
after the last input row (Ichimoku's forward span projections) are
discarded. ``TechnicalIndicators.find_noncausal_columns`` re-runs the
truncation check and is exercised by the unit tests, so a pandas-ta
upgrade that introduces a new look-ahead indicator fails the test suite.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALL_CATEGORIES = [
    "candles",
    "cycles",
    "momentum",
    "overlap",
    "performance",
    "statistics",
    "trend",
    "utility",
    "volatility",
    "volume",
]

# Indicator columns whose value at day t depends on data after t
# (verified empirically with ``find_noncausal_columns`` on pandas-ta
# 0.4.71b0):
#   DPO_          detrended price oscillator, centred by default, which
#                 shifts the close backwards by length / 2 + 1 days
#   ICS_          Ichimoku chikou span: the close shifted back 26 days,
#                 i.e. the price 26 days in the future
#   TOS_STDEVALL  linear regression and standard deviation bands fitted
#                 on the whole sample
#   VHM_          volume heatmap, normalised with future data
#   ZIGZAG        swing pivots, placed on the day they occurred but only
#                 confirmed by later prices
NONCAUSAL_PREFIXES: tuple[str, ...] = (
    "DPO_",
    "ICS_",
    "TOS_STDEVALL",
    "VHM_",
    "ZIGZAG",
)


class TechnicalIndicators:
    """Compute look-ahead-free technical indicators using pandas-ta.

    Applies every pandas-ta indicator (or a filtered subset of
    categories) to a raw OHLCV DataFrame and returns the augmented
    DataFrame, with non-causal indicators removed.

    Parameters
    ----------
    categories : list[str] or None
        Subset of pandas-ta categories to compute. Pass ``None``
        (default) to compute all categories.
    exclude_cols : list[str] or None
        Column names to drop after indicator computation.
    drop_noncausal : bool
        If ``True`` (default), drop columns matching
        ``NONCAUSAL_PREFIXES``. Only set ``False`` to audit the raw
        pandas-ta output.

    Examples
    --------
    >>> ti = TechnicalIndicators()
    >>> df_indicators = ti.compute(df_ohlcv)
    >>> ti.find_noncausal_columns(df_ohlcv)
    []

    """

    def __init__(
        self,
        categories: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        drop_noncausal: bool = True,
    ) -> None:
        self.categories = categories or ALL_CATEGORIES
        self.exclude_cols = exclude_cols or []
        self.drop_noncausal = drop_noncausal

        invalid = set(self.categories) - set(ALL_CATEGORIES)
        if invalid:
            raise ValueError(
                f"Unknown pandas-ta categories: {invalid}. "
                f"Valid options: {ALL_CATEGORIES}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured technical indicators for *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame with columns
            ``["Open", "High", "Low", "Close", "Adj Close", "Volume"]``
            and a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Original columns plus all computed indicator columns, on
            exactly the same index as *df* and with unique column names.

        Raises
        ------
        ImportError
            If ``pandas_ta`` is not installed.

        """
        try:
            import pandas_ta as ta  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError(
                "pandas-ta is required. Install it with: "
                "pip install pandas-ta"
            ) from exc

        result = df.copy()
        result = self._rename_for_ta(result)

        logger.info(
            "Computing indicators for categories: %s", self.categories
        )

        # pandas-ta 0.4.x removed Strategy, so each indicator is called
        # individually and the results are collected.
        indicator_frames: list[pd.DataFrame] = []
        for indicator in self._get_indicator_list():
            try:
                fn = getattr(result.ta, indicator, None)
                if fn is None:
                    continue
                out = fn()
                out = self._to_frame(out)
                if out is None:
                    continue
                # Keep only rows that exist in the input. Ichimoku also
                # returns span projections dated up to 26 days into the
                # future, which would otherwise be appended as fake rows.
                out = out.reindex(result.index)
                indicator_frames.append(out)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping indicator %s: %s", indicator, exc)

        if indicator_frames:
            indicators_df = pd.concat(indicator_frames, axis=1)
            indicators_df = indicators_df.loc[
                :, ~indicators_df.columns.duplicated()
            ]
            # Some indicators echo the input columns back; keep the originals.
            indicators_df = indicators_df.drop(
                columns=[c for c in indicators_df.columns if c in result.columns]
            )
            result = pd.concat([result, indicators_df], axis=1)

        result = self._restore_col_names(result, df)

        cols_to_drop = [c for c in self.exclude_cols if c in result.columns]
        if self.drop_noncausal:
            noncausal = [
                c for c in result.columns
                if str(c).startswith(NONCAUSAL_PREFIXES)
            ]
            if noncausal:
                logger.info("Dropping %d look-ahead columns: %s",
                            len(noncausal), noncausal)
            cols_to_drop += noncausal
        if cols_to_drop:
            result = result.drop(columns=sorted(set(cols_to_drop)))

        n_new = result.shape[1] - df.shape[1]
        logger.info(
            "Indicator computation complete. "
            "Original columns: %d, New indicator columns: %d, Total: %d",
            df.shape[1],
            n_new,
            result.shape[1],
        )
        return result

    def find_noncausal_columns(
        self,
        df: pd.DataFrame,
        cut_points: Sequence[int] | None = None,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> list[str]:
        """Return columns whose past values change when later data is removed.

        For every cut point ``t`` the indicators are recomputed on
        ``df.iloc[: t + 1]`` and every row up to ``t`` is compared with
        the full-sample computation. A causal indicator gives identical
        values on the shared rows. Comparing all rows, not only row ``t``,
        also catches sparse indicators such as ZIGZAG, which place a
        pivot on the day it occurred once later prices confirm it.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame, long enough for the slowest indicator
            to warm up at the earliest cut point.
        cut_points : sequence of int or None
            Row positions to test. Defaults to four points spread over
            the last 40% of the sample.
        rtol, atol : float
            Tolerances passed to ``numpy.isclose``.

        Returns
        -------
        list[str]
            Sorted names of non-causal columns (empty if all are causal).

        """
        n = len(df)
        if cut_points is None:
            cut_points = [int(n * q) for q in (0.6, 0.7, 0.8, 0.9)]

        full = self.compute(df)
        bad: set[str] = set()
        for cut in cut_points:
            part = self.compute(df.iloc[: cut + 1])
            for col in full.columns:
                a = full[col].iloc[: cut + 1].to_numpy(dtype=float)
                if col in part.columns:
                    b = part[col].to_numpy(dtype=float)
                else:
                    b = np.full_like(a, np.nan)
                same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
                if not same.all():
                    bad.add(str(col))
        return sorted(bad)

    def indicator_names(self) -> list[str]:
        """Return the list of indicator function names that will be computed.

        Returns
        -------
        list[str]
            Sorted list of pandas-ta indicator names.

        """
        return sorted(self._get_indicator_list())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_frame(out: object) -> pd.DataFrame | None:
        """Normalise a pandas-ta return value to a DataFrame (or None)."""
        if out is None:
            return None
        if isinstance(out, pd.Series):
            return out.to_frame()
        if isinstance(out, tuple):
            parts = [
                x.to_frame() if isinstance(x, pd.Series) else x
                for x in out
                if isinstance(x, (pd.Series, pd.DataFrame))
            ]
            if not parts:
                return None
            out = pd.concat(parts, axis=1)
        if not isinstance(out, pd.DataFrame) or out.empty:
            return None
        return out

    def _get_indicator_list(self) -> list[str]:
        """Return all pandas-ta indicator names for selected categories.

        Falls back to inspecting the ta accessor directly if the
        Category dict is empty (API changed in 0.4.x).
        """
        try:
            import pandas_ta as ta  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pandas-ta is required.") from exc

        indicators: list[str] = []
        for category in self.categories:
            cat_indicators = getattr(ta, "Category", {}).get(category, [])
            indicators.extend(cat_indicators)

        # Fallback for new API where Category dict may be empty
        if not indicators:
            dummy = pd.DataFrame(
                {
                    "open": [1.0, 2.0, 3.0],
                    "high": [1.5, 2.5, 3.5],
                    "low": [0.5, 1.5, 2.5],
                    "close": [1.2, 2.2, 3.2],
                    "volume": [1000.0, 1100.0, 1200.0],
                }
            )
            skip = {
                "strategy", "indicators", "categories",
                "ticker", "trades", "above", "below",
                "above_value", "below_value", "cross",
                "cross_value", "long_run", "short_run",
                "datetime_ordered", "reverse", "to_utc",
                "adjusted", "cores",
            }
            indicators = [
                m for m in dir(dummy.ta)
                if not m.startswith("_")
                and callable(getattr(dummy.ta, m))
                and m not in skip
            ]
            logger.info(
                "Category dict empty, using %d indicators from accessor.",
                len(indicators),
            )

        return indicators

    @staticmethod
    def _rename_for_ta(df: pd.DataFrame) -> pd.DataFrame:
        """Rename 'Adj Close' → 'Adj_Close' for pandas-ta compatibility."""
        return df.rename(columns={"Adj Close": "Adj_Close"})

    @staticmethod
    def _restore_col_names(
        result: pd.DataFrame, original: pd.DataFrame
    ) -> pd.DataFrame:
        """Restore 'Adj_Close' → 'Adj Close' in the result."""
        if "Adj_Close" in result.columns and "Adj Close" in original.columns:
            result = result.rename(columns={"Adj_Close": "Adj Close"})
        return result

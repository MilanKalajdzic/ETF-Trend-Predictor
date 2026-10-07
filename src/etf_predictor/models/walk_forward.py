"""Expanding-window walk-forward validation for the MLP and LSTM models.

Each fold:
    1. Re-fits the supplied model on the train slice.
    2. Predicts signals on the test slice.
    3. Computes the next-day strategy return  signal_t * actual_return_{t+1}.

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from etf_predictor.models.lstm_value import LSTMValueModel
from etf_predictor.models.mlp_signal import MLPSignalModel

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Container for walk-forward outputs.

    Attributes
    ----------
    predictions : pd.DataFrame
        Index = test dates concatenated across folds.
        Columns: ``["signal", "actual_return", "strategy_return", "fold"]``.
    fold_metrics : pd.DataFrame
        One row per fold: ``["fold", "train_start", "train_end",
        "test_start", "test_end", "n_test", "accuracy", "hit_rate",
        "mean_return", "sharpe"]``.
    model_name : str
        Identifier of the model that produced these results.

    """

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_name: str


class WalkForwardValidator:
    """Expanding-window walk-forward validator.

    Parameters
    ----------
    initial_train_size : int
        Number of rows used for the very first training window.
    test_size : int
        Number of rows in every test fold.
    max_folds : int or None
        Optional cap on number of folds (handy for quick smoke tests).
    price_col : str
        Column in the **unscaled** DataFrame holding the price used
        to compute realised returns. Defaults to ``"Close"``.
    target_col : str
        Column holding the binary target Γ ∈ {+1, -1}. Used by the
        MLP only; LSTM ignores this and works off the price series.

    Notes
    -----
    The validator expects two parallel DataFrames:

    - ``df_scaled``: the modeling-ready frame produced by ``DataPipeline``
      (features in [0, 1] plus the ``Gamma`` column).
    - ``close_unscaled``: the raw ``Close`` series in price units, used
      for the LSTM's return target and for computing strategy returns.

    """

    def __init__(
        self,
        initial_train_size: int,
        test_size: int,
        max_folds: int | None = None,
        price_col: str = "Close",
        target_col: str = "Gamma",
    ) -> None:
        if initial_train_size <= 0 or test_size <= 0:
            raise ValueError(
                "initial_train_size and test_size must be positive."
            )
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.max_folds = max_folds
        self.price_col = price_col
        self.target_col = target_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_mlp(
        self,
        df_scaled: pd.DataFrame,
        close_unscaled: pd.Series,
        model_factory: Callable[[], MLPSignalModel] | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation for the MLP signal model.

        Parameters
        ----------
        df_scaled : pd.DataFrame
            Scaled feature frame including the ``Gamma`` target column.
        close_unscaled : pd.Series
            Unscaled close price aligned with ``df_scaled.index``.
        model_factory : callable
            Returns a fresh ``MLPSignalModel`` per fold. Defaults to
            ``MLPSignalModel()`` with library defaults.

        """
        if self.target_col not in df_scaled.columns:
            raise KeyError(
                f"df_scaled must contain '{self.target_col}' column."
            )
        factory = model_factory or (lambda: MLPSignalModel())
        feature_cols = [c for c in df_scaled.columns if c != self.target_col]

        return self._run(
            df_scaled=df_scaled,
            close_unscaled=close_unscaled,
            model_name="MLP",
            train_fn=lambda train_df: self._train_mlp(
                train_df, feature_cols, factory
            ),
            predict_fn=lambda model, train_df, test_df:
                model.predict_signal(test_df[feature_cols]),
        )

    def run_lstm(
        self,
        df_scaled: pd.DataFrame,
        close_unscaled: pd.Series,
        model_factory: Callable[[], LSTMValueModel] | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation for the LSTM value model.

        Parameters
        ----------
        df_scaled : pd.DataFrame
            Scaled feature frame (may include ``Gamma`` — it is ignored).
        close_unscaled : pd.Series
            Unscaled close price aligned with ``df_scaled.index``.
        model_factory : callable
            Returns a fresh ``LSTMValueModel`` per fold.

        """
        factory = model_factory or (lambda: LSTMValueModel())
        feature_cols = [c for c in df_scaled.columns if c != self.target_col]

        def _train(train_df: pd.DataFrame) -> LSTMValueModel:
            model = factory()
            model.fit(
                train_df[feature_cols],
                close_unscaled.loc[train_df.index],
            )
            return model

        def _predict(
            model: LSTMValueModel,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
        ) -> np.ndarray:
            history = train_df[feature_cols].tail(model.sequence_length)
            return model.predict_signal(test_df[feature_cols], history=history)

        return self._run(
            df_scaled=df_scaled,
            close_unscaled=close_unscaled,
            model_name="LSTM",
            train_fn=_train,
            predict_fn=_predict,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_mlp(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        factory: Callable[[], MLPSignalModel],
    ) -> MLPSignalModel:
        model = factory()
        model.fit(train_df[feature_cols], train_df[self.target_col])
        return model

    def _run(
        self,
        df_scaled: pd.DataFrame,
        close_unscaled: pd.Series,
        model_name: str,
        train_fn: Callable[[pd.DataFrame], object],
        predict_fn: Callable[[object, pd.DataFrame, pd.DataFrame], np.ndarray],
    ) -> WalkForwardResult:
        if not df_scaled.index.equals(close_unscaled.index):
            close_unscaled = close_unscaled.reindex(df_scaled.index)
        if close_unscaled.isna().any():
            raise ValueError(
                "close_unscaled has NaN after aligning to df_scaled.index."
            )

        n = len(df_scaled)
        if n <= self.initial_train_size + self.test_size:
            raise ValueError(
                f"Need at least initial_train_size + test_size + 1 = "
                f"{self.initial_train_size + self.test_size + 1} rows, "
                f"got {n}."
            )

        # Realised next-day simple return on the close series.
        actual_returns = close_unscaled.pct_change().shift(-1)

        fold_rows = []
        pred_chunks = []
        fold_idx = 0
        train_end = self.initial_train_size

        while train_end + self.test_size <= n:
            if self.max_folds is not None and fold_idx >= self.max_folds:
                break
            fold_idx += 1
            test_end = train_end + self.test_size

            train_df = df_scaled.iloc[:train_end]
            test_df = df_scaled.iloc[train_end:test_end]

            logger.info(
                "[%s] fold %d  train=%d  test=%d  (%s → %s)",
                model_name,
                fold_idx,
                len(train_df),
                len(test_df),
                test_df.index[0].date()
                if hasattr(test_df.index[0], "date")
                else test_df.index[0],
                test_df.index[-1].date()
                if hasattr(test_df.index[-1], "date")
                else test_df.index[-1],
            )

            model = train_fn(train_df)
            signal = predict_fn(model, train_df, test_df)
            test_returns = actual_returns.loc[test_df.index]
            strat_returns = signal * test_returns

            chunk = pd.DataFrame(
                {
                    "signal": signal,
                    "actual_return": test_returns.values,
                    "strategy_return": strat_returns.values,
                    "fold": fold_idx,
                },
                index=test_df.index,
            )
            pred_chunks.append(chunk)

            # Per-fold metrics (drop the final row whose forward return is NaN).
            clean = chunk.dropna(subset=["actual_return"])
            up_mask = clean["actual_return"] > 0
            sig_up = clean["signal"] == 1
            accuracy = float((sig_up == up_mask).mean()) if len(clean) else np.nan
            hit_rate = (
                float((clean["strategy_return"] > 0).mean())
                if len(clean) else np.nan
            )
            mean_ret = (
                float(clean["strategy_return"].mean()) if len(clean) else np.nan
            )
            std_ret = (
                float(clean["strategy_return"].std()) if len(clean) else np.nan
            )
            sharpe = (
                float(np.sqrt(252) * mean_ret / std_ret)
                if std_ret and std_ret > 0 else np.nan
            )

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "train_start": train_df.index[0],
                    "train_end": train_df.index[-1],
                    "test_start": test_df.index[0],
                    "test_end": test_df.index[-1],
                    "n_test": len(test_df),
                    "accuracy": accuracy,
                    "hit_rate": hit_rate,
                    "mean_return": mean_ret,
                    "sharpe": sharpe,
                }
            )

            train_end = test_end

        if not pred_chunks:
            raise RuntimeError("Walk-forward produced no folds.")

        predictions = pd.concat(pred_chunks).sort_index()
        fold_metrics = pd.DataFrame(fold_rows)
        return WalkForwardResult(
            predictions=predictions,
            fold_metrics=fold_metrics,
            model_name=model_name,
        )

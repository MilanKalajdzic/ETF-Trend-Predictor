"""LSTM regressor for next-day returns.

The network reads a fixed window of past features and predicts the
next day's simple return of the close. The sign of the prediction is
the trading signal: long when the predicted return is positive, short
otherwise.

Predicting the return rather than the price level matters in a
walk-forward setting. A level target has to be scaled to the training
window's price range, and when the price later trades above anything
seen in training (new highs), the network cannot predict a level that
high. Its prediction then sits below the current price and the signal
is stuck short for as long as the uptrend lasts. Returns are
stationary, so the same target scale holds out of sample.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _LSTMNet(nn.Module):
    """Single-layer LSTM with a fully-connected regression head."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMValueModel:
    """LSTM regressor that predicts next-day returns.

    Parameters
    ----------
    sequence_length : int
    hidden_size : int
    num_layers : int
    dropout : float
    learning_rate : float
    epochs : int
    batch_size : int
    weight_decay : float
    price_col : str
    device : str or None
    random_state : int

    """

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        price_col: str = "Close",
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.price_col = price_col
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: _LSTMNet | None = None
        self._feature_cols: list[str] | None = None
        # Mean and standard deviation of the training returns, used to
        # standardise the regression target and to invert predictions.
        self._ret_mean: float | None = None
        self._ret_std: float | None = None


    def fit(
        self,
        X: pd.DataFrame,
        close_unscaled: pd.Series,
    ) -> LSTMValueModel:
        """Fit the network to predict next-day returns.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled features for the training window.
        close_unscaled : pd.Series
            Close price in price units, aligned with *X*. Only its daily
            returns are used as the target.

        Returns
        -------
        LSTMValueModel
            The fitted model.

        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if not X.index.equals(close_unscaled.index):
            close_unscaled = close_unscaled.reindex(X.index)
        if close_unscaled.isna().any():
            raise ValueError(
                "close_unscaled contains NaN after aligning to X.index."
            )

        self._feature_cols = list(X.columns)
        # Target for the window ending at row i - 1 is the return from
        # row i - 1 to row i, so every target lies inside the training
        # window.
        returns = close_unscaled.pct_change()
        self._ret_mean = float(returns.mean())
        self._ret_std = float(returns.std())
        if not self._ret_std > 0:
            raise ValueError("Close returns have zero variance, cannot scale.")

        target = ((returns - self._ret_mean) / self._ret_std).fillna(0.0)

        seqs, targets = self._make_sequences(
            X.to_numpy(dtype=np.float32),
            target.to_numpy(dtype=np.float32),
        )
        if len(seqs) == 0:
            raise ValueError(
                f"Not enough rows ({len(X)}) for sequence_length="
                f"{self.sequence_length}."
            )

        x_t = torch.from_numpy(seqs).to(self.device)
        y_t = torch.from_numpy(targets).to(self.device)
        loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._model = _LSTMNet(
            n_features=x_t.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimiser = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                optimiser.zero_grad()
                preds = self._model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimiser.step()
                total += loss.item() * xb.size(0)
            avg = total / len(x_t)
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info(
                    "LSTM epoch %d/%d  loss=%.6f",
                    epoch + 1, self.epochs, avg,
                )
        return self

    def predict_return(
        self,
        X: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Predict the next-day return for every row of *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled features to predict on.
        history : pd.DataFrame or None
            Rows preceding *X* (typically the tail of the training
            window), so the first rows of *X* get a full look-back
            window.

        Returns
        -------
        pd.Series
            Predicted return from each day to the next, indexed by
            ``X.index``. NaN where the look-back window is incomplete.

        """
        self._check_fitted()
        if history is None:
            feature_df = X[self._feature_cols]
        else:
            feature_df = pd.concat(
                [history[self._feature_cols], X[self._feature_cols]]
            )

        arr = feature_df.to_numpy(dtype=np.float32)
        windows = []
        for i in range(self.sequence_length - 1, len(arr)):
            windows.append(arr[i - self.sequence_length + 1: i + 1])
        if not windows:
            return pd.Series(
                [np.nan] * len(X), index=X.index, name="pred_return"
            )

        x_t = torch.from_numpy(np.stack(windows)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            preds_scaled = self._model(x_t).cpu().numpy()

        preds = preds_scaled * self._ret_std + self._ret_mean

        # Align predictions with the requested output index. Each window
        # ends at row (sequence_length - 1 + i) of feature_df.
        last_idx = feature_df.index[self.sequence_length - 1:]
        full = pd.Series(preds, index=last_idx, name="pred_return")
        return full.reindex(X.index)

    def predict_signal(
        self,
        X: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Convert predicted returns into a ±1 trading signal.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled features to predict on.
        history : pd.DataFrame or None
            Rows preceding *X*, see :meth:`predict_return`.

        Returns
        -------
        np.ndarray
            Signals in {+1, -1}: +1 (long) when the predicted next-day
            return is positive, -1 (short) otherwise.

        """
        pred = self.predict_return(X, history=history)
        return np.where(pred.fillna(0.0) > 0, 1, -1).astype(np.int8)

    def _make_sequences(
        self, features: np.ndarray, target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (n_samples, seq_len, n_features) and next-day targets."""
        seqs, ys = [], []
        for i in range(self.sequence_length, len(features)):
            seqs.append(features[i - self.sequence_length: i])
            ys.append(target[i])
        return (
            np.stack(seqs) if seqs else np.empty((0,)),
            np.asarray(ys, dtype=np.float32),
        )

    def _check_fitted(self) -> None:
        if self._model is None or self._ret_std is None:
            raise RuntimeError("Call fit() before predict_*().")

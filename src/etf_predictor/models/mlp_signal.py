"""Multilayer perceptron that predicts the trend direction Γ(t).

The target Γ(t) ∈ {+1, -1} is learned directly as a binary
classification task.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _MLPNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPSignalModel:
    """MLP classifier that predicts Γ(t) from indicator features.

    Parameters
    ----------
    hidden_sizes : tuple[int, int]
    dropout : float
    learning_rate : float
    epochs : int
    batch_size : int
    weight_decay : float
    device : str or None
    random_state : int

    """

    def __init__(
        self,
        hidden_sizes: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: _MLPNet | None = None
        self._feature_cols: list[str] | None = None


    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLPSignalModel:
        """Train the MLP on (X, y).

        Parameters
        ----------
        X : pd.DataFrame
            Scaled training features.
        y : pd.Series
            Target labels in {+1, -1}.

        Returns
        -------
        MLPSignalModel

        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._feature_cols = list(X.columns)
        x_arr = X.to_numpy(dtype=np.float32)
        # Map {-1, +1} → {0, 1} for BCEWithLogitsLoss
        y_arr = ((y.to_numpy() + 1) // 2).astype(np.float32)

        x_t = torch.from_numpy(x_arr).to(self.device)
        y_t = torch.from_numpy(y_arr).to(self.device)
        dataset = TensorDataset(x_t, y_t)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self._model = _MLPNet(
            n_features=x_arr.shape[1],
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device)
        optimiser = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        self._model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                optimiser.zero_grad()
                logits = self._model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimiser.step()
                total += loss.item() * xb.size(0)
            avg = total / len(dataset)
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info(
                    "MLP epoch %d/%d  loss=%.4f",
                    epoch + 1, self.epochs, avg,
                )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted P(Γ=+1) for each row of *X*."""
        self._check_fitted()
        X = X[self._feature_cols]
        x_t = torch.from_numpy(X.to_numpy(dtype=np.float32)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self._model(x_t)).cpu().numpy()
        return probs

    def predict_signal(self, X: pd.DataFrame) -> np.ndarray:
        """Return trading signal ∈ {+1, -1} for each row of *X*.

        +1 = long, -1 = short. Probabilities ≥ 0.5 map to +1.
        """
        probs = self.predict_proba(X)
        return np.where(probs >= 0.5, 1, -1).astype(np.int8)



    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_*().")

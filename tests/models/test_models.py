"""Smoke tests for the MLP, LSTM, walk-forward, and equity modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from etf_predictor.models.equity import (
    apply_transaction_costs,
    build_comparison,
    equity_curve,
    summary_metrics,
)
from etf_predictor.models.lstm_value import LSTMValueModel
from etf_predictor.models.mlp_signal import MLPSignalModel
from etf_predictor.models.walk_forward import WalkForwardValidator


def _make_dataset(n: int = 400, n_features: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic dataset with a learnable target and a simulated price."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        index=dates,
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Target tied to first two features so models have a real signal.
    score = X["f0"] - 0.5 * X["f1"] + rng.normal(scale=0.5, size=n)
    gamma = pd.Series(np.where(score > 0, 1, -1), index=dates, name="Gamma")
    # Simulated price proportional to cumulative score.
    price = pd.Series(
        100 + np.cumsum(score * 0.5), index=dates, name="Close",
    ).clip(lower=1.0)
    df = X.copy()
    df["Gamma"] = gamma
    return df, price


def test_mlp_fits_and_predicts_signal_shape() -> None:
    df, _price = _make_dataset()
    X = df.drop(columns=["Gamma"])
    y = df["Gamma"]

    model = MLPSignalModel(epochs=3, batch_size=32)
    model.fit(X, y)

    sig = model.predict_signal(X.head(20))
    assert sig.shape == (20,)
    assert set(np.unique(sig)).issubset({-1, 1})


def test_lstm_fits_and_predicts_return() -> None:
    df, price = _make_dataset(n=200)
    X = df.drop(columns=["Gamma"])

    model = LSTMValueModel(
        sequence_length=10, epochs=2, batch_size=16, hidden_size=16,
    )
    model.fit(X, price)

    pred = model.predict_return(X.tail(30), history=X.iloc[-40:-30])
    assert len(pred) == 30
    assert pred.notna().all()


def test_lstm_not_stuck_short_at_new_highs() -> None:
    # A steady uptrend with uninformative features: once the test period
    # trades above the training range, a price-level target would push
    # every prediction below the current price (all short). A return
    # target should follow the positive drift instead.
    rng = np.random.default_rng(1)
    n, n_train = 400, 300
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.uniform(size=(n, 4)), index=dates)
    returns = 0.002 + rng.normal(scale=0.002, size=n)
    price = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    assert price.iloc[n_train:].min() > price.iloc[:n_train].max() * 0.95

    model = LSTMValueModel(sequence_length=10, epochs=5, hidden_size=16)
    model.fit(X.iloc[:n_train], price.iloc[:n_train])
    signal = model.predict_signal(
        X.iloc[n_train:], history=X.iloc[n_train - 10:n_train]
    )
    assert (signal == 1).mean() > 0.8


def test_walk_forward_mlp_runs() -> None:
    df, price = _make_dataset(n=300)
    validator = WalkForwardValidator(
        initial_train_size=150, test_size=50, max_folds=2,
    )
    result = validator.run_mlp(
        df, price,
        model_factory=lambda: MLPSignalModel(epochs=2, batch_size=16),
    )
    assert result.model_name == "MLP"
    assert not result.predictions.empty
    assert "strategy_return" in result.predictions.columns
    assert len(result.fold_metrics) == 2


def test_walk_forward_lstm_runs() -> None:
    df, price = _make_dataset(n=300)
    validator = WalkForwardValidator(
        initial_train_size=150, test_size=50, max_folds=2,
    )
    result = validator.run_lstm(
        df, price,
        model_factory=lambda: LSTMValueModel(
            sequence_length=10, epochs=2, batch_size=16, hidden_size=16,
        ),
    )
    assert result.model_name == "LSTM"
    assert len(result.fold_metrics) == 2


def test_equity_helpers_align_and_summarise() -> None:
    df, price = _make_dataset(n=300)
    validator = WalkForwardValidator(
        initial_train_size=150, test_size=50, max_folds=2,
    )
    mlp_res = validator.run_mlp(
        df, price,
        model_factory=lambda: MLPSignalModel(epochs=2, batch_size=16),
    )
    lstm_res = validator.run_lstm(
        df, price,
        model_factory=lambda: LSTMValueModel(
            sequence_length=10, epochs=2, batch_size=16, hidden_size=16,
        ),
    )
    curves, metrics = build_comparison([mlp_res, lstm_res], price)
    assert {"MLP", "LSTM", "BuyAndHold"}.issubset(curves.columns)
    assert {"MLP", "LSTM", "BuyAndHold"}.issubset(metrics.index)
    assert (curves.iloc[0] - 1.0).abs().max() < 1e-9

    eq = equity_curve(mlp_res.predictions["strategy_return"])
    assert eq.iloc[0] > 0
    stats = summary_metrics(mlp_res.predictions["strategy_return"])
    assert "sharpe" in stats and "max_drawdown" in stats


def test_transaction_costs_charge_turnover() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    signal = pd.Series([1, 1, -1, -1], index=idx)
    returns = pd.Series(0.01, index=idx)

    net = apply_transaction_costs(signal, returns, cost_bps=10)

    # Entry trades one unit, the long-to-short flip on day 3 trades two.
    expected = [0.01 - 0.001, 0.01, 0.01 - 0.002, 0.01]
    np.testing.assert_allclose(net.to_numpy(), expected)
    pd.testing.assert_series_equal(
        apply_transaction_costs(signal, returns, cost_bps=0), returns
    )

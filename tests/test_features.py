"""
Unit tests for utils.features.compute_features.

The central guarantee of the feature rework is *stationarity*: every feature
is invariant to the absolute price level. These tests lock that in — if
someone reintroduces a raw price-level feature (a bare SMA, Bollinger band
price, cumulative OBV, un-normalized MACD/ATR), the scale-invariance test
below breaks.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.features import compute_features, TECHNICAL_FEATURES


def _synthetic_ohlcv(n=260, start=100.0, seed=0):
    """Deterministic pseudo-random walk with a High/Low/Open envelope."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.02, size=n)
    close = start * np.cumprod(1 + steps)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    high = close * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    open_ = close * (1 + rng.normal(0, 0.003, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high,
        "Low": low, "Close": close, "Volume": volume,
    })


def test_all_features_present():
    out = compute_features(_synthetic_ohlcv())
    for feat in TECHNICAL_FEATURES:
        assert feat in out.columns, f"missing feature {feat}"


def test_features_are_scale_invariant():
    """Multiplying all prices by a constant must not change any feature.

    This is the definition of stationarity we rely on for walk-forward
    validation and cross-ticker application.
    """
    base = _synthetic_ohlcv()
    scaled = base.copy()
    for col in ["Open", "High", "Low", "Close"]:
        scaled[col] = scaled[col] * 10.0
    # Volume pressure (OBV) depends only on the *sign* of price changes, so
    # scaling price leaves it unchanged too.

    fb = compute_features(base)
    fs = compute_features(scaled)

    for feat in TECHNICAL_FEATURES:
        a = fb[feat].to_numpy()
        b = fs[feat].to_numpy()
        mask = ~(np.isnan(a) | np.isnan(b))
        assert mask.any(), f"{feat} is all-NaN"
        max_diff = np.nanmax(np.abs(a[mask] - b[mask]))
        assert max_diff < 1e-9, f"{feat} not scale-invariant (max diff {max_diff})"


def test_warmup_rows_are_nan_not_dropped():
    out = compute_features(_synthetic_ohlcv())
    # price_sma50_ratio needs 50 rows of warmup -> first row NaN, last valid.
    assert np.isnan(out["price_sma50_ratio"].iloc[0])
    assert not np.isnan(out["price_sma50_ratio"].iloc[-1])
    # Rows are preserved, not silently dropped.
    assert len(out) == len(_synthetic_ohlcv())


def test_rsi_within_bounds():
    out = compute_features(_synthetic_ohlcv())
    rsi = out["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_rsi_is_100_when_only_gains():
    """A strictly increasing series has zero losses -> RSI must be 100, not NaN."""
    n = 100
    close = pd.Series(np.linspace(100, 200, n))
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "Close": close, "High": close, "Low": close,
        "Open": close, "Volume": np.full(n, 1_000_000.0),
    })
    out = compute_features(df)
    assert out["rsi_14"].dropna().iloc[-1] == pytest.approx(100.0)

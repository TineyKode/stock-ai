"""Universe backtest summary. No network and no fitted forest."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.backtesting.backtest import backtest_frame, summarize_universe
from scripts.training.compare_models import prepare_comparison


class AlwaysLong:
    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        up = np.full(len(x), 0.9)
        return np.column_stack([1.0 - up, up])


def _prices(ticker, days, drift):
    close = 100.0 * np.cumprod(1.0 + np.full(days, drift))
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=days, freq="B").strftime("%Y-%m-%d"),
        "Ticker": ticker,
        "Close": close,
        "rsi_14": 50.0,
    })


def test_universe_covers_each_ticker_and_reports_the_median():
    frame = pd.concat([
        _prices("AAA", 180, 0.001),
        _prices("BBB", 180, -0.0005),
    ], ignore_index=True)
    rows = backtest_frame(frame, AlwaysLong(), ["rsi_14"], n_splits=2)
    summary = summarize_universe(rows)
    assert summary["n_tickers"] == 2
    assert {row["ticker"] for row in rows} == {"AAA", "BBB"}
    assert summary["median_sharpe"] == float(pd.Series([row["sharpe_ratio"] for row in rows]).median())
    assert summary["beat_buyhold"] <= 2


def test_prepare_comparison_ignores_moves_below_the_cost_threshold():
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Ticker": ["AAPL", "AAPL", "AAPL"],
        "Close": [100.0, 100.01, 100.02],
        "rsi_14": [50.0, 50.0, 50.0],
    })
    _x, y = prepare_comparison(df, ticker="AAPL")
    assert list(y) == [0, 0]


def test_prepare_comparison_does_not_leak_the_next_ticker():
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "Ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "Close": [100.0, 100.01, 50.0, 80.0],
        "rsi_14": [50.0, 50.0, 50.0, 50.0],
    })
    _x, y = prepare_comparison(df, ticker="AAPL")
    assert list(y) == [0]

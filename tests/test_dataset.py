"""
Unit tests for utils.dataset — the per-ticker target and sentiment merge.

The regression these guard against: a naive Close.shift(-1) on a pooled
multi-ticker frame leaks the first row of one ticker into the last row of
the previous ticker. add_target must compute the label strictly within
each ticker.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.dataset import add_target, merge_sentiment


def _two_ticker_frame():
    return pd.DataFrame({
        "Date": ["2020-01-01", "2020-01-02", "2020-01-03",
                 "2020-01-01", "2020-01-02"],
        "Ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
        "Close": [100.0, 101.0, 100.0, 50.0, 60.0],
    })


def test_target_is_per_ticker_and_last_row_is_nan():
    out = add_target(_two_ticker_frame(), threshold=0.0)
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    aaa = out[out["Ticker"] == "AAA"]["Target"].tolist()
    bbb = out[out["Ticker"] == "BBB"]["Target"].tolist()

    # AAA: 100->101 up (1), 101->100 down (0), last has no next day (NaN)
    assert aaa[0] == 1.0
    assert aaa[1] == 0.0
    assert np.isnan(aaa[2])
    # BBB: 50->60 up (1), last NaN. Crucially BBB's first row does NOT leak
    # into AAA's last row (which is NaN, not derived from 50).
    assert bbb[0] == 1.0
    assert np.isnan(bbb[1])


def test_threshold_filters_small_moves():
    # AAA row0 return is exactly +1% (100->101); a 2% threshold zeroes it.
    out = add_target(_two_ticker_frame(), threshold=0.02)
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    aaa = out[out["Ticker"] == "AAA"]["Target"].tolist()
    assert aaa[0] == 0.0  # +1% does not clear the 2% bar


def test_merge_sentiment_absent_file_defaults_neutral():
    df = _two_ticker_frame()
    merged = merge_sentiment(df, path="does/not/exist.csv")
    assert (merged["sentiment_score"] == 0.5).all()


def test_merge_sentiment_fills_missing_with_neutral(tmp_path):
    sent = pd.DataFrame({
        "Date": ["2020-01-01"],
        "Ticker": ["AAA"],
        "sentiment_score": [0.9],
    })
    path = tmp_path / "news_sentiment.csv"
    sent.to_csv(path, index=False)

    merged = merge_sentiment(_two_ticker_frame(), path=str(path))
    merged = merged.set_index(["Ticker", "Date"])["sentiment_score"]

    # The one matching row keeps its value; everything else is neutral 0.5.
    assert merged[("AAA", "2020-01-01")] == 0.9
    assert merged[("AAA", "2020-01-02")] == 0.5
    assert merged[("BBB", "2020-01-01")] == 0.5

"""
Unit tests for the Alpha Vantage sentiment parsing (no network).

These exercise the pure functions score_to_unit and parse_feed_to_daily
against a fixture feed, so the aggregation is verified without an API key.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.fetching.fetch_alpha_sentiment import (
    score_to_unit,
    parse_feed_to_daily,
)


def test_score_to_unit_maps_and_clamps():
    assert score_to_unit(-1.0) == 0.0
    assert score_to_unit(0.0) == 0.5
    assert score_to_unit(1.0) == 1.0
    # Out-of-range values are clamped into [0, 1].
    assert score_to_unit(2.0) == 1.0
    assert score_to_unit(-2.0) == 0.0


FIXTURE_FEED = [
    {
        "time_published": "20220103T120000",
        "overall_sentiment_score": 0.1,
        "ticker_sentiment": [
            {"ticker": "AAPL", "ticker_sentiment_score": "0.2"},
            {"ticker": "MSFT", "ticker_sentiment_score": "0.9"},
        ],
    },
    {
        "time_published": "20220103T180000",
        "overall_sentiment_score": 0.5,
        "ticker_sentiment": [
            {"ticker": "AAPL", "ticker_sentiment_score": "0.4"},
        ],
    },
    {
        # No AAPL entry -> falls back to overall_sentiment_score (0.6) on a new day.
        "time_published": "20220104T090000",
        "overall_sentiment_score": 0.6,
        "ticker_sentiment": [
            {"ticker": "TSLA", "ticker_sentiment_score": "-0.3"},
        ],
    },
]


def test_parse_feed_aggregates_per_day_for_ticker():
    df = parse_feed_to_daily(FIXTURE_FEED, "AAPL")

    assert list(df.columns) == ["Date", "Ticker", "sentiment_score"]
    assert (df["Ticker"] == "AAPL").all()

    by_date = df.set_index("Date")["sentiment_score"]
    # 2022-01-03: two AAPL scores 0.2 and 0.4 -> unit 0.6 and 0.7 -> mean 0.65
    assert by_date["2022-01-03"] == pytest.approx(0.65)
    # 2022-01-04: no AAPL score -> overall 0.6 -> unit 0.8
    assert by_date["2022-01-04"] == pytest.approx(0.8)


def test_parse_feed_empty_returns_typed_empty_frame():
    df = parse_feed_to_daily([], "AAPL")
    assert df.empty
    assert list(df.columns) == ["Date", "Ticker", "sentiment_score"]


def _article(ticker):
    return [{
        "time_published": "20220103T120000",
        "ticker_sentiment": [{"ticker": ticker, "ticker_sentiment_score": "0.2"}],
    }]


def test_batch_stops_at_the_daily_cap_and_resumes_next_day(tmp_path):
    from scripts.fetching.fetch_alpha_sentiment import run_sentiment_batch

    output = tmp_path / "news_sentiment.csv"
    budget = tmp_path / "alpha_vantage_budget.json"
    calls = []

    def fetch(ticker, api_key, time_from, limit):
        calls.append(ticker)
        return _article(ticker)

    names = [f"T{i:02d}" for i in range(30)]
    code = run_sentiment_batch(
        names, "key", str(output), str(budget), "2026-10-17", fetch, budget=25,
    )
    assert code == 0
    assert calls == names[:25]
    saved = set(pd.read_csv(output)["Ticker"])
    assert saved == set(names[:25])

    calls.clear()
    code = run_sentiment_batch(
        names, "key", str(output), str(budget), "2026-10-17", fetch, budget=25, min_days=1,
    )
    assert code == 0
    assert calls == []

    code = run_sentiment_batch(
        names, "key", str(output), str(budget), "2026-10-18", fetch, budget=25, min_days=1,
    )
    assert code == 0
    assert calls == names[25:]
    assert set(pd.read_csv(output)["Ticker"]) == set(names)


def test_batch_stops_when_alpha_vantage_sends_a_note(tmp_path):
    from scripts.fetching.fetch_alpha_sentiment import (
        AlphaVantageLimit,
        run_sentiment_batch,
    )

    output = tmp_path / "news_sentiment.csv"
    budget = tmp_path / "alpha_vantage_budget.json"
    calls = []

    def fetch(ticker, api_key, time_from, limit):
        calls.append(ticker)
        if ticker == "MSFT":
            raise AlphaVantageLimit("No feed for MSFT: Thank you for using Alpha Vantage")
        return _article(ticker)

    code = run_sentiment_batch(
        ["AAPL", "MSFT", "NVDA"], "key", str(output), str(budget),
        "2026-10-17", fetch, budget=25,
    )
    assert code == 0
    assert calls == ["AAPL", "MSFT"]
    assert list(pd.read_csv(output)["Ticker"]) == ["AAPL"]

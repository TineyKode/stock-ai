"""
Fetch real historical news sentiment from Alpha Vantage and write it in the
format the hybrid model consumes (data/processed/news_sentiment.csv, columns
Date, Ticker, sentiment_score with sentiment_score in [0, 1]).

Why this exists
---------------
The BERTweet chain (fetch_newsapi.py -> fetch_news.py) can only reach ~30 days
of headlines on the free NewsAPI tier, so it cannot populate sentiment across
the multi-year training window. Alpha Vantage's NEWS_SENTIMENT endpoint returns
articles with a pre-computed sentiment score per ticker going back to 2022 and
is free (25 requests/day, 5/min), which is enough to backfill daily sentiment
for the pooled training set.

Alpha Vantage returns an overall_sentiment_score per article in roughly
[-1, 1] and a per-ticker ticker_sentiment_score. We use the per-ticker score
(it is the sentiment of that article *about that ticker*), average it per
calendar day, and map [-1, 1] -> [0, 1] so it matches the neutral-0.5
convention the rest of the pipeline already uses.

Usage
-----
  # requires ALPHA_VANTAGE_KEY in the environment / .env
  python scripts/fetching/fetch_alpha_sentiment.py
  python scripts/fetching/fetch_alpha_sentiment.py --time-from 20220101 --limit 1000

Without a key the script prints a notice and exits 0 (the hybrid model then
falls back to neutral 0.5), so it is safe to call from CI unconditionally.
"""
import os
import sys
import time
import argparse

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.tickers import get_tickers_by_country

API_URL = "https://www.alphavantage.co/query"
OUTPUT_PATH = "data/processed/news_sentiment.csv"

# Free tier: 5 requests/minute. Sleep ~13s between calls to stay under it.
RATE_LIMIT_SLEEP_S = 13


def score_to_unit(score):
    """Map an Alpha Vantage sentiment score in [-1, 1] to [0, 1].

    Values outside the expected range are clamped so a malformed feed cannot
    push sentiment_score out of the [0, 1] band the model expects.
    """
    unit = (float(score) + 1.0) / 2.0
    return min(1.0, max(0.0, unit))


def parse_feed_to_daily(feed, ticker):
    """
    Turn one Alpha Vantage NEWS_SENTIMENT `feed` list into a daily sentiment
    frame for `ticker`.

    `feed` is the list under the API response's "feed" key. Each item has a
    "time_published" (YYYYMMDDTHHMMSS) and a "ticker_sentiment" list of
    {ticker, ticker_sentiment_score}. We take this ticker's score from each
    article, average per day, and return columns Date, Ticker, sentiment_score.

    This is a pure function (no network) so it can be unit-tested against a
    fixture. Returns an empty, correctly-typed frame when there is no usable
    data.
    """
    cols = ["Date", "Ticker", "sentiment_score"]
    rows = []
    for item in feed or []:
        published = item.get("time_published", "")
        if len(published) < 8:
            continue
        date = f"{published[0:4]}-{published[4:6]}-{published[6:8]}"

        score = None
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                try:
                    score = float(ts.get("ticker_sentiment_score"))
                except (TypeError, ValueError):
                    score = None
                break
        # Fall back to the article's overall score if this ticker is not
        # itemized but the article was returned for it.
        if score is None and "overall_sentiment_score" in item:
            try:
                score = float(item["overall_sentiment_score"])
            except (TypeError, ValueError):
                score = None
        if score is None:
            continue

        rows.append({"Date": date, "sentiment_score": score_to_unit(score)})

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    daily = df.groupby("Date", as_index=False)["sentiment_score"].mean()
    daily["Ticker"] = ticker
    return daily[cols]


def merge_into_existing(new_df, path=OUTPUT_PATH):
    """
    Union new daily sentiment with any existing file, de-duplicating on
    (Date, Ticker) and keeping the freshly fetched value on conflicts.
    """
    frames = [new_df]
    if os.path.exists(path):
        existing = pd.read_csv(path)
        if {"Date", "Ticker", "sentiment_score"}.issubset(existing.columns):
            frames.insert(0, existing[["Date", "Ticker", "sentiment_score"]])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return combined


def fetch_ticker(ticker, api_key, time_from=None, limit=1000):
    """Call the Alpha Vantage NEWS_SENTIMENT endpoint for one ticker."""
    import requests  # lazy: keep the module importable without requests

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "sort": "EARLIEST",
        "limit": str(limit),
        "apikey": api_key,
    }
    if time_from:
        params["time_from"] = f"{time_from}T0000"

    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    # Alpha Vantage signals throttling/errors with a "Note" or "Information"
    # field and a 200 status; surface it instead of silently writing nothing.
    if "feed" not in payload:
        note = payload.get("Note") or payload.get("Information") or payload
        raise RuntimeError(f"No feed for {ticker}: {note}")
    return payload["feed"]


def main():
    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage news sentiment")
    parser.add_argument("--time-from", default=None,
                        help="Earliest date YYYYMMDD (default: API max history)")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max articles per ticker (Alpha Vantage max 1000)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("ALPHA_VANTAGE_KEY")
    if not api_key:
        print("ALPHA_VANTAGE_KEY not set - skipping sentiment fetch. "
              "The hybrid model will fall back to neutral 0.5.")
        return 0

    tickers = get_tickers_by_country("US")
    print(f"Fetching Alpha Vantage sentiment for {len(tickers)} tickers "
          f"(free tier: ~5/min, this will take a few minutes)...")

    daily_frames = []
    for i, ticker in enumerate(tickers):
        try:
            feed = fetch_ticker(ticker, api_key, args.time_from, args.limit)
            daily = parse_feed_to_daily(feed, ticker)
            daily_frames.append(daily)
            print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(daily)} days")
        except Exception as exc:  # noqa: BLE001 - keep going, report per ticker
            print(f"  [{i+1}/{len(tickers)}] {ticker}: FAILED ({exc})")
        if i < len(tickers) - 1:
            time.sleep(RATE_LIMIT_SLEEP_S)

    if not daily_frames:
        print("No sentiment fetched.")
        return 1

    new_df = pd.concat(daily_frames, ignore_index=True)
    combined = merge_into_existing(new_df, args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

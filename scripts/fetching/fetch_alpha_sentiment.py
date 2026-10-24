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
is free for 25 requests per day. One call is one US ticker, and there are
30 names, so a run stops at the cap and the next UTC day continues with the
names that are not in the output file yet.

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
import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.tickers import get_tickers_by_country

API_URL = "https://www.alphavantage.co/query"
OUTPUT_PATH = "data/processed/news_sentiment.csv"

# Free tier: 25 requests per day, and about 5 per minute.
DAILY_REQUEST_BUDGET = 25
RATE_LIMIT_SLEEP_S = 13


class AlphaVantageLimit(RuntimeError):
    """Alpha Vantage answered with Note or Information instead of a feed."""


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


def budget_path_for(output_path):
    folder = os.path.dirname(output_path) or "."
    return os.path.join(folder, "alpha_vantage_budget.json")


def covered_tickers(path):
    """Tickers that already have at least one row in the sentiment file."""
    if not os.path.exists(path):
        return set()
    existing = pd.read_csv(path)
    if existing.empty or "Ticker" not in existing.columns:
        return set()
    return set(existing["Ticker"].astype(str))


def load_used(path, today):
    """Calls already spent on this UTC date. A new date starts at zero."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0
    if data.get("date") != today:
        return 0
    try:
        return max(0, int(data.get("used", 0)))
    except (TypeError, ValueError):
        return 0


def save_used(path, today, used):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"date": today, "used": int(used)}, handle)


def choose_tickers(all_tickers, covered, used, budget=DAILY_REQUEST_BUDGET):
    """Names still missing, capped by the calls left today."""
    pending = [ticker for ticker in all_tickers if ticker not in covered]
    remaining = max(0, budget - used)
    return pending[:remaining]


def run_sentiment_batch(
    tickers,
    api_key,
    output_path,
    budget_path,
    today,
    fetch_fn,
    sleep_fn=None,
    time_from=None,
    limit=1000,
    budget=DAILY_REQUEST_BUDGET,
):
    """
    Fetch at most the calls left in today's budget.

    Tickers already present in the output file are skipped, so the next day
    continues at the first missing name instead of starting over at AAPL.
    A Note or Information response stops the run and keeps the original text.
    Returns the process exit code.
    """
    covered = covered_tickers(output_path)
    used = load_used(budget_path, today)
    chosen = choose_tickers(tickers, covered, used, budget)
    if not chosen:
        print(f"No Alpha Vantage calls today. Used {used} of {budget}. "
              f"{len(covered)} tickers already saved.")
        return 0

    print(f"Fetching {len(chosen)} of {len(tickers)} tickers "
          f"({used} of {budget} calls already used today).")
    frames = []
    saw_limit = False
    saw_error = False
    for index, ticker in enumerate(chosen):
        if index and sleep_fn:
            sleep_fn()
        try:
            feed = fetch_fn(ticker, api_key, time_from, limit)
            daily = parse_feed_to_daily(feed, ticker)
            if not daily.empty:
                frames.append(daily)
            print(f"  {ticker}: {len(daily)} days")
        except AlphaVantageLimit as exc:
            used += 1
            save_used(budget_path, today, used)
            saw_limit = True
            print(f"  {ticker}: stopped ({exc})")
            break
        except Exception as exc:  # noqa: BLE001 - keep the vendor message, try the next name
            used += 1
            save_used(budget_path, today, used)
            saw_error = True
            print(f"  {ticker}: FAILED ({exc})")
            continue
        used += 1
        save_used(budget_path, today, used)

    if not frames:
        if saw_limit:
            print("Alpha Vantage returned no feed. Saved calls stop here.")
            return 0
        if saw_error:
            print("No sentiment fetched.")
            return 1
        print("Alpha Vantage returned no usable articles.")
        return 0

    new_df = pd.concat(frames, ignore_index=True)
    combined = merge_into_existing(new_df, output_path)
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} rows to {output_path}")
    return 0


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

    # Alpha Vantage signals throttling and errors with a Note or Information
    # field and a 200 status. Keep that text. Stop the day's run on it.
    if "feed" not in payload:
        note = payload.get("Note") or payload.get("Information") or payload
        message = f"No feed for {ticker}: {note}"
        if "Note" in payload or "Information" in payload:
            raise AlphaVantageLimit(message)
        raise RuntimeError(message)
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
    today = datetime.now(timezone.utc).date().isoformat()
    return run_sentiment_batch(
        tickers,
        api_key,
        args.output,
        budget_path_for(args.output),
        today,
        fetch_ticker,
        sleep_fn=lambda: time.sleep(RATE_LIMIT_SLEEP_S),
        time_from=args.time_from,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())

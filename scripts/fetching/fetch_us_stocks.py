"""
Fetch US stock OHLCV data via yfinance and persist to data/us/<ticker>.csv.

Churn control
-------------
This script is run daily by a GitHub Action that commits the result, so
its output must be *stable* day to day. Two rules keep the git diff to
roughly one new line per day instead of rewriting the whole 5-year file:

1. Append-only: rows for dates already on disk are kept exactly as first
   written and never re-serialized. (yfinance auto-adjusts the entire
   history whenever a new dividend/split lands, which otherwise rewrites
   every row.)
2. Rounding: newly appended prices are rounded to 4 dp and volume to an
   integer, so floating-point noise never shows up as a diff.

The on-disk format matches yfinance's native 3-header layout
(Price / Ticker / Date) that preprocess_data.load_us_data expects.
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.tickers import get_tickers_by_country

TICKERS = get_tickers_by_country("US")
DATA_DIR = "data/us/"
PRICE_COLS = ["Close", "High", "Low", "Open", "Volume"]


def _read_existing(path):
    """Read an existing ticker CSV into a Date-indexed OHLCV frame, or None."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, header=[0, 1], skiprows=[2])
        df.columns = [col[0] for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        return df
    except Exception as e:
        print(f"  Could not parse existing {path} ({e}); rewriting from scratch")
        return None


def _round_frame(df):
    """Round price columns to 4 dp and volume to whole shares."""
    df = df.copy()
    for col in ["Close", "High", "Low", "Open"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].round(0)
    return df


def _write_frame(df, path, ticker):
    """Write a Date-indexed OHLCV frame in yfinance's 3-header format."""
    cols = [c for c in PRICE_COLS if c in df.columns]
    out = df[cols].copy()
    out.columns = pd.MultiIndex.from_tuples(
        [(c, ticker) for c in cols], names=["Price", "Ticker"]
    )
    out.index.name = "Date"
    out.to_csv(path)


def fetch_data(ticker, years=5):
    """Fetch historical data for a ticker and append only new dates."""
    import yfinance as yf  # lazy import: the CSV helpers don't need it

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")

    fetched = yf.download(ticker, start=start_date, end=end_date,
                          auto_adjust=True, progress=False)
    if fetched.empty:
        print(f"  No data returned for {ticker}, skipping")
        return

    if isinstance(fetched.columns, pd.MultiIndex):
        fetched.columns = fetched.columns.get_level_values(0)
    fetched.index = pd.to_datetime(fetched.index)
    fetched.index.name = "Date"
    fetched = _round_frame(fetched)

    os.makedirs(DATA_DIR, exist_ok=True)
    path = f"{DATA_DIR}{ticker.lower()}.csv"

    existing = _read_existing(path)
    if existing is not None:
        # Keep existing rows' VALUES (no re-adjustment to new dividend
        # factors); append only genuinely new dates.
        new_rows = fetched[~fetched.index.isin(existing.index)]
        combined = pd.concat([existing, new_rows])
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        added = len(new_rows)
    else:
        combined = fetched.sort_index()
        added = len(combined)

    # Round the whole frame so re-serialized 16-digit floats don't churn.
    # (First run rewrites each file once to 4 dp, then it is stable.)
    combined = _round_frame(combined)
    _write_frame(combined, path, ticker)
    print(f"Saved {ticker} to {DATA_DIR} (+{added} new rows, {len(combined)} total)")


def fetch_all():
    """Fetch data for all configured tickers."""
    for ticker in TICKERS:
        fetch_data(ticker)


if __name__ == "__main__":
    fetch_all()

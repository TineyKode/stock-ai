"""
Shared dataset assembly for training and backtesting.

Centralizes three things that were previously duplicated (and subtly
wrong) across the training scripts:

1. Loading the processed feature table.
2. Building the next-day-up target *per ticker* — a plain Close.shift(-1)
   on a pooled multi-ticker frame leaks the first row of the next ticker
   into the last row of the previous one.
3. Merging daily news sentiment with a neutral (0.5) fill.

Models are trained on ALL tickers pooled, not AAPL alone, so a single
model generalizes across the tickers the dashboard actually serves.
"""
import os
import numpy as np
import pandas as pd

FEATURES_PATH = "data/processed/features.csv"
SENTIMENT_PATH = "data/processed/news_sentiment.csv"


def load_features(path=FEATURES_PATH):
    """Load the processed feature table produced by preprocess_data.py."""
    return pd.read_csv(path)


def add_target(df):
    """
    Add a binary next-day-up Target computed independently per ticker.

    The last row of each ticker has no next day, so its Target is NaN
    (the caller drops it). This prevents cross-ticker leakage that a
    naive Close.shift(-1) on pooled data would introduce.
    """
    df = df.sort_values(["Ticker", "Date"]).copy()
    next_close = df.groupby("Ticker")["Close"].shift(-1)
    df["Target"] = (next_close > df["Close"]).astype(float)
    df.loc[next_close.isna(), "Target"] = np.nan
    return df


def merge_sentiment(df, path=SENTIMENT_PATH):
    """
    Left-merge daily news sentiment, filling missing values with 0.5
    (neutral). Safe to call when the sentiment file is absent.
    """
    if not os.path.exists(path):
        df = df.copy()
        df["sentiment_score"] = 0.5
        return df

    sentiment = pd.read_csv(path)
    group_cols = ["Date", "Ticker"] if "Ticker" in sentiment.columns else ["Date"]
    sentiment_daily = sentiment.groupby(group_cols, as_index=False)["sentiment_score"].mean()
    merge_cols = [c for c in group_cols if c in df.columns]
    merged = pd.merge(df, sentiment_daily, on=merge_cols, how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.5)
    return merged


def build_xy(features, target_col="Target", sort_by="Date"):
    """
    Prepare (X, y) for walk-forward CV: drop warmup/label NaNs and sort
    chronologically so TimeSeriesSplit's folds respect time order across
    the pooled panel.
    """
    df = features.dropna(subset=list(features.columns)).copy()
    df = df.sort_values(sort_by).reset_index(drop=True)
    return df

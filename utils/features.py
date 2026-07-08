"""
Centralized feature computation and model configuration.

Used by: preprocess_data.py, train_technical.py, train_hybrid.py,
         retrain_models.py, backtest.py, compare_models.py, app.py

All training and prediction code imports feature lists and model
parameters from here to guarantee consistency between training,
backtesting, and inference.

Design note — stationarity
--------------------------
Every feature here is *stationary*: its distribution does not depend on
the absolute price level. This matters because models are validated
walk-forward (train on the past, test on the future) and applied across
many tickers. Raw levels such as a 50-day SMA, Bollinger band prices,
cumulative OBV, or the raw MACD (all of which scale with price) would
hand the model values it never saw in training once a stock trends, and
would be meaningless when the same model is applied to a different
ticker. We therefore express each level as a ratio, a percentage, a
band position, or a rolling z-score instead.
"""
import numpy as np
import pandas as pd


# Canonical feature lists — training, backtesting and app.py must use
# these exact lists. All features are stationary (see module docstring).
TECHNICAL_FEATURES = [
    "rsi_14",             # momentum, already bounded 0-100
    "price_sma50_ratio",  # close / SMA50 - 1  (distance from trend)
    "obv_zscore_50",      # rolling z-score of OBV (volume pressure)
    "macd_norm",          # MACD line / close  (trend, price-normalized)
    "macd_signal_norm",   # MACD signal / close
    "bb_pct_b",           # Bollinger %B: position within the bands
    "bb_width",           # Bollinger band width / SMA20 (volatility)
    "atr_pct",            # ATR14 / close  (volatility, price-normalized)
    "return_1d",          # 1-day return
    "return_5d",          # 5-day return
    "return_10d",         # 10-day return
    "volatility_10d",     # 10-day realized volatility
    "volume_ratio_20d",   # volume / 20-day average volume
]

HYBRID_FEATURES = TECHNICAL_FEATURES + ["sentiment_score"]


# --- Signal / target configuration -------------------------------------
# The label is magnitude-aware: a day counts as "UP" only if the next-day
# return clears TARGET_THRESHOLD, so the model learns to flag moves that
# are actually worth trading rather than any move above zero.
TARGET_THRESHOLD = 0.0015   # ~15 bps: covers a 10 bps round-trip + slippage

# At inference/backtest we go long only when the model's P(up) is at least
# SIGNAL_THRESHOLD, so the strategy trades selectively (fewer trades, less
# cost, lower exposure on uncertain days) instead of on every predict==1.
SIGNAL_THRESHOLD = 0.55


# Shared model hyperparameters — one source of truth so training,
# retraining and backtesting all evaluate and ship the SAME model.
MODEL_PARAMS = {
    "technical": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "random_state": 42,
        "n_jobs": -1,
    },
    "hybrid": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "random_state": 42,
        "n_jobs": -1,
    },
}


def compute_features(df):
    """
    Compute all stationary technical features on a single-ticker OHLCV frame.

    Expects columns: Date, Close, Volume.
    Optional columns: Open, High, Low (used for a true-range ATR; a
    close-only proxy is used if High/Low are missing).

    Returns df with the TECHNICAL_FEATURES columns added. NaN rows from
    indicator warmup are NOT dropped here — the caller decides when to
    dropna.
    """
    df = df.sort_values("Date").copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    has_hl = "High" in df.columns and "Low" in df.columns
    if has_hl:
        h = df["High"].astype(float)
        l = df["Low"].astype(float)

    # --- RSI-14 ---
    # A window with no down-days (loss == 0) is maximally overbought -> RSI 100,
    # not NaN. A window with no up-days (gain == 0) -> RSI 0.
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss != 0, 100.0)      # loss == 0 -> 100
    rsi = rsi.where(~((loss == 0) & (gain == 0)), 50.0)  # flat -> neutral 50
    rsi[gain.isna() | loss.isna()] = np.nan  # keep warmup NaN
    df["rsi_14"] = rsi

    # --- Trend: close relative to SMA-50 (stationary) ---
    sma_50 = c.rolling(50).mean()
    df["price_sma50_ratio"] = c / sma_50 - 1.0

    # --- Volume pressure: OBV expressed as a rolling z-score (stationary) ---
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_mean = obv.rolling(50).mean()
    obv_std = obv.rolling(50).std().replace(0, np.nan)
    df["obv_zscore_50"] = (obv - obv_mean) / obv_std

    # --- MACD (12, 26, 9), normalized by price (stationary) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_norm"] = macd_line / c
    df["macd_signal_norm"] = macd_signal / c

    # --- Bollinger Bands (20-day, 2 std) -> %B and width (stationary) ---
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    band = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_pct_b"] = (c - bb_lower) / band
    df["bb_width"] = band / sma20.replace(0, np.nan)

    # --- ATR-14 as a percentage of price (stationary) ---
    if has_hl:
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
    else:
        # Fallback with no High/Low: mean absolute daily close change.
        atr_14 = c.diff().abs().rolling(14).mean()
    df["atr_pct"] = atr_14 / c

    # --- Lagged Returns (already stationary) ---
    df["return_1d"] = c.pct_change(1)
    df["return_5d"] = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)

    # --- Realized Volatility (10-day rolling std of daily returns) ---
    df["volatility_10d"] = c.pct_change(1).rolling(10).std()

    # --- Volume ratio (today's volume / 20-day average volume) ---
    vol_avg = v.rolling(20).mean()
    df["volume_ratio_20d"] = v / vol_avg.replace(0, np.nan)

    return df

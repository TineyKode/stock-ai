import os
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime
from utils.features import (
    compute_features,
    TECHNICAL_FEATURES,
    HYBRID_FEATURES,
    SIGNAL_THRESHOLD,
)
from utils.alerts import send_email_alert, send_discord_alert
from utils.tickers import (
    TICKER_REGISTRY,
    get_ticker_name,
    get_ticker_metadata,
    get_countries,
    get_sectors,
    get_tickers_by_country,
    get_tickers_by_sector,
)

# --- Page config (must be first st command) ---
st.set_page_config(page_title="AI Stock Signal Dashboard", layout="wide")

FEATURES_PATH = "data/processed/features.csv"
# Committed Kenya slice: US data is fetched live from yfinance, but NSE tickers
# have no live source, so this small file ships in the repo so the deployed app
# has Kenya data even though the full features.csv is gitignored.
KENYA_FEATURES_PATH = "data/processed/kenya_features.csv"


def _feature_source(country):
    """Pick the processed feature file to read for a country."""
    if country == "Kenya" and os.path.exists(KENYA_FEATURES_PATH):
        return KENYA_FEATURES_PATH
    return FEATURES_PATH


def _existing_feature_files():
    """Processed feature files present on disk (deduped, Kenya slice first)."""
    seen, paths = set(), []
    for p in (FEATURES_PATH, KENYA_FEATURES_PATH):
        if os.path.exists(p) and p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


# --- Load models safely ---
@st.cache_resource
def load_models():
    """Load trained models with error handling."""
    models = {}
    try:
        models["technical"] = joblib.load("models/technical_model.pkl")
    except FileNotFoundError:
        pass
    try:
        models["hybrid"] = joblib.load("models/hybrid_model.pkl")
    except FileNotFoundError:
        pass
    return models


@st.cache_data(ttl=3600)
def discover_available_tickers():
    """
    Auto-discover tickers from features.csv.
    Returns (set of ticker symbols, dict of ticker->market).
    Falls back to TICKER_REGISTRY if features.csv doesn't exist.
    """
    feature_files = _existing_feature_files()
    if not feature_files:
        all_tickers = set()
        ticker_markets = {}
        for country, sectors in TICKER_REGISTRY.items():
            for sector_entries in sectors.values():
                for entry in sector_entries:
                    all_tickers.add(entry["ticker"])
                    ticker_markets[entry["ticker"]] = country
        return all_tickers, ticker_markets

    # Union the full (local) features.csv and the committed Kenya slice so
    # both markets appear whether or not features.csv is present.
    df = pd.concat(
        [pd.read_csv(p, usecols=["Ticker", "Market"]) for p in feature_files],
        ignore_index=True,
    )
    available = set(df["Ticker"].unique())
    # Map Market column values to country names
    market_to_country = {"US": "US", "Kenya": "Kenya"}
    ticker_markets = {}
    for _, row in df.drop_duplicates("Ticker").iterrows():
        ticker_markets[row["Ticker"]] = market_to_country.get(row["Market"], row["Market"])
    return available, ticker_markets


def build_sidebar_filters(available_tickers, ticker_markets):
    """
    Sidebar filters: Country -> Sector -> Ticker.
    Only shows tickers that are actually in features.csv.
    Returns (selected_ticker, selected_country) or (None, None).
    """
    st.sidebar.header("Filters")

    # Country filter — from available data
    available_countries = sorted(set(ticker_markets.values()))
    if not available_countries:
        available_countries = get_countries()
    selected_country = st.sidebar.selectbox("Country", available_countries)

    # Sector filter — from registry metadata
    sectors = get_sectors(selected_country)
    sector_options = ["All Sectors"] + sectors
    selected_sector = st.sidebar.selectbox("Sector", sector_options)

    # Build filtered ticker list
    if selected_sector == "All Sectors":
        registry_tickers = get_tickers_by_country(selected_country)
    else:
        registry_tickers = get_tickers_by_sector(selected_country, selected_sector)

    # Intersect with what's actually processed in features.csv
    filtered_tickers = [t for t in registry_tickers if t in available_tickers]

    # Also include tickers from features.csv that are NOT in the registry
    # (future-proofing for dynamically added tickers)
    if selected_sector == "All Sectors":
        extra = [
            t for t in available_tickers
            if ticker_markets.get(t) == selected_country
            and t not in filtered_tickers
        ]
        if extra:
            filtered_tickers.extend(sorted(extra))

    if not filtered_tickers:
        st.sidebar.warning(f"No processed data for {selected_country} / {selected_sector}")
        st.sidebar.info("Run preprocessing: `python scripts/preprocessing/preprocess_data.py`")
        return None, selected_country

    # Ticker selectbox with human-readable labels: "AAPL - Apple"
    ticker_labels = {}
    for t in filtered_tickers:
        name = get_ticker_name(t)
        ticker_labels[t] = f"{t} - {name}" if name != t else t

    selected_label = st.sidebar.selectbox("Ticker", list(ticker_labels.values()))

    # Reverse lookup: label -> ticker symbol
    selected_ticker = next(t for t, label in ticker_labels.items() if label == selected_label)

    # Info
    st.sidebar.caption(
        f"{len(filtered_tickers)} tickers in {selected_country}"
        + (f" / {selected_sector}" if selected_sector != "All Sectors" else "")
    )

    return selected_ticker, selected_country


def fetch_latest_from_features(ticker, path=FEATURES_PATH):
    """Load the most recent valid row for a ticker from a processed file."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["Ticker"] == ticker].copy()
    # Drop rows with NaN Date or NaN features
    df = df.dropna(subset=["Date", "Close"])
    df = df.dropna(subset=TECHNICAL_FEATURES)
    if df.empty:
        return None
    df = df.sort_values("Date")
    return df.iloc[-1:].reset_index(drop=True)


def fetch_latest_data(ticker, country):
    """Fetch latest stock data and compute all features."""
    if country == "Kenya":
        # Kenya: use the committed processed slice (yfinance doesn't cover NSE)
        data = fetch_latest_from_features(ticker, _feature_source("Kenya"))
        if data is not None:
            return data
        st.warning(f"No processed data for {ticker}. Run preprocessing first.")
        return None
    else:
        # US ticker — fetch live data from yfinance
        data = yf.download(ticker, period="6mo")
        if data.empty:
            # Fallback: try features.csv
            data = fetch_latest_from_features(ticker)
            if data is not None:
                return data
            st.error(f"No data returned for {ticker}")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        if "Date" not in data.columns and "index" in data.columns:
            data = data.rename(columns={"index": "Date"})

        if len(data) < 60:
            # Not enough rows for SMA-50 + warmup — fall back to features.csv
            fb = fetch_latest_from_features(ticker)
            if fb is not None:
                return fb
            st.warning(f"Only {len(data)} days of data — need at least 60 for indicators.")
            return None

        data = compute_features(data)
        # Take the last row that has all features computed (skip NaN warmup rows)
        valid = data.dropna(subset=TECHNICAL_FEATURES)
        if valid.empty:
            fb = fetch_latest_from_features(ticker)
            if fb is not None:
                return fb
            return None
        return valid.iloc[-1:].reset_index(drop=True)


# --- Discover tickers and build filters ---
models = load_models()
available_tickers, ticker_markets = discover_available_tickers()
selected_ticker, selected_country = build_sidebar_filters(available_tickers, ticker_markets)

# --- Main content ---
st.title("AI Stock Signal Dashboard")
st.markdown("""
**US & Kenyan Market Predictions**
*Data updates EOD | Models retrain weekly*
""")

if selected_ticker is None:
    st.info("Select a ticker from the sidebar to see predictions.")
    if not models:
        st.warning("No models found. Train first:\n"
                    "`python scripts/training/train_technical.py`")
    st.stop()

# Model selector
model_type = st.radio("Model", ["Technical Only", "Hybrid (Technical + Sentiment)"])

# Fetch real data
data = fetch_latest_data(selected_ticker, selected_country)

# Predict
if data is not None:
    if model_type == "Technical Only" and "technical" in models:
        missing = [f for f in TECHNICAL_FEATURES if f not in data.columns]
        if missing:
            st.warning(f"Missing features: {', '.join(missing)}. Re-run preprocessing.")
        else:
            features = data[TECHNICAL_FEATURES].dropna()
            if not features.empty:
                proba = float(models["technical"].predict_proba(features)[:, 1][0])
                signal = "BUY" if proba >= SIGNAL_THRESHOLD else "HOLD/SELL"
                st.metric("Signal", signal, delta=f"P(up) {proba:.0%}")
            else:
                st.warning(
                    f"Feature values contain NaN for {selected_ticker}. "
                    "Try re-running: `python scripts/preprocessing/preprocess_data.py`"
                )
    elif model_type == "Hybrid (Technical + Sentiment)" and "hybrid" in models:
        if "sentiment_score" not in data.columns:
            data["sentiment_score"] = 0.5
        missing = [f for f in HYBRID_FEATURES if f not in data.columns]
        if missing:
            st.warning(f"Missing features: {', '.join(missing)}. Re-run preprocessing.")
        else:
            features = data[HYBRID_FEATURES].dropna()
            if not features.empty:
                proba = float(models["hybrid"].predict_proba(features)[:, 1][0])
                signal = "BUY" if proba >= SIGNAL_THRESHOLD else "HOLD/SELL"
                st.metric("Signal", signal, delta=f"P(up) {proba:.0%}")
            else:
                st.warning(
                    f"Feature values contain NaN for {selected_ticker}. "
                    "Try re-running: `python scripts/preprocessing/preprocess_data.py`"
                )
    else:
        st.info("Model not available. Please train the model first.")

    # --- Price Chart ---
    st.subheader(f"{selected_ticker} Price Chart")

    # Period selector
    period_options = {
        "1 Week": 5,
        "1 Month": 21,
        "3 Months": 63,
        "6 Months": 126,
        "1 Year": 252,
        "All Data": None,
    }
    # Map to yfinance period strings for US live data
    yf_period_map = {
        "1 Week": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "All Data": "5y",
    }
    selected_period = st.selectbox("Chart Period", list(period_options.keys()), index=2)
    n_days = period_options[selected_period]

    if selected_country != "Kenya":
        # US: fetch live data from yfinance for the selected period
        yf_period = yf_period_map[selected_period]
        chart_data = yf.download(selected_ticker, period=yf_period)
        if not chart_data.empty:
            if isinstance(chart_data.columns, pd.MultiIndex):
                chart_data.columns = chart_data.columns.get_level_values(0)
            chart_data = chart_data.reset_index()
            chart_data["Date"] = pd.to_datetime(chart_data["Date"])
            chart_data = chart_data.set_index("Date")
            st.line_chart(chart_data["Close"])
        else:
            st.info(f"No chart data available for {selected_ticker}")
    else:
        # Kenya: show from the committed Kenya slice (or features.csv locally)
        kenya_path = _feature_source("Kenya")
        if os.path.exists(kenya_path):
            hist = pd.read_csv(kenya_path)
            hist = hist[hist["Ticker"] == selected_ticker].copy()
            hist["Date"] = pd.to_datetime(hist["Date"])
            hist = hist.sort_values("Date")
            if n_days is not None:
                hist = hist.tail(n_days)
            if not hist.empty:
                hist = hist.set_index("Date")
                st.line_chart(hist["Close"])
            else:
                st.info(f"No historical chart data for {selected_ticker}")
        else:
            st.info("Run preprocessing first to see Kenya charts.")

# --- Upload NSE Data ---
st.subheader("Upload NSE Kenya Data")
uploaded_file = st.file_uploader("Upload NSE Kenya Data (CSV)")
if uploaded_file:
    nse_data = pd.read_csv(uploaded_file)
    st.write("Latest NSE Data:", nse_data.tail())

# --- Feedback ---
st.subheader("Feedback")
feedback = st.text_area("How can we improve?")
if st.button("Submit Feedback"):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/log.txt", "a") as f:
        f.write(f"{datetime.now()}: {feedback}\n")
    st.success("Thanks for your input!")

# --- Alert section ---
st.subheader("Set Up Alerts")
alert_type = st.radio("Alert Channel", ["Email", "Discord"], key="alert_channel")

if alert_type == "Email":
    if st.button("Enable Email Alerts", key="email_alert_btn"):
        send_email_alert("BUY", selected_ticker)
elif alert_type == "Discord":
    webhook_url = st.text_input("Enter Discord Webhook URL")
    if st.button("Test Discord Alert", key="discord_alert_btn"):
        send_discord_alert("BUY", selected_ticker, webhook_url)

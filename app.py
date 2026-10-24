import os
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime
from utils.features import (
    compute_features,
    TECHNICAL_FEATURES,
    SIGNAL_THRESHOLD,
)
from utils.alerts import send_email_alert, send_discord_alert
from utils.price_chart import (
    load_saved_prices,
    normalize_yahoo,
    price_chart,
    price_span,
    price_window,
)
from utils.signal_view import (
    beat_line,
    format_bar_date,
    load_technical_summary,
    position_label,
    universe_metrics,
)
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
st.set_page_config(
    page_title="Stock signal",
    page_icon=":material/query_stats:",
    layout="wide",
)

FEATURES_PATH = "data/processed/features.csv"
# Committed Kenya slice: US data is fetched live from yfinance, but NSE tickers
# have no live source, so this small file ships in the repo so the deployed app
# has Kenya data even though the full features.csv is gitignored.
KENYA_FEATURES_PATH = "data/processed/kenya_features.csv"
BACKTEST_SUMMARY_PATH = "logs/backtest_universe.json"


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
st.title("Stock signal")
st.caption(
    "US prices come from Yahoo. Kenya prices are the saved NSE file, not a live feed. "
    "Sentiment is off, so this is the technical model only."
)

if selected_ticker is None:
    st.info("Select a ticker from the sidebar to see predictions.")
    if not models:
        st.warning("No models found. Train first:\n"
                    "`python scripts/training/train_technical.py`")
    st.stop()

PERIOD_OPTIONS = {
    "1 Week": 5,
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "All Data": None,
}
YF_PERIODS = {
    "1 Week": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "All Data": "5y",
}
TAB_SIGNAL = ":material/query_stats: Signal"
TAB_PRICE = ":material/show_chart: Price"
TAB_SETTINGS = ":material/settings: Settings"


@st.cache_data(ttl="15m")
def _download_us_prices(ticker, period):
    return normalize_yahoo(yf.download(ticker, period=period))


@st.cache_data(ttl="1h")
def _saved_price_slice(path, ticker):
    return load_saved_prices(path, ticker)


def _signal_from_row(row):
    """Technical signal for the latest feature row. None when the model cannot score it."""
    if row is None or "technical" not in models:
        return None, None, None
    missing = [name for name in TECHNICAL_FEATURES if name not in row.columns]
    if missing:
        return None, None, missing
    features = row[TECHNICAL_FEATURES].dropna()
    if features.empty:
        return None, None, []
    proba = float(models["technical"].predict_proba(features)[:, 1][0])
    bar_date = format_bar_date(row["Date"].iloc[0]) if "Date" in row.columns else None
    return position_label(proba, SIGNAL_THRESHOLD), proba, bar_date


def _show_signal(signal, proba, bar_date, missing):
    if "technical" not in models:
        st.info("Model not available. Please train the model first.")
        return
    if missing:
        st.warning(f"Missing features: {', '.join(missing)}. Re-run preprocessing.")
        return
    if signal is None:
        st.warning(
            f"Feature values contain NaN for {selected_ticker}. "
            "Try re-running: `python scripts/preprocessing/preprocess_data.py`"
        )
        return
    tone = "green" if signal == "Long" else "gray"
    icon = ":material/trending_up:" if signal == "Long" else ":material/account_balance_wallet:"
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(signal, color=tone, icon=icon)
        st.metric("P(up)", f"{proba:.0%}", border=True, width="content")
    note = "Long means the model wants to be in. Cash means it does not."
    if bar_date:
        note = f"Bar date {bar_date}. {note}"
    st.caption(note)
    summary = load_technical_summary(BACKTEST_SUMMARY_PATH)
    metrics = universe_metrics(summary)
    if metrics:
        with st.container(horizontal=True):
            for item in metrics:
                st.metric(
                    item["label"],
                    item["value"],
                    delta=item["detail"],
                    delta_color="off",
                    delta_arrow="off",
                    border=True,
                )
    won = beat_line(summary)
    if won:
        st.caption(won)


def _price_frame(period_label):
    """Rows for the chart, the file end date, and a caption when the series is not live."""
    n_days = PERIOD_OPTIONS[period_label]
    if selected_country != "Kenya":
        live = _download_us_prices(selected_ticker, YF_PERIODS[period_label])
        if not live.empty:
            file_end, window = price_window(live, None)
            return window, None
        saved = _saved_price_slice(FEATURES_PATH, selected_ticker)
        file_end, window = price_window(saved, n_days)
        if window.empty:
            return window, None
        caption = None
        if file_end:
            caption = f"Saved file ends {file_end}. Yahoo did not return a live price."
        return window, caption
    saved = _saved_price_slice(_feature_source("Kenya"), selected_ticker)
    file_end, window = price_window(saved, n_days)
    caption = None
    if file_end:
        caption = f"Kenya file ends {file_end}. This chart is not a live price."
    return window, caption


@st.fragment
def _price_page():
    st.subheader(f"{selected_ticker} price")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        period_label = st.selectbox(
            "Chart period",
            list(PERIOD_OPTIONS),
            index=2,
            key="chart_period",
        )
        style = st.segmented_control(
            "Chart style",
            ["Line", "Area"],
            default="Line",
            key="chart_style",
        )
        scale = st.segmented_control(
            "Price scale",
            ["Fit prices", "Include zero"],
            default="Fit prices",
            key="price_scale",
        )
    window, caption = _price_frame(period_label)
    span = price_span(window)
    if span:
        change = None if span["change"] is None else f"{span['change']:+.2%}"
        with st.container(horizontal=True):
            st.metric("Window start", f"{span['start']:,.2f}", border=True)
            st.metric("Latest close", f"{span['end']:,.2f}", delta=change, border=True)
    spec = price_chart(
        window,
        style=style or "Line",
        include_zero=(scale == "Include zero"),
    )
    if spec is None:
        st.info(f"No chart data available for {selected_ticker}")
        return
    with st.container(border=True):
        st.altair_chart(spec)
        st.caption("Scroll to zoom. Drag to pan. Hover a date to read the close.")
        if caption:
            st.caption(caption)


def _settings_page():
    st.subheader("Model")
    st.caption(
        f"Technical model only. A long signal needs P(up) at or above {SIGNAL_THRESHOLD:.0%}. "
        "Sentiment is off. This screen does not place an order."
    )
    st.subheader("Alerts")
    st.caption("An alert sends the signal on the Signal tab, or nothing when that signal is missing.")
    alert_type = st.segmented_control(
        "Alert channel",
        ["Email", "Discord"],
        default="Email",
        key="alert_channel",
    )
    if alert_type == "Email":
        if st.button("Enable email alerts", key="email_alert_btn"):
            if signal:
                send_email_alert(signal, selected_ticker)
                st.caption(f"Email asked to send {signal} for {selected_ticker}.")
            else:
                st.warning("No signal on screen, so no email was sent.")
    elif alert_type == "Discord":
        webhook_url = st.text_input("Discord webhook URL")
        if st.button("Test Discord alert", key="discord_alert_btn"):
            if signal:
                send_discord_alert(signal, selected_ticker, webhook_url)
                st.caption(f"Discord asked to send {signal} for {selected_ticker}.")
            else:
                st.warning("No signal on screen, so no Discord message was sent.")
    st.subheader("NSE file")
    uploaded_file = st.file_uploader("NSE Kenya CSV")
    if uploaded_file:
        nse_data = pd.read_csv(uploaded_file)
        st.write("Latest NSE data:", nse_data.tail())
    st.subheader("Feedback")
    feedback = st.text_area("How can we improve?")
    if st.button("Submit feedback"):
        os.makedirs("feedback", exist_ok=True)
        with open("feedback/log.txt", "a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now()}: {feedback}\n")
        st.toast("Thanks for your input.")


data = fetch_latest_data(selected_ticker, selected_country)
signal, proba, bar_or_missing = _signal_from_row(data)
missing = bar_or_missing if isinstance(bar_or_missing, list) else None
bar_date = None if missing is not None else bar_or_missing

signal_tab, price_tab, settings_tab = st.tabs(
    [TAB_SIGNAL, TAB_PRICE, TAB_SETTINGS],
    default=TAB_SIGNAL,
    on_change="rerun",
    key="dashboard_tab",
)

if signal_tab.open is not False:
    with signal_tab:
        if data is None:
            st.warning(f"No processed row for {selected_ticker}.")
        else:
            _show_signal(signal, proba, bar_date, missing)
if price_tab.open is not False:
    with price_tab:
        _price_page()
if settings_tab.open is not False:
    with settings_tab:
        _settings_page()

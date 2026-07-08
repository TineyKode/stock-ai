# Stock AI — Full Project Documentation

**Repository:** https://github.com/TineyKode/stock-ai
**Branch:** main
**Python:** 3.11+ (3.13 supported with caveats)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Training](#5-model-training)
6. [Backtesting](#6-backtesting)
7. [Sentiment Analysis](#7-sentiment-analysis)
8. [Streamlit Dashboard](#8-streamlit-dashboard)
9. [Ticker Registry](#9-ticker-registry)
10. [Alerts System](#10-alerts-system)
11. [CI/CD Workflows](#11-cicd-workflows)
12. [Test Suite](#12-test-suite)
13. [Configuration & Environment](#13-configuration--environment)
14. [Known Issues & Limitations](#14-known-issues--limitations)
15. [Model Performance](#15-model-performance)
16. [Dependencies](#16-dependencies)
17. [Getting Started](#17-getting-started)
18. [Adding New Tickers](#18-adding-new-tickers)
19. [Project Activity Summary](#19-project-activity-summary)

---

## 1. Project Overview

Stock AI is an AI-powered stock price prediction system that combines historical market data with news sentiment analysis to generate buy/sell signals for US and Kenyan (NSE) stocks.

**Core capabilities:**

- **Data Pipeline** — US stocks via yfinance (5-year history), Kenyan stocks via CSV/Google Sheets, news via NewsAPI + BERTweet sentiment
- **13 Stationary Technical Features** — RSI-14, close/SMA-50 ratio, OBV z-score, price-normalized MACD, Bollinger %B + width, ATR%, lagged returns, volatility, volume ratio (all price-level independent so one model generalizes across tickers)
- **ML Models** — Technical-only and Hybrid (technical + sentiment) RandomForest models trained on **all tickers pooled** with walk-forward cross-validation
- **Backtesting** — Walk-forward with transaction costs (10 bps), Sharpe ratio, max drawdown, win rate
- **Streamlit Dashboard** — Dynamic sidebar with country/sector/ticker filtering, live charts, model selection, CSV upload, alerts
- **Ticker Registry** — 30 US stocks + 10 Kenyan stocks, centralized in `utils/tickers.py`
- **Alerts** — Email (Gmail SMTP) and Discord webhook notifications
- **CI/CD** — GitHub Actions for code quality, daily data fetch, weekly retraining, Streamlit Cloud deployment

**Markets covered:**

| Market | Tickers | Data Source | History |
|--------|---------|-------------|---------|
| US | 30 stocks | Yahoo Finance (yfinance) | 5 years |
| Kenya (NSE) | 10 stocks | Historical CSVs (2020-2024) | 4 years |

---

## 2. Architecture

```
stock-ai/
├── app.py                              # Streamlit dashboard (dynamic sidebar)
├── requirements.txt                    # ~20 dependencies
├── .env.example                        # API key template
├── README.md                           # Project documentation
│
├── data/
│   ├── kenya/                          # NSE market data (2020-2024)
│   │   ├── nse_2020.csv               # Original NSE data (dd-Mon-yy format)
│   │   ├── nse_latest.csv             # Latest NSE snapshot (different format)
│   │   └── Stock AI Development       # Google Sheets exports per year
│   │       [Market Data] - {2020,2021,2023,2024,MASTER 2}.csv
│   ├── us/                             # US stock CSVs (5 years, ~1,256 rows each)
│   │   ├── aapl.csv, msft.csv, ...    # 30 ticker files
│   └── processed/
│       ├── features.csv                # All tickers with 13 features (~47,630 rows)
│       ├── raw_headlines.csv           # Headlines from NewsAPI
│       ├── scored_headlines.csv        # Headlines with BERTweet scores
│       └── news_sentiment.csv          # Daily aggregated sentiment
│
├── models/
│   ├── technical_model.pkl             # RandomForest (13 features)
│   └── hybrid_model.pkl                # RandomForest (14 features)
│
├── logs/
│   ├── technical_metrics.json          # Walk-forward CV results
│   ├── hybrid_metrics.json
│   ├── model_comparison.json           # RF vs GB vs XGBoost
│   ├── backtest_technical.csv          # Daily backtest results
│   └── backtest_hybrid.csv
│
├── scripts/
│   ├── fetching/
│   │   ├── fetch_us_stocks.py          # yfinance download (5 years, append-only)
│   │   ├── fetch_newsapi.py            # NewsAPI headlines (recent, ~30 days)
│   │   ├── fetch_news.py               # BERTweet sentiment scoring
│   │   ├── fetch_alpha_sentiment.py    # Alpha Vantage historical sentiment (keyed)
│   │   └── scrape_nse.py               # NSE web scraper (standalone, not wired in)
│   ├── preprocessing/
│   │   └── preprocess_data.py          # Main preprocessing (US + committed Kenya slice)
│   ├── training/
│   │   ├── train_technical.py          # Technical model (walk-forward CV)
│   │   ├── train_hybrid.py             # Hybrid model (walk-forward CV)
│   │   ├── retrain_models.py           # Automated retraining
│   │   └── compare_models.py           # RF vs GradientBoosting vs XGBoost
│   └── backtesting/
│       └── backtest.py                 # Walk-forward backtest with costs
│
├── utils/
│   ├── tickers.py                      # Centralized ticker registry
│   ├── features.py                     # Feature definitions + compute_features()
│   ├── evaluation.py                   # Walk-forward CV + metrics
│   └── alerts.py                       # Email + Discord alerts
│
├── tests/
│   ├── test_sentiment.py               # BERTweet model test
│   ├── test_yfinance.py                # yfinance download test
│   ├── test_nse.py                     # NSE scraping test
│   ├── test_nse_scraper.py             # Selenium NSE test
│   └── test_gsheets.py                 # Google Sheets auth test
│
└── .github/workflows/
    ├── ci.yml                          # Code quality on push/PR
    ├── deploy.yml                      # Streamlit Cloud deployment
    ├── fetch_us.yml                    # Daily US data fetch (6 PM UTC)
    └── retrain.yml                     # Weekly retrain (Sundays 6 PM UTC)
```

---

## 3. Data Pipeline

The data pipeline flows through four stages:

```
Fetch → Preprocess → Train → Predict
```

### 3.1 Fetching

**US Stocks (`scripts/fetching/fetch_us_stocks.py`):**
- Downloads 5 years of OHLCV data via `yfinance.download()`
- Ticker list imported from `utils/tickers.py` (30 US tickers)
- Saves to `data/us/{ticker.lower()}.csv`
- yfinance outputs multi-index CSVs with 3 header rows

**Kenya Stocks:**
- Historical data loaded from CSV files in `data/kenya/`
- Supports multiple formats: `nse_2020.csv` (dd-Mon-yy dates), `Stock AI Development` files (YYYY-MM-DD dates)
- Column names vary (`DATE`/`Date`, `CODE`/`Code`) — preprocessing normalizes them
- `nse_latest.csv` has incompatible format (Ticker column instead of CODE) and is skipped

**News Headlines (`scripts/fetching/fetch_newsapi.py`):**
- Fetches from NewsAPI.org (free tier: 100 requests/day, 30 days historical)
- Maps tickers to search queries (e.g., AAPL → "Apple stock OR AAPL")
- Saves to `data/processed/raw_headlines.csv` with columns: Date, Headline, Source, Ticker
- Requires `NEWSAPI_KEY` environment variable

### 3.2 Preprocessing (`scripts/preprocessing/preprocess_data.py`)

The main preprocessing script processes all data in one pass:

1. **US data** — Discovers CSVs via `glob("data/us/*.csv")`, extracts ticker from filename
2. **Kenya data** — Discovers CSVs via `glob("data/kenya/*.csv")`, filters by CODE column per ticker
3. **Feature computation** — Calls `compute_features(df)` from `utils/features.py` per ticker
4. **Deduplication** — Removes duplicate (Date, Ticker) rows from overlapping Kenya files
5. **Output** — Saves `data/processed/features.csv` with columns: Date, Ticker, Market, Close, + 13 features

**Column normalization for Kenya files:**
- All column names uppercased for matching (`DATE`, `CODE`, `DAY PRICE`, etc.)
- Date parsing tries ISO format first (`YYYY-MM-DD`), falls back to `dd-Mon-yy`
- Volume strings cleaned: commas removed, dashes converted to NaN
- Tickers with fewer than 60 rows are skipped (insufficient for indicator warmup)

**Current output:** ~47,630 rows across 40 tickers (30 US + 10 Kenya)

---

## 4. Feature Engineering

All features are defined in `utils/features.py`. This is the single source of truth — all training scripts, backtesting, and the dashboard import from here.

### 4.1 Technical Features (13)

**All features are stationary** — none depends on the absolute price level. This is deliberate: models are validated walk-forward (train on the past, test on the future) and one model serves every ticker, so raw price-level features (a 50-day SMA, Bollinger band *prices*, cumulative OBV, raw MACD, raw ATR — all of which scale with price) would feed the model values it never saw in training and would be meaningless across tickers. Each level is therefore expressed as a ratio, percentage, band position, or rolling z-score.

| # | Feature | Category | Computation |
|---|---------|----------|-------------|
| 1 | `rsi_14` | Momentum | 14-period RSI (bounded 0-100; 100 on a zero-loss window) |
| 2 | `price_sma50_ratio` | Trend | `Close / SMA(50) - 1` (distance from trend) |
| 3 | `obv_zscore_50` | Volume | 50-day rolling z-score of On-Balance Volume |
| 4 | `macd_norm` | Trend | `(EMA12 - EMA26) / Close` |
| 5 | `macd_signal_norm` | Trend | `EMA(9) of MACD line / Close` |
| 6 | `bb_pct_b` | Volatility | Bollinger %B: `(Close - lower) / (upper - lower)` |
| 7 | `bb_width` | Volatility | `(upper - lower) / SMA(20)` |
| 8 | `atr_pct` | Volatility | `ATR(14) / Close` |
| 9 | `return_1d` | Momentum | 1-day percentage change |
| 10 | `return_5d` | Momentum | 5-day percentage change |
| 11 | `return_10d` | Momentum | 10-day percentage change |
| 12 | `volatility_10d` | Risk | 10-day rolling std of daily returns |
| 13 | `volume_ratio_20d` | Volume | Today's volume / 20-day average volume |

### 4.2 Hybrid Features (14)

All 13 technical features + `sentiment_score` (daily average BERTweet sentiment, 0.0-1.0 scale).

### 4.3 `compute_features(df)` Function

- **Input:** DataFrame with Date, Close, Volume (required); High, Low (optional for ATR)
- **Output:** Same DataFrame with all 13 feature columns added
- **Warmup:** First ~50 rows will have NaN (SMA-50 / OBV z-score window); handled by dropna in training
- **ATR fallback:** If High/Low columns are missing, ATR uses the 14-day mean of absolute daily close changes instead of the true range

### 4.4 Constants

```python
TECHNICAL_FEATURES = [
    "rsi_14", "price_sma50_ratio", "obv_zscore_50", "macd_norm",
    "macd_signal_norm", "bb_pct_b", "bb_width", "atr_pct",
    "return_1d", "return_5d", "return_10d", "volatility_10d",
    "volume_ratio_20d",
]
HYBRID_FEATURES = TECHNICAL_FEATURES + ["sentiment_score"]

# Magnitude-aware label + selective-trading thresholds
TARGET_THRESHOLD = 0.0015   # a day is "UP" only if next-day return > ~15 bps
SIGNAL_THRESHOLD = 0.55     # go long only when the model's P(up) >= this

# Shared model hyperparameters (used by training, retraining AND backtest)
MODEL_PARAMS = {
    "technical": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20, ...},
    "hybrid":    {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20, ...},
}
```

`utils/dataset.py` centralizes training-set assembly: `load_features()`, `add_target()` (builds the **magnitude-aware** next-day label **per ticker** — 1 only if the next-day return clears `TARGET_THRESHOLD` — so pooling doesn't leak across ticker boundaries and tiny sub-cost moves aren't labelled as buys), and `merge_sentiment()`.

---

## 5. Model Training

### 5.1 Walk-Forward Cross-Validation (`utils/evaluation.py`)

All training uses `TimeSeriesSplit(n_splits=5)` instead of random train/test splits. This prevents data leakage by ensuring the model never sees future data during training.

**How it works:**
```
Fold 1: Train on [0..N],      Test on [N..2N]
Fold 2: Train on [0..2N],     Test on [2N..3N]
Fold 3: Train on [0..3N],     Test on [3N..4N]
Fold 4: Train on [0..4N],     Test on [4N..5N]
Fold 5: Train on [0..5N],     Test on [5N..6N]
```

The training window expands with each fold while the test window always moves forward in time.

**Metrics per fold:** Accuracy, Precision, Recall, F1-score, ROC AUC
**Aggregate metrics:** Mean and standard deviation across all folds
**Output:** JSON saved to `logs/{model_name}_metrics.json`

### 5.2 Technical Model (`scripts/training/train_technical.py`)

- **Data:** **all 40 tickers pooled** from `features.csv` (~38k valid rows), not AAPL alone
- **Features:** 13 stationary technical features (`TECHNICAL_FEATURES`)
- **Target:** Magnitude-aware, computed **per ticker** — 1 if that ticker's next-day return clears `TARGET_THRESHOLD` (~15 bps), else 0 (so sub-cost noise moves aren't labelled as buys)
- **Model:** `RandomForestClassifier(**MODEL_PARAMS["technical"])` (200 trees, depth 8, min_samples_leaf 20)
- **Output:** `models/technical_model.pkl`, `logs/technical_metrics.json`
- Prints feature importances ranked by weight

### 5.3 Hybrid Model (`scripts/training/train_hybrid.py`)

- **Data:** all tickers pooled from `features.csv` merged with `news_sentiment.csv`
- **Merge:** Left join on (Date, Ticker), missing sentiment filled with 0.5 (neutral)
- **Features:** 14 hybrid features (`HYBRID_FEATURES`)
- **Model:** `RandomForestClassifier(**MODEL_PARAMS["hybrid"])`
- **Output:** `models/hybrid_model.pkl`, `logs/hybrid_metrics.json`

### 5.4 Model Comparison (`scripts/training/compare_models.py`)

Compares three classifiers using walk-forward CV on AAPL:

| Model | Parameters |
|-------|-----------|
| RandomForest | n_estimators=200, max_depth=10 |
| GradientBoosting | n_estimators=200, max_depth=5, learning_rate=0.1 |
| XGBoost | n_estimators=200, max_depth=5, learning_rate=0.1 |

XGBoost requires separate installation (`pip install xgboost`) and is skipped gracefully if not installed.

**Output:** `logs/model_comparison.json` with per-model metrics and best model selection by F1 score.

### 5.5 Automated Retraining (`scripts/training/retrain_models.py`)

Retrains both technical and hybrid models using the latest data. Called by the weekly GitHub Actions workflow. Uses the same walk-forward CV approach and saves updated metrics.

---

## 6. Backtesting

### 6.1 Walk-Forward Backtest (`scripts/backtesting/backtest.py`)

The backtester simulates trading using model predictions on unseen data folds.

**Methodology:**
1. `TimeSeriesSplit` creates chronological folds
2. Train model on historical fold (magnitude-aware `TARGET_THRESHOLD` label)
3. Predict `P(up)` on next fold (unseen data)
4. **Selective trading:** go long only when `P(up) >= SIGNAL_THRESHOLD` (0.55), else hold cash — this trades fewer, higher-confidence days instead of on every `predict==1`
5. Apply transaction costs when position changes
6. Calculate daily portfolio returns across all folds

### 6.2 Transaction Costs

- **Cost:** 10 basis points (0.10%) per trade
- **Applied:** On every position change (entry, exit, or flip)
- **First trade:** Entry cost applied on day 1 if signal is 1

### 6.3 Risk Metrics

| Metric | Formula |
|--------|---------|
| **Sharpe Ratio** | √252 × mean(excess returns) / std(excess returns) |
| **Max Drawdown** | Maximum peak-to-trough decline in cumulative returns |
| **Win Rate** | Fraction of trading days with positive returns |
| **Total Return** | Cumulative product of (1 + daily returns) - 1 |

### 6.4 Benchmark

All metrics are computed for both the strategy and a buy-and-hold benchmark on the same time period, printed side by side.

### 6.5 Output

- `logs/backtest_technical.csv` — Daily results for technical model
- `logs/backtest_hybrid.csv` — Daily results for hybrid model
- Formatted console report with comparison table

---

## 7. Sentiment Analysis

### 7.1 Pipeline

```
NewsAPI → raw_headlines.csv → BERTweet → scored_headlines.csv → aggregate → news_sentiment.csv
```

### 7.2 Headline Fetching (`scripts/fetching/fetch_newsapi.py`)

- **API:** NewsAPI.org (free tier: 100 requests/day, 30 days history)
- **Query mapping:** Each ticker maps to a search query (e.g., AAPL → "Apple stock OR AAPL")
- **Output columns:** Date, Headline, Source, Ticker
- **Saves to:** `data/processed/raw_headlines.csv`

### 7.3 Sentiment Scoring (`scripts/fetching/fetch_news.py`)

- **Model:** `finiteautomata/bertweet-base-sentiment-analysis` (Hugging Face)
- **Batch processing:** 32 headlines per batch to avoid memory issues
- **Headline truncation:** Max 280 characters (BERTweet limit ~128 tokens)
- **Score mapping:**
  - POS → 0.5 + 0.5 × confidence (range: 0.5 to 1.0)
  - NEG → 0.5 - 0.5 × confidence (range: 0.0 to 0.5)
  - NEU → 0.5 (neutral)
- **Aggregation:** Daily average `sentiment_score` per (Date, Ticker)
- **Output:**
  - `data/processed/scored_headlines.csv` (per-headline scores)
  - `data/processed/news_sentiment.csv` (daily aggregates)

### 7.4 Sentiment Merge Strategy

When merging sentiment with features for training or backtesting:
- Always use `how="left"` join (keep all feature rows)
- Fill missing sentiment with 0.5 (neutral) via `.fillna(0.5)`
- This prevents data loss when dates don't match (common with free-tier API limits)

### 7.5 Historical Sentiment (`scripts/fetching/fetch_alpha_sentiment.py`)

The NewsAPI chain above only reaches ~30 days on the free tier, so it cannot
populate sentiment across the multi-year training window — which is why the
hybrid model has been nearly identical to the technical one.
`fetch_alpha_sentiment.py` fills that gap using Alpha Vantage's
`NEWS_SENTIMENT` endpoint (free: 25 req/day, 5/min; history back to 2022):

- Uses the **per-ticker** sentiment score of each article, averaged per day
- Maps Alpha Vantage's `[-1, 1]` score to the pipeline's `[0, 1]` scale
- Writes the same `data/processed/news_sentiment.csv` the hybrid model reads
- **Requires `ALPHA_VANTAGE_KEY`.** Without a key it prints a notice and exits
  0 (hybrid then falls back to neutral 0.5), so it is CI-safe to call blind.

```bash
# populate real historical sentiment (needs ALPHA_VANTAGE_KEY in .env)
python scripts/fetching/fetch_alpha_sentiment.py --time-from 20220101
```

Obtaining and setting the key is the one remaining step to make the hybrid
model train on real sentiment rather than the neutral placeholder.

---

## 8. Streamlit Dashboard

### 8.1 Dynamic Sidebar Filtering

The dashboard auto-discovers available tickers from `data/processed/features.csv` rather than using hardcoded lists.

**Filter hierarchy:**
1. **Country** — US or Kenya (auto-detected from the `Market` column in features.csv)
2. **Sector** — Technology, Banking/Finance, Healthcare, etc. (from `utils/tickers.py` metadata)
3. **Ticker** — Only shows tickers with actual processed data, displayed as "AAPL - Apple"

**Future-proofing:** Tickers found in `features.csv` but not in the registry still appear under "All Sectors" — so adding a new ticker to the pipeline without updating `tickers.py` won't break the UI.

**Fallback:** If `features.csv` doesn't exist, falls back to showing all tickers from the registry.

### 8.2 Predictions

- Loads `technical_model.pkl` and `hybrid_model.pkl` from the `models/` directory
- User selects "Technical Only" or "Hybrid (Technical + Sentiment)"
- Fetches latest data for selected ticker via yfinance (US) or mock data (Kenya)
- Computes all features using `compute_features()`
- Runs prediction and displays BUY or HOLD/SELL signal

### 8.3 Charts

Charts include a **period selector dropdown** with six options:

| Period | US (yfinance) | Kenya (features.csv) |
|--------|---------------|----------------------|
| 1 Week | `period="5d"` | Last 5 rows |
| 1 Month | `period="1mo"` | Last 21 rows |
| 3 Months (default) | `period="3mo"` | Last 63 rows |
| 6 Months | `period="6mo"` | Last 126 rows |
| 1 Year | `period="1y"` | Last 252 rows |
| All Data | `period="5y"` | All available rows |

- **US tickers:** Fetches live data from yfinance for the selected period
- **Kenya tickers:** Shows historical data from `features.csv`, sliced by `tail(n_days)`

### 8.4 Additional Features

- **CSV Upload** — Upload custom NSE Kenya data for viewing
- **Feedback** — Text area saved to `feedback/log.txt`
- **Alerts** — Email (Gmail SMTP) and Discord webhook configuration

### 8.5 Caching

| Decorator | Function | TTL |
|-----------|----------|-----|
| `@st.cache_resource` | `load_models()` | Persistent |
| `@st.cache_data(ttl=3600)` | `discover_available_tickers()` | 1 hour |

### 8.6 Page Configuration

```python
st.set_page_config(page_title="AI Stock Signal Dashboard", layout="wide")
```

---

## 9. Ticker Registry

### 9.1 Centralized Configuration (`utils/tickers.py`)

All ticker definitions live in a single `TICKER_REGISTRY` dict structured as:
```
country → sector → [{ticker, name}, ...]
```

This is imported by `fetch_us_stocks.py`, `preprocess_data.py`, and `app.py`.

### 9.2 US Stocks (30)

| Sector | Count | Tickers |
|--------|-------|---------|
| Technology | 12 | AAPL, MSFT, GOOGL, NVDA, TSLA, INTC, META, AMZN, CRM, ORCL, ADBE, AMD |
| Banking / Finance | 7 | JPM, BAC, GS, MS, WFC, V, MA |
| Healthcare | 3 | JNJ, UNH, PFE |
| Consumer / Industrials | 5 | KO, PG, WMT, DIS, HD |
| ETFs | 3 | SOXS, SPY, QQQ |

### 9.3 Kenyan Stocks (10)

| Sector | Count | Tickers |
|--------|-------|---------|
| Telecom | 1 | SCOM (Safaricom) |
| Banking / Finance | 6 | EQTY, KCB, ABSA, COOP, SCBK, NCBA |
| Consumer | 2 | EABL (East African Breweries), BAT (BAT Kenya) |
| Industrials | 1 | BAMB (Bamburi Cement) |

### 9.4 Helper Functions

| Function | Returns |
|----------|---------|
| `get_countries()` | `["US", "Kenya"]` |
| `get_sectors(country)` | List of sector names |
| `get_tickers_by_country(country)` | Flat list of ticker symbols |
| `get_tickers_by_sector(country, sector)` | Ticker symbols for one sector |
| `get_all_tickers()` | All 40 ticker symbols |
| `get_ticker_name(symbol)` | Human-readable name (fallback: symbol itself) |
| `get_ticker_metadata(symbol)` | `{ticker, name, country, sector}` or None |

---

## 10. Alerts System

### 10.1 Email Alerts (`utils/alerts.py`)

- **Protocol:** Gmail SMTP (port 587, TLS)
- **Authentication:** Gmail app passwords (not regular passwords)
- **Environment variables:** `GMAIL_USER`, `GMAIL_RECIPIENT`, `GMAIL_APP_PASSWORD`
- **Message format:** Subject and body with ticker + signal info

### 10.2 Discord Alerts

- **Method:** Webhook POST request
- **Payload:** `{"username": "Stock Bot", "content": "{ticker} Alert: {signal}"}`
- **Configuration:** Webhook URL passed as parameter or via `DISCORD_WEBHOOK_URL` env var

### 10.3 Alert Generation

The `generate_alert()` function:
1. Formats the signal message
2. Optionally includes historical context (30-day % change)
3. Sends via both Email and Discord if configured
4. Fails silently with a printed warning if credentials are missing

---

## 11. CI/CD Workflows

### 11.1 Code Quality (`ci.yml`)

- **Triggers:** Push to main/master, pull requests
- **Python:** 3.11
- **Steps:** Install dependencies → verify Streamlit version → syntax check `app.py` → **run unit tests** (`pytest tests/`)

### 11.2 Streamlit Deployment (`deploy.yml`)

- **Triggers:** Push to main, manual dispatch
- **Logic:** If `STREAMLIT_TOKEN` secret is set, POST to Streamlit API; otherwise auto-deploy from main
- **Target:** `TineyKode/stock-ai`, main file: `app.py`

### 11.3 Daily US Data Fetch (`fetch_us.yml`)

- **Schedule:** `0 18 * * *` (6 PM UTC daily, after US market close)
- **Manual trigger:** workflow_dispatch
- **Steps:**
  1. Checkout with `GITHUB_TOKEN`
  2. Install yfinance + pandas
  3. Run `fetch_us_stocks.py`
  4. Auto-commit updated CSVs to `data/us/`
- **Error handling:** Creates a GitHub issue on failure with workflow logs link
- **Churn control:** `fetch_us_stocks.py` rounds prices to 4 dp and is **append-only** — existing dated rows are kept as first written (never re-adjusted to new dividend/split factors), so each daily commit adds ~1 line per file instead of rewriting the whole 5-year history. (A one-time rewrite rounded the existing files.)

### 11.4 Weekly Retrain (`retrain.yml`)

- **Schedule:** `0 18 * * 0` (every Sunday, 6 PM UTC)
- **Manual trigger:** workflow_dispatch
- **Steps:**
  1. Checkout with `GITHUB_TOKEN`
  2. Install all dependencies
  3. Run `preprocess_data.py` (recompute features)
  4. Run `retrain_models.py` (retrain both models)
  5. Auto-commit updated models to `models/`

---

## 12. Test Suite

**Unit tests** (deterministic, no network — run in CI):

| Test File | What It Tests |
|-----------|---------------|
| `test_features.py` | `compute_features`: all features present, **scale-invariance** (the stationarity guarantee), warmup NaN preserved, RSI bounds / no-loss=100 |
| `test_dataset.py` | `add_target`: per-ticker label with no cross-ticker leakage, `TARGET_THRESHOLD` filtering; `merge_sentiment` neutral fill |
| `test_alpha_sentiment.py` | Alpha Vantage `score_to_unit` mapping/clamping and `parse_feed_to_daily` daily aggregation (against a fixture) |

**Manual smoke scripts** (network/heavy deps, run by hand — named `check_*` so pytest doesn't collect them):

| Script | What It Checks |
|--------|----------------|
| `check_sentiment.py` | BERTweet model loads and scores sample headlines |
| `check_nse_scraper.py` | NSE scrape via requests + BeautifulSoup |
| `check_nse_selenium.py` | NSE scrape via Selenium (needs ChromeDriver) |

Run the unit tests:
```bash
python -m pytest tests/
```

---

## 13. Configuration & Environment

### 13.1 Environment Variables

Create a `.env` file from the template:
```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `NEWSAPI_KEY` | For recent sentiment | Free from newsapi.org (~30 days history) |
| `ALPHA_VANTAGE_KEY` | For historical sentiment | Free from alphavantage.co; used by `fetch_alpha_sentiment.py` to backfill multi-year sentiment |
| `GMAIL_USER` | For email alerts | Gmail address |
| `GMAIL_APP_PASSWORD` | For email alerts | Gmail app password |
| `GMAIL_RECIPIENT` | For email alerts | Alert recipient |
| `DISCORD_WEBHOOK_URL` | For Discord alerts | Discord webhook URL |
| `GSHEETS_CREDENTIALS` | For Google Sheets | Path to credentials.json |

### 13.2 Git Ignored Files

The `.gitignore` excludes:
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `.venv/`)
- IDE files (`.vscode/`, `.idea/`, `.claude/`)
- Credentials (`*.env`, `credentials.json`)
- Generated files (`data/processed/*`, `logs/`, `feedback/`)
- OS files (`.DS_Store`, `Thumbs.db`, `nul`)

**Committed exceptions:** trained `models/*.pkl` (so they deploy) and
`data/processed/kenya_features.csv` (the small NSE slice — the deployed app
needs it because Kenya has no live data source, while US data is fetched live
from yfinance and the full `features.csv` stays ignored).

---

## 14. Known Issues & Limitations

### 14.1 Technical Issues

| Issue | Workaround |
|-------|------------|
| Python 3.13 + sklearn hangs on import | Kill stuck Python processes before running scripts |
| Kenya `nse_latest.csv` has different format | Automatically skipped (no CODE column) |
| yfinance multi-index CSVs | Parsed with `header=[0,1,2]` in `preprocess_data.py` |
| Free NewsAPI only covers 30 days | Sentiment fills with 0.5 (neutral) for older dates |
| Streamlit not on PATH (Windows) | Use `python -m streamlit run app.py` |
| watchdog + Python 3.13 incompatibility | Upgrade to watchdog 6.0.0+ and streamlit 1.54.0+ |

### 14.2 Model Limitations

- Sentiment defaults to **neutral 0.5** until `ALPHA_VANTAGE_KEY` is set and `fetch_alpha_sentiment.py` is run. The historical-sentiment path now exists (§7.5); running it with a key is the biggest remaining lever on model quality.
- ~0.53 AUC is a **modest** edge, not a strong one
- The selective long/flat strategy **trails buy-and-hold on total return** in a bull market, but with **roughly half the drawdown** and comparable Sharpe (§15.2)
- Minimal hyperparameter tuning (shared `MODEL_PARAMS`, no grid search); `SIGNAL_THRESHOLD`/`TARGET_THRESHOLD` are principled defaults, not fit to the backtest
- The label is directional (return above/below a threshold), not a full return-magnitude regression

---

## 15. Model Performance

### 15.1 Walk-Forward CV Results (all tickers pooled, ~38k samples)

| Model | Accuracy | F1 | AUC | Notes |
|-------|----------|----|-----|-------|
| Technical (RF) | 53.8% | 0.111 | **0.530** | Real (modest) edge, generalizes across 40 tickers |
| Hybrid (RF) | 53.6% | 0.097 | **0.529** | Sentiment neutral (0.5) until Alpha Vantage key is set |

**AUC is the metric that matters here.** F1 is low because it is measured at the default 0.5 cut against the magnitude-aware label (a conservative model rarely crosses 0.5), whereas the strategy actually trades at `P(up) >= 0.55` — so ranking quality (AUC), not the 0.5-cut F1, drives the backtest.

Compared with the previous AAPL-only, non-stationary-feature models (AUC ≈ 0.50, i.e. random), making the features stationary and pooling all tickers moved AUC to ~0.53 — a small but genuine signal measured out-of-sample across the whole universe.

### 15.2 Backtesting Results (AAPL, 10 bps costs, selective trading)

| Metric | Technical Strategy | Hybrid Strategy | Buy & Hold |
|--------|-------------------|----------------|------------|
| Total Return | +38.3% | +53.2% | +113.6% |
| Sharpe Ratio | 0.630 | **0.822** | 0.842 |
| Max Drawdown | **-15.7%** | **-15.3%** | -33.4% |
| Win Rate | 41.0% | 39.5% | — |
| Trades (over 990 days) | 108 | 122 | — |

**Conclusion:** With the magnitude-aware label and selective `SIGNAL_THRESHOLD=0.55` trading, the **hybrid strategy now nearly matches buy-and-hold's Sharpe (0.822 vs 0.842) with less than half the drawdown (−15.3% vs −33.4%)** and trades only ~120 times over ~4 years. It trails on total return, which is expected for a long/flat strategy in a strong bull market — the trade-off is far lower risk. The backtest fits a fresh model per walk-forward fold using the **same `MODEL_PARAMS`** and thresholds that ship, so these numbers describe the deployed model. Running real sentiment (§7.5) is the main lever left.

### 15.3 Feature Importances (Technical Model)

Top features by RandomForest importance:
1. `volume_ratio_20d` (17.2%)
2. `bb_width` (9.6%)
3. `volatility_10d` (9.4%)
4. `return_1d` (9.3%)
5. `atr_pct` (7.1%)

Volume and volatility features dominate; the remaining features are fairly evenly distributed.

---

## 16. Dependencies

### 16.1 requirements.txt

| Category | Packages |
|----------|----------|
| Core | streamlit, pandas, numpy, scikit-learn, joblib, matplotlib |
| Data Fetching | yfinance, requests, beautifulsoup4 |
| Technical Analysis | ta |
| Sentiment | transformers, torch |
| ML Frameworks | tensorflow, xgboost |
| Visualization | plotly |
| Google Sheets | gspread |
| Testing | pytest |

### 16.2 Install

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow and torch are large dependencies. For a lighter install focused on core functionality:
```bash
pip install streamlit pandas numpy scikit-learn joblib yfinance ta plotly
```

---

## 17. Getting Started

### 17.1 Quick Start

```bash
# Clone
git clone https://github.com/TineyKode/stock-ai.git
cd stock-ai

# Install
pip install -r requirements.txt

# Fetch data (US stocks, 5 years)
python scripts/fetching/fetch_us_stocks.py

# Preprocess (compute features for all tickers)
python scripts/preprocessing/preprocess_data.py

# Train models
python scripts/training/train_technical.py
python scripts/training/train_hybrid.py

# Run dashboard
streamlit run app.py
```

### 17.2 Full Pipeline

```bash
# 1. Fetch data
python scripts/fetching/fetch_us_stocks.py

# 2. Preprocess
python scripts/preprocessing/preprocess_data.py

# 3. Sentiment (optional)
#    recent (~30d) via NewsAPI, requires NEWSAPI_KEY:
python scripts/fetching/fetch_newsapi.py
python scripts/fetching/fetch_news.py
#    OR multi-year history via Alpha Vantage, requires ALPHA_VANTAGE_KEY:
python scripts/fetching/fetch_alpha_sentiment.py --time-from 20220101

# 4. Train
python scripts/training/train_technical.py
python scripts/training/train_hybrid.py

# 5. Compare models
python scripts/training/compare_models.py

# 6. Backtest
python scripts/backtesting/backtest.py

# 7. Dashboard
streamlit run app.py
```

### 17.3 Windows (Python 3.13) Notes

If Streamlit is not found on PATH:
```bash
python -m streamlit run app.py
```

If sklearn hangs during import, kill stuck Python processes:
```powershell
Get-Process python* | Stop-Process -Force
```

---

## 18. Adding New Tickers

### 18.1 Add a US Stock

1. Edit `utils/tickers.py` — add entry to the appropriate sector:
```python
"Technology": [
    {"ticker": "AAPL", "name": "Apple"},
    {"ticker": "NEW", "name": "New Company"},  # Add here
],
```

2. Re-run the pipeline:
```bash
python scripts/fetching/fetch_us_stocks.py
python scripts/preprocessing/preprocess_data.py
```

3. The dashboard will automatically pick up the new ticker.

### 18.2 Add a Kenya Stock

1. Edit `utils/tickers.py` — add to the Kenya section:
```python
"Kenya": {
    "Banking / Finance": [
        {"ticker": "EQTY", "name": "Equity Group"},
        {"ticker": "NEW", "name": "New Kenya Stock"},  # Add here
    ],
},
```

2. Ensure the ticker code matches the `CODE` column in your Kenya CSV files.

3. Re-run preprocessing:
```bash
python scripts/preprocessing/preprocess_data.py
```

### 18.3 Add a New Country

1. Add a new top-level key to `TICKER_REGISTRY` in `utils/tickers.py`
2. Add data files to a new directory (e.g., `data/newcountry/`)
3. Add a loading function in `preprocess_data.py`
4. The dashboard sidebar will automatically show the new country

---

## 19. Project Activity Summary

### 19.1 Key Milestones

| Date | Activity |
|------|----------|
| Initial | Basic AAPL prediction with 3 features, random train/test split |
| Session 1 | Fixed date mismatch bugs, Streamlit compatibility, feature mismatches |
| Session 2 | Full project restructuring into subdirectories |
| Session 3 | 6-phase improvement: data expansion, 13 features, walk-forward CV, backtesting, model comparison |
| Session 4 | Added 30 US + 10 Kenya tickers, dynamic dashboard with country/sector filtering |

### 19.2 Recent Git History

| Commit | Description |
|--------|-------------|
| `0f8b0ae` | Modified UI |
| `f7c7eed` | ENV |
| `d1de739` | Preprocess Kenyan Data |
| `39430b5` | Merge remote changes, keep local 5-year stock data |
| `fc76d35` | Real sentiment, Backtesting |
| `25d8b69+` | 23+ auto-update US stock data commits (weekly recurring) |

### 19.3 Code Statistics

| Area | Lines / Size |
|------|-------------|
| `app.py` | 302 lines |
| `scripts/` | 1,551+ lines |
| `utils/` | ~450 lines |
| `tests/` | 5 files |
| `models/` | technical_model.pkl (358 KB), hybrid_model.pkl (3.8 MB) |
| `data/` | US (30 CSVs), Kenya (7 CSVs), processed (features.csv ~13 MB) |
| `requirements.txt` | ~20 dependencies |

---

## License

This project is open source.

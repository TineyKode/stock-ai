# Open Items

Updated 2026-10-17. The code items below are done. What remains needs a key, a live site, or a long backtest run.

## Done in the repo

- `origin` points at `https://github.com/TineyKode/stock-ai.git`.
- `fetch_alpha_sentiment.py` stops at 25 calls per UTC day, writes `alpha_vantage_budget.json`, and the next day continues with tickers missing from `news_sentiment.csv`. A Note or Information response stops the run and keeps Alpha Vantage's text.
- `retrain.yml` does not fail the weekly job when that fetch fails.
- `compare_models.py` labels with `add_target()` and builds the forest from `MODEL_PARAMS`.
- `backtest.py` reports the median across every ticker. The Sharpe 0.822 figure in the README is still the old AAPL run.
- `requirements.txt` no longer installs TensorFlow, `ta`, or XGBoost. XGBoost stays optional inside `compare_models.py`.
- The dashboard image is `Dockerfile` plus `requirements-dashboard.txt`. It does not install torch.

## Still needs a person

### 1. Run the sentiment fetcher with a real key

The parser is unit-tested against a fixture. The live Alpha Vantage payload is not. Until a keyed run writes `news_sentiment.csv` and the models are retrained, hybrid sentiment stays the neutral 0.5 fill.

- Add `ALPHA_VANTAGE_KEY` to `.env` on the machine that runs the fetch. Do not commit it.
- Run `python scripts/fetching/fetch_alpha_sentiment.py --time-from 20220101` on two UTC days so all 30 US names are covered.
- After the file has every name, the weekly workflow can hold the same key as a GitHub secret. One Sunday then spends calls only on names that are still missing.

### 2. Confirm the Streamlit Cloud app

Models and `data/processed/kenya_features.csv` are in git. Confirm the deployed app shows US charts from Yahoo and the Kenya slice, with BUY/HOLD signals.

### 3. Regenerate the universe backtest numbers

`python scripts/backtesting/backtest.py` fits a fresh forest per ticker. That replaces the AAPL-only headline. It needs `data/processed/features.csv` and a long run. Do not treat 0.822 as the universe result.

# Open Items

Follow-up work after the Jul 8 2026 fix pass (commit `2f0ccd8`). Nothing here is
broken; these are loose ends, unverified assumptions, and minor inconsistencies.
Ordered by importance.

## Needs your action / unverified against the real world

### 1. Run the sentiment fetcher against the live API (biggest lever)
`scripts/fetching/fetch_alpha_sentiment.py` has only been unit-tested against a
fixture — the real Alpha Vantage response shape, pagination, and rate limits are
unexercised. Until this runs with a key, the hybrid model is identical to the
technical one (sentiment stays neutral 0.5).

- Get a free key at https://www.alphavantage.co/support/#api-key
- Add `ALPHA_VANTAGE_KEY=...` to `.env`
- Run: `python scripts/fetching/fetch_alpha_sentiment.py --time-from 20220101`
  (free tier is ~5 req/min, so 30 tickers takes a few minutes)
- Re-run preprocessing + retrain, then re-check the hybrid backtest.
- Also set `ALPHA_VANTAGE_KEY` as a GitHub Actions secret so `retrain.yml` uses it.

### 2. Repoint the git remote to the renamed repo
`origin` still points at the old `tonykihu` URL; pushes work via a 301 redirect
but print "This repository moved."

```bash
git remote set-url origin https://github.com/TineyKode/stock-ai.git
```

### 3. Verify the Streamlit Cloud deploy end-to-end
Models are now tracked and `data/processed/kenya_features.csv` ships, so the live
app *should* show both US (live yfinance) and Kenya (committed slice) charts and
BUY/HOLD signals. Confirm on the deployed app — this was not verified locally.

## Caveat on the headline backtest result

### 4. Backtest is AAPL-only
The reported hybrid Sharpe 0.822 / max drawdown -15.3% (§15.2 of the README) are
for a single ticker. For an honest performance figure, extend
`scripts/backtesting/backtest.py` to a portfolio / multi-ticker backtest and
report universe-wide metrics.

## Minor inconsistencies (quick, self-contained fixes)

### 5. `compare_models.py` uses the old target
`scripts/training/compare_models.py` still labels with the old binary target
(`Close.shift(-1) > Close`, no threshold) and its own `max_depth=10`, so its
diagnostic RF/GB/XGB numbers won't line up with the shipped magnitude-aware
model. Align it to `add_target()` / `TARGET_THRESHOLD` and `MODEL_PARAMS` if you
want its output comparable. Not in CI; not broken.

### 6. Prune heavy/unused dependencies
`requirements.txt` carries `tensorflow` and `ta` (likely unused now that features
are hand-rolled) and `xgboost` (only for the optional `compare_models.py`). CI
installs all of them, making it slow. Audit and trim.

### 7. `retrain.yml` sentiment step can fail-hard
If `ALPHA_VANTAGE_KEY` is set but *every* ticker errors, `fetch_alpha_sentiment.py`
exits 1 and reddens the retrain job (per-ticker failures are already caught; only a
total wipeout returns non-zero). Consider making the CI step non-fatal, e.g.
`run: python scripts/fetching/fetch_alpha_sentiment.py || true`.

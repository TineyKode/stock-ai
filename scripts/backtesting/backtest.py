"""
Walk-forward backtesting with transaction costs and risk metrics.

Simulates trading using model predictions on unseen data folds,
computes Sharpe ratio, max drawdown, win rate, and compares to buy-and-hold.

Usage:
  python scripts/backtesting/backtest.py
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.features import (
    TECHNICAL_FEATURES,
    HYBRID_FEATURES,
    MODEL_PARAMS,
    TARGET_THRESHOLD,
    SIGNAL_THRESHOLD,
)


TRANSACTION_COST_BPS = 10  # 10 basis points per trade
TRADING_DAYS_PER_YEAR = 252


def compute_returns(close_prices):
    """Compute daily returns from close prices."""
    return close_prices.pct_change().shift(-1)


def apply_transaction_costs(signals, cost_bps=TRANSACTION_COST_BPS):
    """
    Deduct transaction costs when position changes.
    Returns a Series of cost deductions (negative values).
    """
    cost = cost_bps / 10_000
    trades = signals.diff().abs().fillna(0)
    # First entry also incurs cost if signal is 1
    trades.iloc[0] = abs(signals.iloc[0])
    return -trades * cost


def sharpe_ratio(returns, risk_free_rate=0.0):
    """Annualized Sharpe ratio."""
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std()


def max_drawdown(cumulative_returns):
    """Maximum drawdown from peak equity."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def win_rate(returns):
    """Fraction of positive-return trading days."""
    trading_days = returns[returns != 0]
    if len(trading_days) == 0:
        return 0.0
    return (trading_days > 0).mean()


def backtest_model(model, X, close_prices, feature_names, n_splits=5,
                   cost_bps=TRANSACTION_COST_BPS,
                   target_threshold=TARGET_THRESHOLD,
                   signal_threshold=SIGNAL_THRESHOLD):
    """
    Walk-forward backtest: train on past folds, predict on next fold,
    simulate returns with transaction costs.

    The label is magnitude-aware (next-day return > target_threshold) and
    the strategy trades selectively: go long only when the model's P(up)
    is at least signal_threshold, otherwise hold cash.

    Returns dict of metrics and a DataFrame of daily results.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    daily_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        close_train = close_prices.iloc[train_idx]
        close_test = close_prices.iloc[test_idx]

        # Magnitude-aware target: next-day return clears the threshold
        train_ret = close_train.shift(-1) / close_train - 1.0
        y_train = (train_ret > target_threshold).astype(int).iloc[:-1]
        X_train = X_train.iloc[:-1]

        X_test_aligned = X_test.iloc[:-1]
        close_test_aligned = close_test.iloc[:-1]

        if len(X_train) < 50 or len(X_test_aligned) < 5:
            continue

        # Train, then go long only on high-confidence up-days
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_aligned)[:, 1]
        else:
            proba = model.predict(X_test_aligned).astype(float)
        predictions = (proba >= signal_threshold).astype(int)

        # Daily market returns
        market_returns = close_test_aligned.pct_change().shift(-1).iloc[:-1]
        signals = pd.Series(predictions[:-1], index=market_returns.index)

        # Strategy returns = signal * market return - transaction costs
        strategy_returns = signals * market_returns
        costs = apply_transaction_costs(signals, cost_bps)
        strategy_returns = strategy_returns + costs.values[:len(strategy_returns)]

        for i in range(len(strategy_returns)):
            daily_results.append({
                "fold": fold,
                "date": market_returns.index[i],
                "market_return": market_returns.iloc[i],
                "signal": signals.iloc[i],
                "strategy_return": strategy_returns.iloc[i],
            })

    if not daily_results:
        print("No valid backtest folds produced.")
        return {}, pd.DataFrame()

    results_df = pd.DataFrame(daily_results)

    # Compute metrics
    strat_returns = results_df["strategy_return"]
    mkt_returns = results_df["market_return"]

    strat_cumulative = (1 + strat_returns).cumprod()
    mkt_cumulative = (1 + mkt_returns).cumprod()

    metrics = {
        "total_return_strategy": strat_cumulative.iloc[-1] - 1,
        "total_return_buyhold": mkt_cumulative.iloc[-1] - 1,
        "sharpe_ratio": sharpe_ratio(strat_returns),
        "sharpe_ratio_buyhold": sharpe_ratio(mkt_returns),
        "max_drawdown": max_drawdown(strat_cumulative),
        "max_drawdown_buyhold": max_drawdown(mkt_cumulative),
        "win_rate": win_rate(strat_returns),
        "total_trades": int((results_df["signal"].diff().abs() > 0).sum()),
        "days_tested": len(results_df),
        "cost_bps": cost_bps,
    }

    return metrics, results_df


def print_backtest_report(metrics, label="Model"):
    """Print formatted backtest results."""
    if not metrics:
        print(f"\n{label}: No results to report.")
        return

    print(f"\n{'='*55}")
    print(f" Backtest Report: {label}")
    print(f"{'='*55}")
    print(f"  Days tested:           {metrics['days_tested']}")
    print(f"  Total trades:          {metrics['total_trades']}")
    print(f"  Transaction cost:      {metrics['cost_bps']} bps per trade")
    print(f"")
    print(f"  {'Metric':<25} {'Strategy':>12} {'Buy&Hold':>12}")
    print(f"  {'-'*49}")
    print(f"  {'Total Return':<25} {metrics['total_return_strategy']:>11.2%} {metrics['total_return_buyhold']:>11.2%}")
    print(f"  {'Sharpe Ratio':<25} {metrics['sharpe_ratio']:>12.3f} {metrics['sharpe_ratio_buyhold']:>12.3f}")
    print(f"  {'Max Drawdown':<25} {metrics['max_drawdown']:>11.2%} {metrics['max_drawdown_buyhold']:>11.2%}")
    print(f"  {'Win Rate':<25} {metrics['win_rate']:>11.2%} {'---':>12}")
    print(f"{'='*55}")

    if metrics["sharpe_ratio"] > metrics["sharpe_ratio_buyhold"]:
        print(f"  >> Strategy outperforms buy-and-hold on risk-adjusted basis")
    else:
        print(f"  >> Buy-and-hold has better risk-adjusted returns")


def main():
    """Run backtest on technical and hybrid models for AAPL."""
    features_path = "data/processed/features.csv"
    if not os.path.exists(features_path):
        print(f"Features file not found: {features_path}")
        print("Run preprocessing first: python scripts/preprocessing/preprocess_data.py")
        return

    df = pd.read_csv(features_path)

    # Filter to AAPL (most data)
    ticker = "AAPL"
    if "Ticker" in df.columns:
        df = df[df["Ticker"] == ticker].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Backtesting on {ticker}: {len(df)} rows")

    # --- Technical Model ---
    tech_model_path = "models/technical_model.pkl"
    if os.path.exists(tech_model_path):
        tech_model = joblib.load(tech_model_path)
        feat_cols = [f for f in TECHNICAL_FEATURES if f in df.columns]
        X = df[feat_cols].copy()
        close = df["Close"].copy()

        # Drop rows with NaN features
        valid = X.dropna().index
        X = X.loc[valid]
        close = close.loc[valid]

        from sklearn.ensemble import RandomForestClassifier
        # Fresh model per fold, using the SAME hyperparameters that ship
        # (utils.features.MODEL_PARAMS) so the backtest reflects the
        # deployed model rather than an ad-hoc configuration.
        metrics, results = backtest_model(
            RandomForestClassifier(**MODEL_PARAMS["technical"]),
            X, close, feat_cols,
        )
        print_backtest_report(metrics, f"Technical Model ({ticker})")
    else:
        print(f"\nTechnical model not found at {tech_model_path}, skipping.")

    # --- Hybrid Model ---
    sentiment_path = "data/processed/news_sentiment.csv"
    if os.path.exists(sentiment_path):
        sentiment = pd.read_csv(sentiment_path)
        merge_cols = ["Date"]
        if "Ticker" in sentiment.columns:
            merge_cols.append("Ticker")
        df_hybrid = pd.merge(df, sentiment[merge_cols + ["sentiment_score"]],
                             on=merge_cols, how="left", suffixes=("", "_sent"))
        if "sentiment_score_sent" in df_hybrid.columns:
            df_hybrid["sentiment_score"] = df_hybrid["sentiment_score_sent"]
            df_hybrid.drop(columns=["sentiment_score_sent"], inplace=True)
    else:
        df_hybrid = df.copy()

    if "sentiment_score" not in df_hybrid.columns:
        df_hybrid["sentiment_score"] = 0.5
    else:
        df_hybrid["sentiment_score"] = df_hybrid["sentiment_score"].fillna(0.5)

    feat_cols_h = [f for f in HYBRID_FEATURES if f in df_hybrid.columns]
    X_h = df_hybrid[feat_cols_h].copy()
    close_h = df_hybrid["Close"].copy()

    valid_h = X_h.dropna().index
    X_h = X_h.loc[valid_h]
    close_h = close_h.loc[valid_h]

    from sklearn.ensemble import RandomForestClassifier
    metrics_h, results_h = backtest_model(
        RandomForestClassifier(**MODEL_PARAMS["hybrid"]),
        X_h, close_h, feat_cols_h,
    )
    print_backtest_report(metrics_h, f"Hybrid Model ({ticker})")

    # Save results
    os.makedirs("logs", exist_ok=True)
    try:
        if not results.empty:
            results.to_csv("logs/backtest_technical.csv", index=False)
    except NameError:
        pass
    if not results_h.empty:
        results_h.to_csv("logs/backtest_hybrid.csv", index=False)
    print("\nDetailed results saved to logs/")


if __name__ == "__main__":
    main()

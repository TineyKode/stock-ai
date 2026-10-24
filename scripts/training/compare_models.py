"""
Compare multiple model types using walk-forward cross-validation.

Models: RandomForest, GradientBoosting, XGBoost (if installed).
Reports accuracy, F1, AUC for each model on AAPL technical features.

Usage:
  python scripts/training/compare_models.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.dataset import add_target
from utils.features import MODEL_PARAMS, TECHNICAL_FEATURES


def prepare_comparison(df, ticker="AAPL"):
    """Label one ticker with the same magnitude-aware target the shipped model uses."""
    frame = df.copy()
    if "Ticker" in frame.columns:
        frame = frame[frame["Ticker"] == ticker].copy()
    else:
        frame["Ticker"] = ticker
    labeled = add_target(frame)
    feat_cols = [name for name in TECHNICAL_FEATURES if name in labeled.columns]
    labeled = labeled.dropna(subset=feat_cols + ["Target"])
    x = labeled[feat_cols]
    y = labeled["Target"].astype(int)
    return x, y


def get_models():
    """Return dict of model_name to model instance."""
    models = {
        "RandomForest": RandomForestClassifier(**MODEL_PARAMS["technical"]),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
        )
    except ImportError:
        print("XGBoost not installed, skipping. Install with: pip install xgboost")

    return models


def walk_forward_evaluate(model, X, y, n_splits=5):
    """
    Walk-forward CV returning per-fold and aggregate metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        fold_metrics.append({
            "fold": fold,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "train_size": len(X_train),
            "test_size": len(X_test),
        })

    return fold_metrics


def main():
    features_path = "data/processed/features.csv"
    if not os.path.exists(features_path):
        print(f"Features file not found: {features_path}")
        print("Run preprocessing first: python scripts/preprocessing/preprocess_data.py")
        return

    df = pd.read_csv(features_path)
    ticker = "AAPL"
    X, y = prepare_comparison(df, ticker)
    feat_cols = list(X.columns)

    print(f"Comparing models on {ticker}: {len(X)} samples, {len(feat_cols)} features")
    print(f"Target distribution: {y.mean():.1%} up days\n")

    models = get_models()
    all_results = {}

    print(f"{'Model':<22} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Folds':>7}")
    print("-" * 62)

    for name, model in models.items():
        fold_metrics = walk_forward_evaluate(model, X, y, n_splits=5)

        if not fold_metrics:
            print(f"{name:<22} {'(no valid folds)':>40}")
            continue

        metrics_df = pd.DataFrame(fold_metrics)
        avg_acc = metrics_df["accuracy"].mean()
        avg_f1 = metrics_df["f1"].mean()
        avg_auc = metrics_df["auc"].mean()

        print(f"{name:<22} {avg_acc:>9.3f} {avg_f1:>9.3f} {avg_auc:>9.3f} {len(fold_metrics):>7}")

        all_results[name] = {
            "avg_accuracy": round(avg_acc, 4),
            "avg_f1": round(avg_f1, 4),
            "avg_auc": round(avg_auc, 4),
            "folds": fold_metrics,
        }

    # Determine best model
    if all_results:
        best_name = max(all_results, key=lambda k: all_results[k]["avg_f1"])
        print(f"\nBest model by F1: {best_name} ({all_results[best_name]['avg_f1']:.4f})")

        # Per-fold detail for best model
        best_folds = pd.DataFrame(all_results[best_name]["folds"])
        print(f"\n{best_name} per-fold breakdown:")
        print(f"  {'Fold':>5} {'Train':>7} {'Test':>6} {'Acc':>8} {'F1':>8} {'AUC':>8}")
        for _, row in best_folds.iterrows():
            print(f"  {int(row['fold']):>5} {int(row['train_size']):>7} {int(row['test_size']):>6}"
                  f" {row['accuracy']:>7.3f} {row['f1']:>7.3f} {row['auc']:>7.3f}")

        # Save comparison results
        os.makedirs("logs", exist_ok=True)
        summary = {name: {k: v for k, v in data.items() if k != "folds"}
                   for name, data in all_results.items()}
        summary["best_model"] = best_name
        summary["ticker"] = ticker
        summary["n_samples"] = len(X)

        with open("logs/model_comparison.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nComparison saved to logs/model_comparison.json")


if __name__ == "__main__":
    main()

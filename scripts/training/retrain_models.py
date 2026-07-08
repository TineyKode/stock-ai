import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.features import TECHNICAL_FEATURES, HYBRID_FEATURES, MODEL_PARAMS
from utils.dataset import load_features, add_target, merge_sentiment
from utils.evaluation import walk_forward_validate, save_metrics


def main():
    """Retrain both technical and hybrid models on all tickers pooled."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    retrain_technical_model()
    retrain_hybrid_model()


def retrain_technical_model():
    """Retrain the technical model with walk-forward validation."""
    model_path = "models/technical_model.pkl"
    print("Retraining technical model...")

    try:
        data = load_features()
        data = add_target(data)
        data = data.dropna(subset=TECHNICAL_FEATURES + ["Target"])
        data = data.sort_values("Date").reset_index(drop=True)

        X = data[TECHNICAL_FEATURES]
        y = data["Target"].astype(int)

        if len(X) < 100:
            print(f"  Not enough data ({len(X)} rows). Skipping.")
            return

        model_params = MODEL_PARAMS["technical"]
        cv_results = walk_forward_validate(
            RandomForestClassifier, model_params, X, y, n_splits=5
        )
        agg = cv_results["aggregate"]
        print(f"  CV Mean Accuracy: {agg['mean_accuracy']:.4f}, F1: {agg['mean_f1']:.4f}")

        model = RandomForestClassifier(**model_params)
        model.fit(X, y)

        joblib.dump(model, model_path)
        save_metrics(cv_results, "technical")
        print(f"  Technical model saved ({len(X)} samples)")

    except FileNotFoundError as e:
        print(f"  Data not found: {e}")
    except Exception as e:
        print(f"  Error: {e}")


def retrain_hybrid_model():
    """Retrain the hybrid model with walk-forward validation."""
    model_path = "models/hybrid_model.pkl"
    print("Retraining hybrid model...")

    try:
        data = load_features()
        data = merge_sentiment(data)
        data = add_target(data)
        data = data.dropna(subset=HYBRID_FEATURES + ["Target"])
        data = data.sort_values("Date").reset_index(drop=True)

        X = data[HYBRID_FEATURES]
        y = data["Target"].astype(int)

        if len(X) < 100:
            print(f"  Not enough data ({len(X)} rows). Skipping.")
            return

        model_params = MODEL_PARAMS["hybrid"]
        cv_results = walk_forward_validate(
            RandomForestClassifier, model_params, X, y, n_splits=5
        )
        agg = cv_results["aggregate"]
        print(f"  CV Mean Accuracy: {agg['mean_accuracy']:.4f}, F1: {agg['mean_f1']:.4f}")

        model = RandomForestClassifier(**model_params).fit(X, y)

        joblib.dump(model, model_path)
        save_metrics(cv_results, "hybrid")
        print(f"  Hybrid model saved ({len(X)} samples)")

    except FileNotFoundError as e:
        print(f"  Data not found: {e}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()

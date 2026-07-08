import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.features import HYBRID_FEATURES, MODEL_PARAMS
from utils.dataset import load_features, add_target, merge_sentiment
from utils.evaluation import walk_forward_validate, save_metrics

# Load features, merge sentiment (neutral 0.5 fill), build per-ticker target.
# Trained on ALL tickers pooled (not AAPL alone).
data = load_features()
data = merge_sentiment(data)
data = add_target(data)
data = data.dropna(subset=HYBRID_FEATURES + ["Target"])
data = data.sort_values("Date").reset_index(drop=True)

X = data[HYBRID_FEATURES]
y = data["Target"].astype(int)

n_tickers = data["Ticker"].nunique() if "Ticker" in data.columns else 1
print(f"Training data: {len(X)} samples across {n_tickers} tickers, "
      f"{len(HYBRID_FEATURES)} features")
print(f"Class balance: UP={int(y.sum())}/{len(y)} ({y.mean():.1%})")

# Walk-forward cross-validation (shared hyperparameters)
model_params = MODEL_PARAMS["hybrid"]
cv_results = walk_forward_validate(
    RandomForestClassifier, model_params, X, y, n_splits=5
)

print("\nWalk-Forward CV Results:")
for fold in cv_results["folds"]:
    auc_str = f", AUC={fold['roc_auc']:.4f}" if fold.get("roc_auc") else ""
    print(f"  Fold {fold['fold']}: acc={fold['accuracy']:.4f}, "
          f"f1={fold['f1']:.4f}, prec={fold['precision']:.4f}, "
          f"rec={fold['recall']:.4f}{auc_str} "
          f"(train={fold['train_size']}, test={fold['test_size']})")

agg = cv_results["aggregate"]
print(f"\n  Mean Accuracy:  {agg['mean_accuracy']:.4f} (+/- {agg['std_accuracy']:.4f})")
print(f"  Mean F1:        {agg['mean_f1']:.4f}")
if "mean_roc_auc" in agg:
    print(f"  Mean AUC:       {agg['mean_roc_auc']:.4f}")

# Train final model on all data
model = RandomForestClassifier(**model_params).fit(X, y)

# Save
os.makedirs("models", exist_ok=True)
save_metrics(cv_results, "hybrid")
joblib.dump(model, "models/hybrid_model.pkl")
print(f"\nHybrid model trained on {len(X)} samples and saved.")

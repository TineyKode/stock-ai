import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.features import TECHNICAL_FEATURES, MODEL_PARAMS
from utils.dataset import load_features, add_target
from utils.evaluation import walk_forward_validate, save_metrics

# Load features and build a next-day-up target per ticker.
# Trained on ALL tickers pooled (not AAPL alone) so one model
# generalizes across the tickers the dashboard serves.
data = load_features()
data = add_target(data)
data = data.dropna(subset=TECHNICAL_FEATURES + ["Target"])
data = data.sort_values("Date").reset_index(drop=True)

X = data[TECHNICAL_FEATURES]
y = data["Target"].astype(int)

n_tickers = data["Ticker"].nunique() if "Ticker" in data.columns else 1
print(f"Training data: {len(X)} samples across {n_tickers} tickers, "
      f"{len(TECHNICAL_FEATURES)} features")
print(f"Class balance: UP={int(y.sum())}/{len(y)} ({y.mean():.1%})")

# Walk-forward cross-validation (shared hyperparameters)
model_params = MODEL_PARAMS["technical"]
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
print(f"  Mean Precision: {agg['mean_precision']:.4f}")
print(f"  Mean Recall:    {agg['mean_recall']:.4f}")
if "mean_roc_auc" in agg:
    print(f"  Mean AUC:       {agg['mean_roc_auc']:.4f}")

# Train final model on all data
final_model = RandomForestClassifier(**model_params)
final_model.fit(X, y)

# Feature importances
importances = sorted(zip(TECHNICAL_FEATURES, final_model.feature_importances_),
                     key=lambda x: x[1], reverse=True)
print("\nFeature Importances:")
for name, imp in importances:
    print(f"  {name:20s}: {imp:.4f}")

# Save
os.makedirs("models", exist_ok=True)
save_metrics(cv_results, "technical")
joblib.dump(final_model, "models/technical_model.pkl")
print("\nTechnical model saved to models/technical_model.pkl")

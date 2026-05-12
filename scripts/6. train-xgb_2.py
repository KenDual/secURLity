"""
Train XGBoost classifier (model 2).

Input  : data/processed/xgboost/{train,val,test}_X_xgb.npy
         data/processed/xgboost/{train,val,test}_y_xgb.npy
         data/processed/xgboost/xgb_feature_cols.json
         data/processed/xgboost/xgb_feature_meta.json   (pos_weight)
Output : models/xgb_url_2.ubj                           (XGBoost binary)
         models/results_xgb_2.json                      (metrics + feat importance)

Cấu hình:
  - tree_method = hist + device = cuda (GPU; XGBoost >= 2.0).
  - scale_pos_weight = neg/pos từ train set (≈ 2.33).
  - Early stopping trên val aucpr (PR-AUC nhạy với class imbalance hơn ROC-AUC).
  - max_depth=8, lr=0.05, n_estimators=2000 — early stopping sẽ chốt sớm.
"""

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(r"D:\! secURLity")
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "xgboost"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "xgb_url_2.ubj"
RESULTS_PATH = MODELS_DIR / "results_xgb_2.json"

# ============================================================================
# Hyperparameters
# ============================================================================
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["aucpr", "auc", "logloss"],
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "random_state": 42,
}
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50


# ============================================================================
# Load data
# ============================================================================
def load_split(name: str):
    X = np.load(DATA_DIR / f"{name}_X_xgb.npy")
    y = np.load(DATA_DIR / f"{name}_y_xgb.npy")
    return X, y


def main():
    print("=" * 78)
    print("XGBOOST TRAINING (model 2)")
    print("=" * 78)

    # Load feature columns + meta
    with open(DATA_DIR / "xgb_feature_cols.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open(DATA_DIR / "xgb_feature_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"Feature dim: {len(feature_cols)}")

    # Load splits
    print("\nLoading splits...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"  Train: X{X_train.shape}  y{y_train.shape}  pos={int(y_train.sum()):,}")
    print(f"  Val  : X{X_val.shape}  y{y_val.shape}  pos={int(y_val.sum()):,}")
    print(f"  Test : X{X_test.shape}  y{y_test.shape}  pos={int(y_test.sum()):,}")

    pos_weight = float(meta["pos_weight"])
    print(f"\nscale_pos_weight = neg/pos = {pos_weight:.4f}")

    # Build DMatrix (memory-efficient for ~8M rows)
    print("\nBuilding DMatrices...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = dict(PARAMS)
    params["scale_pos_weight"] = pos_weight

    # Train
    print(f"\nTraining (n_estimators={N_ESTIMATORS}, early_stopping={EARLY_STOPPING_ROUNDS})...")
    evals_result: dict = {}
    t0 = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=25,
    )
    train_secs = time.time() - t0

    best_iter = booster.best_iteration
    print(f"\nDone. best_iteration={best_iter}  best_score(aucpr)={booster.best_score:.6f}")
    print(f"Training wall time: {train_secs/60:.2f} min")

    # Save model
    booster.save_model(str(MODEL_PATH))
    print(f"Saved model -> {MODEL_PATH}")

    # ---- Evaluate on test ----
    print("\nEvaluating on test set...")
    y_proba_test = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    y_pred_test = (y_proba_test >= 0.5).astype(np.int8)

    acc = accuracy_score(y_test, y_pred_test)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average="binary", pos_label=1, zero_division=0
    )
    roc = roc_auc_score(y_test, y_proba_test)
    pr_auc = average_precision_score(y_test, y_proba_test)
    cm = confusion_matrix(y_test, y_pred_test).tolist()  # [[TN,FP],[FN,TP]]
    report = classification_report(
        y_test, y_pred_test, target_names=["benign", "malicious"], digits=4
    )

    print(f"  Accuracy           : {acc:.6f}")
    print(f"  Precision (mal)    : {prec:.6f}")
    print(f"  Recall    (mal)    : {rec:.6f}")
    print(f"  F1        (mal)    : {f1:.6f}")
    print(f"  ROC-AUC            : {roc:.6f}")
    print(f"  PR-AUC             : {pr_auc:.6f}")
    print(f"  Confusion matrix   : {cm}")
    print("\n" + report)

    # Feature importance (gain) — top 30
    importance = booster.get_score(importance_type="gain")
    importance_sorted = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    top30 = [{"feature": k, "gain": float(v)} for k, v in importance_sorted[:30]]
    print("Top 15 features by gain:")
    for row in top30[:15]:
        print(f"  {row['feature']:<32s}  {row['gain']:.4f}")

    # Save results
    results = {
        "model": "xgboost",
        "dataset": "model_2",
        "n_features": len(feature_cols),
        "params": params,
        "n_estimators_max": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(best_iter),
        "best_score_aucpr_val": float(booster.best_score),
        "training_seconds": train_secs,
        "test_metrics": {
            "accuracy": float(acc),
            "precision_malicious": float(prec),
            "recall_malicious": float(rec),
            "f1_malicious": float(f1),
            "roc_auc": float(roc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm,
        },
        "test_classification_report": report,
        "feature_importance_top30_gain": top30,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {RESULTS_PATH}")

    print("\n" + "=" * 78)
    print("XGBOOST TRAINING DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()

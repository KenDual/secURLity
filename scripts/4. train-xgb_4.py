"""
Train XGBoost classifier for model 4 (real-world 19.68M URL dataset).

Reads feature matrices from `data/processed/xgboost_4/` (output of
`3. feature-extract_xgb_4.py`) and trains an XGBoost binary classifier with
early stopping on val ROC-AUC. Saves model + metrics + feature importance.

Outputs:
    models/xgb_url_4.ubj                  # UBJ binary checkpoint
    models/xgb_url_4_metrics.json         # train + val metrics, hyperparams, ES round
    models/xgb_url_4_importance.json      # gain / weight / cover per feature

Hardware notes:
  - Uses CUDA if available (tree_method="hist" + device="cuda"). 3060 Ti has
    8 GB → enough cho 19.68M rows × ~80 features float32 (~5 GB train matrix)
    với DMatrix streaming bên trong XGBoost.
  - If GPU OOM, fall back to CPU automatically via env var XGB_FORCE_CPU=1.

Hyperparams chọn theo "safe baseline cho large tabular":
  - max_depth=8, learning_rate=0.05, n_estimators=2000 (early-stop ~500-800)
  - subsample=0.8, colsample_bytree=0.8 (chống overfit)
  - reg_alpha=0.0, reg_lambda=1.0 (L2 default)
  - scale_pos_weight đọc từ meta (target ~4.0 cho dataset 80/20)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(r"D:\! secURLity")
FEAT_DIR     = PROJECT_ROOT / "data" / "processed" / "xgboost_4"
MODELS_DIR   = PROJECT_ROOT / "models"

MODEL_PATH      = MODELS_DIR / "xgb_url_4.ubj"
METRICS_PATH    = MODELS_DIR / "xgb_url_4_metrics.json"
IMPORTANCE_PATH = MODELS_DIR / "xgb_url_4_importance.json"

# ============================================================================
# Hyperparameters
# ============================================================================
HP = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr", "logloss"],
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "max_bin": 256,
    "random_state": 42,
    "n_jobs": -1,
}
EARLY_STOPPING_ROUNDS = 50


def _load_split(name: str):
    X = np.load(FEAT_DIR / f"{name}_X_xgb.npy", mmap_mode="r")
    y = np.load(FEAT_DIR / f"{name}_y_xgb.npy")
    return X, y


def _device() -> str:
    if os.environ.get("XGB_FORCE_CPU") == "1":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TRAIN XGBoost (model 4 — real-world 19.68M URLs)")
    print("=" * 78)

    # ---- Load splits ----
    print(f"\nLoading splits from {FEAT_DIR}...")
    X_tr, y_tr = _load_split("train")
    X_va, y_va = _load_split("val")
    print(f"  train: X={X_tr.shape}  y={y_tr.shape}  pos={(y_tr==1).sum():,}")
    print(f"  val  : X={X_va.shape}  y={y_va.shape}  pos={(y_va==1).sum():,}")

    # ---- Pos weight from meta (preferred) or recompute ----
    meta_path = FEAT_DIR / "xgb_4_feature_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pos_weight = float(meta["pos_weight"])
        print(f"  pos_weight (from meta): {pos_weight:.4f}")
    else:
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        pos_weight = n_neg / max(n_pos, 1)
        print(f"  pos_weight (computed) : {pos_weight:.4f}")

    feature_cols_path = FEAT_DIR / "xgb_4_feature_cols.json"
    feature_names = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    assert len(feature_names) == X_tr.shape[1], "feature count mismatch"

    # ---- Build model ----
    device = _device()
    print(f"\nDevice: {device}")
    print(f"Hyperparams: {json.dumps(HP, indent=2)}")
    print(f"Early stopping rounds: {EARLY_STOPPING_ROUNDS}")

    model = xgb.XGBClassifier(
        **HP,
        device=device,
        scale_pos_weight=pos_weight,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    # ---- Train ----
    print("\nTraining...")
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=25,
    )
    train_secs = time.time() - t0
    best_iter = int(model.best_iteration)
    print(f"\nDone in {train_secs:.1f}s | best_iteration={best_iter}")

    # ---- Eval on val (with best_iteration) ----
    print("\nVal metrics @ best_iteration:")
    p_va = model.predict_proba(X_va)[:, 1]
    auc_va  = float(roc_auc_score(y_va, p_va))
    ap_va   = float(average_precision_score(y_va, p_va))
    pred_va = (p_va >= 0.5).astype(np.int8)
    f1_va   = float(f1_score(y_va, pred_va))
    prec_va = float(precision_score(y_va, pred_va))
    rec_va  = float(recall_score(y_va, pred_va))
    print(f"  ROC-AUC: {auc_va:.6f}")
    print(f"  PR-AUC : {ap_va:.6f}")
    print(f"  F1@0.5 : {f1_va:.6f}")
    print(f"  P @0.5 : {prec_va:.6f}")
    print(f"  R @0.5 : {rec_va:.6f}")

    # ---- Save model ----
    model.save_model(str(MODEL_PATH))
    print(f"\nSaved model       -> {MODEL_PATH}")

    # ---- Save metrics ----
    metrics = {
        "device": device,
        "train_seconds": train_secs,
        "best_iteration": best_iter,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "hyperparams": HP,
        "scale_pos_weight": pos_weight,
        "n_features": int(X_tr.shape[1]),
        "n_train": int(X_tr.shape[0]),
        "n_val":   int(X_va.shape[0]),
        "val": {
            "roc_auc":   auc_va,
            "pr_auc":    ap_va,
            "f1_thr050": f1_va,
            "p_thr050":  prec_va,
            "r_thr050":  rec_va,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics     -> {METRICS_PATH}")

    # ---- Save feature importance (gain/weight/cover, all 3 useful) ----
    booster = model.get_booster()
    importance = {}
    for kind in ("gain", "weight", "cover", "total_gain", "total_cover"):
        score = booster.get_score(importance_type=kind)
        # XGBoost trả về key dạng f0/f1/... — map về tên cột thật
        named = {feature_names[int(k[1:])]: float(v) for k, v in score.items()}
        importance[kind] = dict(
            sorted(named.items(), key=lambda kv: kv[1], reverse=True)
        )
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    print(f"Saved importance  -> {IMPORTANCE_PATH}")

    # ---- Top-10 by gain (quick sanity check) ----
    print("\nTop 10 features by gain:")
    for i, (name, val) in enumerate(list(importance["gain"].items())[:10], 1):
        print(f"  {i:2d}. {name:<35s}  {val:>12.4f}")

    print("\n" + "=" * 78)
    print("TRAIN DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()

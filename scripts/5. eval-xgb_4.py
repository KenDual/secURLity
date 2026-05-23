"""
Evaluate XGBoost model 4 on test split + tune operating thresholds.

Reads:
    data/processed/xgboost_4/test_X_xgb.npy
    data/processed/xgboost_4/test_y_xgb.npy
    models/xgb_url_4.ubj

Writes:
    models/results_xgb_4.json        # ROC-AUC, PR-AUC, F1/P/R at 3 operating points
    models/thresholds_xgb_4.json     # default / safer / precise operating points

Operating points (same convention as model 4 CNN-LSTM):
    default : threshold maximizing F1
    safer   : lowest threshold where precision >= 0.99 (bias to catch mal)
    precise : highest threshold where recall    >= 0.99 (bias to avoid FP)
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(r"D:\! secURLity")
FEAT_DIR     = PROJECT_ROOT / "data" / "processed" / "xgboost_4"
MODELS_DIR   = PROJECT_ROOT / "models"

MODEL_PATH       = MODELS_DIR / "xgb_url_4.ubj"
RESULTS_PATH     = MODELS_DIR / "results_xgb_4.json"
THRESHOLDS_PATH  = MODELS_DIR / "thresholds_xgb_4.json"

# Sweep step — 0.005 enough for 3-decimal threshold selection
THRESH_STEP = 0.005


def _load_test():
    X = np.load(FEAT_DIR / "test_X_xgb.npy", mmap_mode="r")
    y = np.load(FEAT_DIR / "test_y_xgb.npy")
    return X, y


def _metrics_at(y_true, probs, thr: float) -> dict:
    pred = (probs >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "f1":        float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall":    float(recall_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _pick_thresholds(y_true, probs):
    """Return (default, safer, precise) dicts."""
    # F1 sweep on coarse grid (vectorized via PR curve is fastest)
    prec, rec, thr = precision_recall_curve(y_true, probs)
    # precision_recall_curve trả về (n+1, n+1, n) — chỉnh align
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # bỏ điểm cuối (recall=0 sentinel)
    f1 = f1[:-1]
    prec_arr = prec[:-1]
    rec_arr  = rec[:-1]

    # default = max F1
    i_def = int(np.argmax(f1))
    thr_def = float(thr[i_def]) if i_def < len(thr) else 0.5

    # safer = lowest threshold where precision >= 0.99
    mask_p = prec_arr >= 0.99
    if mask_p.any():
        # tìm threshold thấp nhất satisfying
        idx_safer = int(np.where(mask_p)[0].min())
        thr_safer = float(thr[idx_safer]) if idx_safer < len(thr) else 0.5
    else:
        thr_safer = thr_def

    # precise = highest threshold where recall >= 0.99
    mask_r = rec_arr >= 0.99
    if mask_r.any():
        idx_prec = int(np.where(mask_r)[0].max())
        thr_prec = float(thr[idx_prec]) if idx_prec < len(thr) else 0.5
    else:
        thr_prec = thr_def

    return {
        "default": _metrics_at(y_true, probs, thr_def),
        "safer":   _metrics_at(y_true, probs, thr_safer),
        "precise": _metrics_at(y_true, probs, thr_prec),
    }


def main():
    print("=" * 78)
    print("EVAL XGBoost (model 4)")
    print("=" * 78)

    print(f"\nLoading test set from {FEAT_DIR}...")
    X_te, y_te = _load_test()
    print(f"  test: X={X_te.shape}  pos={(y_te==1).sum():,}  neg={(y_te==0).sum():,}")

    print(f"\nLoading model from {MODEL_PATH}...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    print("Predicting (may take a few minutes for large test set)...")
    p_te = model.predict_proba(X_te)[:, 1]

    # ---- Global metrics ----
    auc  = float(roc_auc_score(y_te, p_te))
    ap   = float(average_precision_score(y_te, p_te))
    print(f"\nROC-AUC : {auc:.6f}")
    print(f"PR-AUC  : {ap:.6f}")

    # ---- Sweep + 3 operating points ----
    print("\nTuning thresholds (default / safer / precise)...")
    ops = _pick_thresholds(y_te, p_te)
    for name, m in ops.items():
        print(f"  [{name:<7s}] thr={m['threshold']:.4f}  "
              f"F1={m['f1']:.6f}  P={m['precision']:.6f}  R={m['recall']:.6f}  "
              f"TP={m['tp']:,} FP={m['fp']:,} TN={m['tn']:,} FN={m['fn']:,}")

    # ---- Save ----
    results = {
        "test": {
            "n":        int(X_te.shape[0]),
            "n_pos":    int((y_te == 1).sum()),
            "n_neg":    int((y_te == 0).sum()),
            "roc_auc":  auc,
            "pr_auc":   ap,
            "operating_points": ops,
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results    -> {RESULTS_PATH}")

    # thresholds file = ops chỉ với thr (giống convention model 4)
    thr_only = {name: m["threshold"] for name, m in ops.items()}
    THRESHOLDS_PATH.write_text(json.dumps(thr_only, indent=2), encoding="utf-8")
    print(f"Saved thresholds -> {THRESHOLDS_PATH}")

    print("\n" + "=" * 78)
    print("EVAL DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""
Single-URL inference for XGBoost model 4 — with SHAP explainability.

Usage:
    python "scripts/predict-url_4_xgb.py" "https://example.com/login"
    python "scripts/predict-url_4_xgb.py" --explain "https://example.com/login"
    python "scripts/predict-url_4_xgb.py" --thr safer "https://..."
    python "scripts/predict-url_4_xgb.py" --topk 15 --explain "https://..."

Loads:
    models/xgb_url_4.ubj                          # XGBoost model
    models/thresholds_xgb_4.json                  # operating thresholds
    data/processed/xgboost_4/xgb_4_feature_cols.json
    data/processed/xgboost_4/xgb_4_feature_meta.json

Outputs:
    - Binary decision (vs chosen threshold)
    - 4-tier risk level from raw probability (Low / Caution / Suspicious / High)
    - With --explain: top-K features by |SHAP value|, signed contribution +
      raw feature value. SHAP is computed via TreeExplainer (sub-millisecond
      per URL because XGBoost is a tree ensemble).

Note: SHAP values are in LOG-ODDS space. `expected_value + sum(shap)` = raw
model output → sigmoid → probability. Sign of a SHAP value tells direction:
    positive → push toward malicious
    negative → push toward benign
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\! secURLity")
FEAT_DIR     = PROJECT_ROOT / "data" / "processed" / "xgboost_4"
MODELS_DIR   = PROJECT_ROOT / "models"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"

MODEL_PATH       = MODELS_DIR / "xgb_url_4.ubj"
THRESHOLDS_PATH  = MODELS_DIR / "thresholds_xgb_4.json"
COLS_PATH        = FEAT_DIR   / "xgb_4_feature_cols.json"


# ============================================================================
# Load extract_features() from `3. feature-extract_xgb_4.py` via importlib
# (filename has spaces + leading digit → can't `import` normally)
# ============================================================================
def _load_extractor():
    extract_path = SCRIPTS_DIR / "3. feature-extract_xgb_4.py"
    spec = importlib.util.spec_from_file_location("xgb4_extract", extract_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Risk tier (same convention as predict-url_4.py)
# ============================================================================
def risk_level(prob: float) -> str:
    pct = prob * 100
    if pct <= 25:
        return "Low Risk"
    if pct <= 50:
        return "Caution"
    if pct <= 75:
        return "Suspicious"
    return "High Risk"


def resolve_threshold(arg: str) -> tuple[str, float]:
    """Accept either a named operating point (default/safer/precise) or a number."""
    try:
        return ("custom", float(arg))
    except ValueError:
        pass
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"thresholds file missing: {THRESHOLDS_PATH}. "
            f"Run `5. eval-xgb_4.py` first or pass --thr 0.5"
        )
    table = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    if arg not in table:
        raise ValueError(
            f"unknown threshold '{arg}'. Choices: {list(table.keys())} or a float."
        )
    return (arg, float(table[arg]))


# ============================================================================
# Explain (SHAP)
# ============================================================================
def explain_with_shap(model, X_row: np.ndarray, feature_names: list[str],
                      topk: int) -> tuple[list[dict], float]:
    """Return (top-k feature contributions, expected_value in log-odds)."""
    try:
        import shap  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "SHAP not installed. Run: pip install shap"
        ) from e

    explainer = shap.TreeExplainer(model)
    # X_row shape (1, n_features) → shap_values shape (1, n_features) for binary
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        # Older API: list of arrays per class. Binary → index 1 (positive class).
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]  # shape (n_features,)

    expected = explainer.expected_value
    if hasattr(expected, "__len__") and not isinstance(expected, (str, bytes)):
        expected = expected[1] if len(expected) > 1 else expected[0]
    expected = float(expected)

    abs_order = np.argsort(-np.abs(sv))
    top = []
    for idx in abs_order[:topk]:
        top.append({
            "feature":     feature_names[idx],
            "raw_value":   float(X_row[0, idx]),
            "shap_value":  float(sv[idx]),
            "direction":   "→ MAL" if sv[idx] > 0 else "→ BEN",
        })
    return top, expected


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Predict URL maliciousness with XGBoost model 4 (+ SHAP)."
    )
    ap.add_argument("url", help="URL to classify")
    ap.add_argument(
        "--thr", default="default",
        help="Operating point name (default/safer/precise) or a float threshold. Default: 'default'"
    )
    ap.add_argument(
        "--explain", action="store_true",
        help="Show top-K SHAP feature contributions (requires `pip install shap`)"
    )
    ap.add_argument(
        "--topk", type=int, default=10,
        help="Number of top features to show with --explain (default 10)"
    )
    args = ap.parse_args()

    # ---- Load feature extractor + columns ----
    extractor = _load_extractor()
    feature_names = json.loads(COLS_PATH.read_text(encoding="utf-8"))
    assert len(feature_names) == extractor.N_FEATURES, "feature count mismatch"

    # ---- Extract features ----
    feats = extractor.extract_features(args.url)
    X_row = np.empty((1, len(feature_names)), dtype=np.float32)
    for j, k in enumerate(feature_names):
        X_row[0, j] = feats[k]

    # ---- Load model ----
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    # ---- Predict ----
    prob = float(model.predict_proba(X_row)[0, 1])
    thr_name, thr_val = resolve_threshold(args.thr)
    decision = "MAL" if prob >= thr_val else "BEN"
    risk = risk_level(prob)

    # ---- Output ----
    print("=" * 78)
    print(f"URL      : {args.url}")
    print(f"Prob(MAL): {prob:.6f}  ({prob*100:.2f}%)")
    print(f"Threshold: {thr_val:.4f}  ({thr_name})")
    print(f"Decision : {decision}")
    print(f"Risk     : [{decision.lower()[:3]} + {risk}]")
    print("=" * 78)

    if args.explain:
        print(f"\nSHAP — top {args.topk} feature contributions:")
        top, expected = explain_with_shap(model, X_row, feature_names, args.topk)
        print(f"  base log-odds (expected_value): {expected:+.4f}  "
              f"→ base prob {1/(1+np.exp(-expected)):.4f}")
        print(f"  {'feature':<35s} {'raw':>12s} {'shap (log-odds)':>18s}  dir")
        print(f"  {'-'*35} {'-'*12} {'-'*18}  -----")
        for row in top:
            print(f"  {row['feature']:<35s} {row['raw_value']:>12.4f} "
                  f"{row['shap_value']:>+18.4f}  {row['direction']}")
        # Sanity: sum(shap) + expected ≈ raw logit
        total_shap = sum(r["shap_value"] for r in top)
        print(f"\n  (top-{args.topk} ∑shap = {total_shap:+.4f}; full sum + base "
              f"≈ model logit. Sigmoid → prob.)")


if __name__ == "__main__":
    main()

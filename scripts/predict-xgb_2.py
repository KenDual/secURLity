"""
CLI predict using XGBoost model 2.

Usage:
    python "scripts/predict-xgb_2.py" https://example.com
    python "scripts/predict-xgb_2.py" url1 url2 url3
    python "scripts/predict-xgb_2.py" --file urls.txt
    python "scripts/predict-xgb_2.py"                     # interactive

Reuses `extract_features()` từ `scripts/5. feature-extract_2.py` để đảm bảo
feature engineering tại inference time **trùng tuyệt đối** với lúc train.
Cũng so sánh thứ tự cột với `data/processed/xgboost/xgb_feature_cols.json`
để cảnh báo nếu file feature-extract đã bị sửa sau khi train.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(r"D:\! secURLity")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "xgboost"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "xgb_url_2.ubj"
COLS_PATH = DATA_DIR / "xgb_feature_cols.json"
FEATURE_EXTRACT_PATH = SCRIPTS_DIR / "5. feature-extract_2.py"


# ============================================================================
# Risk thresholds (đồng bộ với predict-url_2.py)
# ============================================================================
def risk_level(score: float) -> str:
    pct = score * 100
    if pct <= 25:
        return "Low Risk"
    if pct <= 50:
        return "Caution"
    if pct <= 75:
        return "Suspicious"
    return "High Risk"


# ============================================================================
# Dynamic import of `5. feature-extract_2.py`
# (tên file có space + prefix số nên không import được bằng `import` thường)
# ============================================================================
def load_feature_extractor():
    spec = importlib.util.spec_from_file_location(
        "feature_extract_2", FEATURE_EXTRACT_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {FEATURE_EXTRACT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_features, mod.FEATURE_NAMES


# ============================================================================
# Featurization
# ============================================================================
def featurize_batch(urls, extract_features, feature_names) -> np.ndarray:
    """List[str] -> np.ndarray (N, F) float32 theo đúng thứ tự cột training."""
    n = len(urls)
    f = len(feature_names)
    X = np.empty((n, f), dtype=np.float32)
    for i, url in enumerate(urls):
        feats = extract_features(url)
        for j, k in enumerate(feature_names):
            X[i, j] = feats[k]
    return X


# ============================================================================
# Main
# ============================================================================
def main():
    # ---- Load feature extractor + verify column order ----
    print("Loading feature extractor...", end=" ", flush=True)
    t0 = time.time()
    extract_features, feature_names = load_feature_extractor()
    print(f"done ({time.time() - t0:.2f}s)")

    with open(COLS_PATH, "r", encoding="utf-8") as f:
        cols_saved = json.load(f)
    cols_match = cols_saved == feature_names

    # ---- Load XGBoost model ----
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    # Inference trên CPU: single-row predict bị dominate bởi overhead,
    # không có lợi ích từ GPU. Cũng giúp script chạy được trên máy không có CUDA.
    booster.set_param({"device": "cpu"})

    try:
        best_iter = booster.best_iteration
    except AttributeError:
        best_iter = 0
    iter_range = (0, int(best_iter) + 1) if best_iter and best_iter > 0 else None

    # ---- Diagnostics ----
    print("\n=== Diagnostics (XGBoost model 2) ===")
    print(f"Model            : {MODEL_PATH.name}")
    print(f"Feature dim      : {len(feature_names)}  (saved: {len(cols_saved)})")
    print(f"Column order     : {'OK' if cols_match else 'MISMATCH (see warning below)'}")
    print(f"best_iteration   : {best_iter}")
    print(f"Inference device : CPU")
    print("======================================")
    if not cols_match:
        print("\nWARNING: feature column order in xgb_feature_cols.json differs from")
        print("  current scripts/5. feature-extract_2.py — predictions may be invalid.")
        print("  Re-run feature extraction + retrain, hoặc align lại 2 file này.\n")
    else:
        print()

    def predict(urls):
        X = featurize_batch(urls, extract_features, feature_names)
        return booster.inplace_predict(X, iteration_range=iter_range)

    # ---- CLI: single URL / multiple URLs / file ----
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("urls", nargs="*", help="URLs to scan")
        parser.add_argument("--file", "-f", help="Path to .txt file (one URL per line)")
        args = parser.parse_args()

        urls_to_scan = []
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                urls_to_scan = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(urls_to_scan)} URLs from {args.file}\n")
        elif args.urls:
            urls_to_scan = args.urls

        if urls_to_scan:
            t0 = time.time()
            scores = predict(urls_to_scan)
            elapsed = time.time() - t0
            for url, s in zip(urls_to_scan, scores):
                s = float(s)
                print(f"[{risk_level(s):12s}] {s * 100:6.2f}%  {url}")
            print(f"\n{len(urls_to_scan)} URLs scanned in {elapsed:.3f}s "
                  f"({len(urls_to_scan) / max(elapsed, 1e-9):.1f} URL/s)")
            return

    # ---- Interactive mode ----
    print("Interactive mode. Enter URL (or 'exit'):\n")
    while True:
        try:
            url = input("URL > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not url or url.lower() in ("exit", "quit"):
            break
        s = float(predict([url])[0])
        print(f"Score: {s * 100:6.2f}%  ->  {risk_level(s)}\n")


if __name__ == "__main__":
    main()

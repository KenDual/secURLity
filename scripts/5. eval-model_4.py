"""
Eval model 4 trên TEST set (CNN-LSTM-final.md Stage 4, items 1.3, 1.5, 4.11, 8.8).

Mục đích:
    Báo cáo con số CHÍNH THỨC trên TEST set (chưa chạm trong train/tune).
    Áp dụng 3 operating points đã chọn từ `models/thresholds_4.json` (tune trên VAL):
      * default  : balanced (max F1 trên val)
      * safer    : bias về catch mal (precision ≥ 0.99 trên val)
      * precise  : bias về tránh false alarm (recall ≥ 0.99 trên val)

    Output: `models/results_4.json` (merge per split-mode).

    Không pick threshold ở đây — đó là việc của `6. tune-threshold_4.py` trên VAL.
    Script này CHỈ áp dụng + báo cáo.

Tại sao có script eval-only riêng (không gộp vào train)?
    CLAUDE.md Known issues #1: train script trên Windows crash ở test phase do
    DataLoader workers exhaust handles. Script này dùng `num_workers=0` + chỉ
    load 1 split (mmap, <1GB RAM) nên an toàn.

Usage:
    .\\venv\\Scripts\\Activate.ps1
    python "scripts/5. eval-model_4.py" --split-mode random              # default split=test
    python "scripts/5. eval-model_4.py" --split-mode random --split val  # sanity check val F1 match
    python "scripts/5. eval-model_4.py" --split-mode domain
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT    = Path(r"D:\! secURLity")
PROCESSED_ROOT  = PROJECT_ROOT / "data" / "processed" / "model_4"
MODELS_ROOT     = PROJECT_ROOT / "models"
TRAIN_SCRIPT    = PROJECT_ROOT / "scripts" / "4. train-model_4.py"
THRESHOLDS_JSON = MODELS_ROOT / "thresholds_4.json"
RESULTS_JSON    = MODELS_ROOT / "results_4.json"


# ============================================================================
# Helpers
# ============================================================================

def import_train_module():
    """Load CNNLSTM + URLDataset từ file có tên awkward (space + leading digit)."""
    spec = importlib.util.spec_from_file_location("train_model_4", str(TRAIN_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không load được module từ {TRAIN_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def predict_split(model, loader, device):
    """Inference → trả về (probs np.float32, labels np.int64)."""
    model.eval()
    all_probs = []
    all_labels = []
    use_amp = device.type == "cuda"
    for x_seq, x_feat, y in loader:
        x_seq  = x_seq.to(device, non_blocking=True)
        x_feat = x_feat.to(device, non_blocking=True)
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x_seq, x_feat)
        else:
            logits = model(x_seq, x_feat)
        # Sigmoid trên FP32 — tránh FP16 overflow ở edge
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy().astype(np.int64))
    return np.concatenate(all_probs), np.concatenate(all_labels)


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float):
    """Tính P/R/F1/FPR + confusion matrix tại 1 threshold cụ thể."""
    preds = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "fpr": float(fpr),
    }


def prediction_histogram(probs: np.ndarray, labels: np.ndarray):
    """Histogram theo bins [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] — check calibration."""
    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 + 1e-9]
    hist_all = np.histogram(probs, bins=bins)[0]
    hist_pos = np.histogram(probs[labels == 1], bins=bins)[0]
    hist_neg = np.histogram(probs[labels == 0], bins=bins)[0]
    labels_str = ["[0.00,0.10)", "[0.10,0.25)", "[0.25,0.50)",
                  "[0.50,0.75)", "[0.75,0.90)", "[0.90,1.00]"]
    return {
        "bin_labels": labels_str,
        "all":        [int(x) for x in hist_all],
        "label_1":    [int(x) for x in hist_pos],
        "label_0":    [int(x) for x in hist_neg],
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-mode", choices=["random", "domain"], required=True,
                        help="Checkpoint nào — random hoặc domain split.")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Default 'test' — con số chính thức. 'val' chỉ để sanity-check best_f1 từ training.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Default 0 — inference GPU-bound. Per CLAUDE.md #1 + workaround pickle import.")
    parser.add_argument("--fallback-threshold", type=float, default=0.5,
                        help="Threshold dùng khi thresholds_4.json chưa có entry cho mode này.")
    args = parser.parse_args()

    # ----- Paths
    ckpt_path = MODELS_ROOT / f"cnn_lstm_best_4_{args.split_mode}.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] Không tìm thấy checkpoint: {ckpt_path}")
        print(f"        Chạy train trước: python \"scripts/4. train-model_4.py\" --split-mode {args.split_mode}")
        sys.exit(2)

    mode_root = PROCESSED_ROOT / args.split_mode
    split_dir = mode_root / args.split
    for fname in ("X.npy", "y.npy", "feat.npy"):
        if not (split_dir / fname).exists():
            print(f"[ERROR] Thiếu {split_dir / fname}.")
            sys.exit(2)

    # ----- Load thresholds (operating points) ----------------------
    operating_points = None
    if THRESHOLDS_JSON.exists():
        try:
            thr_data = json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8"))
            mode_data = thr_data.get(args.split_mode)
            if mode_data and "operating_points" in mode_data:
                operating_points = mode_data["operating_points"]
        except json.JSONDecodeError:
            pass

    if operating_points is None:
        print(f"[WARN] thresholds_4.json chưa có entry cho mode='{args.split_mode}'.")
        print(f"       Sẽ chỉ eval ở fallback threshold = {args.fallback_threshold}.")
        print(f"       Chạy '6. tune-threshold_4.py' trước để có 3 operating points.")
        operating_points = {
            "default": {"threshold": args.fallback_threshold}
        }

    # ----- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Eval model 4 — split_mode={args.split_mode}, split={args.split}")
    print("=" * 70)
    print(f"Device       : {device}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Split        : {split_dir}")
    print(f"Operating pts: {list(operating_points.keys())}")

    # ----- Import + build model
    print("\n[1/5] Import CNNLSTM + URLDataset từ train script...")
    train_mod = import_train_module()
    CNNLSTM = train_mod.CNNLSTM
    URLDataset = train_mod.URLDataset

    print("[2/5] Load checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    val_metrics_train = ckpt.get("val_metrics", {})
    print(f"  epoch         : {ckpt['epoch']}")
    print(f"  best_val_f1   : {ckpt['best_f1']:.6f}")
    print(f"  max_len       : {cfg['max_len']}")
    print(f"  vocab_size    : {cfg['vocab_size']}")
    print(f"  n_features    : {cfg['n_features']}")

    model = CNNLSTM(
        vocab_size=cfg["vocab_size"],
        n_features=cfg["n_features"],
        pad_idx=cfg["pad_idx"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params        : {n_params:,}")

    # ----- Dataset + loader
    print("[3/5] Load split + inference...")
    ds = URLDataset(
        split_dir / "X.npy",
        split_dir / "y.npy",
        split_dir / "feat.npy",
    )
    print(f"  n_samples     : {len(ds):,}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    t0 = time.time()
    probs, labels = predict_split(model, loader, device)
    dt = time.time() - t0
    print(f"  inference     : {dt:.1f}s  ({len(ds) / max(dt, 1e-6):.0f} URLs/s)")
    print(f"  probs range   : [{probs.min():.6f}, {probs.max():.6f}]")
    print(f"  probs mean    : {probs.mean():.6f}")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"  label distr   : pos={n_pos:,} / neg={n_neg:,}")

    # ----- Curve-level (threshold-independent)
    print("\n[4/5] Curve metrics (threshold-independent)...")
    roc_auc_val = float(roc_auc_score(labels, probs))
    pr_auc_val  = float(average_precision_score(labels, probs))
    print(f"  ROC-AUC       : {roc_auc_val:.6f}")
    print(f"  PR-AUC        : {pr_auc_val:.6f}")

    # ----- Metrics tại mỗi operating point
    print("\n[5/5] Metrics tại mỗi operating point...")
    op_results = {}
    print(f"  {'Mode':<8} {'Thr':>6} {'F1':>10} {'Prec':>10} {'Recall':>10} {'FPR':>12} "
          f"{'TP':>10} {'FP':>8} {'FN':>8} {'TN':>12}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12} "
          f"{'-'*10} {'-'*8} {'-'*8} {'-'*12}")
    for name, op in operating_points.items():
        thr = float(op["threshold"])
        m = metrics_at_threshold(probs, labels, thr)
        op_results[name] = m
        print(f"  {name:<8} {m['threshold']:>6.2f} {m['f1']:>10.6f} {m['precision']:>10.6f} "
              f"{m['recall']:>10.6f} {m['fpr']:>12.2e} {m['tp']:>10,} {m['fp']:>8,} "
              f"{m['fn']:>8,} {m['tn']:>12,}")

    # ----- Histogram (calibration)
    histogram = prediction_histogram(probs, labels)
    print("\n  Prediction histogram (target: spread, KHÔNG U-shape):")
    print(f"  {'Bin':<14} {'All':>14} {'Label 1':>14} {'Label 0':>14}")
    for i, blab in enumerate(histogram["bin_labels"]):
        print(f"  {blab:<14} {histogram['all'][i]:>14,} "
              f"{histogram['label_1'][i]:>14,} {histogram['label_0'][i]:>14,}")

    # ----- Save results (merge với mode khác nếu file đã có)
    print("\n  Save results...")
    existing = {}
    if RESULTS_JSON.exists():
        try:
            existing = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    mode_entry = existing.get(args.split_mode, {})
    mode_entry["split_mode"] = args.split_mode
    mode_entry["checkpoint"] = str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    mode_entry["checkpoint_epoch"] = int(ckpt["epoch"])
    mode_entry["checkpoint_best_val_f1"] = float(ckpt["best_f1"])
    mode_entry["config"] = cfg
    mode_entry["n_params"] = int(n_params)
    mode_entry["val_metrics_from_training"] = val_metrics_train
    # Per-split sub-dict
    mode_entry[args.split] = {
        "n_samples": int(len(ds)),
        "n_positives": int(n_pos),
        "n_negatives": int(n_neg),
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "operating_points": op_results,
        "prediction_histogram": histogram,
        "inference_seconds": float(dt),
        "inference_urls_per_sec": float(len(ds) / max(dt, 1e-6)),
    }
    existing[args.split_mode] = mode_entry
    RESULTS_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"  results JSON saved → {RESULTS_JSON}")

    # ----- Sanity check: nếu --split val, so sánh với best_f1 trong checkpoint
    if args.split == "val" and "default" in op_results:
        diff = abs(op_results["default"]["f1"] - ckpt["best_f1"])
        print(f"\n  [Sanity] Val F1 từ eval = {op_results['default']['f1']:.6f}, "
              f"từ checkpoint = {ckpt['best_f1']:.6f}, diff = {diff:.2e}")
        if diff > 1e-3:
            print(f"  [WARN] Diff > 1e-3 — kiểm tra threshold (default tune trên val) "
                  f"hoặc data drift?")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()

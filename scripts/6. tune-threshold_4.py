"""
Item 7 — Threshold tuning post-train (CNN-LSTM-final.md Stage 4).

Mục đích:
    Sau khi model train xong (Val F1 max), tìm 3 threshold operating points:
      * default  : threshold tối đa F1
      * safer    : threshold thấp nhất sao cho recall ≥ 0.99   (bắt nhiều mal, chấp nhận FP)
      * precise  : threshold cao nhất sao cho precision ≥ 0.99 (tránh false alarm, chấp nhận FN)

    Output: models/thresholds_4.json  (per split-mode: random / domain)
    Optional plots: models/plots/{pr,roc,sweep}_curve_4_<mode>.png

Tại sao chạy trên VAL chứ không phải TEST?
    Test = thước đo cuối cùng, không được nhìn để tune bất cứ gì (kể cả threshold).
    Val đã được model thấy qua early-stop, nhưng KHÔNG fit weights → dùng để pick threshold
    là chuẩn ML practice.

Usage:
    .\\venv\\Scripts\\Activate.ps1
    python "scripts/6. tune-threshold_4.py" --split-mode random
    python "scripts/6. tune-threshold_4.py" --split-mode domain
    python "scripts/6. tune-threshold_4.py" --split-mode random --split test   # chỉ để debug, không pick threshold từ test
    python "scripts/6. tune-threshold_4.py" --split-mode random --no-plots
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
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT   = Path(r"D:\! secURLity")
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "model_4"
MODELS_ROOT    = PROJECT_ROOT / "models"
PLOTS_ROOT     = MODELS_ROOT / "plots"
TRAIN_SCRIPT   = PROJECT_ROOT / "scripts" / "4. train-model_4.py"
THRESHOLDS_JSON = MODELS_ROOT / "thresholds_4.json"


# ============================================================================
# Import CNNLSTM + URLDataset từ train script (tên file có space + bắt đầu bằng số)
# ============================================================================

def import_train_module():
    spec = importlib.util.spec_from_file_location("train_model_4", str(TRAIN_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không load được module từ {TRAIN_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Inference: chạy model trên 1 split, trả về (probs, labels)
# ============================================================================

@torch.no_grad()
def predict_split(model, loader, device):
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
        # sigmoid trên FP32 để tránh overflow ở edge (logits = ±20 → fp16 inf)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ============================================================================
# Threshold sweep (vectorized)
# ============================================================================

def sweep_metrics(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray):
    """
    Với mỗi threshold, tính TP/FP/FN/TN/P/R/F1/FPR.
    Vectorized — không loop từng threshold (sẽ chậm trên 2M samples).
    """
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    # Sort theo prob desc — TP/FP cumulative
    order = np.argsort(-probs, kind="stable")
    probs_sorted  = probs[order]
    labels_sorted = labels[order]

    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(~labels_sorted)

    rows = []
    for thr in thresholds:
        # Số sample có prob >= thr
        # probs_sorted desc → tìm vị trí cuối cùng prob >= thr
        # np.searchsorted yêu cầu sorted asc, dùng -probs_sorted
        k = int(np.searchsorted(-probs_sorted, -thr, side="right"))
        if k == 0:
            tp = fp = 0
        else:
            tp = int(tp_cum[k - 1])
            fp = int(fp_cum[k - 1])
        fn = n_pos - tp
        tn = n_neg - fp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append({
            "threshold": float(thr),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
        })
    return rows


def pick_operating_points(rows, prec_floor: float = 0.99, recall_floor: float = 0.99):
    """
    Pick 3 threshold operating points cho 3 use-case:

      default : max F1 — "trade-off cân bằng nhất theo dữ liệu".

      safer   : threshold THẤP NHẤT mà precision ≥ prec_floor.
                Diễn giải: "tôi sẵn sàng hạ threshold để catch nhiều mal hơn,
                miễn là precision không tụt dưới 0.99".
                → recall tối đa hóa, với precision floor.

      precise : threshold CAO NHẤT mà recall ≥ recall_floor.
                Diễn giải: "tôi sẵn sàng nâng threshold để tránh false alarm,
                miễn là recall không tụt dưới 0.99".
                → precision tối đa hóa, với recall floor.

    Fallback khi không có row nào thỏa mãn constraint:
      safer   → row có recall cao nhất (degrade precision floor)
      precise → row có precision cao nhất (degrade recall floor)
    """
    # default — max F1
    best = max(rows, key=lambda r: r["f1"])

    # safer — precision ≥ floor, lowest threshold (= max recall)
    safer_candidates = [r for r in rows if r["precision"] >= prec_floor]
    if safer_candidates:
        safer = min(safer_candidates, key=lambda r: r["threshold"])
    else:
        safer = max(rows, key=lambda r: r["recall"])

    # precise — recall ≥ floor, highest threshold (= max precision)
    precise_candidates = [r for r in rows if r["recall"] >= recall_floor]
    if precise_candidates:
        precise = max(precise_candidates, key=lambda r: r["threshold"])
    else:
        precise = max(rows, key=lambda r: r["precision"])

    return {"default": best, "safer": safer, "precise": precise}


# ============================================================================
# Plots (optional — graceful skip nếu thiếu matplotlib)
# ============================================================================

def try_plots(probs, labels, rows, ops, out_dir: Path, mode_tag: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib không có — bỏ qua plot.")
        print("       Cài: pip install matplotlib  (nếu muốn xuất PR/ROC PNG)")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    # PR curve
    p, r, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r, p, label=f"PR (AP={ap:.6f})")
    for name, op in ops.items():
        ax.scatter(op["recall"], op["precision"], s=60, label=f"{name} @ thr={op['threshold']:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve — {mode_tag}")
    ax.set_xlim(0.95, 1.001); ax.set_ylim(0.95, 1.001)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_curve_4_{mode_tag}.png", dpi=120)
    plt.close(fig)

    # ROC curve
    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, label=f"ROC (AUC={auc:.6f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    for name, op in ops.items():
        ax.scatter(op["fpr"], op["recall"], s=60, label=f"{name} @ thr={op['threshold']:.2f}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR (Recall)")
    ax.set_title(f"ROC curve — {mode_tag}")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_curve_4_{mode_tag}.png", dpi=120)
    plt.close(fig)

    # Threshold sweep (F1 / P / R / FPR theo threshold)
    thrs = [r["threshold"] for r in rows]
    f1s  = [r["f1"]        for r in rows]
    ps   = [r["precision"] for r in rows]
    rs   = [r["recall"]    for r in rows]
    fprs = [r["fpr"]       for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, f1s, label="F1")
    ax.plot(thrs, ps,  label="Precision")
    ax.plot(thrs, rs,  label="Recall")
    ax.plot(thrs, fprs, label="FPR", alpha=0.5)
    for name, op in ops.items():
        ax.axvline(op["threshold"], linestyle="--", alpha=0.5, label=f"{name}={op['threshold']:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title(f"Threshold sweep — {mode_tag}")
    ax.grid(True, alpha=0.3); ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"sweep_curve_4_{mode_tag}.png", dpi=120)
    plt.close(fig)

    print(f"[PLOT] PNG đã lưu vào {out_dir}")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-mode", choices=["random", "domain"], required=True,
                        help="Checkpoint nào? random hoặc domain")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Split nào? default=val (chuẩn ML). test chỉ để debug.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Default 0 — inference GPU-bound, multi-worker dễ vướng pickle issue "
                             "vì module load qua importlib không có ở worker.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--thr-min", type=float, default=0.01)
    parser.add_argument("--thr-max", type=float, default=0.99)
    parser.add_argument("--thr-step", type=float, default=0.01)
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

    # ----- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Threshold tuning — split_mode={args.split_mode}, split={args.split}")
    print("=" * 70)
    print(f"Device       : {device}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Split        : {split_dir}")

    # ----- Import model + dataset
    print("\n[1/5] Import CNNLSTM + URLDataset từ train script...")
    train_mod = import_train_module()
    CNNLSTM = train_mod.CNNLSTM
    URLDataset = train_mod.URLDataset

    # ----- Load checkpoint
    print("[2/5] Load checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    print(f"  epoch       : {ckpt['epoch']}")
    print(f"  best_f1     : {ckpt['best_f1']:.6f}")
    print(f"  config      : {cfg}")

    model = CNNLSTM(
        vocab_size=cfg["vocab_size"],
        n_features=cfg["n_features"],
        pad_idx=cfg["pad_idx"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  params      : {sum(p.numel() for p in model.parameters()):,}")

    # ----- Dataset + loader
    print("[3/5] Load dataset + inference...")
    ds = URLDataset(
        split_dir / "X.npy",
        split_dir / "y.npy",
        split_dir / "feat.npy",
    )
    print(f"  n_samples   : {len(ds):,}")
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
    print(f"  inference   : {dt:.1f}s  ({len(ds) / max(dt, 1e-6):.0f} URLs/s)")
    print(f"  probs range : [{probs.min():.6f}, {probs.max():.6f}]")
    print(f"  probs mean  : {probs.mean():.6f}")
    print(f"  label distr : pos={int(labels.sum()):,} / neg={int((1 - labels).sum()):,}")

    # ----- Sweep
    print("[4/5] Threshold sweep...")
    thresholds = np.round(np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step), 4)
    print(f"  n_thresholds: {len(thresholds)} từ {thresholds[0]:.2f} → {thresholds[-1]:.2f}")
    rows = sweep_metrics(probs, labels, thresholds)
    ops  = pick_operating_points(rows)

    # ----- Curve-level metrics (independent of threshold)
    roc_auc_val = float(roc_auc_score(labels, probs))
    pr_auc_val  = float(average_precision_score(labels, probs))

    # ----- In bảng
    print()
    print(f"  Curve metrics : ROC-AUC={roc_auc_val:.6f}  PR-AUC={pr_auc_val:.6f}")
    print()
    print(f"  {'Mode':<8} {'Thr':>6} {'F1':>8} {'Prec':>8} {'Recall':>8} {'FPR':>10} {'TP':>8} {'FP':>6} {'FN':>6} {'TN':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*10}")
    for name, op in ops.items():
        print(f"  {name:<8} {op['threshold']:>6.2f} {op['f1']:>8.6f} {op['precision']:>8.6f} "
              f"{op['recall']:>8.6f} {op['fpr']:>10.2e} {op['tp']:>8,} {op['fp']:>6,} "
              f"{op['fn']:>6,} {op['tn']:>10,}")

    # ----- Save JSON (merge với mode khác nếu file đã có)
    print("\n[5/5] Save thresholds JSON + (optional) plots...")
    existing = {}
    if THRESHOLDS_JSON.exists():
        try:
            existing = json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    mode_tag = f"{args.split_mode}_{args.split}" if args.split != "val" else args.split_mode
    existing[mode_tag] = {
        "split_mode": args.split_mode,
        "split": args.split,
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_best_f1": float(ckpt["best_f1"]),
        "n_samples": int(len(ds)),
        "n_positives": int(labels.sum()),
        "n_negatives": int((1 - labels).sum()),
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "operating_points": {name: op for name, op in ops.items()},
        "sweep_summary": {
            "thr_min":  float(thresholds[0]),
            "thr_max":  float(thresholds[-1]),
            "thr_step": float(args.thr_step),
            "n_points": int(len(thresholds)),
        },
    }
    THRESHOLDS_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"  thresholds JSON saved → {THRESHOLDS_JSON}")

    # ----- Plots
    if not args.no_plots:
        try_plots(probs, labels, rows, ops, PLOTS_ROOT, mode_tag)

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()

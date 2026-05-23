"""
Item 11 — Export model 4 sang ONNX FP32 + dynamic INT8 quantization (Stage 6).

Steps:
  1. Load best checkpoint (default: random — production candidate)
  2. Build CNNLSTM (eval mode) + load state_dict
  3. Export ONNX FP32 với dynamic batch axis
  4. Sanity check: ONNX FP32 output vs PyTorch FP32 (max diff < 1e-4 trên 10 URL)
  5. Dynamic INT8 quantization
  6. Benchmark CPU latency: PyTorch FP32 vs ONNX FP32 vs ONNX INT8 (200 URL warmup + 1000 timed)
  7. Accuracy regression: F1 trên test set không drop > 0.5%
  8. Save mọi metric vào `models/results_4.json` dưới key `<split_mode>.deployment`

Usage:
    .\\venv\\Scripts\\Activate.ps1

    # Export random checkpoint (production default)
    python "scripts/8. export-onnx_4.py" --split-mode random

    # Export domain checkpoint (out-of-dist deployment)
    python "scripts/8. export-onnx_4.py" --split-mode domain

    # Skip INT8 nếu chỉ test FP32
    python "scripts/8. export-onnx_4.py" --split-mode random --no-int8

    # Skip benchmark (chỉ export)
    python "scripts/8. export-onnx_4.py" --split-mode random --no-benchmark

    # Skip accuracy regression (skip test set load)
    python "scripts/8. export-onnx_4.py" --split-mode random --no-regression
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
TRAIN_SCRIPT   = PROJECT_ROOT / "scripts" / "4. train-model_4.py"
RESULTS_JSON   = MODELS_ROOT / "results_4.json"


# ============================================================================
# Helpers
# ============================================================================

def import_train_module():
    spec = importlib.util.spec_from_file_location("train_model_4", str(TRAIN_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_model_4"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(split_mode: str, device: torch.device):
    """Load checkpoint + build CNNLSTM (eval mode). Return (model, cfg, ckpt_meta)."""
    train_mod = import_train_module()
    CNNLSTM = train_mod.CNNLSTM

    ckpt_path = MODELS_ROOT / f"cnn_lstm_best_4_{split_mode}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = CNNLSTM(
        vocab_size=cfg["vocab_size"],
        n_features=cfg["n_features"],
        pad_idx=cfg["pad_idx"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


# ============================================================================
# Wrapper exposes attention weights as 2nd output (Item 12 for ONNX deployment)
# ============================================================================

class CNNLSTMWithAttention(torch.nn.Module):
    """
    Wrap base CNNLSTM, replicate forward nhưng expose attention weights.
    Mục đích: ONNX cần explicit graph output cho mỗi tensor ta muốn lấy;
    attention weights trong `attn_pool.forward()` gốc bị discard nội bộ.

    Output: (logit, attn_weights)  với attn_weights shape (B, L/4).
    """

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x_seq: torch.Tensor, x_feat: torch.Tensor):
        b = self.base
        # Sao chép forward() của CNNLSTM gốc, nhưng expose attention weights.
        pad_mask = (x_seq != b.pad_idx).float().unsqueeze(1)
        x = b.embed(x_seq)
        x = x.transpose(1, 2)
        x = F.relu(b.conv1(x))
        x = F.max_pool1d(x, 2)
        pad_mask = F.max_pool1d(pad_mask, 2)
        x = F.relu(b.conv2(x))
        x = F.max_pool1d(x, 2)
        pad_mask = F.max_pool1d(pad_mask, 2)
        x = x.transpose(1, 2)
        x, _ = b.lstm(x)
        pad_mask_bool = pad_mask.squeeze(1).bool()

        # Inline AttentionPool — expose weights
        scores = b.attn_pool.attn(x).squeeze(-1)
        scores = scores.masked_fill(~pad_mask_bool, float("-inf"))
        attn_weights = F.softmax(scores, dim=1)  # (B, L/4) — output thứ 2
        h_seq = (x * attn_weights.unsqueeze(-1)).sum(dim=1)

        h_feat = b.feat_mlp(x_feat)
        h = torch.cat([h_seq, h_feat], dim=1)
        logit = b.head(h).squeeze(-1)
        return logit, attn_weights


# ============================================================================
# Export
# ============================================================================

def export_fp32(model, cfg, onnx_path: Path, device: torch.device) -> None:
    """Export model với 2 outputs: logit + attention_weights (cho --explain)."""
    print(f"\n[Export FP32 — 2 outputs: logit + attention_weights]")
    print(f"  Output: {onnx_path}")
    # Wrap base model để expose attention weights làm output thứ 2
    wrapped = CNNLSTMWithAttention(model).to(device).eval()

    dummy_seq = torch.zeros((1, cfg["max_len"]), dtype=torch.int64, device=device)
    dummy_feat = torch.zeros((1, cfg["n_features"]), dtype=torch.float32, device=device)

    # Verify wrapper trả output đúng shape trước export
    with torch.no_grad():
        out = wrapped(dummy_seq, dummy_feat)
    assert isinstance(out, tuple) and len(out) == 2, \
        f"Wrapper expects 2 outputs, got {type(out)}"
    print(f"  Wrapper outputs: logit={tuple(out[0].shape)}, "
          f"attn_weights={tuple(out[1].shape)}")

    # Dùng LEGACY exporter (dynamo=False) — torch 2.11 dynamo exporter có bug
    # với dynamic batch + LSTM (hardcoded reshape, output 0.02MB → mất weights).
    # Legacy TorchScript-based exporter handles LSTM + dynamic batch tốt hơn.
    # opset 17 — đủ cho LSTM + Linear + Conv1d + softmax mask trong attention.
    torch.onnx.export(
        wrapped,
        (dummy_seq, dummy_feat),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x_seq", "x_feat"],
        output_names=["logit", "attention_weights"],
        dynamic_axes={
            "x_seq":            {0: "batch"},
            "x_feat":           {0: "batch"},
            "logit":            {0: "batch"},
            "attention_weights": {0: "batch"},
        },
        dynamo=False,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"\n[Quantize INT8 dynamic]")
    print(f"  In : {fp32_path}")
    print(f"  Out: {int8_path}")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        # Op types để quantize — LSTM ONNX không phải lúc nào cũng quantize được tốt,
        # nhưng dynamic quant ít rủi ro hơn static
    )
    size_mb = int8_path.stat().st_size / (1024 * 1024)
    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB  (FP32 {fp32_mb:.2f} MB → INT8 {size_mb:.2f} MB, "
          f"compression {fp32_mb/size_mb:.2f}x)")


# ============================================================================
# Sanity (FP32 ONNX == PyTorch FP32)
# ============================================================================

def sanity_check(model, cfg, onnx_path: Path, device: torch.device,
                 n_samples: int = 10) -> dict:
    import onnxruntime as ort
    print(f"\n[Sanity check FP32 ONNX vs PyTorch FP32 — n={n_samples}]")

    rng = np.random.default_rng(seed=42)
    max_len = cfg["max_len"]
    n_feat  = cfg["n_features"]
    vocab   = cfg["vocab_size"]

    x_seq_np  = rng.integers(0, vocab, size=(n_samples, max_len), dtype=np.int64)
    x_feat_np = rng.standard_normal((n_samples, n_feat)).astype(np.float32)

    # PyTorch reference (FP32 trên CPU)
    model_cpu = model.cpu().float()
    with torch.no_grad():
        logit_pt = model_cpu(
            torch.from_numpy(x_seq_np), torch.from_numpy(x_feat_np)
        ).cpu().numpy()

    # ONNX (output[0]=logit, output[1]=attention_weights)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"x_seq": x_seq_np, "x_feat": x_feat_np})
    logit_ox = outs[0]
    attn_ox  = outs[1] if len(outs) > 1 else None
    if attn_ox is not None:
        print(f"  attention output shape: {attn_ox.shape} "
              f"(expect (N, L/4) = ({n_samples}, {cfg['max_len']//4}))")
        # sum per row phải gần 1 (softmax) — nhưng vì input random với toàn-token (no pad),
        # mask không kích hoạt và sum=1 chính xác.
        sums = attn_ox.sum(axis=1)
        print(f"  attention row-sums: min={sums.min():.4f}, max={sums.max():.4f} (expect ~1.0)")

    max_diff = float(np.max(np.abs(logit_pt - logit_ox)))
    mean_diff = float(np.mean(np.abs(logit_pt - logit_ox)))
    print(f"  max_diff  = {max_diff:.6f}  (target < 1e-4)")
    print(f"  mean_diff = {mean_diff:.6f}")
    ok = max_diff < 1e-4
    print(f"  Status    = {'PASS' if ok else 'FAIL (numerical drift!)'}")

    # Restore device
    if device.type == "cuda":
        model.cuda()
    return {"max_diff": max_diff, "mean_diff": mean_diff, "pass": ok}


# ============================================================================
# Benchmark (CPU)
# ============================================================================

def make_benchmark_inputs(cfg, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x_seq = rng.integers(0, cfg["vocab_size"], size=(n, cfg["max_len"]), dtype=np.int64)
    x_feat = rng.standard_normal((n, cfg["n_features"])).astype(np.float32)
    return x_seq, x_feat


def bench_pytorch_cpu(model, cfg, n_warmup: int = 200, n_timed: int = 1000) -> dict:
    """Benchmark batch=1 inference (URL/s, ms/URL) PyTorch FP32 trên CPU."""
    print(f"\n[Bench PyTorch FP32 CPU]")
    model_cpu = model.cpu().float().eval()
    torch.set_num_threads(1)
    x_seq, x_feat = make_benchmark_inputs(cfg, n_warmup + n_timed, seed=1)
    x_seq_t = torch.from_numpy(x_seq)
    x_feat_t = torch.from_numpy(x_feat)

    with torch.no_grad():
        # Warmup
        for i in range(n_warmup):
            _ = model_cpu(x_seq_t[i:i+1], x_feat_t[i:i+1])
        # Timed
        t0 = time.perf_counter()
        for i in range(n_warmup, n_warmup + n_timed):
            _ = model_cpu(x_seq_t[i:i+1], x_feat_t[i:i+1])
        dt = time.perf_counter() - t0

    ms_per = dt * 1000 / n_timed
    urls_per_s = n_timed / dt
    print(f"  n={n_timed} (single-URL), {ms_per:.3f} ms/URL, {urls_per_s:,.0f} URL/s")
    return {"ms_per_url": ms_per, "urls_per_sec": urls_per_s, "threads": 1}


def bench_onnx_cpu(onnx_path: Path, cfg, label: str,
                   n_warmup: int = 200, n_timed: int = 1000) -> dict:
    import onnxruntime as ort
    print(f"\n[Bench {label} CPU]")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])

    x_seq, x_feat = make_benchmark_inputs(cfg, n_warmup + n_timed, seed=2)

    # Warmup
    for i in range(n_warmup):
        sess.run(None, {"x_seq": x_seq[i:i+1], "x_feat": x_feat[i:i+1]})
    # Timed
    t0 = time.perf_counter()
    for i in range(n_warmup, n_warmup + n_timed):
        sess.run(None, {"x_seq": x_seq[i:i+1], "x_feat": x_feat[i:i+1]})
    dt = time.perf_counter() - t0

    ms_per = dt * 1000 / n_timed
    urls_per_s = n_timed / dt
    print(f"  n={n_timed} (single-URL), {ms_per:.3f} ms/URL, {urls_per_s:,.0f} URL/s")
    return {"ms_per_url": ms_per, "urls_per_sec": urls_per_s, "threads": 1}


# ============================================================================
# Accuracy regression (F1 trên test set: PyTorch vs ONNX FP32 vs ONNX INT8)
# ============================================================================

def accuracy_regression(model, cfg, onnx_fp32: Path, onnx_int8: Path | None,
                        split_mode: str, threshold: float,
                        n_samples: int = 50_000) -> dict:
    """
    Đo F1 trên N samples đầu của test set:
      - PyTorch FP32 (GPU nếu có)
      - ONNX FP32 (CPU)
      - ONNX INT8 (CPU) — nếu có
    Yêu cầu: |F1_int8 - F1_fp32| < 0.005
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    import onnxruntime as ort

    print(f"\n[Accuracy regression — N={n_samples:,} từ test set]")
    test_dir = PROCESSED_ROOT / split_mode / "test"
    X = np.load(test_dir / "X.npy", mmap_mode="r")[:n_samples].astype(np.int64)
    y = np.load(test_dir / "y.npy", mmap_mode="r")[:n_samples].astype(np.int64)
    feat = np.load(test_dir / "feat.npy", mmap_mode="r")[:n_samples].astype(np.float32)
    print(f"  Loaded test slice: X={X.shape}, feat={feat.shape}, y={y.shape}")

    # ---- PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batch = 4096
    probs_pt = []
    with torch.no_grad():
        for i in range(0, n_samples, batch):
            xs = torch.from_numpy(X[i:i+batch]).to(device)
            xf = torch.from_numpy(feat[i:i+batch]).to(device)
            logits = model(xs, xf)
            probs_pt.append(torch.sigmoid(logits.float()).cpu().numpy())
    probs_pt = np.concatenate(probs_pt)
    pred_pt = (probs_pt >= threshold).astype(np.int64)
    f1_pt = float(f1_score(y, pred_pt))
    print(f"  PyTorch FP32 (device={device}):  F1 = {f1_pt:.6f}")

    out = {"threshold": float(threshold), "n_samples": int(n_samples),
           "pytorch_fp32_f1": f1_pt}

    # ---- ONNX FP32
    def run_onnx(path: Path) -> np.ndarray:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out_probs = []
        for i in range(0, n_samples, batch):
            o = sess.run(None, {
                "x_seq":  X[i:i+batch],
                "x_feat": feat[i:i+batch],
            })[0]
            out_probs.append(1.0 / (1.0 + np.exp(-o.astype(np.float64))))
        return np.concatenate(out_probs).astype(np.float32)

    probs_fp32 = run_onnx(onnx_fp32)
    pred_fp32 = (probs_fp32 >= threshold).astype(np.int64)
    f1_fp32 = float(f1_score(y, pred_fp32))
    print(f"  ONNX FP32 (CPU):                 F1 = {f1_fp32:.6f}  "
          f"(Δ vs PyTorch = {f1_fp32 - f1_pt:+.6f})")
    out["onnx_fp32_f1"] = f1_fp32
    out["onnx_fp32_delta"] = f1_fp32 - f1_pt

    if onnx_int8 and onnx_int8.exists():
        probs_int8 = run_onnx(onnx_int8)
        pred_int8 = (probs_int8 >= threshold).astype(np.int64)
        f1_int8 = float(f1_score(y, pred_int8))
        delta_int8 = f1_int8 - f1_fp32
        print(f"  ONNX INT8 (CPU):                 F1 = {f1_int8:.6f}  "
              f"(Δ vs ONNX FP32 = {delta_int8:+.6f})")
        out["onnx_int8_f1"] = f1_int8
        out["onnx_int8_delta_vs_fp32"] = delta_int8
        out["int8_regression_pass"] = abs(delta_int8) < 0.005

    return out


# ============================================================================
# Save results to results_4.json
# ============================================================================

def save_deployment_results(split_mode: str, payload: dict) -> None:
    existing = {}
    if RESULTS_JSON.exists():
        try:
            existing = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    mode_entry = existing.get(split_mode, {})
    mode_entry["deployment"] = payload
    existing[split_mode] = mode_entry
    RESULTS_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\n  results saved → {RESULTS_JSON}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--split-mode", choices=("random", "domain"), default="random")
    ap.add_argument("--no-int8", action="store_true",
                    help="Skip INT8 quantization (chỉ FP32)")
    ap.add_argument("--no-benchmark", action="store_true",
                    help="Skip CPU latency benchmark")
    ap.add_argument("--no-regression", action="store_true",
                    help="Skip accuracy regression test")
    ap.add_argument("--regression-n", type=int, default=50_000,
                    help="Số samples test cho accuracy regression. Default 50k.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Threshold dùng cho regression test. Default: đọc từ thresholds_4.json.")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Export model 4 → ONNX  ·  split_mode = {args.split_mode}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device for export: {device}")

    # ----- Load checkpoint + build model
    model, cfg, ckpt = build_model(args.split_mode, device)
    print(f"  Checkpoint: cnn_lstm_best_4_{args.split_mode}.pt  "
          f"(epoch {ckpt['epoch']}, val F1={ckpt['best_f1']:.4f})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    print(f"  Config: max_len={cfg['max_len']}, vocab={cfg['vocab_size']}, "
          f"n_feat={cfg['n_features']}")

    # ----- Output paths
    onnx_fp32 = MODELS_ROOT / f"cnn_lstm_4_{args.split_mode}.onnx"
    onnx_int8 = MODELS_ROOT / f"cnn_lstm_4_{args.split_mode}_int8.onnx"

    # ----- Resolve threshold
    if args.threshold is None:
        thr_json = MODELS_ROOT / "thresholds_4.json"
        if thr_json.exists():
            data = json.loads(thr_json.read_text(encoding="utf-8"))
            op = data.get(args.split_mode, {}).get("operating_points", {}).get("default", {})
            threshold = float(op.get("threshold", 0.5))
        else:
            threshold = 0.5
    else:
        threshold = args.threshold
    print(f"  Threshold (for regression test): {threshold:.4f}")

    # ----- 1. Export FP32
    export_fp32(model, cfg, onnx_fp32, device)

    # ----- 2. Sanity check FP32
    sanity_fp32 = sanity_check(model, cfg, onnx_fp32, device)

    # ----- 3. INT8 quantize
    int8_done = False
    if not args.no_int8:
        try:
            quantize_int8(onnx_fp32, onnx_int8)
            int8_done = True
        except Exception as e:
            print(f"  [WARN] INT8 quantize failed: {e}")
            print(f"         Tiếp tục với FP32 only.")

    # ----- 4. Benchmark CPU
    bench_results = {}
    if not args.no_benchmark:
        bench_results["pytorch_fp32_cpu"] = bench_pytorch_cpu(model, cfg)
        bench_results["onnx_fp32_cpu"]    = bench_onnx_cpu(onnx_fp32, cfg, "ONNX FP32")
        if int8_done:
            bench_results["onnx_int8_cpu"] = bench_onnx_cpu(onnx_int8, cfg, "ONNX INT8")
        # Summary table
        print(f"\n  Latency summary (single-URL, CPU, 1 thread):")
        print(f"  {'Backend':<22} {'ms/URL':>10} {'URL/s':>10} {'speedup':>10}")
        base = bench_results["pytorch_fp32_cpu"]["ms_per_url"]
        for name, r in bench_results.items():
            speedup = base / r["ms_per_url"]
            print(f"  {name:<22} {r['ms_per_url']:>10.3f} {r['urls_per_sec']:>10,.0f} "
                  f"{speedup:>9.2f}x")

    # ----- 5. Accuracy regression
    accuracy = None
    if not args.no_regression:
        try:
            accuracy = accuracy_regression(
                model, cfg, onnx_fp32,
                onnx_int8 if int8_done else None,
                args.split_mode, threshold, n_samples=args.regression_n,
            )
        except Exception as e:
            print(f"  [WARN] Accuracy regression failed: {e}")

    # ----- 6. Save to results_4.json
    payload = {
        "split_mode": args.split_mode,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_best_val_f1": float(ckpt["best_f1"]),
        "threshold": float(threshold),
        "onnx_fp32_path": str(onnx_fp32.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "onnx_fp32_size_mb": onnx_fp32.stat().st_size / (1024 * 1024),
        "sanity_fp32": sanity_fp32,
    }
    if int8_done:
        payload["onnx_int8_path"] = str(onnx_int8.relative_to(PROJECT_ROOT)).replace("\\", "/")
        payload["onnx_int8_size_mb"] = onnx_int8.stat().st_size / (1024 * 1024)
        payload["int8_compression_x"] = onnx_fp32.stat().st_size / max(onnx_int8.stat().st_size, 1)
    if bench_results:
        payload["benchmark_cpu"] = bench_results
    if accuracy:
        payload["accuracy_regression"] = accuracy

    save_deployment_results(args.split_mode, payload)

    print("\n" + "=" * 70)
    print(f"DONE  ·  split_mode = {args.split_mode}")
    print("=" * 70)


if __name__ == "__main__":
    main()

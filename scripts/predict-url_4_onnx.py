"""
Inference CLI dùng ONNX Runtime (Item 11.7).

Wrapper rất gọn cho `predict-url_4.py` — thay PyTorch CNNLSTM bằng
ONNX Runtime InferenceSession. Mục đích: deploy CPU production không cần
PyTorch + CUDA, image footprint nhỏ hơn.

Recommendation từ benchmark (Item 11.5):
  - **ONNX FP32 (default)**: 1.67 ms/URL CPU 1-thread, F1 không đổi vs PyTorch
  - **ONNX INT8 (--int8)**: 6.6 ms/URL (chậm hơn — LSTM không quantize tốt),
    F1 drop ~0.04 pp. Dùng INT8 khi CHỈ cần disk size nhỏ (3.84x compression).

Usage:
    .\\venv\\Scripts\\Activate.ps1

    # Default: ONNX FP32 random checkpoint
    python "scripts/predict-url_4_onnx.py" https://example.com

    # ONNX INT8 (smaller disk, slower inference)
    python "scripts/predict-url_4_onnx.py" https://example.com --int8

    # Domain checkpoint
    python "scripts/predict-url_4_onnx.py" https://example.com --split-mode domain

    # Batch
    python "scripts/predict-url_4_onnx.py" --file urls.txt
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT     = Path(r"D:\! secURLity")
PROCESSED_ROOT   = PROJECT_ROOT / "data" / "processed" / "model_4"
VOCAB_PATH       = PROCESSED_ROOT / "vocab.json"
METADATA_PATH    = PROCESSED_ROOT / "metadata.json"
MODELS_ROOT      = PROJECT_ROOT / "models"
LEXICAL_SCRIPT   = PROJECT_ROOT / "scripts" / "4-feat. extract-lexical_4.py"
PREDICT_SCRIPT   = PROJECT_ROOT / "scripts" / "predict-url_4.py"
THRESHOLDS_JSON  = MODELS_ROOT / "thresholds_4.json"


# ============================================================================
# Risk tier
# ============================================================================

def risk_level(score: float) -> str:
    pct = score * 100
    if pct <= 25: return "Low Risk"
    if pct <= 50: return "Caution"
    if pct <= 75: return "Suspicious"
    return "High Risk"


# ============================================================================
# Re-use helpers từ predict-url_4.py (load_vocab, encode_url, normalize_features,
# resolve_threshold) — import qua importlib.
# ============================================================================

def import_module_from_path(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# ONNX Predictor
# ============================================================================

class ONNXURLPredictor:
    def __init__(self, split_mode: str, use_int8: bool, mode: str,
                 custom_threshold: float | None = None, verbose: bool = True):
        import onnxruntime as ort

        # Paths
        onnx_name = f"cnn_lstm_4_{split_mode}_int8.onnx" if use_int8 \
                    else f"cnn_lstm_4_{split_mode}.onnx"
        onnx_path = MODELS_ROOT / onnx_name
        feat_stats_path = PROCESSED_ROOT / split_mode / "feat_stats.json"
        for p in (onnx_path, feat_stats_path, VOCAB_PATH, METADATA_PATH):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        # Import shared helpers từ predict-url_4.py
        pred_mod = import_module_from_path(PREDICT_SCRIPT, "predict_url_4")
        lex_mod  = import_module_from_path(LEXICAL_SCRIPT, "lexical_4")
        self.encode_url        = pred_mod.encode_url
        self.normalize_features = pred_mod.normalize_features
        self.resolve_threshold = pred_mod.resolve_threshold
        self.extract_features  = lex_mod.extract_features
        # Explain helpers (Item 12, dùng được với ONNX nhờ output 'attention_weights')
        self.render_attention_heatmap = pred_mod.render_attention_heatmap
        self.top_k_char_positions     = pred_mod.top_k_char_positions
        self.top_k_lexical_zscores    = pred_mod.top_k_lexical_zscores
        self.fmt_explanation          = pred_mod.fmt_explanation
        self._pred_mod = pred_mod

        # Vocab + stats
        self.char2idx, self.pad_idx, self.unk_idx = pred_mod.load_vocab(VOCAB_PATH)
        self.stats = pred_mod.load_feat_stats(feat_stats_path)

        # MAX_LEN từ metadata
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.max_len = int(meta["max_len"])
        self.n_features = self.stats["n_features"]

        # ONNX session — CPU, 1 thread (production single-URL)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        # Detect output names — model có thể có 1 output (logit only — model cũ)
        # hoặc 2 outputs (logit + attention_weights — re-exported sau Item 12)
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.has_attention_output = "attention_weights" in self.output_names

        # Threshold
        self.threshold, self.thr_source = self.resolve_threshold(
            split_mode, mode, custom_threshold
        )
        self.split_mode = split_mode
        self.use_int8 = use_int8

        if verbose:
            print("=" * 70)
            print(f"ONNX URL classifier — split_mode={split_mode}, "
                  f"quant={'INT8' if use_int8 else 'FP32'}, mode={mode}")
            print("=" * 70)
            print(f"  ONNX file      : {onnx_path.name}  "
                  f"({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")
            print(f"  Outputs        : {self.output_names}")
            print(f"  Attention      : {'ENABLED (--explain available)' if self.has_attention_output else 'NOT available (re-export to enable)'}")
            print(f"  MAX_LEN        : {self.max_len}")
            print(f"  Threshold      : {self.threshold:.4f}  ({self.thr_source})")
            print(f"  Provider       : CPUExecutionProvider (1 thread)")
            print("=" * 70)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)

    def predict_one(self, url: str) -> dict:
        x_seq = self.encode_url(url, self.char2idx, self.max_len,
                                self.pad_idx, self.unk_idx)[None, :]
        raw_feat = self.extract_features(url).reshape(1, -1)
        feat_norm = self.normalize_features(raw_feat, self.stats)
        outs = self.sess.run(None, {"x_seq": x_seq, "x_feat": feat_norm})
        logit = outs[0]
        prob = float(self._sigmoid(logit).item())
        pred = int(prob >= self.threshold)
        return {
            "url": url,
            "probability": prob,
            "threshold": self.threshold,
            "prediction": pred,
            "label": "MALICIOUS" if pred == 1 else "BENIGN",
            "risk_level": risk_level(prob),
        }

    def predict_one_explained(self, url: str, top_k: int = 5) -> dict:
        """Predict + attention heatmap + lexical z-scores. Cần ONNX có
        output 'attention_weights' (re-export sau Item 12)."""
        if not self.has_attention_output:
            raise RuntimeError(
                "ONNX model không có output 'attention_weights'. "
                "Re-export: `python \"scripts/8. export-onnx_4.py\" --split-mode "
                f"{self.split_mode}`"
            )
        url_strip = url.strip()
        url_trunc = url_strip[:self.max_len]

        x_seq = self.encode_url(url, self.char2idx, self.max_len,
                                self.pad_idx, self.unk_idx)[None, :]
        raw_feat = self.extract_features(url).reshape(1, -1)
        feat_norm = self.normalize_features(raw_feat, self.stats)

        outs = self.sess.run(None, {"x_seq": x_seq, "x_feat": feat_norm})
        logit = outs[0]
        # Tìm attention output (có thể không phải [1] nếu order khác)
        attn_idx = self.output_names.index("attention_weights")
        attn_lq = outs[attn_idx][0]  # (L/4,)

        prob = float(self._sigmoid(logit).item())
        pred = int(prob >= self.threshold)

        # Up-sample L/4 → L, crop về URL length
        attn_per_char = np.repeat(attn_lq, 4)[:self.max_len][:len(url_trunc)]

        heatmap = self.render_attention_heatmap(url_trunc, attn_per_char)
        top_chars = self.top_k_char_positions(url_trunc, attn_per_char, k=top_k)
        top_lex = self.top_k_lexical_zscores(
            feat_norm, raw_feat, self.stats["feature_names"], k=top_k
        )
        return {
            "url": url,
            "url_truncated": url_trunc if len(url_strip) > self.max_len else None,
            "probability": prob,
            "threshold": self.threshold,
            "prediction": pred,
            "label": "MALICIOUS" if pred == 1 else "BENIGN",
            "risk_level": risk_level(prob),
            "explanation": {
                "heatmap": heatmap,
                "attention_weights_per_char": attn_per_char.tolist(),
                "top_chars": top_chars,
                "top_lexical": top_lex,
            },
        }

    def predict_batch(self, urls: list[str], batch_size: int = 256) -> list[dict]:
        results = []
        for start in range(0, len(urls), batch_size):
            chunk = urls[start:start + batch_size]
            x_seq = np.stack([
                self.encode_url(u, self.char2idx, self.max_len,
                                self.pad_idx, self.unk_idx)
                for u in chunk
            ])
            raw_feat = np.stack([self.extract_features(u) for u in chunk])
            feat_norm = self.normalize_features(raw_feat, self.stats)
            logits = self.sess.run(None, {"x_seq": x_seq, "x_feat": feat_norm})[0]
            probs = self._sigmoid(logits)
            for u, p in zip(chunk, probs):
                pr = float(p)
                pred = int(pr >= self.threshold)
                results.append({
                    "url": u,
                    "probability": pr,
                    "threshold": self.threshold,
                    "prediction": pred,
                    "label": "MALICIOUS" if pred == 1 else "BENIGN",
                    "risk_level": risk_level(pr),
                })
        return results


def fmt_result(r: dict, width_url: int = 60) -> str:
    pct = r["probability"] * 100
    decision = "MAL " if r["prediction"] == 1 else "ben "
    url = r["url"]
    if len(url) > width_url:
        url = url[:width_url - 3] + "..."
    return f"[{decision}] {pct:6.2f}%  [{r['risk_level']:<11}]  {url}"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--file", "-f")
    ap.add_argument("--split-mode", choices=("random", "domain"), default="random")
    ap.add_argument("--int8", action="store_true",
                    help="Dùng INT8 model thay vì FP32 (chậm hơn, dung lượng disk nhỏ hơn 3.84x).")
    ap.add_argument("--mode", choices=("default", "safer", "precise"), default="default")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--explain", action="store_true",
                    help="In attention heatmap + top-5 chars + top-5 lexical z-scores. "
                         "Cần ONNX có output 'attention_weights' (re-export sau Item 12).")
    ap.add_argument("--explain-top-k", type=int, default=5)
    args = ap.parse_args()

    try:
        predictor = ONNXURLPredictor(
            split_mode=args.split_mode,
            use_int8=args.int8,
            mode=args.mode,
            custom_threshold=args.threshold,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"[HINT]  Chạy `python \"scripts/8. export-onnx_4.py\" --split-mode "
              f"{args.split_mode}` để generate ONNX files.")
        sys.exit(2)

    urls_to_scan: list[str] = []
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            urls_to_scan = [line.strip() for line in f if line.strip()]
        if not args.quiet:
            print(f"\nLoaded {len(urls_to_scan):,} URLs từ {args.file}")
    elif args.urls:
        urls_to_scan = args.urls

    if urls_to_scan:
        if args.explain:
            results = [predictor.predict_one_explained(u, top_k=args.explain_top_k)
                       for u in urls_to_scan]
        elif len(urls_to_scan) > 1:
            results = predictor.predict_batch(urls_to_scan, args.batch_size)
        else:
            results = [predictor.predict_one(urls_to_scan[0])]
        if args.json:
            if args.explain:
                for r in results:
                    if "explanation" in r:
                        r["explanation"].pop("heatmap", None)
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print()
            for r in results:
                print(fmt_result(r))
                if args.explain:
                    print(predictor.fmt_explanation(r))
        return

    # Interactive
    if not args.quiet:
        print("\nInteractive mode. Nhập URL ('exit' để thoát).\n")
    while True:
        try:
            url = input("URL > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not url or url.lower() in ("exit", "quit"):
            break
        try:
            if args.explain:
                r = predictor.predict_one_explained(url, top_k=args.explain_top_k)
            else:
                r = predictor.predict_one(url)
        except Exception as e:
            print(f"  [ERROR] {e}\n"); continue
        if args.json:
            if args.explain and "explanation" in r:
                r["explanation"].pop("heatmap", None)
            print(json.dumps(r, ensure_ascii=False))
        else:
            print(f"  → Probability : {r['probability']*100:6.2f}%  "
                  f"(threshold={r['threshold']:.4f})")
            print(f"  → Decision    : {r['label']}")
            print(f"  → Risk level  : {r['risk_level']}")
            if args.explain:
                print(predictor.fmt_explanation(r))
            else:
                print()


if __name__ == "__main__":
    main()

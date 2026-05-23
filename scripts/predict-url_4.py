"""
Inference CLI cho model 4 (CNN-LSTM-final.md Items 2.3, 4.11, 7.4).

Hybrid CNN-LSTM + lexical features. Đặc trưng so với predict-url_3.py:
  - **KHÔNG lowercase** URL input (Item 2.3) — vocab case-preserved
  - Trích 30 lexical features tại runtime (Item 4.11), dùng `feat_stats.json`
    để normalize (log1p + standardize) — y hệt training
  - Đọc threshold + mode operating points từ `models/thresholds_4.json` (Item 7.4)
  - Đọc MAX_LEN, vocab_size, n_features, pad_idx từ checkpoint config (Item 1.3)
  - Hỗ trợ chọn checkpoint random hoặc domain qua `--split-mode`

Risk levels (theo CLAUDE.md, độc lập với threshold binary decision):
  0–25%  → Low Risk    | 26–50% → Caution
  51–75% → Suspicious  | 76–100% → High Risk

Threshold mode (Item 7.4):
  default : balanced (max F1 trên val)
  safer   : bias về catch mal (precision ≥ 0.99 trên val)
  precise : bias về tránh false alarm (recall ≥ 0.99 trên val)
  custom  : --threshold X.XX để override (overrides --mode)

Usage:
    .\\venv\\Scripts\\Activate.ps1

    # Single URL — random checkpoint, default threshold
    python "scripts/predict-url_4.py" https://example.com

    # Multiple URLs
    python "scripts/predict-url_4.py" https://example.com https://phish.example

    # Batch từ file (one URL per line)
    python "scripts/predict-url_4.py" --file urls.txt

    # Interactive mode (no args)
    python "scripts/predict-url_4.py"

    # Dùng domain checkpoint (con số "thật" production)
    python "scripts/predict-url_4.py" https://example.com --split-mode domain

    # Đổi operating point
    python "scripts/predict-url_4.py" https://example.com --mode safer

    # Custom threshold (override mode)
    python "scripts/predict-url_4.py" https://example.com --threshold 0.85
"""

import argparse
import importlib.util
import json
import sys
import types
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

PROJECT_ROOT     = Path(r"D:\! secURLity")
PROCESSED_ROOT   = PROJECT_ROOT / "data" / "processed" / "model_4"
VOCAB_PATH       = PROCESSED_ROOT / "vocab.json"
MODELS_ROOT      = PROJECT_ROOT / "models"
TRAIN_SCRIPT     = PROJECT_ROOT / "scripts" / "4. train-model_4.py"
LEXICAL_SCRIPT   = PROJECT_ROOT / "scripts" / "4-feat. extract-lexical_4.py"
THRESHOLDS_JSON  = MODELS_ROOT / "thresholds_4.json"


# ============================================================================
# Risk tier (theo CLAUDE.md)
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
# Dynamic imports — load CNNLSTM + extract_features từ training scripts
# ============================================================================

def import_module(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không load được module từ {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Encode URL (char-level, KHÔNG lowercase — Item 2.3)
# ============================================================================

def load_vocab(path: Path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"vocab.json không phải dict — got {type(data)}")
    char2idx = {k: int(v) for k, v in data.items()}
    pad_idx = int(char2idx.get("<PAD>", 0))
    unk_idx = int(char2idx.get("<UNK>", 1))
    return char2idx, pad_idx, unk_idx


def encode_url(url: str, char2idx: dict, max_len: int,
               pad_idx: int, unk_idx: int) -> np.ndarray:
    """Encode URL → int64 array (max_len,). KHÔNG lowercase (Item 2.3)."""
    url = url.strip()  # Chỉ strip, KHÔNG lowercase
    seq = [char2idx.get(c, unk_idx) for c in url[:max_len]]
    if len(seq) < max_len:
        seq.extend([pad_idx] * (max_len - len(seq)))
    return np.array(seq, dtype=np.int64)


# ============================================================================
# Normalize lexical features dùng feat_stats.json (giống training)
# ============================================================================

def load_feat_stats(path: Path):
    with path.open(encoding="utf-8") as f:
        d = json.load(f)
    return {
        "feature_names":  list(d["feature_names"]),
        "log1p_features": set(d["log1p_features"]),
        "mean":           np.array(d["mean"], dtype=np.float32),
        "std":            np.array(d["std"],  dtype=np.float32),
        "n_features":     int(d["n_features"]),
    }


def normalize_features(raw: np.ndarray, stats: dict) -> np.ndarray:
    """Apply log1p (cho LOG1P features) + standardize. raw shape: (N, 30)."""
    arr = raw.astype(np.float32, copy=True)
    log1p_mask = np.array(
        [name in stats["log1p_features"] for name in stats["feature_names"]],
        dtype=bool,
    )
    arr[:, log1p_mask] = np.log1p(arr[:, log1p_mask])
    arr = (arr - stats["mean"]) / stats["std"]
    if not np.isfinite(arr).all():
        raise ValueError("NaN/Inf trong features đã normalize — URL có thể quá lạ.")
    return arr


# ============================================================================
# Threshold + mode picking
# ============================================================================

def resolve_threshold(split_mode: str, mode: str,
                      custom_threshold: float | None) -> tuple[float, str]:
    """
    Trả về (threshold, source_description).
    Priority: custom_threshold > thresholds_4.json[split_mode][mode] > 0.5 fallback
    """
    if custom_threshold is not None:
        return custom_threshold, f"custom (CLI --threshold {custom_threshold})"

    if not THRESHOLDS_JSON.exists():
        print(f"[WARN] {THRESHOLDS_JSON} không tồn tại — fallback threshold=0.5.")
        print(f"       Chạy `6. tune-threshold_4.py --split-mode {split_mode}` "
              f"để có operating points.")
        return 0.5, "fallback 0.5 (thresholds_4.json missing)"

    try:
        data = json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[WARN] {THRESHOLDS_JSON} parse fail — fallback threshold=0.5.")
        return 0.5, "fallback 0.5 (thresholds_4.json invalid)"

    mode_data = data.get(split_mode)
    if not mode_data or "operating_points" not in mode_data:
        print(f"[WARN] thresholds_4.json không có entry cho split_mode='{split_mode}' "
              f"— fallback threshold=0.5.")
        return 0.5, f"fallback 0.5 (no entry for {split_mode})"

    op = mode_data["operating_points"].get(mode)
    if not op:
        print(f"[WARN] mode='{mode}' không tồn tại trong thresholds_4.json[{split_mode}] "
              f"— fallback threshold=0.5.")
        return 0.5, f"fallback 0.5 (mode={mode} missing)"

    thr = float(op["threshold"])
    return thr, f"thresholds_4.json[{split_mode}][{mode}]"


# ============================================================================
# Predictor (wrap state + inference)
# ============================================================================

class URLPredictor:
    def __init__(self, split_mode: str, mode: str,
                 custom_threshold: float | None = None,
                 verbose: bool = True):
        self.split_mode = split_mode
        self.mode = mode

        # ---- Paths
        ckpt_path = MODELS_ROOT / f"cnn_lstm_best_4_{split_mode}.pt"
        feat_stats_path = PROCESSED_ROOT / split_mode / "feat_stats.json"
        for p in (ckpt_path, feat_stats_path, VOCAB_PATH):
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")

        # ---- Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Imports
        train_mod = import_module(TRAIN_SCRIPT, "train_model_4")
        lex_mod = import_module(LEXICAL_SCRIPT, "lexical_4")
        self.CNNLSTM = train_mod.CNNLSTM
        self.extract_features = lex_mod.extract_features  # single-URL function

        # ---- Vocab
        self.char2idx, self.pad_idx, self.unk_idx = load_vocab(VOCAB_PATH)

        # ---- Feat stats
        self.stats = load_feat_stats(feat_stats_path)

        # ---- Checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.max_len    = int(cfg["max_len"])
        self.vocab_size = int(cfg["vocab_size"])
        self.n_features = int(cfg["n_features"])
        self.cfg_pad    = int(cfg["pad_idx"])

        if self.n_features != self.stats["n_features"]:
            raise ValueError(
                f"n_features mismatch: checkpoint={self.n_features} "
                f"vs feat_stats={self.stats['n_features']}"
            )
        if self.cfg_pad != self.pad_idx:
            raise ValueError(
                f"pad_idx mismatch: checkpoint={self.cfg_pad} vs vocab={self.pad_idx}"
            )

        self.model = self.CNNLSTM(
            vocab_size=self.vocab_size,
            n_features=self.n_features,
            pad_idx=self.pad_idx,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # ---- Attention capture (Item 12) — monkey-patch attn_pool.forward
        # để stash weights vào pool._last_weights mỗi forward pass.
        # Lazy install — chỉ install khi predict_one_explained() được gọi.
        self._attn_hook_installed = False

        # ---- Threshold
        self.threshold, self.thr_source = resolve_threshold(
            split_mode, mode, custom_threshold
        )

        if verbose:
            print("=" * 70)
            print(f"Model 4 URL classifier — split_mode={split_mode}, mode={mode}")
            print("=" * 70)
            print(f"  Device         : {self.device}")
            print(f"  Checkpoint     : {ckpt_path.name}  "
                  f"(epoch {ckpt['epoch']}, val F1={ckpt['best_f1']:.4f})")
            print(f"  Vocab size     : {len(self.char2idx)} "
                  f"(case-preserved, UNK={self.unk_idx})")
            print(f"  MAX_LEN        : {self.max_len}")
            print(f"  n_features     : {self.n_features}")
            print(f"  Threshold      : {self.threshold:.4f}  ({self.thr_source})")
            print(f"  Params         : {sum(p.numel() for p in self.model.parameters()):,}")
            print("=" * 70)

    def _install_attention_hook(self) -> None:
        """Monkey-patch model.attn_pool.forward → cũng stash weights vào pool._last_weights.
        Logic giữ nguyên forward gốc (xem AttentionPool trong train script)."""
        if self._attn_hook_installed:
            return
        pool = self.model.attn_pool

        def forward_capturing(p_self, x: torch.Tensor,
                              mask: torch.Tensor = None) -> torch.Tensor:
            scores = p_self.attn(x).squeeze(-1)  # (B, L/4)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=1)  # (B, L/4)
            p_self._last_weights = weights.detach().cpu().numpy()
            return (x * weights.unsqueeze(-1)).sum(dim=1)

        pool.forward = types.MethodType(forward_capturing, pool)
        self._attn_hook_installed = True

    @torch.no_grad()
    def predict_one_explained(self, url: str, top_k: int = 5) -> dict:
        """
        Như predict_one() nhưng kèm explanation:
          - attention heatmap (ANSI colored)
          - top-K char positions theo attention weight
          - top-K lexical features theo |z-score|
        """
        self._install_attention_hook()

        # Encode + extract
        url_strip = url.strip()
        url_trunc = url_strip[:self.max_len]  # phần model thấy
        x_seq_np = encode_url(url, self.char2idx, self.max_len,
                              self.pad_idx, self.unk_idx)
        x_seq = torch.from_numpy(x_seq_np).unsqueeze(0).to(self.device)

        raw_feat = self.extract_features(url).reshape(1, -1)  # (1, 30) RAW
        feat_norm = normalize_features(raw_feat, self.stats)   # (1, 30) z-score
        x_feat = torch.from_numpy(feat_norm).to(self.device)

        logit = self.model(x_seq, x_feat)
        prob = float(torch.sigmoid(logit.float()).item())
        pred = int(prob >= self.threshold)

        # Get attention weights (L/4,) — up-sample × 4 → (L,)
        attn_lq = self.model.attn_pool._last_weights[0]      # (L/4,)
        attn_per_char = np.repeat(attn_lq, 4)[:self.max_len]  # (L,)
        # Crop về độ dài URL thật (phần PAD đã bị mask → weight ~0 nhưng vẫn crop)
        n_chars = len(url_trunc)
        attn_per_char = attn_per_char[:n_chars]

        # Build explanation
        heatmap = render_attention_heatmap(url_trunc, attn_per_char)
        top_chars = top_k_char_positions(url_trunc, attn_per_char, k=top_k)
        top_lex = top_k_lexical_zscores(
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

    @torch.no_grad()
    def predict_one(self, url: str) -> dict:
        """Trả về dict {probability, threshold, prediction (0/1), risk_level, label}."""
        # Encode sequence
        x_seq_np = encode_url(url, self.char2idx, self.max_len,
                              self.pad_idx, self.unk_idx)
        x_seq = torch.from_numpy(x_seq_np).unsqueeze(0).to(self.device)

        # Extract + normalize lexical features
        raw_feat = self.extract_features(url).reshape(1, -1)  # (1, 30)
        feat_norm = normalize_features(raw_feat, self.stats)
        x_feat = torch.from_numpy(feat_norm).to(self.device)

        # Forward → prob
        logit = self.model(x_seq, x_feat)
        prob = float(torch.sigmoid(logit.float()).item())

        pred = int(prob >= self.threshold)
        return {
            "url": url,
            "probability": prob,
            "threshold": self.threshold,
            "prediction": pred,
            "label": "MALICIOUS" if pred == 1 else "BENIGN",
            "risk_level": risk_level(prob),
        }

    @torch.no_grad()
    def predict_batch(self, urls: list[str], batch_size: int = 256) -> list[dict]:
        """Batch inference cho file mode — nhanh hơn loop predict_one."""
        results = []
        for start in range(0, len(urls), batch_size):
            chunk = urls[start:start + batch_size]
            # Encode all
            x_seq_np = np.stack([
                encode_url(u, self.char2idx, self.max_len,
                           self.pad_idx, self.unk_idx)
                for u in chunk
            ])
            raw_feat = np.stack([self.extract_features(u) for u in chunk])
            feat_norm = normalize_features(raw_feat, self.stats)

            x_seq = torch.from_numpy(x_seq_np).to(self.device)
            x_feat = torch.from_numpy(feat_norm).to(self.device)
            logits = self.model(x_seq, x_feat)
            probs = torch.sigmoid(logits.float()).cpu().numpy()

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


# ============================================================================
# Attention explainability (Item 12 — Option C)
# ============================================================================

# 256-color background gradient: cold → hot. Từ trung tính (no color) qua
# vàng/cam tới đỏ tươi. ANSI escape: `\033[48;5;{N}m ... \033[0m`.
_HEATMAP_BG = [
    None,   # bucket 0: no color (low attention)
    229,    # bucket 1: pale yellow
    228,
    227,
    226,    # bright yellow
    220,    # gold
    214,    # orange
    208,    # darker orange
    202,    # red-orange
    196,    # bright red
]


def _ansi_color_char(ch: str, intensity: float) -> str:
    """
    Trả về char đã wrap trong ANSI bg color theo intensity ∈ [0, 1].
    intensity = 0 → không tô; 1 → đỏ rực.
    """
    if intensity <= 0.05:
        return ch
    idx = min(int(intensity * (len(_HEATMAP_BG) - 1)), len(_HEATMAP_BG) - 1)
    bg = _HEATMAP_BG[idx]
    if bg is None:
        return ch
    # Foreground đen khi bg sáng (vàng nhạt), trắng khi bg đỏ — đọc dễ hơn
    fg = "30" if idx <= 4 else "97"
    return f"\033[48;5;{bg}m\033[{fg}m{ch}\033[0m"


def render_attention_heatmap(url: str, weights_per_char: np.ndarray) -> str:
    """
    Render URL với mỗi char tô màu theo attention weight.
    weights_per_char: shape (len(url),), giá trị raw từ softmax (sum=1 across L/4).
    """
    if weights_per_char.size == 0:
        return url
    # Normalize về [0, 1] bằng max để hiển thị tương đối
    w_max = float(weights_per_char.max())
    if w_max <= 0:
        return url
    norm = weights_per_char / w_max
    parts = [_ansi_color_char(c, float(n)) for c, n in zip(url, norm)]
    return "".join(parts)


def top_k_char_positions(url: str, weights_per_char: np.ndarray,
                         k: int = 5) -> list[dict]:
    """
    Top-K char positions theo attention. Vì sau upsample ×4, các chars liền kề
    thường cùng weight (cùng L/4 bucket), group lại để tránh duplicate.
    """
    n = len(url)
    if n == 0 or weights_per_char.size == 0:
        return []
    w = weights_per_char[:n]
    # Dedupe-by-value runs: lấy 1 đại diện cho mỗi L/4 block
    seen_blocks = set()
    candidates = []
    for i in range(n):
        block = i // 4
        if block in seen_blocks:
            continue
        seen_blocks.add(block)
        # Đại diện block = ký tự đầu của block (hoặc snippet 4 chars)
        snippet_end = min(i + 4, n)
        snippet = url[i:snippet_end]
        candidates.append({
            "position": i,
            "block": block,
            "snippet": snippet,
            "weight": float(w[i]),
        })
    candidates.sort(key=lambda c: c["weight"], reverse=True)
    return candidates[:k]


def top_k_lexical_zscores(feat_norm: np.ndarray, raw_feat: np.ndarray,
                          feature_names: list[str], k: int = 5) -> list[dict]:
    """
    Top-K lexical features sorted theo |z-score|.
    `feat_norm` ĐÃ là z-score (output của normalize_features = (log1p? - mean)/std).
    """
    z = feat_norm.flatten()
    raw = raw_feat.flatten()
    order = np.argsort(-np.abs(z))
    out = []
    for idx in order[:k]:
        out.append({
            "feature": feature_names[idx],
            "raw_value": float(raw[idx]),
            "z_score": float(z[idx]),
            "direction": "ABOVE train mean" if z[idx] > 0 else "BELOW train mean",
        })
    return out


# ============================================================================
# Output formatting
# ============================================================================

def fmt_result(r: dict, width_url: int = 60) -> str:
    """Format 1 dòng kết quả."""
    pct = r["probability"] * 100
    decision = "MAL " if r["prediction"] == 1 else "ben "
    url = r["url"]
    if len(url) > width_url:
        url = url[:width_url - 3] + "..."
    return (
        f"[{decision}] {pct:6.2f}%  "
        f"[{r['risk_level']:<11}]  {url}"
    )


def fmt_explanation(r: dict) -> str:
    """
    Format full explanation block — gọi sau fmt_result.
    Yêu cầu r có thêm key 'explanation' với 'heatmap', 'top_chars', 'top_lexical'.
    """
    exp = r["explanation"]
    lines = []
    lines.append("")
    lines.append(f"  Attention heatmap (màu càng đỏ = model focus càng cao):")
    lines.append(f"    {exp['heatmap']}")
    lines.append("")
    lines.append(f"  Top {len(exp['top_chars'])} char positions (sau max-pool ×4, mỗi position cover 4 chars):")
    lines.append(f"    {'#':>3} {'pos':>5} {'weight':>10}  snippet")
    for i, c in enumerate(exp["top_chars"], 1):
        # Escape control chars trong snippet
        snippet = c["snippet"].replace("\n", "\\n").replace("\r", "\\r")
        lines.append(f"    {i:>3} {c['position']:>5} {c['weight']:>10.4f}  {snippet!r}")
    lines.append("")
    lines.append(f"  Top {len(exp['top_lexical'])} lexical features (sorted theo |z-score| khỏi train mean):")
    lines.append(f"    {'#':>3} {'feature':<28} {'raw':>14} {'z-score':>10}  direction")
    for i, f in enumerate(exp["top_lexical"], 1):
        lines.append(f"    {i:>3} {f['feature']:<28} {f['raw_value']:>14.4f} "
                     f"{f['z_score']:>10.3f}  {f['direction']}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Inference URL malicious detector (model 4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("urls", nargs="*", help="URL(s) cần scan")
    ap.add_argument("--file", "-f", help="Đường dẫn file .txt (1 URL/dòng)")
    ap.add_argument("--split-mode", choices=("random", "domain"), default="random",
                    help="Checkpoint nào? 'random'=F1 in-dist cao, 'domain'=số 'thật' "
                         "production. Default: random.")
    ap.add_argument("--mode", choices=("default", "safer", "precise"), default="default",
                    help="Operating point từ thresholds_4.json. "
                         "default=balanced, safer=bias catch mal, precise=bias avoid FP.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Custom threshold (override --mode). Vd 0.85.")
    ap.add_argument("--json", action="store_true",
                    help="Output JSON thay vì format đẹp.")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Bỏ qua diagnostics header.")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Batch size khi đọc từ --file. Default 256.")
    ap.add_argument("--explain", action="store_true",
                    help="Item 12 — In attention heatmap (ANSI color) + top-5 char positions "
                         "+ top-5 lexical features theo |z-score|. Áp dụng cho từng URL "
                         "(slow hơn ~10ms/URL so với không --explain).")
    ap.add_argument("--explain-top-k", type=int, default=5,
                    help="Số top chars + lexical features hiển thị khi --explain. Default 5.")
    args = ap.parse_args()

    # ---- Build predictor
    try:
        predictor = URLPredictor(
            split_mode=args.split_mode,
            mode=args.mode,
            custom_threshold=args.threshold,
            verbose=not args.quiet,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

    # ---- Collect URLs
    urls_to_scan: list[str] = []
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            urls_to_scan = [line.strip() for line in f if line.strip()]
        if not args.quiet:
            print(f"\nLoaded {len(urls_to_scan):,} URLs từ {args.file}")
    elif args.urls:
        urls_to_scan = args.urls

    # ---- Batch mode
    if urls_to_scan:
        if args.explain:
            # Explain mode: chạy từng URL với attention capture (không batch được
            # cleanly do mỗi URL có heatmap riêng + per-URL ANSI in console)
            results = [
                predictor.predict_one_explained(u, top_k=args.explain_top_k)
                for u in urls_to_scan
            ]
        elif len(urls_to_scan) > 1:
            results = predictor.predict_batch(urls_to_scan, batch_size=args.batch_size)
        else:
            results = [predictor.predict_one(urls_to_scan[0])]

        if args.json:
            # JSON output không in ANSI escape — strip cho clean
            if args.explain:
                for r in results:
                    if "explanation" in r:
                        # Heatmap chứa ANSI → bỏ khỏi JSON (đã có per-char weights)
                        r["explanation"].pop("heatmap", None)
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print()
            for r in results:
                print(fmt_result(r))
                if args.explain:
                    print(fmt_explanation(r))
        return

    # ---- Interactive mode
    if not args.quiet:
        print("\nInteractive mode. Nhập URL (gõ 'exit' hoặc Ctrl+C để thoát).\n")
    while True:
        try:
            url = input("URL > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not url or url.lower() in ("exit", "quit"):
            break
        try:
            if args.explain:
                r = predictor.predict_one_explained(url, top_k=args.explain_top_k)
            else:
                r = predictor.predict_one(url)
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            continue
        if args.json:
            if args.explain and "explanation" in r:
                r["explanation"].pop("heatmap", None)
            print(json.dumps(r, ensure_ascii=False))
        else:
            pct = r["probability"] * 100
            decision = "MALICIOUS" if r["prediction"] == 1 else "BENIGN"
            print(f"  → Probability : {pct:6.2f}%  (threshold={r['threshold']:.4f})")
            print(f"  → Decision    : {decision}")
            print(f"  → Risk level  : {r['risk_level']}")
            if args.explain:
                print(fmt_explanation(r))
            else:
                print()


if __name__ == "__main__":
    main()

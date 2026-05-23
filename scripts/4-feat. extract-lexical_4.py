"""
Trích lexical features cho model 4 — Stage 1 Item 4 (data part).

Hybrid CNN-LSTM (Item 4) cần 1 vector lexical feature song song với chuỗi char.
Script này:
  1. Đọc URLs từ `<mode>/<split>/split.csv` (do preprocess đã tạo)
  2. Trích 30 lexical features mỗi URL (counts, ratios, flags)
  3. Tính mean/std từ TRAIN ONLY → save `feat_stats.json`
  4. Apply log1p (cho features kiểu count/length) + standardize → save `feat.npy`
  5. Val/test dùng cùng stats (KHÔNG re-fit) → tránh leak

30 features (xem `FEATURE_NAMES`):
    Lengths      : url_length, host_length, path_length, query_length
    Counts       : num_dots, num_hyphens, num_underscores, num_slashes,
                   num_question_marks, num_equals, num_amps, num_ats,
                   num_percents, num_digits, num_subdomains,
                   subdomain_length_max, path_depth, query_param_count
    Ratios       : digit_ratio, letter_ratio, vowel_ratio, special_char_ratio
    Binary flags : has_ip_host, has_port, is_https, tld_is_vn,
                   tld_is_common, has_punycode, has_double_slash_in_path
    Entropy      : host_entropy  (Shannon)

Normalization:
    LOG1P_FEATURES (counts + lengths) → log1p() trước rồi standardize
    Other features                    → standardize trực tiếp
    Binary features cũng standardize  → MLP xử lý fine, rare-1 thành outlier
                                        có signal mạnh

Output structure (cạnh các file khác của preprocess):
    data/processed/model_4/
    ├── random/
    │   ├── feat_stats.json
    │   ├── train/feat.npy   (N_train, 30) float32
    │   ├── val/feat.npy
    │   └── test/feat.npy
    └── domain/
        └── (tương tự)

Usage:
    .\\venv\\Scripts\\Activate.ps1

    python "scripts/4-feat. extract-lexical_4.py"                # both modes
    python "scripts/4-feat. extract-lexical_4.py" --mode random
    python "scripts/4-feat. extract-lexical_4.py" --mode domain
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

# Force UTF-8 stdout cho Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

# Bigger CSV field — URL có thể dài bất thường
csv.field_size_limit(10_000_000)


# ============================================================================
# Config
# ============================================================================

PROJECT_ROOT   = Path(r"D:\! secURLity")
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "model_4"

FEATURE_NAMES = [
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equals",
    "num_amps",
    "num_ats",
    "num_percents",
    "num_digits",
    "digit_ratio",
    "letter_ratio",
    "vowel_ratio",
    "special_char_ratio",
    "has_ip_host",
    "has_port",
    "is_https",
    "num_subdomains",
    "subdomain_length_max",
    "tld_is_vn",
    "tld_is_common",
    "host_entropy",
    "has_punycode",
    "path_depth",
    "has_double_slash_in_path",
    "query_param_count",
]
N_FEATURES = len(FEATURE_NAMES)  # 30
assert N_FEATURES == 30, f"Expected 30 features, got {N_FEATURES}"

# Features apply log1p trước khi standardize (heavy-tailed counts/lengths)
LOG1P_FEATURES = {
    "url_length", "host_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_question_marks", "num_equals", "num_amps", "num_ats",
    "num_percents", "num_digits",
    "num_subdomains", "subdomain_length_max",
    "path_depth", "query_param_count",
}

COMMON_TLDS = frozenset({".com", ".net", ".org", ".io", ".dev", ".app"})
VOWELS      = frozenset("aeiouAEIOU")


# ============================================================================
# Helpers
# ============================================================================

def is_ip(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or not (0 <= int(p) <= 255):
            return False
    return True


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def get_last_tld(host: str) -> str:
    """Trả về `.<last_label>` (lowercase). Không TLD-aware như `2. eda 4.py`,
    chỉ dùng cho check tld_is_common với 6 TLD đơn-label phổ biến."""
    if "." not in host:
        return ""
    return "." + host.rsplit(".", 1)[-1].lower()


# ============================================================================
# Feature extraction (per-URL)
# ============================================================================

def extract_features(url: str) -> np.ndarray:
    """Trả về np.ndarray(N_FEATURES,) float32 ở dạng RAW (chưa normalize)."""
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    if not url:
        return feat

    # ---- url_length ----
    n_url = len(url)
    feat[0] = float(n_url)

    # ---- parse ----
    try:
        if "://" not in url:
            parts = urlsplit("http://" + url)
            scheme = ""
        else:
            parts = urlsplit(url)
            scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        path = parts.path or ""
        query = parts.query or ""
    except ValueError:
        # URL malformed (vd IPv6 bracket sai)
        scheme, netloc, path, query = "", "", "", ""

    # Split host:port
    host = netloc
    port = ""
    if ":" in netloc:
        h, p = netloc.rsplit(":", 1)
        if p.isdigit():
            host, port = h, p

    # ---- component lengths ----
    feat[1] = float(len(host))
    feat[2] = float(len(path))
    feat[3] = float(len(query))

    # ---- char counts (toàn URL) ----
    feat[4]  = float(url.count("."))
    feat[5]  = float(url.count("-"))
    feat[6]  = float(url.count("_"))
    feat[7]  = float(url.count("/"))
    feat[8]  = float(url.count("?"))
    feat[9]  = float(url.count("="))
    feat[10] = float(url.count("&"))
    feat[11] = float(url.count("@"))
    feat[12] = float(url.count("%"))

    n_digits  = sum(c.isdigit() for c in url)
    n_letters = sum(c.isalpha() for c in url)
    n_vowels  = sum(c in VOWELS for c in url)
    feat[13]  = float(n_digits)

    # ---- ratios ----
    if n_url > 0:
        feat[14] = n_digits / n_url               # digit_ratio
        feat[15] = n_letters / n_url              # letter_ratio
        feat[17] = (n_url - n_digits - n_letters) / n_url  # special_char_ratio
    feat[16] = (n_vowels / n_letters) if n_letters else 0.0  # vowel_ratio (trên chữ cái)

    # ---- binary flags ----
    host_is_ip = is_ip(host)
    feat[18] = 1.0 if host_is_ip else 0.0
    feat[19] = 1.0 if port else 0.0
    feat[20] = 1.0 if scheme == "https" else 0.0

    # ---- subdomain count + max length ----
    if host and not host_is_ip:
        labels = [l for l in host.split(".") if l]
        if len(labels) > 2:
            sub_labels = labels[:-2]
            feat[21] = float(len(sub_labels))
            feat[22] = float(max(len(s) for s in sub_labels))
        else:
            feat[21] = 0.0
            feat[22] = 0.0

    # ---- TLD flags ----
    feat[23] = 1.0 if host.endswith(".vn") else 0.0
    tld = get_last_tld(host)
    feat[24] = 1.0 if tld in COMMON_TLDS else 0.0

    # ---- host entropy (Shannon) ----
    feat[25] = float(shannon_entropy(host))

    # ---- punycode ----
    feat[26] = 1.0 if "xn--" in host else 0.0

    # ---- path features ----
    feat[27] = float(path.count("/"))
    feat[28] = 1.0 if "//" in path else 0.0

    # ---- query params ----
    if query:
        feat[29] = float(query.count("&") + 1)

    return feat


def extract_features_batch(urls, progress_every: int = 500_000) -> np.ndarray:
    """Trả về np.ndarray(N, N_FEATURES) float32 (RAW)."""
    n = len(urls)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        out[i] = extract_features(urls[i])
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n - i - 1) / max(rate, 1.0)
            print(f"      ... {i+1:>12,}/{n:,}  "
                  f"({rate:,.0f}/s, eta {eta:.0f}s)", flush=True)
    return out


# ============================================================================
# Normalization
# ============================================================================

def compute_stats(raw: np.ndarray, log1p_mask: np.ndarray):
    """Apply log1p tới các cột trong log1p_mask, rồi tính mean/std."""
    arr = raw.copy()
    arr[:, log1p_mask] = np.log1p(arr[:, log1p_mask])
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    # Tránh std=0 (feature constant): set lên 1 để không chia 0
    std = np.where(std < 1e-8, np.float32(1.0), std)
    return mean, std


def apply_stats(raw: np.ndarray, mean: np.ndarray, std: np.ndarray,
                log1p_mask: np.ndarray) -> np.ndarray:
    """Apply log1p + standardize tại RAW. Trả về float32, đảm bảo finite."""
    arr = raw.copy()
    arr[:, log1p_mask] = np.log1p(arr[:, log1p_mask])
    arr = (arr - mean) / std
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        n_bad = int(np.isfinite(arr).sum() != arr.size)
        raise ValueError(f"NaN/Inf detected in normalized features (n_bad={n_bad})")
    return arr


# ============================================================================
# Main
# ============================================================================

def process_mode(mode: str, log1p_mask: np.ndarray) -> None:
    mode_dir = PROCESSED_ROOT / mode
    # Yêu cầu đầy đủ 3 split + split.csv của mỗi cái — nếu thiếu thì preprocess
    # chưa xong hoặc bị interrupt giữa chừng. Báo rõ ràng thay vì silent skip.
    missing = []
    if not mode_dir.exists():
        missing.append(str(mode_dir))
    else:
        for split in ("train", "val", "test"):
            csv_path = mode_dir / split / "split.csv"
            if not csv_path.exists():
                missing.append(str(csv_path))
    if missing:
        print(f"\n[ERROR] mode={mode}: thiếu các file/thư mục sau:")
        for m in missing:
            print(f"          {m}")
        print(f"        → Có thể `3. preprocess-data 4.py` chưa chạy xong "
              f"hoặc bị interrupt.")
        print(f"        → Chỉ chạy lexical SAU KHI thấy `metadata.json` "
              f"trong {PROCESSED_ROOT}")
        sys.exit(2)

    print(f"\n{'='*70}")
    print(f"Mode: {mode}   ({mode_dir})")
    print(f"{'='*70}")

    # ---- TRAIN: extract, compute stats, save ----
    train_csv = mode_dir / "train" / "split.csv"
    print(f"\n[TRAIN] {train_csv}")
    t0 = time.time()
    df_train = pd.read_csv(train_csv, dtype={"url": "string", "label": "int8"})
    urls_train = df_train["url"].astype(str).tolist()
    del df_train
    print(f"      rows: {len(urls_train):,}  (loaded in {time.time()-t0:.1f}s)")

    print(f"      Extracting raw features...")
    t0 = time.time()
    raw_train = extract_features_batch(urls_train)
    print(f"      Done in {time.time()-t0:.1f}s  shape={raw_train.shape}")
    del urls_train

    print(f"      Computing stats (log1p + mean/std)...")
    mean, std = compute_stats(raw_train, log1p_mask)
    stats = {
        "feature_names": FEATURE_NAMES,
        "log1p_features": sorted(LOG1P_FEATURES),
        "n_features": N_FEATURES,
        "mean": mean.tolist(),
        "std":  std.tolist(),
    }
    stats_path = mode_dir / "feat_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"      Saved {stats_path}")

    # Print stats summary (mean/std mỗi feature)
    print(f"\n      {'Feature':<28} {'mean (log1p)':>12} {'std (log1p)':>12}")
    for name, m, s in zip(FEATURE_NAMES, mean, std):
        mark = " *log1p" if name in LOG1P_FEATURES else ""
        print(f"      {name:<28} {m:>12.4f} {s:>12.4f}{mark}")

    print(f"\n      Normalizing TRAIN...")
    feat_train = apply_stats(raw_train, mean, std, log1p_mask)
    del raw_train
    train_out = mode_dir / "train" / "feat.npy"
    np.save(train_out, feat_train)
    print(f"      Saved {train_out}  shape={feat_train.shape}  "
          f"dtype={feat_train.dtype}")
    del feat_train

    # ---- VAL / TEST: dùng stats của TRAIN ----
    for split in ("val", "test"):
        split_csv = mode_dir / split / "split.csv"
        print(f"\n[{split.upper()}] {split_csv}")
        df = pd.read_csv(split_csv, dtype={"url": "string", "label": "int8"})
        urls = df["url"].astype(str).tolist()
        del df
        print(f"      rows: {len(urls):,}")

        print(f"      Extracting raw features...")
        t0 = time.time()
        raw = extract_features_batch(urls)
        del urls
        print(f"      Done in {time.time()-t0:.1f}s  shape={raw.shape}")

        feat = apply_stats(raw, mean, std, log1p_mask)
        del raw
        out_path = mode_dir / split / "feat.npy"
        np.save(out_path, feat)
        print(f"      Saved {out_path}  shape={feat.shape}")
        del feat


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("random", "domain", "both"),
                    default="both", help="Mode to process (default: both)")
    ap.add_argument("--force", action="store_true",
                    help="Bỏ qua check metadata.json (KHÔNG khuyên — nếu preprocess "
                         "chưa xong sẽ chạy trên data thiếu)")
    args = ap.parse_args(argv)

    # ---- Sanity check: preprocess đã xong chưa? ----
    # metadata.json là file CUỐI CÙNG mà preprocess script ghi — nó tồn tại
    # đồng nghĩa toàn bộ NPY + CSV đã ghi xong.
    metadata_path = PROCESSED_ROOT / "metadata.json"
    if not metadata_path.exists() and not args.force:
        print(f"[ERROR] Không thấy {metadata_path}")
        print(f"        → Preprocess (`3. preprocess-data 4.py`) CHƯA xong "
              f"hoặc bị interrupt.")
        print(f"        → Nếu đang chạy: đợi tới khi thấy log `[5/5] Done` "
              f"rồi chạy lại lệnh này.")
        print(f"        → Nếu đã crash:  chạy lại preprocess từ đầu.")
        print(f"        → Override:      thêm flag --force (KHÔNG khuyên).")
        sys.exit(2)

    log1p_mask = np.array([name in LOG1P_FEATURES for name in FEATURE_NAMES])
    print(f"N_FEATURES        = {N_FEATURES}")
    print(f"log1p features    = {sum(log1p_mask)}")
    print(f"standardize-only  = {N_FEATURES - sum(log1p_mask)}")

    modes = ("random", "domain") if args.mode == "both" else (args.mode,)
    t_global = time.time()
    for mode in modes:
        process_mode(mode, log1p_mask)
    print(f"\n[DONE] all modes in {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    main()

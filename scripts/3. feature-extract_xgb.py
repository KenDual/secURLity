"""
Feature Extraction for XGBoost (model 2).

Đọc các split đã được tạo bởi `3. preprocess-data 2.py`
(`data/processed/model_2/{train,val,test}.csv`) và trích xuất feature vector
cho từng URL. Lưu kết quả ra:
    data/processed/xgboost/{train,val,test}_X_xgb.npy   (float32, (N, F))
    data/processed/xgboost/{train,val,test}_y_xgb.npy   (int8,    (N,))
    data/processed/xgboost/xgb_feature_cols.json        (tên cột)
    data/processed/xgboost/xgb_feature_meta.json        (tld list, pos_weight)

Tinh thần feature engineering:
  - Tính theo bản chất generic (entropy, digit/vowel ratio, longest run,...)
    thay vì hardcode tín hiệu Char-RNN cụ thể của dataset này. Mục tiêu là
    feature có ý nghĩa real-world chứ không chỉ memorization dataset.
  - Không dùng known domain pools (ecommerce/news/tech_saas/...) làm feature —
    đó là pure memorization của benign generator.
  - Brand mention tách 2 góc: brand_in_registered_domain (benign signal) vs
    brand_in_path_or_sub_only (phishing signal).
  - TLD: top-50 one-hot + tld_other + tld_missing.

Yêu cầu: `pip install tldextract xgboost` (đã ghi trong checklist mục 0).
"""

import json
import math
import os
import re
import string
from collections import Counter
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import tldextract
from tqdm import tqdm

# ============================================================================
# Paths & parallelism
# ============================================================================
PROJECT_ROOT = Path(r"D:\! secURLity")
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "model_2"   # nguồn CSV splits
OUT_DIR    = PROJECT_ROOT / "data" / "processed" / "xgboost"   # đầu ra .npy + JSON

# Số worker process. Để cpu_count()-1, chừa 1 core cho main + OS.
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
# Số URL mỗi batch gửi sang worker. Đủ lớn để amortize IPC overhead,
# đủ nhỏ để cân bằng tải khi chunk cuối lệch tiến độ.
CHUNK_SIZE = 4000

# ============================================================================
# Word pools (đồng bộ với `scripts/2. eda 2.py`)
# ============================================================================
SUSPICIOUS_KEYWORDS = (
    "login", "secure", "update", "confirm", "signin", "verify", "restore",
    "alert", "validate", "verify-account", "authenticate", "urgent",
    "limited", "security-alert", "warning",
)
SCAM_BAIT = (
    "verify-account", "restore-access", "confirm-password",
    "act-now", "update-payment",
)
BRANDS = (
    "google", "amazon", "aws", "github", "paypal", "facebook", "azure",
    "ubs", "linkedin", "gitlab", "dropbox", "twitter", "instagram",
    "microsoft", "docker",
)
C2_PATHS = (
    "/i", "/update", "/data", "/plugin", "/report", "/config",
    "/gate", "/sync", "/cmd", "/c2",
)
MALWARE_EXTS = {".exe", ".apk", ".dmg", ".sh", ".scr", ".dll", ".bat", ".so", ".msi"}
CDN_EXTS = {".js", ".json", ".css", ".xml", ".ts", ".jpg", ".yaml", ".gif",
            ".cjs", ".png", ".woff", ".woff2", ".svg", ".ico"}
BENIGN_WORDS = {
    "article", "tag", "search", "v1", "v2", "v3", "news", "docs", "releases",
    "user", "comments", "posts", "api", "category", "files", "detail",
    "product", "review", "cart", "order", "watch", "status", "blog",
    "courses", "overview",
}

# Top 50 TLDs lấy từ eda-result 2.txt (label 0 + label 1 + một số TLD nghi vấn)
TOP_TLDS = [
    "com", "net", "org", "app", "io", "vn", "ly", "de", "co.uk", "edu",
    "com.br", "ru", "com.au", "jp", "pl", "gd", "ac.uk", "it", "us",
    "com.vn", "info", "site", "co", "top", "ga", "ml", "tk", "fr",
    "tv", "fm", "social", "cloud", "so", "link", "cn", "uk", "in",
    "me", "biz", "br", "es", "ca", "au", "ch", "nl", "se", "no",
    "kr", "tr", "mx",
]

VOWELS = set("aeiou")
CONSONANTS = set(string.ascii_lowercase) - VOWELS
DIGITS_SET = set(string.digits)
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SPLIT_RE = re.compile(r"[/_\-.]")

# Use bundled PSL snapshot only (no online fetch each run)
_tld_extractor = tldextract.TLDExtract(
    suffix_list_urls=(), fallback_to_snapshot=True, cache_dir=False
)


# ============================================================================
# Helpers
# ============================================================================
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def longest_run(s: str, charset: set) -> int:
    longest = run = 0
    for ch in s:
        if ch in charset:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return longest


def safe_parse(url: str):
    try:
        return urlparse(url)
    except Exception:
        return None


def n_query_params(query: str) -> int:
    if not query:
        return 0
    try:
        return len(parse_qs(query, keep_blank_values=True))
    except Exception:
        return query.count("&") + 1


# ============================================================================
# Feature extraction (returns dict — first row sets column order)
# ============================================================================
def extract_features(url: str) -> dict:
    url = (url or "").strip().lower()
    parsed = safe_parse(url)
    scheme = parsed.scheme if parsed else ""
    host = (parsed.hostname or "") if parsed else ""
    try:
        port = parsed.port if parsed else None
    except (ValueError, TypeError):
        port = None
    path = (parsed.path or "") if parsed else ""
    query = (parsed.query or "") if parsed else ""
    fragment = (parsed.fragment or "") if parsed else ""

    ext = _tld_extractor(url) if url else None
    suffix = ext.suffix if ext else ""
    registered = ext.domain if ext else ""  # phần "google" trong google.com
    subdomain = ext.subdomain if ext else ""

    feats: dict = {}

    # ---- Structural ----
    feats["url_len"] = len(url)
    feats["host_len"] = len(host)
    feats["path_len"] = len(path)
    feats["query_len"] = len(query)
    feats["fragment_len"] = len(fragment)
    feats["path_depth"] = sum(1 for s in path.split("/") if s)
    feats["num_query_params"] = n_query_params(query)
    feats["has_query"] = int(bool(query))
    feats["has_fragment"] = int(bool(fragment))
    feats["has_port"] = int(port is not None)

    # ---- Char composition (whole URL) ----
    digit_count = sum(c.isdigit() for c in url)
    feats["digit_count_url"] = digit_count
    feats["digit_density_url"] = digit_count / max(len(url), 1)
    feats["hyphen_count_url"] = url.count("-")
    feats["dot_count_url"] = url.count(".")
    feats["at_count"] = url.count("@")
    feats["pct_count"] = url.count("%")
    feats["amp_count"] = url.count("&")
    feats["eq_count"] = url.count("=")
    feats["qmark_count"] = url.count("?")

    # ---- Host-specific ----
    host_digits = sum(c.isdigit() for c in host)
    feats["host_digit_count"] = host_digits
    feats["host_digit_density"] = host_digits / max(len(host), 1)
    feats["host_hyphen_count"] = host.count("-")
    feats["host_dot_count"] = host.count(".")
    feats["host_num_subdomains"] = (
        len([s for s in subdomain.split(".") if s]) if subdomain else 0
    )
    feats["host_starts_with_www"] = int(host.startswith("www."))
    feats["host_longest_digit_run"] = longest_run(host, DIGITS_SET)
    feats["host_longest_consonant_run"] = longest_run(host, CONSONANTS)

    # Vowel ratio trên ký tự alphabet của host (capture DGA/Char-RNN gibberish)
    host_letters = [c for c in host if c.isalpha()]
    feats["host_vowel_ratio"] = (
        sum(c in VOWELS for c in host_letters) / len(host_letters)
        if host_letters else 0.0
    )
    feats["host_letter_count"] = len(host_letters)

    # ---- Entropy ----
    feats["entropy_url"] = shannon_entropy(url)
    feats["entropy_host"] = shannon_entropy(host)
    feats["entropy_path"] = shannon_entropy(path)

    # ---- Scheme & host shape ----
    feats["is_http"] = int(scheme == "http")
    feats["is_https"] = int(scheme == "https")
    feats["is_ip_host"] = int(bool(IP_RE.match(host)))

    # ---- TLD one-hot ----
    for tld in TOP_TLDS:
        key = "tld_" + tld.replace(".", "_")
        feats[key] = int(suffix == tld)
    feats["tld_other"] = int(bool(suffix) and suffix not in TOP_TLDS)
    feats["tld_missing"] = int(not suffix)
    feats["tld_label_count"] = suffix.count(".") + 1 if suffix else 0

    # ---- Lexical heuristic flags ----
    path_query = path + "?" + query  # search trên cả path lẫn query

    # Suspicious keywords
    n_sus = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in path_query)
    feats["num_suspicious_keywords"] = n_sus
    feats["has_suspicious_keyword"] = int(n_sus > 0)

    # Scam bait (phrase-level)
    feats["has_scam_bait"] = int(any(b in url for b in SCAM_BAIT))

    # C2-style path prefixes
    n_c2 = 0
    for p in C2_PATHS:
        if path == p or path.startswith(p + "/") or path.startswith(p + "?"):
            n_c2 += 1
    feats["has_c2_path"] = int(n_c2 > 0)
    feats["num_c2_path_hits"] = n_c2

    # Brand mention — split 2 góc
    brand_in_reg = any(b in registered for b in BRANDS)
    brand_anywhere = any(b in url for b in BRANDS)
    feats["brand_in_registered_domain"] = int(brand_in_reg)
    feats["brand_in_path_or_sub_only"] = int(brand_anywhere and not brand_in_reg)

    # File extension category — chỉ lấy extension của segment cuối path
    last_seg = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
    if "." in last_seg:
        ext_str = "." + last_seg.rsplit(".", 1)[-1]
    else:
        ext_str = ""
    feats["has_malware_ext"] = int(ext_str in MALWARE_EXTS)
    feats["has_cdn_ext"] = int(ext_str in CDN_EXTS)
    feats["has_any_ext"] = int(bool(ext_str))

    # Benign word hits (mâu thuẫn signal — giảm score nếu xuất hiện)
    path_tokens = set(t for t in SPLIT_RE.split(path) if t)
    bw_hits = path_tokens & BENIGN_WORDS
    feats["num_benign_words"] = len(bw_hits)
    feats["has_benign_word"] = int(bool(bw_hits))

    return feats


# ============================================================================
# Feature order — chốt 1 lần bằng cách extract một URL probe ở module level.
# Worker process re-import module sẽ tự dựng lại cùng list (deterministic).
# ============================================================================
_PROBE_URL = "https://www.example.com/path/to/file.html?a=1&b=2"
FEATURE_NAMES: list[str] = list(extract_features(_PROBE_URL).keys())
N_FEATURES: int = len(FEATURE_NAMES)


def _worker_batch(urls: list[str]) -> np.ndarray:
    """Worker: trả về numpy array (len(urls), N_FEATURES) float32.

    Trả numpy thay vì list[dict] để pickle nhanh + tiết kiệm RAM khi
    main process gộp.
    """
    out = np.empty((len(urls), N_FEATURES), dtype=np.float32)
    for i, u in enumerate(urls):
        feats = extract_features(u)
        for j, k in enumerate(FEATURE_NAMES):
            out[i, j] = feats[k]
    return out


def _chunked(seq, size):
    it = iter(seq)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


# ============================================================================
# Driver (multiprocess)
# ============================================================================
def process_split(name: str, pool: Pool) -> None:
    """Extract features for one split, using shared worker pool."""
    csv_path = SPLITS_DIR / f"{name}.csv"
    print(f"\n[{name}] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df["url"] = df["url"].astype(str)
    n = len(df)
    print(f"  rows: {n:,}  | workers: {N_WORKERS}  | chunk: {CHUNK_SIZE}")

    X = np.empty((n, N_FEATURES), dtype=np.float32)
    urls = df["url"].tolist()
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    write_idx = 0
    # imap giữ thứ tự — quan trọng vì labels align theo row index
    for arr in tqdm(
        pool.imap(_worker_batch, _chunked(urls, CHUNK_SIZE)),
        total=n_chunks,
        desc=f"  {name}",
        unit="chunk",
    ):
        X[write_idx : write_idx + len(arr)] = arr
        write_idx += len(arr)
    assert write_idx == n, f"row count mismatch: {write_idx} vs {n}"

    y = df["label"].astype(np.int8).values

    np.save(OUT_DIR / f"{name}_X_xgb.npy", X)
    np.save(OUT_DIR / f"{name}_y_xgb.npy", y)
    print(f"  saved -> {name}_X_xgb.npy {X.shape}  /  {name}_y_xgb.npy {y.shape}")


def main():
    print("=" * 78)
    print("FEATURE EXTRACTION (XGBoost — model 2)")
    print("=" * 78)
    print(f"Splits dir : {SPLITS_DIR}")
    print(f"Output dir : {OUT_DIR}")
    print(f"Features   : {N_FEATURES}")
    print(f"Workers    : {N_WORKERS}  | chunk size: {CHUNK_SIZE}")

    with Pool(processes=N_WORKERS) as pool:
        for split in ("train", "val", "test"):
            process_split(split, pool)

    # Save schema + meta
    cols_path = OUT_DIR / "xgb_feature_cols.json"
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"\nSaved feature columns -> {cols_path}  ({N_FEATURES} cols)")

    # Compute pos_weight từ train labels để khỏi recompute lúc train
    y_train = np.load(OUT_DIR / "train_y_xgb.npy")
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    meta = {
        "n_features": N_FEATURES,
        "top_tlds": TOP_TLDS,
        "pos_weight": pos_weight,
        "train_label_counts": {"0": n_neg, "1": n_pos},
        "notes": (
            "Generic lexical / statistical features. KHÔNG dùng known-domain "
            "pools làm feature. Tính chất Char-RNN artifact được bắt gián tiếp "
            "qua entropy_host / host_vowel_ratio / host_longest_consonant_run."
        ),
    }
    meta_path = OUT_DIR / "xgb_feature_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta            -> {meta_path}")
    print(f"  pos_weight (neg/pos): {pos_weight:.4f}")

    print("\n" + "=" * 78)
    print("FEATURE EXTRACTION DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()

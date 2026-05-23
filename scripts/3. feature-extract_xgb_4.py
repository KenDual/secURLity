"""
Feature Extraction for XGBoost (model 4 — real-world dataset 19.68M URLs).

Đọc các split flat đã được tạo bởi `3. preprocess-data 4.py` ở chế độ random:
    data/processed/model_4/random/{train,val,test}/split.csv     # url,label QUOTE_ALL

Trích xuất feature vector mỗi URL → lưu ra:
    data/processed/xgboost_4/{train,val,test}_X_xgb.npy   # float32 (N, F)
    data/processed/xgboost_4/{train,val,test}_y_xgb.npy   # int8   (N,)
    data/processed/xgboost_4/xgb_4_feature_cols.json      # tên cột (deterministic)
    data/processed/xgboost_4/xgb_4_feature_meta.json      # TLD lists, pos_weight, notes

Khác biệt so với phiên bản cũ (model 2 / synthetic):
  1. SPLITS_DIR đổi sang model_4/random, output sang xgboost_4.
  2. Bỏ SCAM_BAIT/C2_PATHS/BRANDS/BENIGN_WORDS — đó là dấu vết của synthetic
     generator (model 2). Trên real-world threat-feed URLs, các pool này gần
     như zero-hit cho cả 2 label → noise.
  3. SUSPICIOUS_KEYWORDS tỉa lại theo phishing thật (login/verify/account/...).
  4. TOP_TLDS rebuild từ EDA 4 — gộp benign-heavy (.vn, .gov.uk, .com.vn, .fm,
     .ie, .int) và mal-heavy (.app, .dev, .stream, .click, .sbs, .cfd, .icu,
     .digital, .online, .shop, .live, .xyz, .top).
  5. Thêm aggregate TLD buckets `tld_is_vn / _suspicious / _common` — EDA cho
     thấy .vn TLD = 19.85% benign vs 0.18% mal (signal cực mạnh).
  6. Thêm case features (mixed_case_ratio, has_upper) — EDA: benign 11.83%
     mixed-case vs mal 5.50% (2.15× chênh).
  7. Thêm charset anomalies (has_control_chars, non_ascii_ratio, has_punycode)
     — EDA: 101 chars CHỈ xuất hiện trong mal đều là control bytes / "<>#"".
  8. Thêm has_double_slash_in_path, path_to_url_ratio, is_bare_hostname (giúp
     XGBoost xử lý bare-hostname FP issue mà model 4 CNN-LSTM gặp).

Yêu cầu: `pip install tldextract xgboost`.
"""

import csv
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
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "model_4" / "random"
OUT_DIR    = PROJECT_ROOT / "data" / "processed" / "xgboost_4"

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
CHUNK_SIZE = 4000

# ============================================================================
# Word pools (tỉa lại từ phishing thực tế, không phải synthetic generator)
# ============================================================================
# Keyword phổ biến trong phishing kits / credential-harvest pages
PHISHING_KEYWORDS = (
    "login", "signin", "secure", "verify", "account", "password",
    "bank", "wallet", "confirm", "auth", "session", "recover",
    "unlock", "reset", "support", "billing",
)

MALWARE_EXTS = {".exe", ".apk", ".dmg", ".sh", ".scr", ".dll", ".bat", ".so", ".msi"}
CDN_EXTS = {".js", ".json", ".css", ".xml", ".ts", ".jpg", ".yaml", ".gif",
            ".cjs", ".png", ".woff", ".woff2", ".svg", ".ico"}

# ============================================================================
# TLD pools (rebuilt từ EDA 4 — real distribution của 19.68M URLs)
# ============================================================================
# Top 50 TLDs theo count trong dataset 4 (gộp cả benign-heavy + mal-heavy).
TOP_TLDS = [
    "com", "vn", "org", "co.uk", "net", "gov.uk", "com.vn", "gov", "ie", "fm",
    "app", "it", "ru", "int", "dev", "org.vn", "stream", "io", "top", "click",
    "de", "xyz", "info", "digital", "online", "ca", "sbs", "fr", "com.br", "co",
    "shop", "tk", "icu", "uk", "cc", "cn", "site", "me", "pl", "id",
    "in", "cfd", "us", "nl", "live", "com.au", "za", "club", "ar", "edu",
]

# Aggregate buckets — capture signal beyond individual one-hot.
VN_TLDS = {"vn", "com.vn", "org.vn", "gov.vn", "edu.vn", "net.vn", "ac.vn"}
# TLD rẻ / mới / phổ biến trong threat feeds (EDA 4: chiếm tỉ lệ lớn ở label=1)
SUSPICIOUS_TLDS = {
    "top", "xyz", "click", "sbs", "cfd", "icu", "stream", "digital",
    "online", "shop", "live", "tk", "ml", "ga", "gd", "site", "club",
    "fit", "rest", "loan", "men", "review", "trade",
}
# Baseline benign / institutional
COMMON_TLDS = {"com", "org", "net", "edu", "gov"}

VOWELS = set("aeiou")
CONSONANTS = set(string.ascii_lowercase) - VOWELS
DIGITS_SET = set(string.digits)
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SPLIT_RE = re.compile(r"[/_\-.]")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Bundled PSL snapshot only (no online fetch)
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
def extract_features(url_raw: str) -> dict:
    url_raw = (url_raw or "").strip()

    # ---- Case-sensitive / charset metrics MUST be computed before lower() ----
    ascii_letters = [c for c in url_raw if c.isalpha() and ord(c) < 128]
    n_upper = sum(1 for c in ascii_letters if c.isupper())
    mixed_case_ratio = n_upper / max(len(ascii_letters), 1)
    has_upper = int(n_upper > 0)
    has_control_chars = int(bool(CONTROL_RE.search(url_raw)))
    n_non_ascii = sum(1 for c in url_raw if ord(c) > 127)
    non_ascii_ratio = n_non_ascii / max(len(url_raw), 1)
    has_non_ascii = int(n_non_ascii > 0)

    # Lowercase for parsing & token-matching
    url = url_raw.lower()
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
    subdomain = ext.subdomain if ext else ""

    feats: dict = {}

    # ---- Structural ----
    feats["url_len"] = len(url_raw)
    feats["host_len"] = len(host)
    feats["path_len"] = len(path)
    feats["query_len"] = len(query)
    feats["fragment_len"] = len(fragment)
    feats["path_depth"] = sum(1 for s in path.split("/") if s)
    feats["num_query_params"] = n_query_params(query)
    feats["has_query"] = int(bool(query))
    feats["has_fragment"] = int(bool(fragment))
    feats["has_port"] = int(port is not None)
    feats["path_to_url_ratio"] = len(path) / max(len(url_raw), 1)
    # Bare-hostname flag — giúp model xử lý FP trên URL như "https://google.com"
    # (model 4 CNN-LSTM bị bug này, doc trong CLAUDE.md mục Known issues #6)
    feats["is_bare_hostname"] = int(path in ("", "/"))

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
    feats["has_double_slash_in_path"] = int("//" in path)

    # ---- Case (raw, before lower) ----
    feats["mixed_case_ratio"] = mixed_case_ratio
    feats["has_upper"] = has_upper

    # ---- Charset anomalies ----
    feats["has_control_chars"] = has_control_chars
    feats["has_non_ascii"] = has_non_ascii
    feats["non_ascii_ratio"] = non_ascii_ratio
    feats["has_punycode"] = int("xn--" in host)

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

    # ---- TLD one-hot (top 50) ----
    for tld in TOP_TLDS:
        key = "tld_" + tld.replace(".", "_")
        feats[key] = int(suffix == tld)
    feats["tld_other"] = int(bool(suffix) and suffix not in TOP_TLDS)
    feats["tld_missing"] = int(not suffix)
    feats["tld_label_count"] = suffix.count(".") + 1 if suffix else 0

    # ---- TLD aggregate buckets (EDA 4 driven) ----
    feats["tld_is_vn"] = int(suffix in VN_TLDS)
    feats["tld_is_suspicious"] = int(suffix in SUSPICIOUS_TLDS)
    feats["tld_is_common"] = int(suffix in COMMON_TLDS)

    # ---- Phishing keyword hits ----
    path_query = path + "?" + query
    n_phish = sum(1 for kw in PHISHING_KEYWORDS if kw in path_query)
    feats["num_phishing_keywords"] = n_phish
    feats["has_phishing_keyword"] = int(n_phish > 0)

    # ---- File extension ----
    last_seg = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
    if "." in last_seg:
        ext_str = "." + last_seg.rsplit(".", 1)[-1]
    else:
        ext_str = ""
    feats["has_malware_ext"] = int(ext_str in MALWARE_EXTS)
    feats["has_cdn_ext"] = int(ext_str in CDN_EXTS)
    feats["has_any_ext"] = int(bool(ext_str))

    return feats


# ============================================================================
# Feature order — chốt 1 lần qua probe URL (deterministic across workers)
# ============================================================================
_PROBE_URL = "https://www.Example.com/path/to/file.html?a=1&b=2"
FEATURE_NAMES: list[str] = list(extract_features(_PROBE_URL).keys())
N_FEATURES: int = len(FEATURE_NAMES)


def _worker_batch(urls: list[str]) -> np.ndarray:
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


def _read_input_csv(csv_path: Path) -> pd.DataFrame:
    """Read split.csv — QUOTE_ALL format from preprocess-data 4.py."""
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline()
    is_quoted = header.lstrip().startswith('"')
    if is_quoted:
        return pd.read_csv(
            csv_path,
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            doublequote=True,
        )
    return pd.read_csv(csv_path)


# ============================================================================
# Driver (multiprocess)
# ============================================================================
def process_split(name: str, pool: Pool) -> None:
    csv_path = SPLITS_DIR / name / "split.csv"
    print(f"\n[{name}] Loading {csv_path}...")
    df = _read_input_csv(csv_path)
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
    print("FEATURE EXTRACTION (XGBoost — model 4, real-world 19.68M URLs)")
    print("=" * 78)
    print(f"Splits dir : {SPLITS_DIR}")
    print(f"Output dir : {OUT_DIR}")
    print(f"Features   : {N_FEATURES}")
    print(f"Workers    : {N_WORKERS}  | chunk size: {CHUNK_SIZE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with Pool(processes=N_WORKERS) as pool:
        for split in ("train", "val", "test"):
            process_split(split, pool)

    cols_path = OUT_DIR / "xgb_4_feature_cols.json"
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"\nSaved feature columns -> {cols_path}  ({N_FEATURES} cols)")

    y_train = np.load(OUT_DIR / "train_y_xgb.npy")
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    meta = {
        "n_features": N_FEATURES,
        "top_tlds": TOP_TLDS,
        "vn_tlds": sorted(VN_TLDS),
        "suspicious_tlds": sorted(SUSPICIOUS_TLDS),
        "common_tlds": sorted(COMMON_TLDS),
        "phishing_keywords": list(PHISHING_KEYWORDS),
        "malware_exts": sorted(MALWARE_EXTS),
        "cdn_exts": sorted(CDN_EXTS),
        "pos_weight": pos_weight,
        "train_label_counts": {"0": n_neg, "1": n_pos},
        "source_splits": str(SPLITS_DIR),
        "notes": (
            "Feature set v2 cho XGBoost model 4 (real-world 19.68M URLs). "
            "Loại bỏ SCAM_BAIT/C2_PATHS/BRANDS/BENIGN_WORDS (synthetic-era artifacts "
            "từ model 2). Thêm: tld_is_vn / tld_is_suspicious / tld_is_common, "
            "mixed_case_ratio, has_upper, has_control_chars, has_non_ascii, "
            "non_ascii_ratio, has_punycode, has_double_slash_in_path, "
            "path_to_url_ratio, is_bare_hostname. TOP_TLDS rebuilt from EDA 4. "
            "Phishing keywords tỉa về real phishing kits. Pos_weight target ~4.0 "
            "(dataset 80/20 benign/mal)."
        ),
    }
    meta_path = OUT_DIR / "xgb_4_feature_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta            -> {meta_path}")
    print(f"  pos_weight (neg/pos): {pos_weight:.4f}")

    print("\n" + "=" * 78)
    print("FEATURE EXTRACTION DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""
CNN-LSTM lexical feature extraction (30 features) — kept in sync with
scripts/4-feat. extract-lexical_4.py in the training repo.

Returns np.ndarray(30,) float32 (RAW, before normalization).
Normalization is done separately using feat_stats.json from domain split.
"""
import math
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

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
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 30, f"Expected 30 features, got {N_FEATURES}"

LOG1P_FEATURES = {
    "url_length", "host_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_question_marks", "num_equals", "num_amps", "num_ats",
    "num_percents", "num_digits",
    "num_subdomains", "subdomain_length_max",
    "path_depth", "query_param_count",
}

COMMON_TLDS = frozenset({".com", ".net", ".org", ".io", ".dev", ".app"})
VOWELS = frozenset("aeiouAEIOU")


def _is_ip(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or not (0 <= int(p) <= 255):
            return False
    return True


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _get_last_tld(host: str) -> str:
    if "." not in host:
        return ""
    return "." + host.rsplit(".", 1)[-1].lower()


def extract_features(url: str) -> np.ndarray:
    """Return np.ndarray(N_FEATURES,) float32 RAW (not normalized)."""
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    if not url:
        return feat

    n_url = len(url)
    feat[0] = float(n_url)

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
        scheme, netloc, path, query = "", "", "", ""

    host = netloc
    port = ""
    if ":" in netloc:
        h, p = netloc.rsplit(":", 1)
        if p.isdigit():
            host, port = h, p

    feat[1] = float(len(host))
    feat[2] = float(len(path))
    feat[3] = float(len(query))

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

    if n_url > 0:
        feat[14] = n_digits / n_url
        feat[15] = n_letters / n_url
        feat[17] = (n_url - n_digits - n_letters) / n_url
    feat[16] = (n_vowels / n_letters) if n_letters else 0.0

    host_is_ip = _is_ip(host)
    feat[18] = 1.0 if host_is_ip else 0.0
    feat[19] = 1.0 if port else 0.0
    feat[20] = 1.0 if scheme == "https" else 0.0

    if host and not host_is_ip:
        labels = [la for la in host.split(".") if la]
        if len(labels) > 2:
            sub_labels = labels[:-2]
            feat[21] = float(len(sub_labels))
            feat[22] = float(max(len(s) for s in sub_labels))

    feat[23] = 1.0 if host.endswith(".vn") else 0.0
    tld = _get_last_tld(host)
    feat[24] = 1.0 if tld in COMMON_TLDS else 0.0

    feat[25] = float(_shannon_entropy(host))
    feat[26] = 1.0 if "xn--" in host else 0.0
    feat[27] = float(path.count("/"))
    feat[28] = 1.0 if "//" in path else 0.0

    if query:
        feat[29] = float(query.count("&") + 1)

    return feat

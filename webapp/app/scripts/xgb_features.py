"""
XGBoost feature extraction (105 features) — kept in sync with
scripts/3. feature-extract_xgb_4.py in the training repo.

Returns a dict with feature name → float value.
FEATURE_NAMES and N_FEATURES are derived deterministically via a probe URL.
"""
import math
import re
import string
from collections import Counter
from urllib.parse import parse_qs, urlparse

import tldextract

PHISHING_KEYWORDS = (
    "login", "signin", "secure", "verify", "account", "password",
    "bank", "wallet", "confirm", "auth", "session", "recover",
    "unlock", "reset", "support", "billing",
)

MALWARE_EXTS = {".exe", ".apk", ".dmg", ".sh", ".scr", ".dll", ".bat", ".so", ".msi"}
CDN_EXTS = {".js", ".json", ".css", ".xml", ".ts", ".jpg", ".yaml", ".gif",
            ".cjs", ".png", ".woff", ".woff2", ".svg", ".ico"}

TOP_TLDS = [
    "com", "vn", "org", "co.uk", "net", "gov.uk", "com.vn", "gov", "ie", "fm",
    "app", "it", "ru", "int", "dev", "org.vn", "stream", "io", "top", "click",
    "de", "xyz", "info", "digital", "online", "ca", "sbs", "fr", "com.br", "co",
    "shop", "tk", "icu", "uk", "cc", "cn", "site", "me", "pl", "id",
    "in", "cfd", "us", "nl", "live", "com.au", "za", "club", "ar", "edu",
]

VN_TLDS = {"vn", "com.vn", "org.vn", "gov.vn", "edu.vn", "net.vn", "ac.vn"}
SUSPICIOUS_TLDS = {
    "top", "xyz", "click", "sbs", "cfd", "icu", "stream", "digital",
    "online", "shop", "live", "tk", "ml", "ga", "gd", "site", "club",
    "fit", "rest", "loan", "men", "review", "trade",
}
COMMON_TLDS = {"com", "org", "net", "edu", "gov"}

VOWELS = set("aeiou")
CONSONANTS = set(string.ascii_lowercase) - VOWELS
DIGITS_SET = set(string.digits)
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Bundled PSL snapshot only — no network calls, no disk cache
_tld_extractor = tldextract.TLDExtract(
    suffix_list_urls=(), fallback_to_snapshot=True, cache_dir=False
)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _longest_run(s: str, charset: set) -> int:
    longest = run = 0
    for ch in s:
        if ch in charset:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return longest


def _safe_parse(url: str):
    try:
        return urlparse(url)
    except Exception:
        return None


def _n_query_params(query: str) -> int:
    if not query:
        return 0
    try:
        return len(parse_qs(query, keep_blank_values=True))
    except Exception:
        return query.count("&") + 1


def extract_features(url_raw: str) -> dict:
    """Return dict of feature_name → float value (105 features)."""
    url_raw = (url_raw or "").strip()

    ascii_letters = [c for c in url_raw if c.isalpha() and ord(c) < 128]
    n_upper = sum(1 for c in ascii_letters if c.isupper())
    mixed_case_ratio = n_upper / max(len(ascii_letters), 1)
    has_upper = int(n_upper > 0)
    has_control_chars = int(bool(CONTROL_RE.search(url_raw)))
    n_non_ascii = sum(1 for c in url_raw if ord(c) > 127)
    non_ascii_ratio = n_non_ascii / max(len(url_raw), 1)
    has_non_ascii = int(n_non_ascii > 0)

    url = url_raw.lower()
    parsed = _safe_parse(url)
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

    feats["url_len"] = len(url_raw)
    feats["host_len"] = len(host)
    feats["path_len"] = len(path)
    feats["query_len"] = len(query)
    feats["fragment_len"] = len(fragment)
    feats["path_depth"] = sum(1 for s in path.split("/") if s)
    feats["num_query_params"] = _n_query_params(query)
    feats["has_query"] = int(bool(query))
    feats["has_fragment"] = int(bool(fragment))
    feats["has_port"] = int(port is not None)
    feats["path_to_url_ratio"] = len(path) / max(len(url_raw), 1)
    feats["is_bare_hostname"] = int(path in ("", "/"))

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

    feats["mixed_case_ratio"] = mixed_case_ratio
    feats["has_upper"] = has_upper

    feats["has_control_chars"] = has_control_chars
    feats["has_non_ascii"] = has_non_ascii
    feats["non_ascii_ratio"] = non_ascii_ratio
    feats["has_punycode"] = int("xn--" in host)

    host_digits = sum(c.isdigit() for c in host)
    feats["host_digit_count"] = host_digits
    feats["host_digit_density"] = host_digits / max(len(host), 1)
    feats["host_hyphen_count"] = host.count("-")
    feats["host_dot_count"] = host.count(".")
    feats["host_num_subdomains"] = (
        len([s for s in subdomain.split(".") if s]) if subdomain else 0
    )
    feats["host_starts_with_www"] = int(host.startswith("www."))
    feats["host_longest_digit_run"] = _longest_run(host, DIGITS_SET)
    feats["host_longest_consonant_run"] = _longest_run(host, CONSONANTS)

    host_letters = [c for c in host if c.isalpha()]
    feats["host_vowel_ratio"] = (
        sum(c in VOWELS for c in host_letters) / len(host_letters)
        if host_letters else 0.0
    )
    feats["host_letter_count"] = len(host_letters)

    feats["entropy_url"] = _shannon_entropy(url)
    feats["entropy_host"] = _shannon_entropy(host)
    feats["entropy_path"] = _shannon_entropy(path)

    feats["is_http"] = int(scheme == "http")
    feats["is_https"] = int(scheme == "https")
    feats["is_ip_host"] = int(bool(IP_RE.match(host)))

    for tld in TOP_TLDS:
        key = "tld_" + tld.replace(".", "_")
        feats[key] = int(suffix == tld)
    feats["tld_other"] = int(bool(suffix) and suffix not in TOP_TLDS)
    feats["tld_missing"] = int(not suffix)
    feats["tld_label_count"] = suffix.count(".") + 1 if suffix else 0

    feats["tld_is_vn"] = int(suffix in VN_TLDS)
    feats["tld_is_suspicious"] = int(suffix in SUSPICIOUS_TLDS)
    feats["tld_is_common"] = int(suffix in COMMON_TLDS)

    path_query = path + "?" + query
    n_phish = sum(1 for kw in PHISHING_KEYWORDS if kw in path_query)
    feats["num_phishing_keywords"] = n_phish
    feats["has_phishing_keyword"] = int(n_phish > 0)

    last_seg = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
    if "." in last_seg:
        ext_str = "." + last_seg.rsplit(".", 1)[-1]
    else:
        ext_str = ""
    feats["has_malware_ext"] = int(ext_str in MALWARE_EXTS)
    feats["has_cdn_ext"] = int(ext_str in CDN_EXTS)
    feats["has_any_ext"] = int(bool(ext_str))

    return feats


# Feature order is fixed by this probe URL (deterministic)
_PROBE_URL = "https://www.Example.com/path/to/file.html?a=1&b=2"
FEATURE_NAMES: list[str] = list(extract_features(_PROBE_URL).keys())
N_FEATURES: int = len(FEATURE_NAMES)

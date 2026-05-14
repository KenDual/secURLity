"""
Detailed EDA for `dataset/dataset 3 (vn).csv`
=============================================

Dataset 3 = ~12.5M benign  (REAL URLs crawl từ ~275 seed VN-targeted)
          + ~2.6M malicious (REAL URLs từ 13 feed công khai: URLhaus, ThreatFox,
                             Phishing.Database ACTIVE/INACTIVE/NEW-today,
                             Phishing.Army, OpenPhish, tweetfeed, DigitalSide,
                             CertPL, cybercrime-tracker, Hagezi TIF, Malsilo)

Khác EDA 2:
  - Cả 2 phía ĐỀU LÀ REAL (không Char-RNN/rule-based generator).
  - Class ratio tự nhiên ~86/14 (benign/malicious) thay vì 70/30 ép buộc.
  - CSV input là QUOTE_ALL (sản phẩm của `2. prepare-dataset 3.py`) — csv.DictReader
    handle transparent, không cần option đặc biệt.
  - Section MỚI: VIETNAMESE TARGETING — đếm URL có .vn TLD hoặc brand VN match,
    chia theo label, để biết model 3 thực sự "phục vụ người Việt" tới đâu.
  - VN sub-TLDs (.com.vn, .edu.vn, .gov.vn, .org.vn, .net.vn, .ac.vn) thêm
    vào KNOWN_TLDS để TLD analysis chính xác.
  - "CHAR-RNN ARTIFACT" section đổi tên thành "SUSPICIOUS HOST ARTIFACTS" —
    cùng heuristics, vẫn hữu ích vì DGA/typosquat thực tế cũng có artifact này.
  - BỎ "BENIGN WORD COVERAGE" — set tiếng Anh không phù hợp data crawl VN
    (URL VN dùng `tin-tuc`, `bai-viet`, ...). Section này trong eda 2 chỉ
    để validate generator, ở real data không có generator để validate.
  - "KNOWN DOMAIN POOLS" -> "ANCHOR DOMAINS": pool nhỏ ~30-40 domain mỗi
    category chỉ phủ vài % real data. Giữ lại như sanity check (xác nhận
    các site lớn có trong dataset), KHÔNG phải coverage report.
  - `classify_url()` bỏ các nhánh `benign_ecommerce/news/tech_saas/...` vì
    pool quá nhỏ so với real data — kết quả sẽ mostly "unknown" (đúng với
    bản chất long-tail của real data).

Usage (PowerShell):
    .\\venv\\Scripts\\Activate.ps1
    python "scripts/2. eda 3.py"
    python "scripts/2. eda 3.py" --path "dataset/dataset 3 (vn).csv" --max-rows 1000000
    python "scripts/2. eda 3.py" --output "dataset/eda-result 3.txt"
"""

import argparse
import csv
import math
import random
import re
import string as _string
import sys
import time
from collections import Counter, defaultdict
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

# Tăng csv field size (URL có thể dài bất thường)
csv.field_size_limit(10_000_000)


# ---------------------------------------------------------------------------
# Word pools — giữ nguyên từ EDA 2 (vẫn hữu ích cho categorization)
# ---------------------------------------------------------------------------

# NOTE: BENIGN_WORDS đã bị BỎ ở model 3.
# Lý do: trong eda 2 set này mirror generator để validate coverage. Real data
# không có generator để validate, và URL VN dùng path words tiếng Việt
# (`tin-tuc`, `bai-viet`, `san-pham`, ...) — set tiếng Anh sẽ ra số nhỏ và
# vô nghĩa. Coi top-host frequency hữu dụng hơn nhiều.

# ANCHOR pools (dùng ở section ANCHOR DOMAINS): KHÔNG phải coverage list,
# chỉ là sanity check vài site lớn có xuất hiện trong dataset không.
ECOMMERCE_DOMAINS = {
    "amazon.com", "ebay.com", "etsy.com", "shopify.com", "walmart.com",
    "bestbuy.com", "target.com", "newegg.com", "alibaba.com", "aliexpress.com",
    "rakuten.co.jp", "mercadolibre.com.ar", "flipkart.com", "tokopedia.com",
    "lazada.com", "shopee.com", "jd.com", "taobao.com", "tmall.com",
    "tiki.vn", "sendo.vn", "shopee.vn", "lazada.vn", "fptshop.com.vn",
    "thegioididong.com", "dienmayxanh.com", "nguyenkim.com", "cellphones.com.vn",
    "hoanghamobile.com", "concung.com", "bibomart.com.vn",
}

NEWS_DOMAINS = {
    "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com", "apnews.com",
    "bloomberg.com", "ft.com", "nytimes.com", "theguardian.com",
    "techcrunch.com", "wired.com", "theverge.com", "engadget.com",
    "vnexpress.net", "tuoitre.vn", "thanhnien.vn", "vietnamnet.vn",
    "dantri.com.vn", "kenh14.vn", "znews.vn", "zingnews.vn", "laodong.vn",
    "baomoi.com", "nhandan.vn", "vov.vn", "vtv.vn", "vtc.vn",
    "cand.com.vn", "qdnd.vn", "vietnamplus.vn", "24h.com.vn",
    "tienphong.vn", "plo.vn", "sggp.org.vn", "baochinhphu.vn",
    "cafef.vn", "cafebiz.vn", "vneconomy.vn", "vietstock.vn",
    "afamily.vn", "eva.vn", "webtretho.com", "soha.vn",
}

TECH_SAAS_DOMAINS = {
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com",
    "aws.amazon.com", "console.cloud.google.com", "portal.azure.com",
    "cloudflare.com", "vercel.com", "netlify.com", "heroku.com",
    "notion.so", "atlassian.com", "trello.com", "asana.com",
    "figma.com", "canva.com", "miro.com",
    "slack.com", "discord.com", "zoom.us", "teams.microsoft.com",
    "dropbox.com", "drive.google.com", "onedrive.live.com",
    "stripe.com", "paypal.com", "momo.vn", "zalopay.vn", "vnpay.vn",
}

EDUCATION_DOMAINS = {
    "coursera.org", "udemy.com", "edx.org", "khanacademy.org",
    "wikipedia.org", "en.wikipedia.org", "vi.wikipedia.org",
    "vi.wiktionary.org", "scholar.google.com",
    "vnu.edu.vn", "vnuhcm.edu.vn", "hcmus.edu.vn", "hust.edu.vn",
    "hcmut.edu.vn", "neu.edu.vn", "ueh.edu.vn", "ftu.edu.vn",
    "uef.edu.vn", "fpt.edu.vn", "rmit.edu.vn", "vinuni.edu.vn",
    "tdtu.edu.vn", "iuh.edu.vn", "hcmute.edu.vn",
    "hocmai.vn", "hoc24.vn", "vuihoc.vn", "olm.vn", "vietjack.com",
    "loigiaihay.com", "tailieu.vn", "vndoc.com", "123docz.net",
    "thuvienphapluat.vn", "luatvietnam.vn",
}

SOCIAL_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "reddit.com", "linkedin.com", "pinterest.com", "tumblr.com",
    "tiktok.com", "youtube.com", "twitch.tv",
    "discord.com", "telegram.org", "whatsapp.com",
    "zalo.me", "voz.vn", "tinhte.vn", "lamchame.com",
    "webtretho.com", "nhattao.com",
}

CDN_PREFIXES = {
    "cdn", "cdn1", "cdn2", "cdn-us", "cdn-eu",
    "static", "static1", "assets", "media", "img", "images",
    "files", "uploads", "content", "resources",
    "s3", "storage", "blob",
    "prod", "production", "staging", "dev",
}

CDN_EXTENSIONS = {
    ".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx",
    ".css", ".scss", ".less",
    ".min.js", ".min.css", ".bundle.js", ".bundle.css",
    ".chunk.js", ".worker.js", ".map",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".webm", ".ogg",
    ".json", ".xml",
}

SHORTLINK_DOMAINS = {
    "bit.ly", "tinyurl.com", "ow.ly", "t.co", "goo.gl",
    "is.gd", "v.gd", "tiny.cc", "buff.ly", "j.mp", "youtu.be",
    "imgur.com", "i.imgur.com", "pastebin.com",
}

COMMON_SUBDOMAINS = {
    "mail", "webmail", "email", "smtp", "pop", "imap", "mx",
    "api", "api-v1", "api-v2", "rest", "graphql",
    "www", "www2", "web", "portal", "app", "apps",
    "m", "mobile", "touch", "lite", "amp",
    "cdn", "static", "assets", "media", "img", "images",
    "auth", "sso", "login", "id", "accounts", "oauth",
    "dev", "staging", "test", "qa", "beta", "preview",
    "admin", "manage", "panel", "console",
    "search", "shop", "blog", "docs", "help", "support",
    "download", "downloads", "update", "updates",
    "us", "eu", "asia", "ap", "vn",
}

# Common TLDs — thêm VN sub-TLDs để get_tld() phân loại chính xác
KNOWN_TLDS = sorted({
    ".co.uk", ".co.jp", ".co.in", ".co.nz", ".com.au", ".com.br", ".com.cn",
    ".com.vn", ".net.vn", ".edu.vn", ".gov.vn", ".org.vn", ".ac.vn",
    ".net.au", ".org.uk", ".ac.uk", ".gov.uk",
    ".com", ".org", ".net", ".edu", ".gov", ".mil", ".int",
    ".io", ".dev", ".app", ".info", ".biz", ".mobi", ".name",
    ".blog", ".shop", ".tech", ".site", ".online", ".space", ".store",
    ".de", ".fr", ".cn", ".au", ".ca", ".jp", ".kr", ".vn", ".ru", ".uk",
    ".br", ".in", ".it", ".es", ".nl", ".pl", ".cz", ".ro", ".se", ".no",
    ".tk", ".ml", ".ga", ".cf", ".top", ".xyz", ".click", ".club",
    ".live", ".cc", ".pw", ".to", ".bid", ".download", ".party", ".faith",
    ".work", ".gdn", ".trade", ".stream", ".red", ".win", ".cricket",
    ".loan", ".racing", ".review", ".ws", ".me", ".tv", ".fm",
    ".social", ".page",
}, key=len, reverse=True)

LEGIT_BRANDS = {
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "tesla",
    "facebook", "instagram", "twitter", "linkedin", "dropbox", "slack",
    "github", "gitlab", "docker", "aws", "azure",
    "wellsfargo", "bankofamerica", "chase", "citibank", "hsbc",
    "dhl", "fedex", "ups", "usps",
    # VN brands
    "vietcombank", "techcombank", "vietinbank", "vpbank", "tpbank",
    "sacombank", "agribank", "bidv", "mbbank", "acb", "hdbank",
    "momo", "zalopay", "vnpay", "shopeepay",
    "shopee", "tiki", "lazada", "sendo",
    "viettel", "vnpt", "mobifone", "vinaphone", "fpt",
    "vietnamairlines", "vietjet", "bamboo",
}

SUSPICIOUS_WORDS = [
    "secure", "verify", "update", "confirm", "alert", "warning",
    "unlock", "restore", "validate", "authenticate", "suspended",
    "limited", "urgent", "action-required", "login", "signin",
    "unusual-activity", "verify-account", "security-alert",
    "immediate-action", "click-here", "act-now",
]

C2_PATHS = [
    "/cmd", "/c2", "/gate", "/bot", "/beacon", "/checkin", "/tasks",
    "/poll", "/update", "/ping", "/report", "/data",
    "/i", "/bin.sh", "/shell", "/exec", "/run", "/config",
    "/sync", "/push", "/pull", "/fetch", "/load", "/plugin",
]

SCAM_BAIT = [
    "you-won", "claim-prize", "free-gift", "limited-offer", "congratulations",
    "lucky-winner", "exclusive-deal", "earn-money-fast", "lose-weight-now",
    "get-rich-quick", "act-now", "final-notice",
    "verify-account", "confirm-password", "update-payment", "restore-access",
    "claim-reward", "free-trial", "limited-time", "last-chance",
]

MALWARE_EXECUTABLES = {
    ".exe", ".bat", ".ps1", ".msi", ".dmg", ".apk",
    ".deb", ".rpm", ".iso", ".dll", ".scr", ".vbs",
    ".jar", ".so", ".dylib", ".sh", ".elf",
}

MALWARE_LURE_WORDS = {
    "update", "install", "setup", "patch", "fix", "crack", "keygen",
    "loader", "installer", "driver", "tool", "utility", "codec",
    "player", "reader", "viewer", "converter", "optimizer", "cleaner",
    "antivirus", "booster",
}

# ALLOWED_CHARS từ CNN-LSTM pipeline (49 chars)
ALLOWED_CHARS = set(_string.ascii_lowercase + _string.digits + "/:.-_?=&#@%+~")
VOWELS = set("aeiou")


# ---------------------------------------------------------------------------
# Compiled regex — gộp keyword sets thành 1 pattern duy nhất.
# Trên 15M URLs, substring check kiểu `any(w in url for w in WORDS)` rất chậm
# (N comparisons/URL); regex C-level alternation nhanh hơn 5-10x.
# ---------------------------------------------------------------------------

_SUSPICIOUS_RE     = re.compile("|".join(re.escape(w) for w in SUSPICIOUS_WORDS))
_SCAM_RE           = re.compile("|".join(re.escape(b) for b in SCAM_BAIT))
_MALWARE_LURE_RE   = re.compile("|".join(re.escape(w) for w in MALWARE_LURE_WORDS))
_LEGIT_BRANDS_RE   = re.compile("|".join(re.escape(b) for b in LEGIT_BRANDS))
_C2_PATHS_RE       = re.compile("^(?:" + "|".join(re.escape(p) for p in C2_PATHS) + ")")


def _make_host_suffix_re(domains):
    """Match nếu host bằng hoặc kết thúc bằng '.<domain>'."""
    return re.compile(
        r"(?:^|\.)(?:" + "|".join(re.escape(d) for d in domains) + r")$"
    )


_ECOMMERCE_HOST_RE = _make_host_suffix_re(ECOMMERCE_DOMAINS)
_NEWS_HOST_RE      = _make_host_suffix_re(NEWS_DOMAINS)
_TECH_SAAS_HOST_RE = _make_host_suffix_re(TECH_SAAS_DOMAINS)
_EDUCATION_HOST_RE = _make_host_suffix_re(EDUCATION_DOMAINS)
_SOCIAL_HOST_RE    = _make_host_suffix_re(SOCIAL_DOMAINS)


# ---------------------------------------------------------------------------
# VIETNAMESE TARGETING heuristics — đồng bộ với scripts/fetch-malicious-all.py
# ---------------------------------------------------------------------------

VN_TLD_RE = re.compile(r"\.vn(?:[:/?#]|$)", re.IGNORECASE)

VN_BRAND_RE = re.compile(
    r"vietcom|techcom|vietin|vpbank|tpbank|sacomb|agribank|"
    r"\bbidv\b|mbbank|acbbank|hdbank|seabank|lpbank|eximbank|"
    r"\bmsb\b|\bocb\b|vietbank|nam-?a-?bank|namabank|saigonbank|abbank|"
    r"momo|zalopay|vnpay|shopeepay|finhay|tikop|"
    r"shopee|tiki|lazada|sendo|tgdd|thegioididong|dienmayxanh|"
    r"fptshop|cellphones|hoanghamobile|nguyenkim|"
    r"vnpost|vnpt|viettel|mobifone|vinaphone|fpt-?telecom|"
    r"vietnamairlines|vietjet|bambooairways|vinpearl|vinmec|vinfast|"
    r"chinhphu|baohiemxahoi|bhxh|gov-?vn|"
    r"vietnam|vietnamese|\bvn-?[a-z]{2,}|"
    r"hanoi|saigon|hochiminh|danang|haiphong|cantho",
    re.IGNORECASE,
)

# Sub-TLDs VN cần phân biệt riêng
VN_SUB_TLDS = (".com.vn", ".edu.vn", ".gov.vn", ".org.vn", ".net.vn", ".ac.vn")


def is_vn_tld(host: str) -> bool:
    return bool(host) and host.lower().endswith(".vn") or False


def is_vn_target(url: str, host: str) -> bool:
    """Cờ tổng: TLD .vn HOẶC URL chứa brand/keyword VN."""
    if is_vn_tld(host):
        return True
    if VN_TLD_RE.search(url):
        return True
    if VN_BRAND_RE.search(url):
        return True
    return False


# ---------------------------------------------------------------------------
# Streaming utilities
# ---------------------------------------------------------------------------

class RunningStats:
    """Welford online mean/variance."""
    __slots__ = ("count", "mean", "M2", "min", "max")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min: Optional[float] = None
        self.max: Optional[float] = None

    def update(self, value: float) -> None:
        self.count += 1
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class ReservoirSampler:
    """Reservoir sampling (Algorithm R) — phân phối đều khi N không biết trước."""
    __slots__ = ("size", "n", "sample", "rng")

    def __init__(self, size: int, seed: int = 42) -> None:
        self.size = size
        self.n = 0
        self.sample: List = []
        self.rng = random.Random(seed)

    def add(self, value) -> None:
        self.n += 1
        if len(self.sample) < self.size:
            self.sample.append(value)
            return
        j = self.rng.randrange(self.n)
        if j < self.size:
            self.sample[j] = value


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = int(round((p / 100) * (len(s) - 1)))
    return float(s[idx])


# ---------------------------------------------------------------------------
# URL parsing helpers
# ---------------------------------------------------------------------------

def is_ip(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        v = int(part)
        if v < 0 or v > 255:
            return False
    return True


def split_host_port(netloc: str) -> Tuple[str, Optional[str]]:
    if not netloc:
        return "", None
    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        if port.isdigit():
            return host, port
    return netloc, None


def get_tld(host: str) -> str:
    h = host.lower()
    for tld in KNOWN_TLDS:
        if h.endswith(tld):
            return tld
    if "." in h:
        return "." + h.rsplit(".", 1)[-1]
    return "<none>"


def get_subdomain(host: str, tld: str) -> Optional[str]:
    if not host or host == "<none>":
        return None
    labels = host.split(".")
    tld_labels = tld.strip(".").split(".")
    if len(labels) <= len(tld_labels) + 1:
        return None
    return labels[0]


def host_endswith(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def parse_url(raw_url: str):
    url = raw_url.strip()
    if not url:
        return "", "", "", "", "", None
    # Python 3.12+ raises ValueError trên IPv6 bracket sai hoặc netloc lạ.
    # Real malicious URLs đôi khi có [, ] ở vị trí bất thường -> phải tolerate.
    try:
        if "://" not in url:
            parts = urlsplit("http://" + url)
            scheme = ""
        else:
            parts = urlsplit(url)
            scheme = parts.scheme.lower()
        host, port = split_host_port(parts.netloc.lower())
        return url, scheme, host, parts.path or "/", parts.query or "", port
    except ValueError:
        # URL malformed (vd IPv6 không đóng bracket) — coi như không có host/path
        return url, "", "", "/", "", None


# ---------------------------------------------------------------------------
# Suspicious host artifact detectors (vẫn hữu ích cho real malicious)
# ---------------------------------------------------------------------------

def host_quality_signals(host: str) -> dict:
    out = {
        "has_double_dot": ".." in host,
        "no_dot": "." not in host,
        "starts_with_dot": host.startswith("."),
        "ends_with_dot": host.endswith("."),
        "very_short_tld_label": False,
        "long_repeat_run": False,
        "low_vowel_ratio": False,
    }
    if "." in host:
        last = host.rsplit(".", 1)[-1]
        if 1 <= len(last) <= 2:
            out["very_short_tld_label"] = True

    run = 1
    for i in range(1, len(host)):
        if host[i] == host[i - 1] and host[i].isalnum():
            run += 1
            if run >= 4:
                out["long_repeat_run"] = True
                break
        else:
            run = 1

    base = host.split(".")[0] if host else ""
    base_alpha = "".join(c for c in base if c.isalpha())
    if len(base_alpha) >= 8:
        vowel_ratio = sum(c in VOWELS for c in base_alpha) / len(base_alpha)
        if vowel_ratio < 0.2:
            out["low_vowel_ratio"] = True

    return out


def looks_like_dga(host: str) -> bool:
    base = host.split(".")[0] if host else ""
    if len(base) < 10:
        return False
    base_alnum = "".join(c for c in base if c.isalnum())
    if not base_alnum:
        return False
    digit_ratio = sum(c.isdigit() for c in base_alnum) / len(base_alnum)
    base_alpha = "".join(c for c in base if c.isalpha())
    if base_alpha:
        vowel_ratio = sum(c in VOWELS for c in base_alpha) / len(base_alpha)
    else:
        vowel_ratio = 1.0
    return digit_ratio > 0.30 or vowel_ratio < 0.18


# ---------------------------------------------------------------------------
# Heuristic categorization
# ---------------------------------------------------------------------------

_SUSPICIOUS_TLDS = frozenset({
    ".tk", ".ml", ".ga", ".cf", ".top", ".xyz", ".click", ".online", ".site",
})


def classify_url(url_lower: str, scheme: str, host: str, path: str,
                 query: str, tld: str, ext: Optional[str]) -> str:
    """Lưu ý: nhận url_lower (đã lowercase) để tránh double-work — caller phải pass lower_url."""
    if is_ip(host):
        return "malicious_ip_based"
    if "/download/" in path and ext in MALWARE_EXECUTABLES:
        return "malicious_malware_dist"
    if _SCAM_RE.search(url_lower):
        return "malicious_spam_scam"
    if host and _LEGIT_BRANDS_RE.search(host) and (
        "/login" in path or "/signin" in path or "/account" in path
        or "secure" in host or "auth" in host
    ):
        return "malicious_phishing_brand_spoof"
    if looks_like_dga(host):
        return "malicious_dga_like"
    if _C2_PATHS_RE.match(path):
        return "malicious_c2_path"
    if _SUSPICIOUS_RE.search(url_lower) and tld in _SUSPICIOUS_TLDS:
        return "malicious_suspicious_keyword"

    # NOTE: bỏ các nhánh benign theo domain pool — pool quá nhỏ (~30-40 domain)
    # so với long-tail thực tế (hàng nghìn host khác nhau). Chỉ giữ 2 nhánh
    # benign generic dựa trên signal cấu trúc (CDN ext, common subdomain).
    if host in SHORTLINK_DOMAINS:
        return "benign_shortlink"
    if ext in CDN_EXTENSIONS:
        return "benign_cdn"
    sub = get_subdomain(host, tld)
    if sub and sub in COMMON_SUBDOMAINS:
        return "benign_subdomain"

    return "unknown"


# ---------------------------------------------------------------------------
# Reporting helpers (tee to stdout + optional file)
# ---------------------------------------------------------------------------

def make_printer(output_path: Optional[str]):
    f_out = open(output_path, "w", encoding="utf-8") if output_path else None
    sinks = [sys.stdout] + ([f_out] if f_out else [])

    def _print(*args, **kw):
        for fp in sinks:
            print(*args, file=fp, **kw)

    def _close():
        if f_out:
            f_out.close()
    return _print, _close


def section(title: str, _print) -> None:
    _print("\n" + "=" * 90)
    _print(title)
    _print("=" * 90)


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "0.00%"
    return f"{(num / den) * 100:.2f}%"


def print_counter(counter: Counter, _print, top_n: int = 15, label: str = "") -> None:
    if label:
        _print(f"\n{label}")
    for key, val in counter.most_common(top_n):
        _print(f"  {str(key):>30} : {val:,}")


# ---------------------------------------------------------------------------
# Main EDA loop
# ---------------------------------------------------------------------------

def run_eda(csv_path: str, max_rows: int, sample_size: int,
            progress_every: int, output_path: Optional[str]) -> None:
    start = time.time()
    _print, _close = make_printer(output_path)

    # ---- Counters / state ----
    total = 0
    missing = 0
    invalid_label = 0
    label_counts = Counter()
    scheme_counts = Counter()
    scheme_by_label = defaultdict(Counter)
    tld_counts = Counter()
    tld_by_label = defaultdict(Counter)
    port_counts = Counter()

    url_len_stats = RunningStats()
    url_len_by_label = defaultdict(RunningStats)
    url_len_sample = ReservoirSampler(sample_size)
    url_len_sample_by_label = defaultdict(lambda: ReservoirSampler(sample_size))

    host_len_stats = RunningStats()
    host_len_by_label = defaultdict(RunningStats)
    path_len_stats = RunningStats()
    path_len_by_label = defaultdict(RunningStats)
    query_len_stats = RunningStats()
    query_len_by_label = defaultdict(RunningStats)

    path_depth_stats = RunningStats()
    path_depth_by_label = defaultdict(RunningStats)
    path_depth_sample = ReservoirSampler(sample_size)

    digit_count_stats = RunningStats()
    digit_count_by_label = defaultdict(RunningStats)
    hyphen_count_stats = RunningStats()
    hyphen_count_by_label = defaultdict(RunningStats)
    dot_count_stats = RunningStats()
    dot_count_by_label = defaultdict(RunningStats)
    special_count_stats = RunningStats()
    special_count_by_label = defaultdict(RunningStats)

    host_dots_stats = RunningStats()
    host_dots_by_label = defaultdict(RunningStats)
    host_hyphens_stats = RunningStats()
    host_hyphens_by_label = defaultdict(RunningStats)
    host_digits_stats = RunningStats()
    host_digits_by_label = defaultdict(RunningStats)

    query_count = Counter()
    query_params_stats = RunningStats()
    query_params_by_label = defaultdict(RunningStats)

    ip_count_by_label = Counter()
    https_count_by_label = Counter()

    ext_counts = Counter()
    cdn_ext_count = Counter()
    malware_ext_count = Counter()
    cdn_ext_by_label = Counter()
    malware_ext_by_label = Counter()

    suspicious_hits = Counter()
    scam_hits = Counter()
    brand_hits = Counter()
    c2_hits = Counter()

    suspicious_by_label = Counter()
    scam_by_label = Counter()
    brand_by_label = Counter()
    c2_by_label = Counter()
    malware_lure_by_label = Counter()

    artifact_signals = ["has_double_dot", "no_dot", "starts_with_dot",
                        "ends_with_dot", "very_short_tld_label",
                        "long_repeat_run", "low_vowel_ratio"]
    artifact_by_label = defaultdict(Counter)
    dga_by_label = Counter()

    char_freq = Counter()
    out_of_vocab_chars = Counter()
    out_of_vocab_by_label = Counter()

    category_counts = Counter()
    category_by_label = defaultdict(Counter)

    known_pool_hits = defaultdict(Counter)
    top_hosts = Counter()
    top_hosts_by_label = defaultdict(Counter)

    malformed_sample_by_label = defaultdict(lambda: ReservoirSampler(20))

    # --- VN-specific counters ---
    vn_tld_by_label = Counter()           # host kết thúc .vn
    vn_brand_by_label = Counter()         # URL match VN_BRAND_RE
    vn_target_by_label = Counter()        # union 2 cờ trên
    vn_subtld_counts = Counter()          # .com.vn / .edu.vn / .gov.vn / ...
    vn_subtld_by_label = defaultdict(Counter)
    vn_target_sample_by_label = defaultdict(lambda: ReservoirSampler(20))

    section("EDA START", _print)
    _print(f"File           : {csv_path}")
    _print(f"Max rows       : {'ALL' if max_rows <= 0 else f'{max_rows:,}'}")
    _print(f"Sample size    : {sample_size:,}")
    _print(f"Output file    : {output_path or '(stdout only)'}")

    # ---------------- Main pass ----------------
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows > 0 and total >= max_rows:
                break
            total += 1

            url = row.get("url", "").strip()
            label_raw = row.get("label", "").strip()
            if not url or label_raw == "":
                missing += 1
                continue
            try:
                label = int(label_raw)
            except ValueError:
                invalid_label += 1
                continue

            label_counts[label] += 1

            parsed_url, scheme, host, path, query, port = parse_url(url)
            if not parsed_url:
                missing += 1
                continue

            scheme_counts[scheme or "<none>"] += 1
            scheme_by_label[label][scheme or "<none>"] += 1
            if scheme == "https":
                https_count_by_label[label] += 1
            if port:
                port_counts[port] += 1

            tld = get_tld(host) if host else "<none>"
            tld_counts[tld] += 1
            tld_by_label[label][tld] += 1

            url_len = len(url)
            url_len_stats.update(url_len)
            url_len_by_label[label].update(url_len)
            url_len_sample.add(url_len)
            url_len_sample_by_label[label].add(url_len)

            host_len_stats.update(len(host))
            host_len_by_label[label].update(len(host))
            path_len_stats.update(len(path))
            path_len_by_label[label].update(len(path))
            query_len_stats.update(len(query))
            query_len_by_label[label].update(len(query))

            digits = sum(c.isdigit() for c in url)
            hyphens = url.count("-")
            dots = url.count(".")
            specials = sum(c in "?=&#@%+~_" for c in url)
            digit_count_stats.update(digits)
            digit_count_by_label[label].update(digits)
            hyphen_count_stats.update(hyphens)
            hyphen_count_by_label[label].update(hyphens)
            dot_count_stats.update(dots)
            dot_count_by_label[label].update(dots)
            special_count_stats.update(specials)
            special_count_by_label[label].update(specials)

            if host:
                hd = host.count(".")
                hh = host.count("-")
                hg = sum(c.isdigit() for c in host)
                host_dots_stats.update(hd)
                host_dots_by_label[label].update(hd)
                host_hyphens_stats.update(hh)
                host_hyphens_by_label[label].update(hh)
                host_digits_stats.update(hg)
                host_digits_by_label[label].update(hg)

            segments = [s for s in path.split("/") if s]
            depth = len(segments)
            path_depth_stats.update(depth)
            path_depth_by_label[label].update(depth)
            path_depth_sample.add(depth)

            has_query = bool(query)
            query_count["with_query" if has_query else "no_query"] += 1
            if has_query:
                params = [p for p in query.split("&") if p]
                query_params_stats.update(len(params))
                query_params_by_label[label].update(len(params))

            if is_ip(host):
                ip_count_by_label[label] += 1

            ext = None
            if segments:
                last = segments[-1]
                if "." in last:
                    ext = "." + last.split(".")[-1].lower()
                    ext_counts[ext] += 1
                    if ext in CDN_EXTENSIONS:
                        cdn_ext_count[ext] += 1
                        cdn_ext_by_label[label] += 1
                    if ext in MALWARE_EXECUTABLES:
                        malware_ext_count[ext] += 1
                        malware_ext_by_label[label] += 1

            lower_url = url.lower()

            # ---- Char frequency + OOV: dùng Counter.update (C-level) ----
            char_freq.update(lower_url)
            n_oov = 0
            for c in lower_url:
                if c not in ALLOWED_CHARS:
                    out_of_vocab_chars[c] += 1
                    n_oov += 1
            if n_oov:
                out_of_vocab_by_label[label] += n_oov

            # ---- Keyword hits: dùng compiled regex; set() để mỗi keyword chỉ
            # đếm 1 lần/URL (giống semantics eda 2 cũ) ----
            susp_matches = _SUSPICIOUS_RE.findall(lower_url)
            if susp_matches:
                suspicious_by_label[label] += 1
                suspicious_hits.update(set(susp_matches))

            scam_matches = _SCAM_RE.findall(lower_url)
            if scam_matches:
                scam_by_label[label] += 1
                scam_hits.update(set(scam_matches))

            if host:
                brand_matches = _LEGIT_BRANDS_RE.findall(host)
                if brand_matches:
                    brand_by_label[label] += 1
                    brand_hits.update(set(brand_matches))

            c2_m = _C2_PATHS_RE.match(path)
            if c2_m:
                c2_by_label[label] += 1
                c2_hits[c2_m.group(0)] += 1

            if _MALWARE_LURE_RE.search(lower_url):
                malware_lure_by_label[label] += 1

            if host:
                sigs = host_quality_signals(host)
                for k in artifact_signals:
                    if sigs[k]:
                        artifact_by_label[label][k] += 1
                        if k in ("has_double_dot", "no_dot", "very_short_tld_label"):
                            malformed_sample_by_label[label].add(url)
                if looks_like_dga(host):
                    dga_by_label[label] += 1

                top_hosts[host] += 1
                top_hosts_by_label[label][host] += 1

            # Domain pool: dùng compiled regex (1 search/pool thay vì N substring)
            if host:
                if _ECOMMERCE_HOST_RE.search(host):
                    known_pool_hits["ecommerce"][label] += 1
                if _NEWS_HOST_RE.search(host):
                    known_pool_hits["news"][label] += 1
                if _TECH_SAAS_HOST_RE.search(host):
                    known_pool_hits["tech_saas"][label] += 1
                if _EDUCATION_HOST_RE.search(host):
                    known_pool_hits["education"][label] += 1
                if _SOCIAL_HOST_RE.search(host):
                    known_pool_hits["social"][label] += 1
                if host in SHORTLINK_DOMAINS:
                    known_pool_hits["shortlink"][label] += 1

            # ---- VN targeting analysis ----
            vn_tld_hit = bool(host) and host.endswith(".vn")
            vn_brand_hit = bool(VN_BRAND_RE.search(url))
            if vn_tld_hit:
                vn_tld_by_label[label] += 1
            if vn_brand_hit:
                vn_brand_by_label[label] += 1
            if vn_tld_hit or vn_brand_hit:
                vn_target_by_label[label] += 1
                vn_target_sample_by_label[label].add(url)
            for sub in VN_SUB_TLDS:
                if host.endswith(sub):
                    vn_subtld_counts[sub] += 1
                    vn_subtld_by_label[label][sub] += 1
                    break
            else:
                if host.endswith(".vn"):
                    vn_subtld_counts[".vn (other)"] += 1
                    vn_subtld_by_label[label][".vn (other)"] += 1

            cat = classify_url(lower_url, scheme, host, path, query, tld, ext)
            category_counts[cat] += 1
            category_by_label[label][cat] += 1

            if total % progress_every == 0:
                elapsed = time.time() - start
                rate = total / max(1.0, elapsed)
                print(f"[{total:,}] processed - {rate:,.0f} rows/s",
                      file=sys.stdout, flush=True)

    # ---------------- Reporting ----------------
    section("BASIC COUNTS", _print)
    _print(f"Rows processed : {total:,}")
    _print(f"Missing rows   : {missing:,}")
    _print(f"Invalid labels : {invalid_label:,}")
    for lbl, cnt in sorted(label_counts.items()):
        name = "benign" if lbl == 0 else "malicious" if lbl == 1 else f"label-{lbl}"
        _print(f"Label {lbl} ({name:>9}) : {cnt:,} ({fmt_pct(cnt, total)})")
    if 0 in label_counts and 1 in label_counts and label_counts[1] > 0:
        ratio = label_counts[0] / label_counts[1]
        _print(f"\nClass ratio benign:mal = {ratio:.2f} : 1")
        _print(f"pos_weight (= N_neg/N_pos) = {ratio:.4f}")

    section("VIETNAMESE TARGETING", _print)
    _print("Đếm URL có dấu hiệu nhắm mục tiêu Việt Nam:")
    _print("  vn_tld_hit  = host kết thúc '.vn'")
    _print("  vn_brand_hit = URL match regex brand/keyword VN "
           "(vietcom*, momo, shopee, viettel, hanoi, ...)")
    _print("")
    total_vn_tld = sum(vn_tld_by_label.values())
    total_vn_brand = sum(vn_brand_by_label.values())
    total_vn_target = sum(vn_target_by_label.values())
    _print(f"VN TLD hits      : {total_vn_tld:,} ({fmt_pct(total_vn_tld, total)})")
    _print(f"VN BRAND hits    : {total_vn_brand:,} ({fmt_pct(total_vn_brand, total)})")
    _print(f"VN TARGET (any)  : {total_vn_target:,} ({fmt_pct(total_vn_target, total)})")
    _print("")
    for lbl in sorted(label_counts):
        name = "benign" if lbl == 0 else "malicious"
        denom = label_counts[lbl]
        _print(f"Label {lbl} ({name}):")
        _print(f"  vn_tld   : {vn_tld_by_label[lbl]:,} ({fmt_pct(vn_tld_by_label[lbl], denom)})")
        _print(f"  vn_brand : {vn_brand_by_label[lbl]:,} ({fmt_pct(vn_brand_by_label[lbl], denom)})")
        _print(f"  vn_target: {vn_target_by_label[lbl]:,} ({fmt_pct(vn_target_by_label[lbl], denom)})")

    _print("\nVN sub-TLD breakdown:")
    print_counter(vn_subtld_counts, _print, top_n=10, label="Overall")
    for lbl in sorted(label_counts):
        print_counter(vn_subtld_by_label[lbl], _print, top_n=10, label=f"Label {lbl}")

    _print("\nSample VN-targeted URLs (max 20/label):")
    for lbl in sorted(label_counts):
        _print(f"\n--- Label {lbl} ---")
        for u in vn_target_sample_by_label[lbl].sample[:20]:
            _print(f"  {u}")

    section("SCHEME DISTRIBUTION", _print)
    print_counter(scheme_counts, _print, top_n=10, label="Overall")
    for lbl in sorted(label_counts):
        print_counter(scheme_by_label[lbl], _print, top_n=10, label=f"Label {lbl}")
    _print("")
    for lbl in sorted(label_counts):
        h = https_count_by_label[lbl]
        _print(f"Label {lbl} HTTPS share: {h:,} / {label_counts[lbl]:,} = "
               f"{fmt_pct(h, label_counts[lbl])}")

    section("URL LENGTH (chars)", _print)
    _print(f"Overall : mean={url_len_stats.mean:.2f}  std={url_len_stats.std:.2f}  "
           f"min={url_len_stats.min}  max={url_len_stats.max}")
    p50 = percentile(url_len_sample.sample, 50)
    p90 = percentile(url_len_sample.sample, 90)
    p95 = percentile(url_len_sample.sample, 95)
    p99 = percentile(url_len_sample.sample, 99)
    _print(f"P50/P90/P95/P99 (sample): {p50} / {p90} / {p95} / {p99}")
    for lbl in sorted(label_counts):
        s = url_len_by_label[lbl]
        ssamp = url_len_sample_by_label[lbl].sample
        _print(f"Label {lbl}: mean={s.mean:.2f}  std={s.std:.2f}  "
               f"min={s.min}  max={s.max}  "
               f"P95={percentile(ssamp,95)}  P99={percentile(ssamp,99)}")

    section("HOST / PATH / QUERY LENGTH", _print)
    _print(f"Host len  : mean={host_len_stats.mean:.2f}  std={host_len_stats.std:.2f}  max={host_len_stats.max}")
    _print(f"Path len  : mean={path_len_stats.mean:.2f}  std={path_len_stats.std:.2f}  max={path_len_stats.max}")
    _print(f"Query len : mean={query_len_stats.mean:.2f}  std={query_len_stats.std:.2f}  max={query_len_stats.max}")
    for lbl in sorted(label_counts):
        _print(f"Label {lbl} host/path/query mean: "
               f"{host_len_by_label[lbl].mean:.2f} / "
               f"{path_len_by_label[lbl].mean:.2f} / "
               f"{query_len_by_label[lbl].mean:.2f}")

    section("PATH DEPTH (segments)", _print)
    _print(f"Mean={path_depth_stats.mean:.2f}  std={path_depth_stats.std:.2f}  "
           f"min={path_depth_stats.min}  max={path_depth_stats.max}")
    p50 = percentile(path_depth_sample.sample, 50)
    p90 = percentile(path_depth_sample.sample, 90)
    p95 = percentile(path_depth_sample.sample, 95)
    p99 = percentile(path_depth_sample.sample, 99)
    _print(f"P50/P90/P95/P99 (sample): {p50} / {p90} / {p95} / {p99}")
    for lbl in sorted(label_counts):
        s = path_depth_by_label[lbl]
        _print(f"Label {lbl}: mean={s.mean:.2f}  std={s.std:.2f}")

    section("QUERY STRINGS", _print)
    wq = query_count["with_query"]
    nq = query_count["no_query"]
    _print(f"With query : {wq:,} ({fmt_pct(wq, total)})")
    _print(f"No query   : {nq:,} ({fmt_pct(nq, total)})")
    _print(f"Params (when present) mean={query_params_stats.mean:.2f}  "
           f"std={query_params_stats.std:.2f}")
    for lbl in sorted(label_counts):
        s = query_params_by_label[lbl]
        if s.count:
            _print(f"Label {lbl}: params mean={s.mean:.2f}  std={s.std:.2f}  count={s.count:,}")

    section("HOST STATISTICS", _print)
    ip_total = sum(ip_count_by_label.values())
    _print(f"IP-based hosts: {ip_total:,} ({fmt_pct(ip_total, total)})")
    for lbl in sorted(label_counts):
        c = ip_count_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")

    _print(f"\nHost dots    : mean={host_dots_stats.mean:.2f}  std={host_dots_stats.std:.2f}")
    _print(f"Host hyphens : mean={host_hyphens_stats.mean:.2f}  std={host_hyphens_stats.std:.2f}")
    _print(f"Host digits  : mean={host_digits_stats.mean:.2f}  std={host_digits_stats.std:.2f}")
    for lbl in sorted(label_counts):
        _print(f"Label {lbl}: dots={host_dots_by_label[lbl].mean:.2f} "
               f"hyphens={host_hyphens_by_label[lbl].mean:.2f} "
               f"digits={host_digits_by_label[lbl].mean:.2f}")

    section("TLD DISTRIBUTION", _print)
    print_counter(tld_counts, _print, top_n=25, label="Top TLDs (overall)")
    for lbl in sorted(label_counts):
        print_counter(tld_by_label[lbl], _print, top_n=25, label=f"Top TLDs (Label {lbl})")
    print_counter(port_counts, _print, top_n=10, label="Ports")

    section("CHARACTER USAGE (whole URL)", _print)
    _print(f"Digits   : mean={digit_count_stats.mean:.2f}  std={digit_count_stats.std:.2f}")
    _print(f"Hyphens  : mean={hyphen_count_stats.mean:.2f}  std={hyphen_count_stats.std:.2f}")
    _print(f"Dots     : mean={dot_count_stats.mean:.2f}  std={dot_count_stats.std:.2f}")
    _print(f"Specials : mean={special_count_stats.mean:.2f}  std={special_count_stats.std:.2f}")
    for lbl in sorted(label_counts):
        _print(f"Label {lbl}: digits={digit_count_by_label[lbl].mean:.2f} "
               f"hyphens={hyphen_count_by_label[lbl].mean:.2f} "
               f"dots={dot_count_by_label[lbl].mean:.2f} "
               f"specials={special_count_by_label[lbl].mean:.2f}")

    section("CHARACTER COVERAGE vs CNN-LSTM ALLOWED_CHARS", _print)
    _print(f"Vocab size (CNN-LSTM): {len(ALLOWED_CHARS)}")
    _print(f"Total OOV chars      : {sum(out_of_vocab_chars.values()):,}")
    print_counter(out_of_vocab_chars, _print, top_n=20, label="Top OOV chars (will be <UNK>)")
    for lbl in sorted(label_counts):
        c = out_of_vocab_by_label[lbl]
        _print(f"Label {lbl} OOV total: {c:,}")

    section("EXTENSIONS", _print)
    print_counter(ext_counts, _print, top_n=20, label="Top file extensions")
    print_counter(cdn_ext_count, _print, top_n=10, label="CDN-style extensions")
    print_counter(malware_ext_count, _print, top_n=10, label="Malware-like extensions")
    for lbl in sorted(label_counts):
        _print(f"Label {lbl}: CDN ext hits={cdn_ext_by_label[lbl]:,}  "
               f"Malware ext hits={malware_ext_by_label[lbl]:,}")

    section("KEYWORD HITS", _print)
    _print(f"Suspicious-word URLs : {sum(suspicious_by_label.values()):,}")
    for lbl in sorted(label_counts):
        c = suspicious_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")
    print_counter(suspicious_hits, _print, top_n=15, label="Top suspicious words")

    _print(f"\nScam-bait URLs       : {sum(scam_by_label.values()):,}")
    for lbl in sorted(label_counts):
        c = scam_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")
    print_counter(scam_hits, _print, top_n=10, label="Top scam bait")

    _print(f"\nBrand-mention URLs   : {sum(brand_by_label.values()):,}")
    for lbl in sorted(label_counts):
        c = brand_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")
    print_counter(brand_hits, _print, top_n=20, label="Top brand mentions")

    _print(f"\nC2-path URLs         : {sum(c2_by_label.values()):,}")
    for lbl in sorted(label_counts):
        c = c2_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")
    print_counter(c2_hits, _print, top_n=10, label="Top C2 paths")

    _print(f"\nMalware-lure URLs    : {sum(malware_lure_by_label.values()):,}")
    for lbl in sorted(label_counts):
        c = malware_lure_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")

    section("ANCHOR DOMAINS (sanity check)", _print)
    _print("CHÚ Ý: Đây KHÔNG phải coverage report. Mỗi pool chỉ ~30-40 domain")
    _print("nổi tiếng — trên long-tail thực tế (hàng nghìn host) số match sẽ thấp.")
    _print("Mục đích: xác nhận các site lớn có thật sự xuất hiện trong dataset.")
    for pool_name in ["ecommerce", "news", "tech_saas", "education", "social", "shortlink"]:
        c = known_pool_hits[pool_name]
        total_pool = sum(c.values())
        _print(f"\n{pool_name.upper():>12}: {total_pool:,} ({fmt_pct(total_pool, total)})")
        for lbl in sorted(label_counts):
            _print(f"  Label {lbl}: {c[lbl]:,} ({fmt_pct(c[lbl], label_counts[lbl])})")

    section("TOP HOSTS", _print)
    print_counter(top_hosts, _print, top_n=30, label="Top hosts (overall)")
    for lbl in sorted(label_counts):
        print_counter(top_hosts_by_label[lbl], _print, top_n=30,
                      label=f"Top hosts (Label {lbl})")

    section("SUSPICIOUS HOST ARTIFACTS", _print)
    _print("Heuristics phát hiện host bất thường (DGA, typosquat, IP-only, ...).")
    for k in artifact_signals:
        total_sig = sum(artifact_by_label[lbl][k] for lbl in label_counts)
        _print(f"\n{k}: {total_sig:,} ({fmt_pct(total_sig, total)})")
        for lbl in sorted(label_counts):
            c = artifact_by_label[lbl][k]
            _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")

    _print(f"\nDGA-like hosts (digit_ratio>0.30 OR vowel_ratio<0.18):")
    for lbl in sorted(label_counts):
        c = dga_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")

    section("MALFORMED URL SAMPLES (max 20/label)", _print)
    for lbl in sorted(label_counts):
        _print(f"\n--- Label {lbl} ---")
        for u in malformed_sample_by_label[lbl].sample[:20]:
            _print(f"  {u}")

    section("HEURISTIC CATEGORY BREAKDOWN", _print)
    _print("Phân loại URL bằng heuristic (không dùng label thực).")
    for cat, val in category_counts.most_common():
        _print(f"{cat:>34} : {val:,} ({fmt_pct(val, total)})")
    _print("\nLabel breakdown:")
    for cat, val in category_counts.most_common():
        l0 = category_by_label[0].get(cat, 0)
        l1 = category_by_label[1].get(cat, 0)
        _print(f"{cat:>34} : "
               f"L0 {l0:,} ({fmt_pct(l0, max(1, label_counts.get(0,0)))}) | "
               f"L1 {l1:,} ({fmt_pct(l1, max(1, label_counts.get(1,0)))})")

    section("EDA DONE", _print)
    elapsed = time.time() - start
    _print(f"Elapsed: {elapsed:.2f}s | {total / max(1.0, elapsed):,.0f} rows/s")
    if output_path:
        _print(f"Report saved -> {output_path}")
    _close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detailed EDA for dataset/dataset 3 (vn).csv"
    )
    parser.add_argument("--path", default=r"D:\! secURLity\dataset\dataset 3 (vn).csv",
                        help="CSV path (default: dataset/dataset 3 (vn).csv)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 = all rows")
    parser.add_argument("--sample-size", type=int, default=200_000,
                        help="Reservoir sample size (default 200k)")
    parser.add_argument("--progress-every", type=int, default=1_000_000,
                        help="Progress print interval")
    parser.add_argument("--output", default=r"D:\! secURLity\dataset\eda-result 3.txt",
                        help="Save report to file (in addition to stdout). "
                             "Pass empty string to disable.")
    args = parser.parse_args()

    out = args.output if args.output else None
    run_eda(args.path, args.max_rows, args.sample_size,
            args.progress_every, out)


if __name__ == "__main__":
    main()

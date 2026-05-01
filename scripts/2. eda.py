import argparse
import csv
import math
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

# ─────────────────────────────────────────────────────────────────────────────
# Constants derived from the generator (v3 FIXED)
# ─────────────────────────────────────────────────────────────────────────────

BENIGN_WORDS = [
    "home", "about", "contact", "login", "register", "search", "profile",
    "settings", "dashboard", "feed", "news", "blog", "article", "category",
    "product", "item", "detail", "checkout", "cart", "order", "account",
    "help", "support", "faq", "terms", "privacy", "careers", "team",
    "services", "solutions", "features", "pricing", "docs", "api",
    "download", "upload", "gallery", "photo", "video", "stream",
    "library", "archive", "forum", "community", "events", "schedule",
    "report", "analysis", "research", "papers", "projects", "portfolio",
    "guide", "tutorial", "howto", "learn", "course", "lesson",
    "review", "rating", "comment", "share", "like", "follow",
    "buy", "sell", "shop", "store", "inventory", "warehouse", "shipping",
    "payment", "subscription", "membership", "tier", "plan", "bundle",
    "user", "member", "friend", "follower", "trending", "explore", "discover",
    "notification", "message", "inbox", "outbox", "thread", "conversation",
    "image", "media", "content", "channel", "playlist", "album", "track",
    "podcast", "webinar", "broadcast", "live", "recording",
    "export", "import", "sync", "backup", "restore", "migrate", "transfer",
    "database", "table", "collection", "document", "record", "field",
    "admin", "panel", "console", "server", "status", "health", "metrics",
    "logs", "audit", "config", "preferences", "options",
    "permission", "role", "access", "security", "certificate",
    "filter", "sort", "paginate", "view", "edit", "create", "delete",
]

BENIGN_TLDS = [
    ".com", ".org", ".net", ".edu", ".gov", ".mil",
    ".co.uk", ".co.jp", ".co.in", ".de", ".fr", ".cn", ".au", ".ca",
    ".io", ".dev", ".app", ".info", ".biz", ".mobi", ".name",
    ".blog", ".shop", ".tech", ".site", ".online", ".space",
]

ECOMMERCE_DOMAINS = {
    "amazon.com", "ebay.com", "etsy.com", "shopify.com", "walmart.com",
    "bestbuy.com", "target.com", "newegg.com", "alibaba.com", "rakuten.co.jp",
    "mercadolibre.com.ar", "flipkart.com", "tokopedia.com", "lazada.com",
    "wish.com", "geek.com", "aliexpress.com", "cdkeys.com", "fanatical.com",
}

NEWS_DOMAINS = {
    "bbc.com", "cnn.com", "reuters.com", "apnews.com", "nytimes.com",
    "theguardian.com", "washingtonpost.com", "forbes.com", "foxnews.com",
    "cnbc.com", "techcrunch.com", "wired.com", "zdnet.com", "theverge.com",
    "engadget.com", "vice.com", "vox.com", "buzzfeed.com", "huffpost.com",
    "axios.com", "politico.com", "breitbart.com", "drudge.com", "medium.com",
}

CDN_PREFIXES = {
    "cdn", "static", "assets", "media", "img", "images", "files", "s3",
    "cloudfront", "akamai", "fastly", "cloudflare", "jsdelivr", "unpkg",
    "cdn-images", "static-assets", "production", "staging", "v1", "v2",
}

CDN_EXTENSIONS = {
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".webm",
    ".min.js", ".min.css", ".bundle.js", ".map",
}

LEGIT_BRANDS = {
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "tesla",
    "facebook", "instagram", "twitter", "linkedin", "dropbox", "slack",
    "github", "gitlab", "bitbucket", "jira", "confluence", "docker",
    "kubernetes", "jenkins", "ansible", "terraform", "aws", "azure",
    "wellsfargo", "bankofamerica", "chase", "citibank", "hsbc", "bnp",
    "barclays", "lloyds", "deutsche", "ubs", "credit-suisse",
    "dhl", "fedex", "ups", "usps", "dpd", "gls", "tnt", "hermes",
    "nike", "adidas", "puma", "reebok", "zara", "hm", "gap", "primark",
    "disney", "paramount", "sony", "warner", "universal",
    "mcdonald", "kfc", "starbucks", "coca-cola", "pepsi", "nestle",
}

SHORTLINK_DOMAINS = {
    "bit.ly", "tinyurl.com", "ow.ly", "t.co", "shorte.st", "goo.gl",
    "is.gd", "v.gd", "short.link", "tiny.cc", "imgur.com", "buff.ly",
    "adf.ly", "j.mp", "youtu.be", "redd.it", "lnkd.in", "medium.com",
}

COMMON_SUBDOMAINS = {
    "mail", "ftp", "smtp", "pop", "imap",
    "api", "api-v1", "api-v2", "api-v3",
    "www", "web",
    "cdn", "static", "assets", "images",
    "download", "downloads",
    "secure", "vpn", "ssl",
    "internal", "intranet",
    "us", "eu", "asia", "apac", "jp", "au",
    "search", "map", "drive", "docs", "mail",
    "monitor", "status", "health", "dashboard",
}

MALICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".top", ".xyz", ".click", ".club",
    ".online", ".site", ".live", ".cc", ".pw", ".in", ".ru", ".to",
    ".bid", ".download", ".party", ".faith", ".work", ".gdn", ".trade",
    ".stream", ".red", ".win", ".cricket", ".loan", ".racing", ".review",
}

SUSPICIOUS_WORDS = [
    "secure", "verify", "update", "confirm", "alert", "warning",
    "unlock", "restore", "validate", "authenticate", "suspended",
    "limited", "urgent", "action-required", "login", "signin", "re-confirm",
    "reconfirm", "unusual-activity", "confirm-identity", "verify-account",
    "security-alert", "immediate-action", "click-here", "act-now",
]

C2_PATHS = [
    "/cmd", "/c2", "/gate", "/bot", "/beacon", "/checkin", "/tasks",
    "/poll", "/update", "/ping", "/report", "/data", "/admin", "/api",
    "/i", "/bin.sh", "/shell", "/exec", "/run", "/config", "/status",
    "/sync", "/push", "/pull", "/fetch", "/load", "/plugin",
]

SCAM_BAIT = [
    "you-won", "claim-prize", "free-gift", "limited-offer", "congratulations",
    "lucky-winner", "exclusive-deal", "earn-money-fast", "lose-weight-now",
    "get-rich-quick", "click-here", "act-now", "urgent", "final-notice",
    "verify-account", "confirm-password", "update-payment", "restore-access",
    "claim-reward", "free-trial", "limited-time", "last-chance",
]

MALWARE_EXECUTABLES = {
    ".exe", ".bat", ".ps1", ".msi", ".zip", ".rar", ".dmg", ".apk",
    ".deb", ".rpm", ".tar.gz", ".iso", ".img", ".dll", ".scr", ".vbs",
    ".js", ".jar", ".class", ".so", ".dylib", ".sh", ".elf",
}

MALWARE_LURE_WORDS = {
    "update", "install", "setup", "patch", "fix", "crack", "keygen",
    "loader", "installer", "driver", "tool", "utility", "codec",
    "player", "reader", "viewer", "converter", "optimizer", "cleaner",
    "antivirus", "security", "booster", "accelerator", "manager",
}

SOCIAL_DOMAINS = {"twitter.com", "facebook.com", "instagram.com", "reddit.com"}
API_BASE_DOMAINS = {"google.com", "github.com", "microsoft.com"}
CDN_BASE_DOMAINS = {"google.com", "facebook.com", "cloudflare.com", "akamai.net"}

KNOWN_TLDS = sorted(set(BENIGN_TLDS) | set(MALICIOUS_TLDS), key=len, reverse=True)


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = None
        self.max = None

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
    def __init__(self, size: int, seed: int = 42) -> None:
        self.size = size
        self.n = 0
        self.sample: List[int] = []
        self.rng = random.Random(seed)

    def add(self, value: int) -> None:
        self.n += 1
        if len(self.sample) < self.size:
            self.sample.append(value)
            return
        j = self.rng.randrange(self.n)
        if j < self.size:
            self.sample[j] = value


def percentile(values: List[int], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    idx = int(round((p / 100) * (len(values) - 1)))
    return float(values[idx])


def is_ip(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        val = int(part)
        if val < 0 or val > 255:
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
    host = host.lower()
    for tld in KNOWN_TLDS:
        if host.endswith(tld):
            return tld
    if "." in host:
        return "." + host.rsplit(".", 1)[-1]
    return "<none>"


def get_subdomain(host: str, tld: str) -> Optional[str]:
    if not host or host == "<none>":
        return None
    labels = host.split(".")
    tld_labels = tld.strip(".").split(".")
    if len(labels) <= len(tld_labels) + 1:
        return None
    return labels[0]


def host_endswith_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def looks_like_dga(host: str) -> bool:
    base = host.split(".")[0]
    if len(base) < 10:
        return False
    digit_ratio = sum(ch.isdigit() for ch in base) / max(1, len(base))
    vowel_ratio = sum(ch in "aeiou" for ch in base) / max(1, len(base))
    return digit_ratio > 0.15 or vowel_ratio < 0.2


def classify_generator(
    url: str,
    scheme: str,
    host: str,
    path: str,
    query: str,
    tld: str,
    has_query: bool,
    ext: Optional[str],
) -> str:
    lower_url = url.lower()

    if is_ip(host):
        return "malicious_ip_based"

    if "/download/" in path and ext in MALWARE_EXECUTABLES:
        return "malicious_malware_dist"

    if path.startswith("/ad/") and has_query:
        return "malicious_drive_by"

    if any(bait in lower_url for bait in SCAM_BAIT):
        return "malicious_spam_scam"

    if any(brand in host for brand in LEGIT_BRANDS) and (
        "/login" in path or "/signin" in path or "/account" in path
        or "secure" in host or "login" in host or "auth" in host
    ):
        return "malicious_phishing"

    if tld in MALICIOUS_TLDS and looks_like_dga(host):
        return "malicious_dga"

    if any(host_endswith_domain(host, d) for d in ECOMMERCE_DOMAINS):
        if "/dp/" in path or "/product/" in path or "/category/" in path or "/cart" in path:
            return "benign_ecommerce"
        return "benign_ecommerce"

    if any(host_endswith_domain(host, d) for d in CDN_BASE_DOMAINS):
        if ext in CDN_EXTENSIONS:
            return "benign_cdn"
        if host.split(".")[0] in CDN_PREFIXES:
            return "benign_cdn"

    if any(host_endswith_domain(host, d) for d in API_BASE_DOMAINS):
        if host.startswith("api.") or path.startswith("/v1/") or path.startswith("/v2/") or path.startswith("/v3/"):
            return "benign_api"

    if any(host_endswith_domain(host, d) for d in NEWS_DOMAINS):
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
            return "benign_news"
        return "benign_news"

    if host in SHORTLINK_DOMAINS:
        return "benign_shortlink"

    if host in SOCIAL_DOMAINS:
        return "benign_social"

    sub = get_subdomain(host, tld)
    if sub and sub in COMMON_SUBDOMAINS:
        return "benign_common_subdomain"

    return "unknown"


def parse_url(raw_url: str) -> Tuple[str, str, str, str, str, Optional[str]]:
    url = raw_url.strip()
    if not url:
        return "", "", "", "", "", None
    if "://" not in url:
        parts = urlsplit("http://" + url)
        scheme = ""
    else:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
    host, port = split_host_port(parts.netloc.lower())
    return url, scheme, host, parts.path or "/", parts.query or "", port


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def format_pct(num: int, den: int) -> str:
    if den == 0:
        return "0.00%"
    return f"{(num / den) * 100:.2f}%"


def print_counter(counter: Counter, top_n: int = 15, label: str = "") -> None:
    if label:
        print(f"\n{label}")
    for key, val in counter.most_common(top_n):
        print(f"  {key:>15} : {val:,}")


def run_eda(csv_path: str, max_rows: int, sample_size: int, progress_every: int) -> None:
    start = time.time()

    total = 0
    missing_rows = 0
    invalid_label = 0

    label_counts = Counter()
    scheme_counts = Counter()
    scheme_by_label = defaultdict(Counter)
    tld_counts = Counter()
    tld_by_label = defaultdict(Counter)
    port_counts = Counter()

    url_len_stats = RunningStats()
    url_len_stats_by_label = defaultdict(RunningStats)
    url_len_sample = ReservoirSampler(sample_size)
    url_len_sample_by_label = defaultdict(lambda: ReservoirSampler(sample_size))

    path_depth_stats = RunningStats()
    path_depth_stats_by_label = defaultdict(RunningStats)
    path_depth_sample = ReservoirSampler(sample_size)

    digit_count_stats = RunningStats()
    digit_count_stats_by_label = defaultdict(RunningStats)

    query_count = Counter()
    query_params_stats = RunningStats()
    query_params_stats_by_label = defaultdict(RunningStats)

    ip_count = Counter()
    https_count = Counter()

    ext_counts = Counter()
    cdn_ext_count = Counter()
    malware_ext_count = Counter()

    suspicious_hits = Counter()
    scam_hits = Counter()
    brand_hits = Counter()
    c2_hits = Counter()
    benign_word_hits = Counter()

    suspicious_by_label = Counter()
    scam_by_label = Counter()
    brand_by_label = Counter()
    c2_by_label = Counter()
    cdn_ext_by_label = Counter()
    malware_ext_by_label = Counter()

    category_counts = Counter()
    category_by_label = defaultdict(Counter)

    known_domain_counts = Counter()

    print_section("EDA START")
    print(f"File: {csv_path}")
    max_rows_display = "ALL" if max_rows <= 0 else f"{max_rows:,}"
    print(f"Max rows: {max_rows_display}")
    print(f"Reservoir sample size: {sample_size:,}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows > 0 and total >= max_rows:
                break
            total += 1

            url = row.get("url", "").strip()
            label_raw = row.get("label", "").strip()
            if not url or label_raw == "":
                missing_rows += 1
                continue
            try:
                label = int(label_raw)
            except ValueError:
                invalid_label += 1
                continue

            label_counts[label] += 1

            parsed_url, scheme, host, path, query, port = parse_url(url)
            if not parsed_url or not host:
                missing_rows += 1
                continue

            scheme_counts[scheme or "<none>"] += 1
            scheme_by_label[label][scheme or "<none>"] += 1
            if scheme == "https":
                https_count[label] += 1

            if port:
                port_counts[port] += 1

            tld = get_tld(host)
            tld_counts[tld] += 1
            tld_by_label[label][tld] += 1

            url_len = len(url)
            url_len_stats.update(url_len)
            url_len_stats_by_label[label].update(url_len)
            url_len_sample.add(url_len)
            url_len_sample_by_label[label].add(url_len)

            digits = sum(ch.isdigit() for ch in url)
            digit_count_stats.update(digits)
            digit_count_stats_by_label[label].update(digits)

            segments = [seg for seg in path.split("/") if seg]
            depth = len(segments)
            path_depth_stats.update(depth)
            path_depth_stats_by_label[label].update(depth)
            path_depth_sample.add(depth)

            has_query = bool(query)
            query_count["with_query" if has_query else "no_query"] += 1
            if has_query:
                params = [p for p in query.split("&") if p]
                query_params_stats.update(len(params))
                query_params_stats_by_label[label].update(len(params))

            if is_ip(host):
                ip_count[label] += 1

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
            if any(word in lower_url for word in SUSPICIOUS_WORDS):
                suspicious_by_label[label] += 1
                for word in SUSPICIOUS_WORDS:
                    if word in lower_url:
                        suspicious_hits[word] += 1
            if any(bait in lower_url for bait in SCAM_BAIT):
                scam_by_label[label] += 1
                for bait in SCAM_BAIT:
                    if bait in lower_url:
                        scam_hits[bait] += 1
            if any(brand in host for brand in LEGIT_BRANDS):
                brand_by_label[label] += 1
                for brand in LEGIT_BRANDS:
                    if brand in host:
                        brand_hits[brand] += 1
            if any(path.startswith(p) for p in C2_PATHS):
                c2_by_label[label] += 1
                for p in C2_PATHS:
                    if path.startswith(p):
                        c2_hits[p] += 1

            for seg in segments:
                if seg in BENIGN_WORDS:
                    benign_word_hits[seg] += 1

            if any(host_endswith_domain(host, d) for d in ECOMMERCE_DOMAINS):
                known_domain_counts["ecommerce_domains"] += 1
            if any(host_endswith_domain(host, d) for d in NEWS_DOMAINS):
                known_domain_counts["news_domains"] += 1
            if host in SHORTLINK_DOMAINS:
                known_domain_counts["shortlink_domains"] += 1
            if host in SOCIAL_DOMAINS:
                known_domain_counts["social_domains"] += 1
            if any(host_endswith_domain(host, d) for d in CDN_BASE_DOMAINS):
                known_domain_counts["cdn_domains"] += 1

            category = classify_generator(
                url=url,
                scheme=scheme,
                host=host,
                path=path,
                query=query,
                tld=tld,
                has_query=has_query,
                ext=ext,
            )
            category_counts[category] += 1
            category_by_label[label][category] += 1

            if total % progress_every == 0:
                elapsed = time.time() - start
                rate = total / max(1.0, elapsed)
                print(f"[{total:,}] rows processed - {rate:,.0f} rows/s")

    print_section("BASIC COUNTS")
    print(f"Rows processed : {total:,}")
    print(f"Missing rows   : {missing_rows:,}")
    print(f"Invalid labels : {invalid_label:,}")
    for label, count in sorted(label_counts.items()):
        print(f"Label {label}       : {count:,} ({format_pct(count, total)})")

    print_section("SCHEME DISTRIBUTION")
    print_counter(scheme_counts, top_n=10, label="Overall")
    for label in sorted(label_counts):
        print_counter(scheme_by_label[label], top_n=10, label=f"Label {label}")

    print_section("URL LENGTH")
    print(f"Mean   : {url_len_stats.mean:.2f} | Std: {url_len_stats.std:.2f}")
    print(f"Min/Max: {url_len_stats.min} / {url_len_stats.max}")
    p50 = percentile(url_len_sample.sample, 50)
    p90 = percentile(url_len_sample.sample, 90)
    p95 = percentile(url_len_sample.sample, 95)
    p99 = percentile(url_len_sample.sample, 99)
    print(f"Percentiles (sample) P50/P90/P95/P99: {p50} / {p90} / {p95} / {p99}")
    for label in sorted(label_counts):
        stats = url_len_stats_by_label[label]
        print(f"Label {label} Mean/Std: {stats.mean:.2f} / {stats.std:.2f}")

    print_section("PATH DEPTH")
    print(f"Mean   : {path_depth_stats.mean:.2f} | Std: {path_depth_stats.std:.2f}")
    print(f"Min/Max: {path_depth_stats.min} / {path_depth_stats.max}")
    p50 = percentile(path_depth_sample.sample, 50)
    p90 = percentile(path_depth_sample.sample, 90)
    p95 = percentile(path_depth_sample.sample, 95)
    p99 = percentile(path_depth_sample.sample, 99)
    print(f"Percentiles (sample) P50/P90/P95/P99: {p50} / {p90} / {p95} / {p99}")

    print_section("QUERY STATS")
    print(f"With query : {query_count['with_query']:,} ({format_pct(query_count['with_query'], total)})")
    print(f"No query   : {query_count['no_query']:,} ({format_pct(query_count['no_query'], total)})")
    print(f"Query params mean: {query_params_stats.mean:.2f} | Std: {query_params_stats.std:.2f}")
    for label in sorted(label_counts):
        stats = query_params_stats_by_label[label]
        if stats.count:
            print(f"Label {label} query params mean: {stats.mean:.2f} | Std: {stats.std:.2f}")

    print_section("HOST / TLD")
    ip_total = sum(ip_count.values())
    print(f"IP-based hosts: {ip_total:,} ({format_pct(ip_total, total)})")
    for label in sorted(label_counts):
        print(f"Label {label} IP hosts: {ip_count[label]:,} ({format_pct(ip_count[label], label_counts[label])})")
    print_counter(tld_counts, top_n=15, label="Top TLDs")
    for label in sorted(label_counts):
        print_counter(tld_by_label[label], top_n=15, label=f"Top TLDs (Label {label})")
    print_counter(port_counts, top_n=10, label="Ports")

    print_section("DIGIT DENSITY")
    print(f"Digits per URL mean: {digit_count_stats.mean:.2f} | Std: {digit_count_stats.std:.2f}")
    for label in sorted(label_counts):
        stats = digit_count_stats_by_label[label]
        print(f"Label {label} digits mean: {stats.mean:.2f} | Std: {stats.std:.2f}")

    print_section("EXTENSIONS")
    print_counter(ext_counts, top_n=15, label="Top extensions")
    print_counter(cdn_ext_count, top_n=10, label="CDN extensions")
    print_counter(malware_ext_count, top_n=10, label="Malware-like extensions")
    for label in sorted(label_counts):
        print(f"Label {label} CDN ext hits: {cdn_ext_by_label[label]:,}")
        print(f"Label {label} Malware ext hits: {malware_ext_by_label[label]:,}")

    print_section("KEYWORD HITS")
    print(f"Suspicious-word URLs: {suspicious_by_label[0] + suspicious_by_label[1]:,} ({format_pct(suspicious_by_label[0] + suspicious_by_label[1], total)})")
    for label in sorted(label_counts):
        print(f"Label {label} suspicious URLs: {suspicious_by_label[label]:,} ({format_pct(suspicious_by_label[label], label_counts[label])})")
    print_counter(suspicious_hits, top_n=15, label="Top suspicious words")

    print(f"\nScam-bait URLs: {scam_by_label[0] + scam_by_label[1]:,} ({format_pct(scam_by_label[0] + scam_by_label[1], total)})")
    for label in sorted(label_counts):
        print(f"Label {label} scam URLs: {scam_by_label[label]:,} ({format_pct(scam_by_label[label], label_counts[label])})")
    print_counter(scam_hits, top_n=10, label="Top scam bait")

    print(f"\nBrand-hit URLs: {brand_by_label[0] + brand_by_label[1]:,} ({format_pct(brand_by_label[0] + brand_by_label[1], total)})")
    for label in sorted(label_counts):
        print(f"Label {label} brand-hit URLs: {brand_by_label[label]:,} ({format_pct(brand_by_label[label], label_counts[label])})")
    print_counter(brand_hits, top_n=15, label="Top brands")

    print(f"\nC2-path URLs: {c2_by_label[0] + c2_by_label[1]:,} ({format_pct(c2_by_label[0] + c2_by_label[1], total)})")
    for label in sorted(label_counts):
        print(f"Label {label} C2-path URLs: {c2_by_label[label]:,} ({format_pct(c2_by_label[label], label_counts[label])})")
    print_counter(c2_hits, top_n=10, label="Top C2 paths")

    print_section("BENIGN WORDS IN PATH")
    print_counter(benign_word_hits, top_n=20, label="Top benign words (path segments)")

    print_section("KNOWN DOMAIN SETS (from generator)")
    for key, val in known_domain_counts.most_common():
        print(f"{key:>20} : {val:,} ({format_pct(val, total)})")

    print_section("GENERATOR-STYLE CATEGORY HEURISTICS")
    print("Note: heuristic classification based on generator patterns.")
    for cat, val in category_counts.most_common():
        print(f"{cat:>28} : {val:,} ({format_pct(val, total)})")
    print("\nLabel breakdown by category:")
    for cat, val in category_counts.most_common():
        label0 = category_by_label[0].get(cat, 0)
        label1 = category_by_label[1].get(cat, 0)
        print(
            f"{cat:>28} : "
            f"L0 {label0:,} ({format_pct(label0, max(1, label_counts[0]))}) | "
            f"L1 {label1:,} ({format_pct(label1, max(1, label_counts[1]))})"
        )

    print_section("EDA DONE")
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s | Rows/s: {total / max(1.0, elapsed):,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed EDA for urls_synthetic_10m.csv")
    parser.add_argument("--path", default="urls_synthetic_10m.csv", help="CSV path")
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--sample-size", type=int, default=200_000, help="Reservoir sample size for percentiles")
    parser.add_argument("--progress-every", type=int, default=1_000_000, help="Progress interval")
    args = parser.parse_args()

    run_eda(args.path, args.max_rows, args.sample_size, args.progress_every)


if __name__ == "__main__":
    main()

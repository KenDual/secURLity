"""
Detailed EDA for `dataset/dataset 2.csv`
=========================================

Dataset 2 = 7M benign (rule-based, model-2 expanded pools)
          + 3M malicious (Char-RNN, học từ ~413k real-world malicious URLs).

Khác EDA 1:
  - Word pools mở rộng (TECH_SAAS, EDUCATION, SOCIAL, QUERY_KEYS, etc.) để
    bám sát generator benign mới.
  - Thêm phân tích dành riêng cho malicious Char-RNN (không có pattern cứng):
    * Char-RNN artifacts: double-dot, no-dot host, very-short-TLD,
      repeated-char run, vowel/consonant ratio bất thường.
    * Coverage character set vs ALLOWED_CHARS (49 chars của CNN-LSTM).
  - Thêm phân tích host: số dots, hyphens, digits trong host.
  - Phân loại heuristic được điều chỉnh: malicious không còn phân nhánh theo
    generator mà gom theo signal (DGA-like, IP-based, brand-spoof,
    suspicious-keyword, malware-ext, scam-bait, c2-path, generic-malicious).
  - Streaming + constant memory (RunningStats + ReservoirSampler) — file 535MB
    chạy được trên RAM thường.

Usage (PowerShell):
    .\\venv\\Scripts\\Activate.ps1
    python "scripts/2. eda 2.py"
    python "scripts/2. eda 2.py" --path "dataset/dataset 2.csv" --max-rows 1000000
    python "scripts/2. eda 2.py" --output "dataset/eda-result 2.txt"
"""

import argparse
import csv
import math
import random
import string as _string
import sys
import time
from collections import Counter, defaultdict
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

# Tăng csv field size (URL có thể dài bất thường)
csv.field_size_limit(10_000_000)


# ---------------------------------------------------------------------------
# Word pools — sync với scripts/1. url-generator-benign (model 2).py
# ---------------------------------------------------------------------------

BENIGN_WORDS = {
    "home", "about", "about-us", "contact", "contact-us", "sitemap", "index",
    "login", "signin", "signup", "register", "logout", "auth", "oauth",
    "profile", "account", "settings", "preferences", "dashboard", "overview",
    "feed", "timeline", "activity", "history", "notifications", "alerts",
    "news", "blog", "blogs", "post", "posts", "article", "articles", "story",
    "stories", "press", "releases", "announcements", "updates", "changelog",
    "category", "categories", "tag", "tags", "topic", "topics", "section",
    "archive", "archives", "rss", "atom",
    "gallery", "photo", "photos", "image", "images", "picture", "pictures",
    "video", "videos", "media", "stream", "streaming", "live", "watch",
    "playlist", "album", "albums", "track", "tracks", "podcast", "podcasts",
    "webinar", "broadcast", "recording", "recordings", "clip", "clips",
    "product", "products", "item", "items", "detail", "details", "listing",
    "shop", "store", "market", "marketplace", "catalog", "catalogue",
    "checkout", "cart", "basket", "wishlist", "order", "orders",
    "payment", "invoice", "receipt", "refund", "return", "returns",
    "shipping", "delivery", "tracking", "inventory", "stock",
    "subscription", "subscriptions", "membership", "plan", "plans",
    "pricing", "price", "quote", "bundle", "offer", "deals", "sale",
    "discount", "coupon", "voucher", "promo", "promotion",
    "help", "support", "faq", "faqs", "knowledgebase", "kb", "docs",
    "documentation", "manual", "guide", "guides", "tutorial", "tutorials",
    "howto", "learn", "course", "courses", "lesson", "lessons", "training",
    "terms", "privacy", "legal", "compliance", "gdpr",
    "community", "forum", "forums", "discussion", "discussions",
    "comment", "comments", "review", "reviews", "rating", "ratings",
    "feedback", "share", "like", "follow",
    "friend", "follower", "trending", "popular", "explore", "discover",
    "message", "messages", "inbox", "outbox", "thread", "chat", "group",
    "team", "careers", "jobs", "hire", "work",
    "services", "solutions", "features", "integrations", "partners",
    "partner", "affiliate", "referral", "reseller", "enterprise", "business",
    "events", "event", "conference",
    "api", "graphql", "rest", "webhook", "webhooks", "sdk",
    "developer", "developers", "dev", "portal", "console", "panel",
    "admin", "manage", "manager", "control",
    "status", "health", "uptime", "metrics", "analytics", "stats",
    "logs", "audit", "monitor", "monitoring",
    "config", "deploy", "deployment", "release", "version", "migration",
    "server", "servers", "node", "cluster",
    "download", "downloads", "upload", "uploads", "file", "files",
    "export", "import", "sync", "backup", "restore", "migrate", "transfer",
    "report", "reports", "data", "dataset", "analysis",
    "document", "documents", "spreadsheet", "template", "templates",
    "user", "users", "member", "members", "role", "roles",
    "permission", "access", "security", "verify", "confirm", "activate",
    "password", "reset", "sso", "saml",
    "map", "maps", "location", "directory", "search", "results",
    "filter", "sort", "compare", "embed", "widget", "plugin", "extension",
    "app", "apps", "mobile", "ios", "android", "web", "desktop",
    "open", "public", "shared", "private", "secure",
    "old", "new", "beta", "alpha", "preview", "test", "demo", "sandbox",
    "v1", "v2", "v3", "2023", "2024", "en", "us", "uk", "au", "ca",
}

ECOMMERCE_DOMAINS = {
    "amazon.com", "ebay.com", "etsy.com", "shopify.com", "walmart.com",
    "bestbuy.com", "target.com", "newegg.com", "alibaba.com", "aliexpress.com",
    "wish.com", "geek.com", "cdkeys.com", "fanatical.com", "overstock.com",
    "wayfair.com", "chewy.com", "zappos.com", "nordstrom.com", "macys.com",
    "homedepot.com", "lowes.com", "costco.com", "samsclub.com",
    "rakuten.co.jp", "mercadolibre.com.ar", "flipkart.com", "tokopedia.com",
    "lazada.com", "shopee.com", "jd.com", "taobao.com", "tmall.com",
    "bukalapak.com", "tiki.vn", "sendo.vn", "thegioididong.com",
    "zalando.com", "asos.com", "otto.de", "bol.com", "cdiscount.com",
    "allegro.pl", "emag.ro", "heureka.cz",
    "steamstore.steampowered.com", "g2a.com", "kinguin.net", "humble.com",
    "bandcamp.com", "gumroad.com", "payhip.com",
}

NEWS_DOMAINS = {
    "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com", "apnews.com",
    "bloomberg.com", "ft.com", "economist.com", "time.com",
    "nytimes.com", "washingtonpost.com", "wsj.com", "usatoday.com",
    "theguardian.com", "independent.co.uk", "telegraph.co.uk", "mirror.co.uk",
    "techcrunch.com", "wired.com", "zdnet.com", "theverge.com", "engadget.com",
    "arstechnica.com", "gizmodo.com", "cnet.com", "tomshardware.com",
    "anandtech.com", "pcmag.com", "digitaltrends.com", "9to5mac.com",
    "androidauthority.com", "xda-developers.com",
    "forbes.com", "fortune.com", "businessinsider.com", "cnbc.com",
    "marketwatch.com", "investopedia.com", "thestreet.com",
    "foxnews.com", "huffpost.com", "vox.com", "vice.com", "axios.com",
    "buzzfeed.com", "politico.com", "theatlantic.com", "slate.com",
    "salon.com", "thedailybeast.com", "newsweek.com",
    "dw.com", "france24.com", "aljazeera.com", "rt.com", "scmp.com",
    "straitstimes.com", "abc.net.au", "smh.com.au", "theage.com.au",
    "vnexpress.net", "tuoitre.vn", "dantri.com.vn", "thanhnien.vn",
    "kenh14.vn", "zingnews.vn", "24h.com.vn",
}

TECH_SAAS_DOMAINS = {
    "aws.amazon.com", "console.cloud.google.com", "portal.azure.com",
    "app.digitalocean.com", "console.hetzner.cloud", "linode.com",
    "cloudflare.com", "fastly.com", "heroku.com", "render.com",
    "vercel.com", "netlify.com", "railway.app",
    "github.com", "gitlab.com", "bitbucket.org", "sourcegraph.com",
    "npmjs.com", "pypi.org", "packagist.org", "crates.io", "hub.docker.com",
    "stackoverflow.com", "stackexchange.com", "codepen.io", "jsfiddle.net",
    "replit.com", "codesandbox.io", "gitpod.io",
    "notion.so", "atlassian.com", "jira.atlassian.com", "confluence.atlassian.com",
    "trello.com", "asana.com", "monday.com", "clickup.com", "airtable.com",
    "figma.com", "canva.com", "miro.com", "lucidchart.com",
    "slack.com", "discord.com", "zoom.us", "meet.google.com", "teams.microsoft.com",
    "dropbox.com", "box.com", "onedrive.live.com", "drive.google.com",
    "analytics.google.com", "mixpanel.com", "amplitude.com", "segment.com",
    "hotjar.com", "fullstory.com", "hubspot.com", "salesforce.com",
    "mailchimp.com", "sendgrid.com", "twilio.com", "intercom.com",
    "stripe.com", "paypal.com", "square.com", "braintreepayments.com",
    "plaid.com", "coinbase.com", "binance.com", "kraken.com",
}

EDUCATION_DOMAINS = {
    "coursera.org", "udemy.com", "edx.org", "khanacademy.org",
    "udacity.com", "pluralsight.com", "linkedin.com",
    "skillshare.com", "masterclass.com", "duolingo.com",
    "mit.edu", "harvard.edu", "stanford.edu", "ox.ac.uk", "cam.ac.uk",
    "wikipedia.org", "en.wikipedia.org", "wikimedia.org",
    "scholar.google.com", "researchgate.net", "academia.edu",
    "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "semanticscholar.org",
    "quora.com", "medium.com", "substack.com", "dev.to", "hashnode.com",
}

SOCIAL_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "reddit.com", "linkedin.com", "pinterest.com", "tumblr.com",
    "snapchat.com", "tiktok.com", "youtube.com", "twitch.tv",
    "discord.com", "telegram.org", "whatsapp.com", "signal.org",
    "mastodon.social", "threads.net", "bluesky.app",
    "github.com", "deviantart.com", "behance.net", "dribbble.com",
    "500px.com", "flickr.com", "vimeo.com", "dailymotion.com",
    "quora.com", "goodreads.com", "last.fm", "soundcloud.com",
}

CDN_PREFIXES = {
    "cdn", "cdn1", "cdn2", "cdn3", "cdn-us", "cdn-eu", "cdn-ap",
    "static", "static1", "static2", "static-assets", "static-content",
    "assets", "assets1", "assets2", "public", "public-assets",
    "media", "media1", "media2", "media-cdn",
    "img", "img1", "img2", "images", "thumbs", "thumbnails",
    "files", "uploads", "content", "resources",
    "s3", "storage", "blob", "object",
    "cloudfront", "akamai", "fastly", "cloudflare",
    "unpkg", "cdnjs", "skypack",
    "prod", "production", "staging", "dev", "uat",
    "r2", "b2",
    "v1", "v2", "v3",
}

CDN_EXTENSIONS = {
    ".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx",
    ".css", ".scss", ".less",
    ".min.js", ".min.css", ".bundle.js", ".bundle.css",
    ".chunk.js", ".worker.js", ".map",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif", ".ico",
    ".bmp", ".tiff",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".ogg", ".aac", ".flac",
    ".mp4", ".webm", ".ogv", ".m3u8",
    ".json", ".xml", ".yaml", ".toml",
}

SHORTLINK_DOMAINS = {
    "bit.ly", "tinyurl.com", "ow.ly", "t.co", "goo.gl",
    "is.gd", "v.gd", "short.link", "tiny.cc", "buff.ly",
    "j.mp", "youtu.be", "redd.it", "lnkd.in",
    "medium.com", "rb.gy", "cutt.ly", "shorturl.at",
    "su.pr", "soo.gd", "cuttly.com", "bl.ink",
    "snip.ly", "rebrand.ly", "smart.link", "sho.rt",
    "2.gp", "clck.ru",
    "imgur.com", "i.imgur.com", "pastebin.com", "hastebin.com",
}

COMMON_SUBDOMAINS = {
    "mail", "webmail", "email", "smtp", "pop", "imap", "mx",
    "api", "api-v1", "api-v2", "api-v3", "api-gateway",
    "rest", "graphql", "grpc", "ws", "wss",
    "www", "www2", "web", "portal", "app", "apps",
    "m", "mobile", "touch", "lite", "amp",
    "cdn", "static", "assets", "media", "img", "images",
    "upload", "uploads", "files", "resources", "public",
    "auth", "sso", "login", "id", "accounts", "oauth",
    "dev", "staging", "test", "qa", "beta", "preview", "sandbox",
    "uat", "preprod", "demo",
    "admin", "manage", "panel", "console", "cp", "back", "backoffice",
    "services", "micro", "gateway", "proxy",
    "search", "maps", "shop", "blog", "docs", "help", "support",
    "status", "monitor", "health", "metrics", "analytics",
    "download", "downloads", "update", "updates",
    "ns1", "ns2", "dns", "vpn", "git", "svn", "ci", "cd",
    "jenkins", "gitlab", "grafana", "kibana", "prometheus",
    "us", "us-east", "us-west", "eu", "eu-west", "eu-central",
    "asia", "ap", "apac", "au", "jp", "sg", "in",
    "uk", "de", "fr", "br",
}

# Common TLDs (parser-friendly, expanded)
KNOWN_TLDS = sorted({
    ".co.uk", ".co.jp", ".co.in", ".co.nz", ".com.au", ".com.br", ".com.cn",
    ".com.vn", ".net.au", ".org.uk", ".ac.uk", ".gov.uk",
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
    "barclays", "lloyds", "deutsche", "ubs",
    "dhl", "fedex", "ups", "usps", "dpd",
    "nike", "adidas", "puma", "zara",
    "disney", "paramount", "sony", "warner",
    "mcdonald", "kfc", "starbucks", "coca-cola", "pepsi", "nestle",
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
    "antivirus", "booster", "accelerator",
}

# ALLOWED_CHARS từ CNN-LSTM pipeline (49 chars)
ALLOWED_CHARS = set(_string.ascii_lowercase + _string.digits + "/:.-_?=&#@%+~")
VOWELS = set("aeiou")


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
    if "://" not in url:
        parts = urlsplit("http://" + url)
        scheme = ""
    else:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
    host, port = split_host_port(parts.netloc.lower())
    return url, scheme, host, parts.path or "/", parts.query or "", port


# ---------------------------------------------------------------------------
# Char-RNN artifact detectors
# ---------------------------------------------------------------------------

def host_quality_signals(host: str) -> dict:
    """Detect suspicious host artifacts from Char-RNN generation."""
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

def classify_url(url: str, scheme: str, host: str, path: str,
                 query: str, tld: str, ext: Optional[str]) -> str:
    lower_url = url.lower()
    if is_ip(host):
        return "malicious_ip_based"
    if "/download/" in path and ext in MALWARE_EXECUTABLES:
        return "malicious_malware_dist"
    if any(bait in lower_url for bait in SCAM_BAIT):
        return "malicious_spam_scam"
    if any(brand in host for brand in LEGIT_BRANDS) and (
        "/login" in path or "/signin" in path or "/account" in path
        or "secure" in host or "auth" in host
    ):
        return "malicious_phishing_brand_spoof"
    if looks_like_dga(host):
        return "malicious_dga_like"
    if any(path.startswith(p) for p in C2_PATHS):
        return "malicious_c2_path"
    if any(w in lower_url for w in SUSPICIOUS_WORDS) and tld in {
        ".tk", ".ml", ".ga", ".cf", ".top", ".xyz", ".click", ".online", ".site",
    }:
        return "malicious_suspicious_keyword"

    if any(host_endswith(host, d) for d in ECOMMERCE_DOMAINS):
        return "benign_ecommerce"
    if any(host_endswith(host, d) for d in NEWS_DOMAINS):
        return "benign_news"
    if any(host_endswith(host, d) for d in TECH_SAAS_DOMAINS):
        return "benign_tech_saas"
    if any(host_endswith(host, d) for d in EDUCATION_DOMAINS):
        return "benign_education"
    if any(host_endswith(host, d) for d in SOCIAL_DOMAINS):
        return "benign_social"
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
    benign_word_hits = Counter()

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

            for c in url.lower():
                char_freq[c] += 1
                if c not in ALLOWED_CHARS:
                    out_of_vocab_chars[c] += 1
                    out_of_vocab_by_label[label] += 1

            lower_url = url.lower()
            if any(w in lower_url for w in SUSPICIOUS_WORDS):
                suspicious_by_label[label] += 1
                for w in SUSPICIOUS_WORDS:
                    if w in lower_url:
                        suspicious_hits[w] += 1
            if any(b in lower_url for b in SCAM_BAIT):
                scam_by_label[label] += 1
                for b in SCAM_BAIT:
                    if b in lower_url:
                        scam_hits[b] += 1
            if host and any(brand in host for brand in LEGIT_BRANDS):
                brand_by_label[label] += 1
                for brand in LEGIT_BRANDS:
                    if brand in host:
                        brand_hits[brand] += 1
            if any(path.startswith(p) for p in C2_PATHS):
                c2_by_label[label] += 1
                for p_ in C2_PATHS:
                    if path.startswith(p_):
                        c2_hits[p_] += 1
            if any(w in lower_url for w in MALWARE_LURE_WORDS):
                malware_lure_by_label[label] += 1

            for seg in segments:
                if seg in BENIGN_WORDS:
                    benign_word_hits[seg] += 1

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

            if any(host_endswith(host, d) for d in ECOMMERCE_DOMAINS):
                known_pool_hits["ecommerce"][label] += 1
            if any(host_endswith(host, d) for d in NEWS_DOMAINS):
                known_pool_hits["news"][label] += 1
            if any(host_endswith(host, d) for d in TECH_SAAS_DOMAINS):
                known_pool_hits["tech_saas"][label] += 1
            if any(host_endswith(host, d) for d in EDUCATION_DOMAINS):
                known_pool_hits["education"][label] += 1
            if any(host_endswith(host, d) for d in SOCIAL_DOMAINS):
                known_pool_hits["social"][label] += 1
            if host in SHORTLINK_DOMAINS:
                known_pool_hits["shortlink"][label] += 1

            cat = classify_url(url, scheme, host, path, query, tld, ext)
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
    print_counter(tld_counts, _print, top_n=20, label="Top TLDs (overall)")
    for lbl in sorted(label_counts):
        print_counter(tld_by_label[lbl], _print, top_n=20, label=f"Top TLDs (Label {lbl})")
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
    print_counter(brand_hits, _print, top_n=15, label="Top brand mentions")

    _print(f"\nC2-path URLs         : {sum(c2_by_label.values()):,}")
    for lbl in sorted(label_counts):
        c = c2_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")
    print_counter(c2_hits, _print, top_n=10, label="Top C2 paths")

    _print(f"\nMalware-lure URLs    : {sum(malware_lure_by_label.values()):,}")
    for lbl in sorted(label_counts):
        c = malware_lure_by_label[lbl]
        _print(f"  Label {lbl}: {c:,} ({fmt_pct(c, label_counts[lbl])})")

    section("BENIGN WORD COVERAGE (path segments)", _print)
    print_counter(benign_word_hits, _print, top_n=25, label="Top benign words")

    section("KNOWN DOMAIN POOLS", _print)
    for pool_name in ["ecommerce", "news", "tech_saas", "education", "social", "shortlink"]:
        c = known_pool_hits[pool_name]
        total_pool = sum(c.values())
        _print(f"\n{pool_name.upper():>12}: {total_pool:,} ({fmt_pct(total_pool, total)})")
        for lbl in sorted(label_counts):
            _print(f"  Label {lbl}: {c[lbl]:,} ({fmt_pct(c[lbl], label_counts[lbl])})")

    section("TOP HOSTS", _print)
    print_counter(top_hosts, _print, top_n=20, label="Top hosts (overall)")
    for lbl in sorted(label_counts):
        print_counter(top_hosts_by_label[lbl], _print, top_n=20,
                      label=f"Top hosts (Label {lbl})")

    section("CHAR-RNN ARTIFACT / QUALITY SIGNALS (host)", _print)
    _print("Heuristics nhằm bắt URL malformed do model Char-RNN sinh ra.")
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
        description="Detailed EDA for dataset/dataset 2.csv"
    )
    parser.add_argument("--path", default=r"D:\! secURLity\dataset\dataset 2.csv",
                        help="CSV path (default: dataset/dataset 2.csv)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 = all rows")
    parser.add_argument("--sample-size", type=int, default=200_000,
                        help="Reservoir sample size (default 200k)")
    parser.add_argument("--progress-every", type=int, default=1_000_000,
                        help="Progress print interval")
    parser.add_argument("--output", default=r"D:\! secURLity\dataset\eda-result 2.txt",
                        help="Save report to file (in addition to stdout). "
                             "Pass empty string to disable.")
    args = parser.parse_args()

    out = args.output if args.output else None
    run_eda(args.path, args.max_rows, args.sample_size,
            args.progress_every, out)


if __name__ == "__main__":
    main()

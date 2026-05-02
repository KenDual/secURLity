"""
Synthetic URL Dataset Generator (v3 FIXED - One-Time Real Data + Full Synthetic)
==================================================================================
Output : urls_synthetic_10m.csv  (columns: url, label)
         label = 0 → benign  |  label = 1 → malicious

FIXED: Protocol distribution (only change from v3):
  - Benign  : 95% HTTPS / 5%  HTTP  (legacy/dev sites)
  - Malicious: 87% HTTPS / 13% HTTP  (2024-2026 real-world stat, APWG)

Everything else (word pools, generators, strategy) identical to v3.
"""

import csv
import random
import string
import math
from pathlib import Path
from typing import Set, List, Tuple

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# REAL DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_real_legitimate_domains(filepath: str) -> List[str]:
    """Load ALL real legitimate domains from Alexa top 1M (no limit)."""
    domains = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, fieldnames=["url"])
            for i, row in enumerate(reader):
                domain = row["url"].strip()
                if domain and domain != "url":
                    domains.append(domain)
    except Exception as e:
        print(f"[!] Could not load legitimate domains: {e}")
    return domains


def load_real_malicious_urls(filepath: str) -> Set[str]:
    """Load ALL real malicious URLs from URLhaus (no limit)."""
    urls = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url:
                    urls.add(url)
    except Exception as e:
        print(f"[!] Could not load malicious URLs: {e}")
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL HELPERS  ← ONLY NEW CODE vs v3
# ─────────────────────────────────────────────────────────────────────────────

def scheme_benign() -> str:
    """95% HTTPS / 5% HTTP — mirrors modern legitimate web traffic."""
    return "https://" if random.random() < 0.95 else "http://"


def scheme_malicious() -> str:
    """
    87% HTTPS / 13% HTTP.
    Source: APWG Q4 2024; Kaspersky Securelist 2024.
    Attackers use free DV (Domain Validation) certs to appear legitimate.
    """
    return "https://" if random.random() < 0.87 else "http://"


# ─────────────────────────────────────────────────────────────────────────────
# WORD POOLS & CONSTANTS  ← IDENTICAL TO v3
# ─────────────────────────────────────────────────────────────────────────────

BENIGN_WORDS = [
    # Navigation
    "home", "about", "contact", "login", "register", "search", "profile",
    "settings", "dashboard", "feed", "news", "blog", "article", "category",
    "product", "item", "detail", "checkout", "cart", "order", "account",
    "help", "support", "faq", "terms", "privacy", "careers", "team",
    # Content
    "services", "solutions", "features", "pricing", "docs", "api",
    "download", "upload", "gallery", "photo", "video", "stream",
    "library", "archive", "forum", "community", "events", "schedule",
    "report", "analysis", "research", "papers", "projects", "portfolio",
    "guide", "tutorial", "howto", "learn", "course", "lesson",
    "review", "rating", "comment", "share", "like", "follow",
    # E-commerce
    "buy", "sell", "shop", "store", "inventory", "warehouse", "shipping",
    "payment", "subscription", "membership", "tier", "plan", "bundle",
    # Social/Interaction
    "user", "member", "friend", "follower", "trending", "explore", "discover",
    "notification", "message", "inbox", "outbox", "thread", "conversation",
    # Media
    "image", "media", "content", "channel", "playlist", "album", "track",
    "podcast", "webinar", "broadcast", "live", "stream", "recording",
    # Data
    "export", "import", "sync", "backup", "restore", "migrate", "transfer",
    "database", "table", "collection", "document", "record", "field",
    # Admin/System
    "admin", "panel", "console", "server", "status", "health", "metrics",
    "logs", "audit", "config", "settings", "preferences", "options",
    "permission", "role", "access", "security", "certificate",
    # Other
    "search", "filter", "sort", "paginate", "view", "edit", "create", "delete",
]

BENIGN_TLDS = [
    ".com", ".org", ".net", ".edu", ".gov", ".mil",
    ".co.uk", ".co.jp", ".co.in", ".de", ".fr", ".cn", ".au", ".ca",
    ".io", ".dev", ".app", ".info", ".biz", ".mobi", ".name",
    ".blog", ".shop", ".tech", ".site", ".online", ".space",
]

ECOMMERCE_DOMAINS = [
    "amazon.com", "ebay.com", "etsy.com", "shopify.com", "walmart.com",
    "bestbuy.com", "target.com", "newegg.com", "alibaba.com", "rakuten.co.jp",
    "mercadolibre.com.ar", "flipkart.com", "tokopedia.com", "lazada.com",
    "wish.com", "geek.com", "aliexpress.com", "cdkeys.com", "fanatical.com",
]

NEWS_DOMAINS = [
    "bbc.com", "cnn.com", "reuters.com", "apnews.com", "nytimes.com",
    "theguardian.com", "washingtonpost.com", "forbes.com", "foxnews.com",
    "cnbc.com", "techcrunch.com", "wired.com",
    "zdnet.com", "theverge.com", "engadget.com",
    "vice.com", "vox.com", "buzzfeed.com", "huffpost.com", "axios.com",
    "politico.com", "breitbart.com", "drudge.com",
]

CDN_PREFIXES = [
    "cdn", "static", "assets", "media", "img", "images", "files", "s3",
    "cloudfront", "akamai", "fastly", "cloudflare", "jsdelivr", "unpkg",
    "cdn-images", "static-assets", "production", "staging", "v1", "v2",
]

CDN_EXTENSIONS = [
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".webm",
    ".min.js", ".min.css", ".bundle.js", ".map",
]

LEGIT_BRANDS = [
    # Tech
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "tesla",
    "facebook", "instagram", "twitter", "linkedin", "dropbox", "slack",
    "github", "gitlab", "bitbucket", "jira", "confluence", "docker",
    "kubernetes", "jenkins", "ansible", "terraform", "aws", "azure",
    # Finance
    "wellsfargo", "bankofamerica", "chase", "citibank", "hsbc", "bnp",
    "barclays", "lloyds", "deutsche", "ubs", "credit-suisse",
    # Logistics
    "dhl", "fedex", "ups", "usps", "dpd", "gls", "tnt", "hermes",
    # Retail
    "nike", "adidas", "puma", "reebok", "zara", "hm", "gap", "primark",
    # Media/Entertainment
    "disney", "paramount", "sony", "warner", "universal",
    # Food/Beverage
    "mcdonald", "kfc", "starbucks", "coca-cola", "pepsi", "nestle",
]

SHORTLINK_DOMAINS = [
    "bit.ly", "tinyurl.com", "ow.ly", "t.co", "shorte.st", "goo.gl",
    "is.gd", "v.gd", "short.link", "tiny.cc", "imgur.com", "buff.ly",
    "adf.ly", "j.mp", "youtu.be", "redd.it", "lnkd.in", "medium.com",
]

COMMON_SUBDOMAINS = [
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
]

MALICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".top", ".xyz", ".click", ".club",
    ".online", ".site", ".live", ".cc", ".pw", ".in", ".ru", ".to",
    ".bid", ".download", ".party", ".faith", ".work", ".gdn", ".trade",
    ".stream", ".red", ".win", ".cricket", ".loan", ".racing", ".review",
]

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

MALWARE_EXECUTABLES = [
    ".exe", ".bat", ".ps1", ".msi", ".zip", ".rar", ".dmg", ".apk",
    ".deb", ".rpm", ".tar.gz", ".iso", ".img", ".dll", ".scr", ".vbs",
    ".js", ".jar", ".class", ".so", ".dylib", ".sh", ".elf",
]

MALWARE_LURE_WORDS = [
    "update", "install", "setup", "patch", "fix", "crack", "keygen",
    "loader", "installer", "driver", "tool", "utility", "codec",
    "player", "reader", "viewer", "converter", "optimizer", "cleaner",
    "antivirus", "security", "booster", "accelerator", "manager",
]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS  ← IDENTICAL TO v3
# ─────────────────────────────────────────────────────────────────────────────

def rand_str(length: int, charset=string.ascii_lowercase) -> str:
    return "".join(random.choices(charset, k=length))

def rand_hex(length: int) -> str:
    return "".join(random.choices(string.hexdigits[:16], k=length))

def rand_slug(word_count: int = 3) -> str:
    return "-".join(random.choices(BENIGN_WORDS, k=word_count))

def rand_digits(n: int) -> str:
    return "".join(random.choices(string.digits, k=n))

def rand_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

def dga_domain(length: int = None) -> str:
    length = length or random.randint(10, 22)
    charset = string.ascii_lowercase + string.digits
    domain = rand_str(length, charset)
    tld = random.choice(MALICIOUS_TLDS)
    return domain + tld

def typosquat(brand: str) -> str:
    techniques = [
        lambda b: b.replace("o", "0").replace("i", "1").replace("l", "1"),
        lambda b: b + rand_digits(2),
        lambda b: b + "-" + random.choice(["secure", "login", "auth"]),
        lambda b: b[:-1] + random.choice(string.ascii_lowercase),
        lambda b: b + b[-1],
        lambda b: b.replace("a", "4").replace("e", "3"),
    ]
    return random.choice(techniques)(brand)


# ─────────────────────────────────────────────────────────────────────────────
# BENIGN GENERATORS  ← scheme() calls added, logic identical to v3
# ─────────────────────────────────────────────────────────────────────────────

def gen_benign_popular(popular_domains: List[str]) -> str:
    s = scheme_benign()                                          # ← FIXED
    domain = random.choice(popular_domains)
    depth = random.randint(0, 3)
    if depth == 0:
        return f"{s}{domain}/"
    path = "/".join(random.choices(BENIGN_WORDS, k=depth))
    if random.random() < 0.2:
        param = random.choice(["q", "id", "page", "ref", "tab", "sort"])
        value = rand_str(random.randint(3, 10)) if random.random() < 0.5 else rand_digits(4)
        return f"{s}{domain}/{path}?{param}={value}"
    return f"{s}{domain}/{path}"


def gen_benign_ecommerce() -> str:
    s = scheme_benign()                                          # ← FIXED
    domain = random.choice(ECOMMERCE_DOMAINS)
    patterns = [
        lambda: f"{s}www.{domain}/dp/{rand_str(10, string.ascii_uppercase + string.digits)}",
        lambda: f"{s}www.{domain}/product/{rand_slug(2)}-{rand_digits(6)}",
        lambda: f"{s}www.{domain}/{random.choice(BENIGN_WORDS)}/{rand_slug(2)}?page={random.randint(1,10)}",
        lambda: f"{s}www.{domain}/cart?item={rand_digits(8)}&qty={random.randint(1,5)}",
        lambda: f"{s}www.{domain}/category/{rand_slug(2)}?sort=price",
    ]
    return random.choice(patterns)()


def gen_benign_cdn() -> str:
    s = scheme_benign()                                          # ← FIXED
    domain = random.choice(["google.com", "facebook.com", "cloudflare.com", "akamai.net"])
    prefix = random.choice(CDN_PREFIXES)
    ext = random.choice(CDN_EXTENSIONS)
    version = f"v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}"
    filename = rand_slug(2).replace("-", "_") + ext
    return f"{s}{prefix}.{domain}/{version}/{filename}"


def gen_benign_api() -> str:
    s = scheme_benign()
    domain = random.choice(["google.com", "github.com", "microsoft.com"])
    version = f"v{random.randint(1,3)}"
    resources = ["users", "products", "orders", "posts", "metrics", "reports"]
    resource = random.choice(resources)
    return f"{s}api.{domain}/{version}/{resource}/{rand_digits(5)}"


def gen_benign_news() -> str:
    s = scheme_benign()
    domain = random.choice(NEWS_DOMAINS + ["medium.com"])
    year = random.randint(2018, 2024)
    month = str(random.randint(1, 12)).zfill(2)
    slug = rand_slug(random.randint(4, 7))
    return f"{s}www.{domain}/{year}/{month}/{slug}"


def gen_benign_shortlink() -> str:
    s = scheme_benign()
    domain = random.choice(SHORTLINK_DOMAINS)
    code = rand_str(random.randint(4, 8), string.ascii_letters + string.digits)
    return f"{s}{domain}/{code}"


def gen_benign_social() -> str:
    s = scheme_benign()
    domain = random.choice(["twitter.com", "facebook.com", "instagram.com", "reddit.com"])
    return f"{s}{domain}/{rand_slug(1)}/{rand_digits(10)}"


def gen_benign_subdomains(popular_domains: List[str]) -> str:
    s = scheme_benign()
    domain = random.choice(popular_domains[:500])
    subdomain = random.choice(COMMON_SUBDOMAINS)
    path_choice = random.randint(0, 2)
    if path_choice == 0:
        return f"{s}{subdomain}.{domain}/{random.choice(BENIGN_WORDS)}"
    elif path_choice == 1:
        return f"{s}{subdomain}.{domain}/?id={rand_digits(6)}"
    else:
        return f"{s}{subdomain}.{domain}/"


# ─────────────────────────────────────────────────────────────────────────────
# MALICIOUS GENERATORS  ← scheme() calls added, logic identical to v3
# NOTE: gen_malicious_ip_based stays http:// — IP-based C2 is always HTTP
# ─────────────────────────────────────────────────────────────────────────────

def gen_malicious_phishing() -> str:
    s = scheme_malicious()
    brand = random.choice(LEGIT_BRANDS)
    technique = random.randint(1, 3)
    if technique == 1:
        fake_domain = rand_str(random.randint(8, 18)) + random.choice(MALICIOUS_TLDS)
        path = "/" + random.choice(["login", "verify", "secure", "account"])
        return f"{s}{brand}.{fake_domain}{path}.php"
    elif technique == 2:
        fake = typosquat(brand) + random.choice([".com", ".net"])
        return f"{s}www.{fake}/login"
    else:
        fake_domain = rand_str(random.randint(6, 15)) + random.choice(MALICIOUS_TLDS)
        return f"{s}{fake_domain}/account/signin?redirect={rand_hex(16)}"


def gen_malicious_malware_dist() -> str:
    s = scheme_malicious()                                       # ← FIXED
    domain = rand_str(random.randint(6, 15)) + random.choice(MALICIOUS_TLDS)
    filename = random.choice(MALWARE_LURE_WORDS) + "_" + rand_digits(4) + random.choice(MALWARE_EXECUTABLES)
    return f"{s}{domain}/download/{filename}"


def gen_malicious_ip_based() -> str:
    """IP-based C2: always HTTP — browsers block HTTPS on raw IPs."""
    ip = rand_ip()
    port = random.choice([80, 443, 8080, 8443, 4444, 1337])
    path = random.choice(C2_PATHS)
    return f"http://{ip}:{port}{path}?id={rand_hex(8)}"              # stays http


def gen_malicious_dga() -> str:
    s = scheme_malicious()
    domain = dga_domain()
    path = random.choice(C2_PATHS) if random.random() < 0.6 else ""
    return f"{s}{domain}{path}"


def gen_malicious_drive_by() -> str:
    s = scheme_malicious()
    domain = rand_str(random.randint(5, 12)) + random.choice(MALICIOUS_TLDS)
    return f"{s}{domain}/ad/{rand_hex(4)}?{rand_hex(4)}={rand_hex(16)}"


def gen_malicious_spam_scam() -> str:
    s = scheme_malicious()
    bait = random.choice(SCAM_BAIT)
    tld = random.choice(MALICIOUS_TLDS + [".com"])
    return f"{s}{bait}-{rand_str(4)}{tld}/claim?ref={rand_digits(8)}"


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER  ← IDENTICAL TO v3
# ─────────────────────────────────────────────────────────────────────────────

BENIGN_GENERATORS = [
    (gen_benign_ecommerce, 0.18),
    (gen_benign_cdn,       0.12),
    (gen_benign_api,       0.12),
    (gen_benign_news,      0.12),
    (gen_benign_shortlink, 0.12),
    (gen_benign_social,    0.10),
    (gen_benign_subdomains, 0.14),
]

MALICIOUS_GENERATORS = [
    (gen_malicious_phishing,     0.25),
    (gen_malicious_malware_dist, 0.20),
    (gen_malicious_ip_based,     0.20),
    (gen_malicious_dga,          0.15),
    (gen_malicious_drive_by,     0.10),
    (gen_malicious_spam_scam,    0.10),
]

def _pick(generators):
    fns, weights = zip(*generators)
    return random.choices(fns, weights=weights, k=1)[0]()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR  ← logic identical to v3; real benign variants use scheme()
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_one_time_only(
    n_benign: int = 8_500_000,
    n_malicious: int = 1_500_000,
    output_path: str = "urls_synthetic_10m.csv",
    real_benign_file: str = "top-1m-legitimate-url.csv",
    real_malicious_file: str = "urlhaus-malicious-url.csv",
    batch_size: int = 100_000,
) -> None:
    print(f"[*] Generating {n_benign:,} benign + {n_malicious:,} malicious URLs")
    print(f"    Protocol: Benign 95% HTTPS | Malicious 87% HTTPS  [FIXED]\n")

    print("  [>] Loading real data...")
    real_benign_list  = load_real_legitimate_domains(real_benign_file)
    real_malicious_list = list(load_real_malicious_urls(real_malicious_file))
    print(f"      [+] {len(real_benign_list):,} legitimate domains loaded")
    print(f"      [+] {len(real_malicious_list):,} malicious URLs loaded")

    n_benign_real_with_variants = len(real_benign_list) * random.randint(3, 4)
    n_benign_synthetic  = n_benign  - n_benign_real_with_variants
    n_malicious_real    = len(real_malicious_list)
    n_malicious_synthetic = n_malicious - n_malicious_real

    print(f"\n  [>] Plan:")
    print(f"      Benign:    {n_benign_real_with_variants:,} real variants + {n_benign_synthetic:,} synthetic")
    print(f"      Malicious: {n_malicious_real:,} real (URLhaus) + {n_malicious_synthetic:,} synthetic")

    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])

        benign_written   = 0
        malicious_written = 0
        batch      = []
        batch_seen = set()

        # ── REAL BENIGN VARIANTS ─────────────────────────────────────────────
        print("\n  [>] Real benign variants...")
        variants_per_domain = random.randint(3, 4)
        real_written = 0
        for domain in random.sample(real_benign_list, len(real_benign_list)):
            if real_written >= n_benign - n_benign_synthetic:
                break
            s = scheme_benign()
            variants = [
                f"{s}{domain}/",
                f"{s}{domain}/{random.choice(BENIGN_WORDS)}",
                f"{s}www.{domain}/" if not domain.startswith("www.") else None,
                f"{s}{domain}/api/{random.choice(BENIGN_WORDS)}",
            ]
            variants = [v for v in variants if v]

            for url in variants[:variants_per_domain]:
                if real_written >= n_benign - n_benign_synthetic:
                    break
                if url not in batch_seen:
                    batch_seen.add(url)
                    batch.append((url, 0))
                    real_written  += 1
                    benign_written += 1

                    if len(batch) >= batch_size:
                        writer.writerows(batch)
                        batch = []
                        batch_seen = set()
                        print(f"      [{benign_written:,} / {n_benign:,}] benign written")

        print(f"      [+] {real_written:,} real benign variants written")

        # ── SYNTHETIC BENIGN ─────────────────────────────────────────────────
        print("  [>] Synthetic benign...")
        benign_gens = [
            (gen_benign_ecommerce, 0.18),
            (gen_benign_cdn,       0.12),
            (gen_benign_api,       0.12),
            (gen_benign_news,      0.12),
            (gen_benign_shortlink, 0.12),
            (gen_benign_social,    0.10),
            (lambda: gen_benign_subdomains(real_benign_list),      0.14),
            (lambda: gen_benign_popular(real_benign_list[:1000]),  0.04),
        ]
        synth_written = 0
        attempts      = 0
        while synth_written < n_benign_synthetic:
            attempts += 1
            if attempts > n_benign_synthetic * 2:
                print(f"      [!] Stopping at {synth_written:,} synthetic benign")
                break
            url = _pick(benign_gens)
            if url not in batch_seen:
                batch_seen.add(url)
                batch.append((url, 0))
                synth_written  += 1
                benign_written += 1
                if len(batch) >= batch_size:
                    writer.writerows(batch)
                    batch = []
                    batch_seen = set()
                    print(f"      [{benign_written:,} / {n_benign:,}] benign written")

        print(f"      [+] {synth_written:,} synthetic benign written")

        if batch:
            writer.writerows(batch)
            batch = []
            batch_seen = set()

        # ── REAL MALICIOUS ───────────────────────────────────────────────────
        print("  [>] Real malicious (URLhaus, used once)...")
        real_written = 0
        for url in random.sample(real_malicious_list, len(real_malicious_list)):
            if real_written >= n_malicious_real:
                break
            if url not in batch_seen:
                batch_seen.add(url)
                batch.append((url, 1))
                real_written      += 1
                malicious_written += 1
                if len(batch) >= batch_size:
                    writer.writerows(batch)
                    batch = []
                    batch_seen = set()
                    print(f"      [{malicious_written:,} / {n_malicious:,}] malicious written")

        print(f"      [+] {real_written:,} real malicious written")

        # ── SYNTHETIC MALICIOUS ──────────────────────────────────────────────
        print("  [>] Synthetic malicious...")
        synth_written = 0
        attempts      = 0
        while synth_written < n_malicious_synthetic:
            attempts += 1
            if attempts > n_malicious_synthetic * 2:
                print(f"      [!] Stopping at {synth_written:,} synthetic malicious")
                break
            url = _pick(MALICIOUS_GENERATORS)
            if url not in batch_seen:
                batch_seen.add(url)
                batch.append((url, 1))
                synth_written      += 1
                malicious_written  += 1
                if len(batch) >= batch_size:
                    writer.writerows(batch)
                    batch = []
                    batch_seen = set()
                    print(f"      [{malicious_written:,} / {n_malicious:,}] malicious written")

        print(f"      [+] {synth_written:,} synthetic malicious written")

        if batch:
            writer.writerows(batch)

    total = benign_written + malicious_written
    print(f"\n[OK] Done → {output_path}")
    print(f"    Total     : {total:,}")
    print(f"    Benign    : {benign_written:,}  ({benign_written/total*100:.1f}%)")
    print(f"    Malicious : {malicious_written:,}  ({malicious_written/total*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset_one_time_only(
        n_benign=8_500_000,
        n_malicious=1_500_000,
        output_path="urls_synthetic_10m.csv",
        real_benign_file="top-1m-legitimate-url.csv",
        real_malicious_file="urlhaus-malicious-url.csv",
        batch_size=100_000,
    )
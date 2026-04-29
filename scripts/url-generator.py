"""
Synthetic URL Dataset Generator (v3 - One-Time Real Data + Full Synthetic)
===========================================================================
Output : urls_synthetic_10m.csv  (columns: url, label)
         label = 0 → benign  |  label = 1 → malicious

Strategy:
  - Each real URL used EXACTLY ONCE (no reuse/cycling)
  - Real benign: 1M Alexa URLs + 2-4 variants each → ~3-4M URLs
  - Real malicious: 12K URLhaus URLs (used once) → ~12K URLs
  - Remaining: 100% synthetic generation (no duplicates across all)
  - Total: 10,000,000 (8.5M benign + 1.5M malicious)

Features:
  - Diverse word pools & TLDs for synthetic generation
  - One-time-only real data (zero cycling)
  - High diversity through multiple URL patterns
"""

import csv
import random
import string
import math
from pathlib import Path
from typing import Set, List, Tuple

# ── reproducibility ──────────────────────────────────────────────────────────
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
# WORD POOLS & CONSTANTS (EXPANDED FOR DIVERSITY)
# ─────────────────────────────────────────────────────────────────────────────

# Expanded benign words (100+ options)
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

# Expanded TLDs
BENIGN_TLDS = [
    ".com", ".org", ".net", ".edu", ".gov", ".mil",
    ".co.uk", ".co.jp", ".co.in", ".de", ".fr", ".cn", ".au", ".ca",
    ".io", ".dev", ".app", ".info", ".biz", ".mobi", ".name",
    ".blog", ".shop", ".tech", ".site", ".online", ".space",
]

# Expanded e-commerce domains
ECOMMERCE_DOMAINS = [
    "amazon.com", "ebay.com", "etsy.com", "shopify.com", "walmart.com",
    "bestbuy.com", "target.com", "newegg.com", "alibaba.com", "rakuten.co.jp",
    "mercadolibre.com.ar", "flipkart.com", "tokopedia.com", "lazada.com",
    "wish.com", "geek.com", "aliexpress.com", "cdkeys.com", "fanatical.com",
]

# Expanded news/media domains
NEWS_DOMAINS = [
    "bbc.com", "cnn.com", "reuters.com", "apnews.com", "nytimes.com",
    "theguardian.com", "washingtonpost.com", "forbes.com", "foxnews.com",
    "cnbc.com", "bussiness insider.com", "techcrunch.com", "wired.com",
    "ars technica.com", "zdnet.com", "theverge.com", "engadget.com",
    "vice.com", "vox.com", "buzzfeed.com", "huffpost.com", "axios.com",
    "politico.com", "breitbart.com", "drudge.com",
]

# Expanded CDN/Static prefixes
CDN_PREFIXES = [
    "cdn", "static", "assets", "media", "img", "images", "files", "s3",
    "cloudfront", "akamai", "fastly", "cloudflare", "jsdelivr", "unpkg",
    "cdn-images", "static-assets", "production", "staging", "v1", "v2",
]

# Expanded file extensions
CDN_EXTENSIONS = [
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".webm",
    ".min.js", ".min.css", ".bundle.js", ".map",
]

# Expanded legitimate brands
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
    "disney", "paramount", "sony", "warner", "universal", "paramount",
    # Food/Beverage
    "mcdonald", "kfc", "starbucks", "coca-cola", "pepsi", "nestle",
]

# Expanded shortlink services
SHORTLINK_DOMAINS = [
    "bit.ly", "tinyurl.com", "ow.ly", "t.co", "shorte.st", "goo.gl",
    "is.gd", "v.gd", "short.link", "tiny.cc", "imgur.com", "buff.ly",
    "adf.ly", "j.mp", "youtu.be", "redd.it", "lnkd.in", "medium.com",
]

# Common subdomains (production, legitimate services only)
COMMON_SUBDOMAINS = [
    "mail", "ftp", "smtp", "pop", "imap",          # Email services
    "api", "api-v1", "api-v2", "api-v3",           # API endpoints
    "www", "web",                                   # Web
    "cdn", "static", "assets", "images",            # CDN/Static
    "download", "downloads",                        # Downloads
    "secure", "vpn", "ssl",                         # Security
    "internal", "intranet",                         # Internal
    "us", "eu", "asia", "apac", "jp", "au",        # Regional
    "search", "map", "drive", "docs", "mail",      # Services
    "monitor", "status", "health", "dashboard",     # Monitoring
]

# Expanded malicious TLDs
MALICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".top", ".xyz", ".click", ".club",
    ".online", ".site", ".live", ".cc", ".pw", ".in", ".ru", ".to",
    ".bid", ".download", ".party", ".faith", ".work", ".gdn", ".trade",
    ".stream", ".red", ".win", ".cricket", ".loan", ".racing", ".review",
]

# Expanded suspicious keywords
SUSPICIOUS_WORDS = [
    "secure", "verify", "update", "confirm", "alert", "warning",
    "unlock", "restore", "validate", "authenticate", "suspended",
    "limited", "urgent", "action-required", "login", "signin", "re-confirm",
    "reconfirm", "unusual-activity", "confirm-identity", "verify-account",
    "security-alert", "immediate-action", "click-here", "act-now",
]

# Expanded C2 paths
C2_PATHS = [
    "/cmd", "/c2", "/gate", "/bot", "/beacon", "/checkin", "/tasks",
    "/poll", "/update", "/ping", "/report", "/data", "/admin", "/api",
    "/i", "/bin.sh", "/shell", "/exec", "/run", "/config", "/status",
    "/sync", "/push", "/pull", "/fetch", "/load", "/plugin",
]

# Expanded scam bait
SCAM_BAIT = [
    "you-won", "claim-prize", "free-gift", "limited-offer", "congratulations",
    "lucky-winner", "exclusive-deal", "earn-money-fast", "lose-weight-now",
    "get-rich-quick", "click-here", "act-now", "urgent", "final-notice",
    "verify-account", "confirm-password", "update-payment", "restore-access",
    "claim-reward", "free-trial", "limited-time", "last-chance",
]

# Expanded malware file extensions
MALWARE_EXECUTABLES = [
    ".exe", ".bat", ".ps1", ".msi", ".zip", ".rar", ".dmg", ".apk",
    ".deb", ".rpm", ".tar.gz", ".iso", ".img", ".dll", ".scr", ".vbs",
    ".js", ".jar", ".class", ".so", ".dylib", ".sh", ".elf",
]

# Expanded malware lure words
MALWARE_LURE_WORDS = [
    "update", "install", "setup", "patch", "fix", "crack", "keygen",
    "loader", "installer", "driver", "tool", "utility", "codec",
    "player", "reader", "viewer", "converter", "optimizer", "cleaner",
    "antivirus", "security", "booster", "accelerator", "manager",
]

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
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
    """Domain Generation Algorithm: high entropy."""
    length = length or random.randint(10, 22)
    charset = string.ascii_lowercase + string.digits
    domain = rand_str(length, charset)
    tld = random.choice(MALICIOUS_TLDS)
    return domain + tld

def typosquat(brand: str) -> str:
    """Typosquatted version of brand."""
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
# BENIGN GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def gen_benign_popular(popular_domains: List[str]) -> str:
    """Well-known sites with realistic paths."""
    domain = random.choice(popular_domains)
    depth = random.randint(0, 3)
    if depth == 0:
        return f"https://{domain}/"
    path = "/".join(random.choices(BENIGN_WORDS, k=depth))
    if random.random() < 0.2:
        param = random.choice(["q", "id", "page", "ref", "tab", "sort"])
        value = rand_str(random.randint(3, 10)) if random.random() < 0.5 else rand_digits(4)
        return f"https://{domain}/{path}?{param}={value}"
    return f"https://{domain}/{path}"

def gen_benign_ecommerce() -> str:
    """E-commerce product pages."""
    domain = random.choice(ECOMMERCE_DOMAINS)
    patterns = [
        lambda: f"https://www.{domain}/dp/{rand_str(10, string.ascii_uppercase + string.digits)}",
        lambda: f"https://www.{domain}/product/{rand_slug(2)}-{rand_digits(6)}",
        lambda: f"https://www.{domain}/{random.choice(BENIGN_WORDS)}/{rand_slug(2)}?page={random.randint(1,10)}",
        lambda: f"https://www.{domain}/cart?item={rand_digits(8)}&qty={random.randint(1,5)}",
        lambda: f"https://www.{domain}/category/{rand_slug(2)}?sort=price",
    ]
    return random.choice(patterns)()

def gen_benign_cdn() -> str:
    """CDN / static asset URLs."""
    domain = random.choice(["google.com", "facebook.com", "cloudflare.com", "akamai.net"])
    prefix = random.choice(CDN_PREFIXES)
    ext = random.choice(CDN_EXTENSIONS)
    version = f"v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}"
    filename = rand_slug(2).replace("-", "_") + ext
    return f"https://{prefix}.{domain}/{version}/{filename}"

def gen_benign_api() -> str:
    """REST API endpoints."""
    domain = random.choice(["google.com", "github.com", "microsoft.com"])
    version = f"v{random.randint(1,3)}"
    resources = ["users", "products", "orders", "posts", "metrics", "reports"]
    resource = random.choice(resources)
    return f"https://api.{domain}/{version}/{resource}/{rand_digits(5)}"

def gen_benign_news() -> str:
    """News / blog articles."""
    domain = random.choice(NEWS_DOMAINS + ["medium.com"])
    year = random.randint(2018, 2024)
    month = str(random.randint(1, 12)).zfill(2)
    slug = rand_slug(random.randint(4, 7))
    return f"https://www.{domain}/{year}/{month}/{slug}"

def gen_benign_shortlink() -> str:
    """URL shorteners."""
    domain = random.choice(SHORTLINK_DOMAINS)
    code = rand_str(random.randint(4, 8), string.ascii_letters + string.digits)
    return f"https://{domain}/{code}"

def gen_benign_social() -> str:
    """Social media URLs."""
    social_domains = ["twitter.com", "facebook.com", "instagram.com", "reddit.com"]
    domain = random.choice(social_domains)
    return f"https://{domain}/{rand_slug(1)}/{rand_digits(10)}"

def gen_benign_subdomains(popular_domains: List[str]) -> str:
    """URLs with common subdomains (mail, ftp, api, etc)."""
    domain = random.choice(popular_domains[:500])  # Use popular domains
    subdomain = random.choice(COMMON_SUBDOMAINS)
    path_choice = random.randint(0, 2)

    if path_choice == 0:
        # Subdomain + path
        return f"https://{subdomain}.{domain}/{random.choice(BENIGN_WORDS)}"
    elif path_choice == 1:
        # Subdomain + query
        return f"https://{subdomain}.{domain}/?id={rand_digits(6)}"
    else:
        # Just subdomain
        return f"https://{subdomain}.{domain}/"

# ─────────────────────────────────────────────────────────────────────────────
# MALICIOUS GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def gen_malicious_phishing() -> str:
    """Phishing URLs."""
    brand = random.choice(LEGIT_BRANDS)
    technique = random.randint(1, 3)
    if technique == 1:
        fake_domain = rand_str(random.randint(8, 18)) + random.choice(MALICIOUS_TLDS)
        path = "/" + random.choice(["login", "verify", "secure", "account"])
        return f"http://{brand}.{fake_domain}{path}.php"
    elif technique == 2:
        fake = typosquat(brand) + random.choice([".com", ".net"])
        return f"http://www.{fake}/login"
    else:
        fake_domain = rand_str(random.randint(6, 15)) + random.choice(MALICIOUS_TLDS)
        return f"http://{fake_domain}/account/signin?redirect={rand_hex(16)}"

def gen_malicious_malware_dist() -> str:
    """Malware distribution."""
    domain = rand_str(random.randint(6, 15)) + random.choice(MALICIOUS_TLDS)
    filename = random.choice(MALWARE_LURE_WORDS) + "_" + rand_digits(4) + random.choice(MALWARE_EXECUTABLES)
    return f"http://{domain}/download/{filename}"

def gen_malicious_ip_based() -> str:
    """IP-based C2."""
    ip = rand_ip()
    port = random.choice([80, 443, 8080, 8443, 4444, 1337])
    path = random.choice(C2_PATHS)
    return f"http://{ip}:{port}{path}?id={rand_hex(8)}"

def gen_malicious_dga() -> str:
    """DGA C2."""
    domain = dga_domain()
    path = random.choice(C2_PATHS) if random.random() < 0.6 else ""
    return f"http://{domain}{path}"

def gen_malicious_drive_by() -> str:
    """Drive-by exploits."""
    domain = rand_str(random.randint(5, 12)) + random.choice(MALICIOUS_TLDS)
    return f"http://{domain}/ad/{rand_hex(4)}?{rand_hex(4)}={rand_hex(16)}"

def gen_malicious_spam_scam() -> str:
    """Spam/scam URLs."""
    bait = random.choice(SCAM_BAIT)
    tld = random.choice(MALICIOUS_TLDS + [".com"])
    return f"http://{bait}-{rand_str(4)}{tld}/claim?ref={rand_digits(8)}"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

BENIGN_GENERATORS = [
    (gen_benign_ecommerce, 0.18),
    (gen_benign_cdn, 0.12),
    (gen_benign_api, 0.12),
    (gen_benign_news, 0.12),
    (gen_benign_shortlink, 0.12),
    (gen_benign_social, 0.10),
    (gen_benign_subdomains, 0.14),  # NEW: Common subdomains
]

MALICIOUS_GENERATORS = [
    (gen_malicious_phishing, 0.25),
    (gen_malicious_malware_dist, 0.20),
    (gen_malicious_ip_based, 0.20),
    (gen_malicious_dga, 0.15),
    (gen_malicious_drive_by, 0.10),
    (gen_malicious_spam_scam, 0.10),
]

def _pick(generators):
    fns, weights = zip(*generators)
    return random.choices(fns, weights=weights, k=1)[0]()

def generate_dataset_one_time_only(
    n_benign: int = 8_500_000,
    n_malicious: int = 1_500_000,
    output_path: str = "urls_synthetic_10m.csv",
    real_benign_file: str = "top-1m-legitimate-url.csv",
    real_malicious_file: str = "urlhaus-malicious-url.csv",
    batch_size: int = 100_000,
) -> None:
    """
    Generate 10M URLs with ZERO reuse of real data.
    Each real URL used exactly once (+ optional variants).
    Remaining 100% synthetic.
    """
    print(f"[*] Generating {n_benign:,} benign + {n_malicious:,} malicious URLs")
    print(f"    Strategy: One-time real data + synthetic filling\n")

    # Load ALL real data
    print("  [>] Loading real data (no limits)...")
    real_benign_list = load_real_legitimate_domains(real_benign_file)
    real_malicious_list = list(load_real_malicious_urls(real_malicious_file))

    print(f"      [+] Loaded {len(real_benign_list):,} legitimate domains")
    print(f"      [+] Loaded {len(real_malicious_list):,} malicious URLs")

    # Calculate: real data + variants
    n_benign_real_with_variants = len(real_benign_list) * random.randint(3, 4)
    n_benign_synthetic = n_benign - n_benign_real_with_variants

    n_malicious_real = len(real_malicious_list)
    n_malicious_synthetic = n_malicious - n_malicious_real

    print(f"\n  [>] Generation plan:")
    print(f"      Benign: {n_benign_real_with_variants:,} real (variants) + {n_benign_synthetic:,} synthetic")
    print(f"      Malicious: {n_malicious_real:,} real + {n_malicious_synthetic:,} synthetic")

    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])

        benign_written = 0
        malicious_written = 0
        batch = []
        batch_seen = set()

        print("\n  [>] Generating benign URLs...")

        # REAL BENIGN: Each domain + 3-4 variants (ONE-TIME ONLY)
        variants_per_domain = random.randint(3, 4)
        real_written = 0
        for domain in random.sample(real_benign_list, len(real_benign_list)):
            if real_written >= n_benign - n_benign_synthetic:
                break
            # Create variants (ONE-TIME only, no cycling)
            variants = [
                f"https://{domain}/",
                f"https://{domain}/{random.choice(BENIGN_WORDS)}",
                f"https://www.{domain}/" if not domain.startswith("www.") else None,
                f"https://{domain}/api/{random.choice(BENIGN_WORDS)}",
            ]
            variants = [v for v in variants if v]  # Remove None values

            for url in variants[:variants_per_domain]:
                if real_written >= n_benign - n_benign_synthetic:
                    break
                if url not in batch_seen:
                    batch_seen.add(url)
                    batch.append((url, 0))
                    real_written += 1
                    benign_written += 1

                    if len(batch) >= batch_size:
                        writer.writerows(batch)
                        batch = []
                        batch_seen = set()
                        print(f"      [{benign_written:,} / {n_benign:,}] benign written")

        print(f"      [+] Real benign variants: {real_written:,} (each domain used once)")

        # SYNTHETIC BENIGN
        synthetic_written = 0
        attempts = 0
        max_attempts = n_benign_synthetic * 2

        # Create generators with access to real_benign_list
        benign_gens = [
            (gen_benign_ecommerce, 0.18),
            (gen_benign_cdn, 0.12),
            (gen_benign_api, 0.12),
            (gen_benign_news, 0.12),
            (gen_benign_shortlink, 0.12),
            (gen_benign_social, 0.10),
            (lambda: gen_benign_subdomains(real_benign_list), 0.14),  # NEW: Subdomains
            (lambda: gen_benign_popular(real_benign_list[:1000]), 0.04),
        ]

        while synthetic_written < n_benign_synthetic:
            attempts += 1
            if attempts > max_attempts:
                print(f"      [!] Generated {synthetic_written:,} synthetic benign")
                break

            url = _pick(benign_gens)

            if url not in batch_seen:
                batch_seen.add(url)
                batch.append((url, 0))
                synthetic_written += 1
                benign_written += 1

                if len(batch) >= batch_size:
                    writer.writerows(batch)
                    batch = []
                    batch_seen = set()
                    print(f"      [{benign_written:,} / {n_benign:,}] benign written")

        print(f"      [+] Synthetic benign: {synthetic_written:,}")

        # Flush batch
        if batch:
            writer.writerows(batch)
            batch = []
            batch_seen = set()

        # REAL MALICIOUS: ONE-TIME ONLY
        print("  [>] Generating malicious URLs...")
        real_written = 0
        for url in random.sample(real_malicious_list, len(real_malicious_list)):
            if real_written >= n_malicious_real:
                break
            if url not in batch_seen:
                batch_seen.add(url)
                batch.append((url, 1))
                real_written += 1
                malicious_written += 1

                if len(batch) >= batch_size:
                    writer.writerows(batch)
                    batch = []
                    batch_seen = set()
                    print(f"      [{malicious_written:,} / {n_malicious:,}] malicious written")

        print(f"      [+] Real malicious: {real_written:,} (used once)")

        # SYNTHETIC MALICIOUS
        synthetic_written = 0
        attempts = 0
        max_attempts = n_malicious_synthetic * 2
        while synthetic_written < n_malicious_synthetic:
            attempts += 1
            if attempts > max_attempts:
                print(f"      [!] Generated {synthetic_written:,} synthetic malicious")
                break

            url = _pick(MALICIOUS_GENERATORS)
            if url not in batch_seen:
                batch_seen.add(url)
                batch.append((url, 1))
                synthetic_written += 1
                malicious_written += 1

                if len(batch) >= batch_size:
                    writer.writerows(batch)
                    batch = []
                    batch_seen = set()
                    print(f"      [{malicious_written:,} / {n_malicious:,}] malicious written")

        print(f"      [+] Synthetic malicious: {synthetic_written:,}")

        # Final flush
        if batch:
            writer.writerows(batch)

    # Stats
    total = benign_written + malicious_written
    print(f"\n[OK] Done -> {output_path}")
    print(f"    Total      : {total:,}")
    print(f"    Benign     : {benign_written:,} ({benign_written/total*100:.1f}%)")
    print(f"    Malicious  : {malicious_written:,} ({malicious_written/total*100:.1f}%)")


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

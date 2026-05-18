"""
Crawl INTERNATIONAL benign URLs for a real-world CNN-LSTM training dataset.

Pipeline (mỗi seed):
  1. crt.sh         -> enumerate subdomain qua Certificate Transparency logs.
  2. sitemap.xml    -> đọc robots.txt + thử path mặc định; parse urlset/sitemapindex
                       (handle nested + .xml.gz).
  3. BFS crawler    -> fallback khi sitemap không có / quá ít URL.
                       Respect robots.txt, depth-limited, per-host rate limit,
                       chỉ follow link cùng eTLD+1 với seed.

Output:
  dataset/intl-benign/urls.csv : url, seed, source(sitemap|crawl|crtsh_root), depth
  dataset/intl-benign/log.txt  : per-seed summary

Tốc độ: chạy song song nhiều seed cùng lúc qua ThreadPoolExecutor
(--workers, default 8). Per-host rate-limit vẫn được tôn trọng (mỗi seed có
host riêng nên không chặn nhau).

Usage:
  pip install requests beautifulsoup4 tldextract tqdm
  python "scripts/crawl-benign-urls.py"
  python "scripts/crawl-benign-urls.py" --seeds dataset/intl-benign/seeds.txt --max-per-host 2000
  python "scripts/crawl-benign-urls.py" --workers 16              # tăng song song
  python "scripts/crawl-benign-urls.py" --no-crtsh --no-crawler   # chỉ sitemap
"""

import argparse
import csv
import gzip
import re
import sys
import threading
import time
import urllib.parse
import urllib.robotparser
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import tldextract
from bs4 import BeautifulSoup
from tqdm import tqdm

# ============================================================================
# Config
# ============================================================================
PROJECT_ROOT = Path(r"D:\! secURLity")
OUT_DIR = PROJECT_ROOT / "dataset" / "intl-benign"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SEEDS = OUT_DIR / "seeds.txt"
DEFAULT_OUTPUT = OUT_DIR / "urls.csv"
DEFAULT_LOG = OUT_DIR / "log.txt"
DEFAULT_PROCESSED_SEEDS = OUT_DIR / "processed_seeds.txt"
DEFAULT_WORKERS = 8

USER_AGENT = (
    "secURLityResearchCrawler/0.1 "
    "(+research; respects robots.txt; contact: maiphuhai123@gmail.com)"
)
REQUEST_TIMEOUT = 15
PER_HOST_DELAY = 1.0          # giây — khoảng cách tối thiểu giữa 2 request cùng host
MAX_SITEMAPS_PER_HOST = 50    # chống sitemap-bomb (nested vô hạn)
CRTSH_TIMEOUT = 30
CRTSH_RETRIES = 3
MAX_SUBDOMAINS_PER_SEED = 30  # crt.sh có thể trả vài trăm — giới hạn để khỏi crawl quá rộng

# Crawler defaults — có thể override qua CLI
DEFAULT_MAX_PER_HOST = 1500
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_PAGES_TO_FETCH = 300  # mỗi host crawl tối đa N HTML pages (mỗi page có thể chứa nhiều link)

# Regex chỉ <loc> trong sitemap — robust hơn BS4 cho XML (không cần lxml)
LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)

_tld = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True, cache_dir=False)

session = requests.Session()
# Pool đủ lớn cho 8-16 worker dùng chung session (urllib3 default = 10 / host
# → tăng lên 32 để tránh "Connection pool is full" warning).
_adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
session.mount("https://", _adapter)
session.mount("http://", _adapter)
session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})

_last_request_time: dict[str, float] = {}
_rate_lock = threading.Lock()        # bảo vệ _last_request_time
_csv_lock = threading.Lock()         # bảo vệ writer + out_f.flush
_seen_lock = threading.Lock()        # bảo vệ global_seen
_processed_lock = threading.Lock()   # bảo vệ processed_f + processed_seeds set


# ============================================================================
# URL utils
# ============================================================================
def etld_plus_1(host: str) -> str:
    ext = _tld(host)
    if not ext.domain:
        return host.lower()
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return ext.domain.lower()


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Resolve relative against base, drop fragment, lowercase scheme/host.
    Trả về None nếu URL không phải http/https."""
    if not url:
        return None
    try:
        if base:
            url = urllib.parse.urljoin(base, url)
        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            return None
        if not parsed.hostname:
            return None
        host = parsed.hostname.lower()
        netloc = host + (f":{parsed.port}" if parsed.port else "")
        rebuilt = urllib.parse.urlunparse((
            parsed.scheme.lower(),
            netloc,
            parsed.path or "/",
            parsed.params,
            parsed.query,
            "",  # drop fragment
        ))
        return rebuilt
    except Exception:
        return None


# ============================================================================
# Politeness layer
# ============================================================================
def polite_get(url: str) -> requests.Response | None:
    """GET với per-host rate limit. Trả None nếu request fail.

    Thread-safe: _last_request_time được bảo vệ bằng _rate_lock. Khi nhiều
    worker chạm cùng 1 host, chúng sẽ tự xếp hàng (mỗi worker chờ đúng phần
    còn lại của PER_HOST_DELAY rồi chiếm slot tiếp theo).
    """
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return None
    with _rate_lock:
        last = _last_request_time.get(host, 0.0)
        wait = PER_HOST_DELAY - (time.time() - last)
        # Pre-claim slot: ghi nhận thời điểm dự kiến gửi request để worker
        # khác cùng host phải chờ tiếp PER_HOST_DELAY sau đó (không kẹt lock
        # trong khi sleep).
        scheduled = max(time.time(), last + PER_HOST_DELAY)
        _last_request_time[host] = scheduled
    if wait > 0:
        time.sleep(wait)
    try:
        return session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return None


class RobotsCache:
    """Cache RobotFileParser per (scheme, netloc). Thread-safe."""

    def __init__(self):
        self.cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._cache_lock = threading.Lock()
        # Per-key lock để tránh hai worker cùng tải robots.txt của cùng 1 host
        self._key_locks: dict[str, threading.Lock] = {}

    def _key(self, url: str) -> str:
        p = urllib.parse.urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._cache_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _load(self, key: str) -> urllib.robotparser.RobotFileParser | None:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{key}/robots.txt")
        try:
            resp = polite_get(f"{key}/robots.txt")
            if resp is not None and resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                return rp
        except Exception:
            pass
        return None

    def _ensure_loaded(self, key: str) -> urllib.robotparser.RobotFileParser | None:
        with self._cache_lock:
            if key in self.cache:
                return self.cache[key]
        # Load ngoài lock — chỉ 1 worker / key thực sự fetch nhờ key_lock.
        key_lock = self._get_key_lock(key)
        with key_lock:
            with self._cache_lock:
                if key in self.cache:
                    return self.cache[key]
            rp = self._load(key)
            with self._cache_lock:
                self.cache[key] = rp
            return rp

    def is_allowed(self, url: str) -> bool:
        rp = self._ensure_loaded(self._key(url))
        if rp is None:
            return True  # No robots.txt -> default allow
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True

    def sitemap_urls(self, key_url: str) -> list[str]:
        rp = self._ensure_loaded(self._key(key_url))
        if rp is None:
            return []
        try:
            sm = rp.site_maps()
            return list(sm) if sm else []
        except Exception:
            return []


# ============================================================================
# crt.sh — subdomain enumeration qua Certificate Transparency logs
# ============================================================================
def fetch_subdomains_crtsh(domain: str) -> set[str]:
    """Trả set subdomain (FQDN) cùng eTLD+1 với `domain`. KHÔNG bao gồm root."""
    url = f"https://crt.sh/?q=%25.{domain}&output=json"
    for attempt in range(CRTSH_RETRIES):
        try:
            resp = session.get(url, timeout=CRTSH_TIMEOUT)
            if resp.status_code == 200 and resp.text.strip():
                data = resp.json()
                break
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    else:
        return set()

    root = etld_plus_1(domain)
    subs: set[str] = set()
    for entry in data:
        name = (entry.get("name_value") or "").strip().lower()
        for line in name.split("\n"):
            line = line.strip().lstrip("*.")
            if not line or line == root:
                continue
            if etld_plus_1(line) != root:
                continue
            if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for c in line):
                continue
            subs.add(line)
    return subs


# ============================================================================
# Sitemap discovery + parse
# ============================================================================
def _decode_sitemap(resp: requests.Response, url: str) -> str | None:
    content = resp.content
    is_gz = (
        url.lower().endswith(".gz")
        or resp.headers.get("Content-Type", "").lower().endswith("gzip")
        or (len(content) >= 2 and content[:2] == b"\x1f\x8b")
    )
    if is_gz:
        try:
            content = gzip.decompress(content)
        except Exception:
            return None
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return None


def fetch_sitemap_urls(host: str, robots: RobotsCache) -> tuple[set[str], int]:
    """BFS các sitemap (xử lý sitemapindex lồng). Trả (URLs, số sitemap đã visit)."""
    urls: set[str] = set()
    visited: set[str] = set()
    queue: deque[str] = deque()

    # Từ robots.txt
    for scheme in ("https", "http"):
        queue.extend(robots.sitemap_urls(f"{scheme}://{host}/"))
        break  # robots.txt cache key dùng cả scheme — chỉ cần 1 lần

    # Default paths
    for default in ("sitemap.xml", "sitemap_index.xml", "sitemap-index.xml", "sitemap1.xml"):
        queue.append(f"https://{host}/{default}")

    while queue and len(visited) < MAX_SITEMAPS_PER_HOST:
        sm = queue.popleft()
        sm_norm = normalize_url(sm)
        if not sm_norm or sm_norm in visited:
            continue
        visited.add(sm_norm)

        resp = polite_get(sm_norm)
        if resp is None or resp.status_code != 200:
            continue
        text = _decode_sitemap(resp, sm_norm)
        if not text:
            continue

        # Heuristic: nếu có thẻ <sitemapindex> thì tất cả <loc> là sitemap con
        is_index = "<sitemapindex" in text.lower()
        for loc in LOC_RE.findall(text):
            loc_norm = normalize_url(loc.strip())
            if not loc_norm:
                continue
            if is_index or loc_norm.lower().endswith((".xml", ".xml.gz")):
                queue.append(loc_norm)
            else:
                urls.add(loc_norm)

    return urls, len(visited)


# ============================================================================
# BFS crawler (fallback)
# ============================================================================
def crawl_bfs(
    start_url: str,
    robots: RobotsCache,
    max_depth: int,
    max_urls: int,
    max_pages: int,
) -> set[str]:
    """Crawl HTML từ start_url, follow link cùng eTLD+1. Trả set URL đã thấy."""
    seed_etld = etld_plus_1(urllib.parse.urlparse(start_url).hostname or "")
    seen: set[str] = set()
    pages_fetched = 0
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])

    while queue and len(seen) < max_urls and pages_fetched < max_pages:
        url, depth = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        if depth >= max_depth:
            continue
        if not robots.is_allowed(url):
            continue

        resp = polite_get(url)
        pages_fetched += 1
        if resp is None or resp.status_code != 200:
            continue
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype.lower():
            continue

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            continue

        for tag in soup.find_all("a", href=True):
            child = normalize_url(tag["href"], base=url)
            if not child:
                continue
            child_host = urllib.parse.urlparse(child).hostname or ""
            if etld_plus_1(child_host) != seed_etld:
                continue
            if child not in seen:
                queue.append((child, depth + 1))
                seen.add(child)  # add sớm để dedup
                if len(seen) >= max_urls:
                    break

    return seen


# ============================================================================
# Per-seed pipeline
# ============================================================================
def process_seed(
    seed_url: str,
    writer: "csv.writer",
    out_f,
    global_seen: set[str],
    robots: RobotsCache,
    args,
) -> dict:
    """Trả dict summary cho log. Thread-safe: ghi CSV + cập nhật global_seen
    qua lock; chỉ phần I/O mạng (sitemap fetch, crawler) chạy không lock."""
    parsed = urllib.parse.urlparse(seed_url if "://" in seed_url else f"https://{seed_url}")
    root_host = (parsed.hostname or "").lower()
    if not root_host:
        return {"seed": seed_url, "error": "invalid host"}

    summary = {
        "seed": seed_url,
        "root_host": root_host,
        "subdomains_found": 0,
        "hosts_processed": 0,
        "sitemap_urls": 0,
        "crawl_urls": 0,
        "new_urls_written": 0,
    }

    def _commit(urls_iter, source: str, depth: int) -> int:
        """Atomically dedupe + ghi batch URL. Trả số dòng mới."""
        rows: list[list] = []
        with _seen_lock:
            for u in urls_iter:
                if u not in global_seen:
                    global_seen.add(u)
                    rows.append([u, seed_url, source, depth])
        if not rows:
            return 0
        with _csv_lock:
            writer.writerows(rows)
            out_f.flush()
        return len(rows)

    # 1. crt.sh
    subdomains: set[str] = set()
    if args.crtsh:
        subdomains = fetch_subdomains_crtsh(root_host)
        # Giới hạn — crt.sh có thể trả hàng trăm
        subdomains = set(list(sorted(subdomains))[:MAX_SUBDOMAINS_PER_SEED])
        summary["subdomains_found"] = len(subdomains)

    hosts_to_process = [root_host] + sorted(subdomains - {root_host})

    # Per host: sitemap -> fallback crawler
    for host in hosts_to_process:
        summary["hosts_processed"] += 1

        # ---- Sitemap ----
        sitemap_urls: set[str] = set()
        try:
            sitemap_urls, _ = fetch_sitemap_urls(host, robots)
        except Exception:
            pass
        summary["sitemap_urls"] += len(sitemap_urls)
        summary["new_urls_written"] += _commit(sitemap_urls, "sitemap", 0)

        # ---- Crawler fallback ----
        if args.crawler and len(sitemap_urls) < args.crawl_threshold:
            start = f"https://{host}/"
            try:
                crawled = crawl_bfs(
                    start,
                    robots,
                    max_depth=args.max_depth,
                    max_urls=args.max_per_host,
                    max_pages=args.max_pages,
                )
            except Exception:
                crawled = set()
            new_crawl = crawled - sitemap_urls
            summary["crawl_urls"] += len(new_crawl)
            summary["new_urls_written"] += _commit(new_crawl, "crawl", -1)

    return summary


# ============================================================================
# Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Crawl international benign URLs for CNN-LSTM training.")
    p.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS,
                   help=f"File list seed (default: {DEFAULT_SEEDS})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output CSV (default: {DEFAULT_OUTPUT})")
    p.add_argument("--log", type=Path, default=DEFAULT_LOG,
                   help=f"Log file (default: {DEFAULT_LOG})")
    p.add_argument("--no-crtsh", dest="crtsh", action="store_false",
                   help="Bỏ qua subdomain enumeration qua crt.sh")
    p.add_argument("--no-crawler", dest="crawler", action="store_false",
                   help="Bỏ qua BFS crawler fallback (chỉ sitemap)")
    p.add_argument("--max-per-host", type=int, default=DEFAULT_MAX_PER_HOST,
                   help=f"Số URL tối đa thu được mỗi host (crawler). Default {DEFAULT_MAX_PER_HOST}")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help=f"Depth crawler BFS. Default {DEFAULT_MAX_DEPTH}")
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES_TO_FETCH,
                   help=f"Số HTML page tối đa fetch mỗi host. Default {DEFAULT_MAX_PAGES_TO_FETCH}")
    p.add_argument("--crawl-threshold", type=int, default=100,
                   help="Nếu sitemap < N URLs thì mới chạy crawler fallback. Default 100")
    p.add_argument("--append", action="store_true",
                   help="Append vào output CSV thay vì ghi đè")
    p.add_argument("--resume", action="store_true",
                   help="Skip seed đã có trong processed_seeds.txt + load URL đã có "
                        "trong output CSV vào global_seen để không ghi trùng. "
                        "Tự bootstrap processed_seeds.txt từ cột `seed` của CSV cũ "
                        "nếu file chưa tồn tại. Implies --append.")
    p.add_argument("--processed-seeds", type=Path, default=DEFAULT_PROCESSED_SEEDS,
                   help=f"File track seed đã xử lý (default: {DEFAULT_PROCESSED_SEEDS})")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Số seed chạy song song. Default {DEFAULT_WORKERS}. "
                        f"Per-host rate limit vẫn được giữ.")
    return p.parse_args()


def load_seeds(path: Path) -> list[str]:
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seeds.append(line)
    return seeds


def load_processed_seeds(processed_path: Path, csv_path: Path) -> set[str]:
    """Trả set seed đã xử lý.

    Nếu `processed_path` chưa tồn tại nhưng CSV cũ có (vd: chạy round 1 trước
    khi tính năng `--resume` được thêm), bootstrap từ cột `seed` của CSV và ghi
    luôn `processed_path` để lần sau không phải scan lại.
    """
    if processed_path.exists():
        return set(
            line.strip()
            for line in processed_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not csv_path.exists():
        return set()
    print(f"  Không có {processed_path.name} — bootstrap từ cột 'seed' của {csv_path.name}...")
    seeds: set[str] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return set()
        try:
            seed_idx = header.index("seed")
        except ValueError:
            return set()
        for row in reader:
            if len(row) > seed_idx and row[seed_idx]:
                seeds.add(row[seed_idx])
    # Persist để lần sau khỏi quét lại
    processed_path.write_text("\n".join(sorted(seeds)) + "\n", encoding="utf-8")
    print(f"  Bootstrap xong: {len(seeds)} seed đã processed -> {processed_path}")
    return seeds


def load_existing_urls(csv_path: Path) -> set[str]:
    """Load cột `url` của CSV cũ vào set để dedupe khi append."""
    if not csv_path.exists():
        return set()
    seen: set[str] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return set()
        try:
            url_idx = header.index("url")
        except ValueError:
            url_idx = 0
        for row in reader:
            if len(row) > url_idx and row[url_idx]:
                seen.add(row[url_idx])
    return seen


def main():
    args = parse_args()
    if args.resume:
        args.append = True   # --resume implies --append

    seeds = load_seeds(args.seeds)
    if not seeds:
        sys.exit(f"No seeds found in {args.seeds}")

    print("=" * 78)
    print("INTERNATIONAL BENIGN URL CRAWLER")
    print("=" * 78)
    print(f"Seeds          : {args.seeds}  ({len(seeds)} seeds)")
    print(f"Output         : {args.output}  (mode={'append' if args.append else 'overwrite'})")
    print(f"Resume         : {'ON' if args.resume else 'OFF'}")
    print(f"crt.sh         : {'ON' if args.crtsh else 'OFF'}")
    print(f"Crawler        : {'ON' if args.crawler else 'OFF'}  "
          f"(threshold={args.crawl_threshold}, depth={args.max_depth}, "
          f"max/host={args.max_per_host}, max_pages={args.max_pages})")
    print(f"Per-host delay : {PER_HOST_DELAY}s")
    print(f"Workers        : {args.workers}  (seed chạy song song)")
    print("=" * 78)

    # ---- Resume state ----
    processed_seeds: set[str] = set()
    global_seen: set[str] = set()
    if args.resume:
        print("\n[resume] Loading state...")
        t0 = time.time()
        processed_seeds = load_processed_seeds(args.processed_seeds, args.output)
        print(f"  processed_seeds : {len(processed_seeds):,} seed sẽ bị skip")
        global_seen = load_existing_urls(args.output)
        print(f"  global_seen     : {len(global_seen):,} URL đã có trong CSV (dedupe)")
        print(f"  loaded in {time.time() - t0:.1f}s\n")

    robots = RobotsCache()

    mode = "a" if args.append else "w"
    output_existed = args.output.exists() and args.output.stat().st_size > 0
    with open(args.output, mode, encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        # Chỉ ghi header khi đang tạo file mới
        if mode == "w" or not output_existed:
            writer.writerow(["url", "seed", "source", "depth"])

        # File state để append seed đã xong (mở persistent ở mode append)
        processed_f = open(args.processed_seeds, "a", encoding="utf-8")

        # Filter seed cần làm
        pending_seeds = [s for s in seeds if s not in processed_seeds]
        skipped = len(seeds) - len(pending_seeds)
        if skipped:
            print(f"[resume] Skip {skipped} seed đã xử lý.")

        summaries: list[dict] = []
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                future_to_seed = {
                    ex.submit(process_seed, seed, writer, out_f, global_seen, robots, args): seed
                    for seed in pending_seeds
                }
                with tqdm(total=len(pending_seeds), desc="seeds", unit="seed") as pbar:
                    for fut in as_completed(future_to_seed):
                        seed = future_to_seed[fut]
                        try:
                            summary = fut.result()
                        except KeyboardInterrupt:
                            print("\nInterrupted — cancelling remaining seeds...")
                            for f in future_to_seed:
                                f.cancel()
                            break
                        except Exception as e:
                            summary = {"seed": seed, "error": str(e)}
                        summaries.append(summary)
                        tqdm.write(
                            f"  {seed}  subs={summary.get('subdomains_found', 0)}  "
                            f"sitemap={summary.get('sitemap_urls', 0)}  "
                            f"crawl={summary.get('crawl_urls', 0)}  "
                            f"new={summary.get('new_urls_written', 0)}"
                        )
                        # Đánh dấu seed đã xong (chỉ khi không có error)
                        if "error" not in summary:
                            with _processed_lock:
                                processed_f.write(seed + "\n")
                                processed_f.flush()
                                processed_seeds.add(seed)
                        pbar.update(1)
        finally:
            processed_f.close()

    # Log file
    with open(args.log, "w", encoding="utf-8") as f:
        f.write(f"Total seeds in file: {len(seeds)}\n")
        f.write(f"Seeds processed this run: {len(summaries)}\n")
        f.write(f"Total unique URLs in memory: {len(global_seen)}\n\n")
        for s in summaries:
            f.write(f"{s}\n")

    print("\n" + "=" * 78)
    print(f"DONE.")
    print(f"  Seeds processed this run : {len(summaries)}")
    print(f"  Seeds total (incl. skipped): {len(processed_seeds)}")
    print(f"  Unique URLs (in memory)   : {len(global_seen):,}")
    print(f"  Output : {args.output}")
    print(f"  State  : {args.processed_seeds}")
    print(f"  Log    : {args.log}")
    print("=" * 78)


if __name__ == "__main__":
    main()

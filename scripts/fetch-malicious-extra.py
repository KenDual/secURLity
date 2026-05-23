"""
Fetch malicious URLs — EXTRA sources (mở rộng `fetch-malicious-url.py`).

Lấy URL THẬT (không synthetic) từ các nguồn chưa khai thác, **chạy song song**
(mỗi source = 1 worker thread). Gộp toàn bộ kết quả vào `urls-all.csv` của
script cũ — KHÔNG tách file VN-target riêng nữa (cột `is_vn_target` vẫn còn
trong CSV để downstream lọc).

**Lưu ý sau khi test thực tế**: Tier A free tier hầu như vô dụng cho bulk pull
(URLScan 403 phần lớn query, Pulsedive 429 vô tận, OTX 502). Tier B (no-auth
GitHub mirrors) đóng góp ~99% yield. Default đã skip pulsedive vì nó hay treo.

Tier A — API auth free (đọc key từ env var, skip nếu thiếu):
    OTX_API_KEY       AlienVault OTX
    URLSCAN_API_KEY   URLScan.io Search
    PULSEDIVE_API_KEY Pulsedive (**default OFF**, dùng --include pulsedive để bật)

Tier B — No-auth mirrors:
    phishdb_domains, blocklistproject, spam404, viriback, botvrij, bigbl_hacked

Output:
    dataset/vn-malicious/urls-all.csv               (APPEND + dedupe, ghi incremental)
    dataset/vn-malicious/sources-extra-summary.txt  (log)

Tính năng quan trọng:
  - **Incremental write**: mỗi source xong là flush ngay xuống file. Ctrl+C
    chỉ mất source đang chạy, các source kia đã an toàn trên đĩa.
  - **Per-source timeout** (--source-timeout, default 1800s = 30 phút): worker
    tự kiểm tra deadline mỗi vòng pagination, vượt thì cắt và return URLs đã có.
  - **Max retry 429/5xx** (3 lần) để tránh loop vô hạn.

Usage:
    $env:OTX_API_KEY     = "..."
    $env:URLSCAN_API_KEY = "..."
    python "scripts/fetch-malicious-extra.py"
    python "scripts/fetch-malicious-extra.py" --workers 6
    python "scripts/fetch-malicious-extra.py" --skip otx urlscan          # chỉ Tier B
    python "scripts/fetch-malicious-extra.py" --include pulsedive         # bật Pulsedive
    python "scripts/fetch-malicious-extra.py" --separate                  # ghi file riêng
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(r"D:\! secURLity")
OUT_DIR      = PROJECT_ROOT / "dataset" / "vn-malicious"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL      = OUT_DIR / "urls-all.csv"
OUT_SEPARATE = OUT_DIR / "urls-extra-all.csv"
OUT_SUMMARY  = OUT_DIR / "sources-extra-summary.txt"

MAX_RETRY    = 3       # số lần retry tối đa khi 429/5xx trước khi bỏ source

# ============================================================================
# VN-target detection
# ============================================================================
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


def is_vn_target(url: str) -> bool:
    if not url:
        return False
    if VN_TLD_RE.search(url):
        return True
    if VN_BRAND_RE.search(url):
        return True
    try:
        host = urlparse(url).hostname or ""
        if host.endswith(".vn"):
            return True
    except Exception:
        pass
    return False


# ============================================================================
# HTTP helpers
# ============================================================================
HEADERS_BASE = {
    "User-Agent": "secURLity-research/1.0 (+research; contact: maiphuhai123@gmail.com)",
    "Accept": "*/*",
}

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_session.headers.update(HEADERS_BASE)


def _norm_url(u: str) -> str | None:
    if not u:
        return None
    u = u.strip().strip('"').strip("'")
    if not u:
        return None
    if u.lower().startswith(("http://", "https://")):
        return u
    return None


def _wrap_domain(d: str) -> str | None:
    if not d:
        return None
    s = d.strip()
    if not s or s.startswith(("#", "!", ";")):
        return None
    parts = s.split()
    dom = parts[-1].strip().lower()
    if "." not in dom or "/" in dom or ":" in dom or "*" in dom:
        return None
    if dom in ("localhost", "broadcasthost", "ip6-localhost", "ip6-loopback"):
        return None
    return f"https://{dom}/"


def _download(url: str, timeout: int = 600) -> bytes | None:
    try:
        r = _session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        tqdm.write(f"    [download err] {url}: {type(e).__name__}: {e}")
        return None


def _deadline_exceeded(deadline: float | None) -> bool:
    return deadline is not None and time.time() > deadline


# ============================================================================
# Tier A — API auth sources
# ============================================================================
def fetch_otx(api_key: str, max_pages: int = 500, deadline: float | None = None) -> list[str]:
    base = "https://otx.alienvault.com/api/v1/pulses/subscribed"
    headers = {"X-OTX-API-KEY": api_key, **HEADERS_BASE}
    urls: list[str] = []
    seen: set[str] = set()
    page = 1
    retries = 0
    while page <= max_pages:
        if _deadline_exceeded(deadline):
            tqdm.write(f"    [otx] deadline exceeded; returning {len(urls)} URLs")
            return urls
        try:
            r = _session.get(base, params={"limit": 50, "page": page},
                             headers=headers, timeout=60)
        except requests.RequestException as e:
            tqdm.write(f"    [otx] network err page={page}: {e}")
            break
        if r.status_code == 429 or r.status_code >= 500:
            retries += 1
            if retries > MAX_RETRY:
                tqdm.write(f"    [otx] gave up after {MAX_RETRY} retries (last={r.status_code})")
                break
            time.sleep(min(30, 5 * retries))
            continue
        if r.status_code != 200:
            tqdm.write(f"    [otx] http {r.status_code} page={page}; stop")
            break
        retries = 0
        try:
            data = r.json()
        except Exception:
            break
        results = data.get("results") or []
        if not results:
            break
        for pulse in results:
            for ind in pulse.get("indicators") or []:
                if ind.get("type") not in ("URL", "URI"):
                    continue
                u = _norm_url(ind.get("indicator", ""))
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
        if not data.get("next"):
            break
        page += 1
    return urls


def fetch_urlscan(api_key: str, max_pages: int = 950, deadline: float | None = None) -> list[str]:
    base = "https://urlscan.io/api/v1/search/"
    headers = {"API-Key": api_key, **HEADERS_BASE}
    queries = [
        "verdicts.malicious:true",
        "verdicts.urlscan.malicious:true",
        "verdicts.engines.malicious:true",
        "task.tags:phishing",
        "verdicts.community.malicious:true",
    ]
    urls: list[str] = []
    seen: set[str] = set()
    pages_used = 0

    for q in queries:
        if pages_used >= max_pages or _deadline_exceeded(deadline):
            break
        search_after: str | None = None
        retries = 0
        while pages_used < max_pages:
            if _deadline_exceeded(deadline):
                tqdm.write(f"    [urlscan] deadline exceeded; returning {len(urls)} URLs")
                return urls
            params = {"q": q, "size": 100}
            if search_after:
                params["search_after"] = search_after
            try:
                r = _session.get(base, params=params, headers=headers, timeout=30)
            except requests.RequestException as e:
                tqdm.write(f"    [urlscan] network err: {e}")
                break
            if r.status_code == 429 or r.status_code >= 500:
                retries += 1
                if retries > MAX_RETRY:
                    tqdm.write(f"    [urlscan] {MAX_RETRY} retries trên q={q}; next")
                    break
                time.sleep(min(60, 10 * retries))
                continue
            if r.status_code != 200:
                tqdm.write(f"    [urlscan] http {r.status_code} on q={q}; next")
                break
            retries = 0
            pages_used += 1
            try:
                data = r.json()
            except Exception:
                break
            results = data.get("results") or []
            if not results:
                break
            for hit in results:
                task = hit.get("task") or {}
                page = hit.get("page") or {}
                u = _norm_url(task.get("url") or page.get("url") or "")
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
            if not data.get("has_more"):
                break
            last_sort = results[-1].get("sort")
            if not last_sort:
                break
            search_after = ",".join(str(s) for s in last_sort)
            time.sleep(0.1)
    return urls


def fetch_pulsedive(api_key: str, max_per_query: int = 20000, deadline: float | None = None) -> list[str]:
    base = "https://pulsedive.com/api/explore.php"
    queries = [
        "ioc=url&risk=critical",
        "ioc=url&risk=high",
        "ioc=url&risk=medium&threat=phishing",
        "ioc=url&risk=medium&threat=malware",
    ]
    urls: list[str] = []
    seen: set[str] = set()

    for q in queries:
        if _deadline_exceeded(deadline):
            break
        offset = 0
        retries = 0
        while offset < max_per_query:
            if _deadline_exceeded(deadline):
                tqdm.write(f"    [pulsedive] deadline; return {len(urls)} URLs")
                return urls
            params = {"q": q, "limit": 100, "offset": offset,
                      "pretty": 1, "key": api_key}
            try:
                r = _session.get(base, params=params, timeout=30)
            except requests.RequestException as e:
                tqdm.write(f"    [pulsedive] network err: {e}")
                break
            if r.status_code == 429 or r.status_code >= 500:
                retries += 1
                if retries > MAX_RETRY:
                    tqdm.write(f"    [pulsedive] {MAX_RETRY} retries trên q={q}; next")
                    break
                time.sleep(min(60, 10 * retries))
                continue
            if r.status_code == 402:
                tqdm.write(f"    [pulsedive] 402 paid/quota; next q")
                break
            if r.status_code != 200:
                tqdm.write(f"    [pulsedive] http {r.status_code} on q={q}; next")
                break
            retries = 0
            try:
                data = r.json()
            except Exception:
                break
            results = data.get("results") or []
            if not results:
                break
            for ind in results:
                u = _norm_url(ind.get("indicator", ""))
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
            if len(results) < 100:
                break
            offset += 100
            time.sleep(2.0)
    return urls


# ============================================================================
# Tier B — No-auth mirrors
# ============================================================================
def fetch_phishdb_domains() -> list[str]:
    sources = [
        "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt",
        "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-INACTIVE.txt",
        "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-NEW-today.txt",
    ]
    urls: list[str] = []
    for src in sources:
        raw = _download(src)
        if not raw:
            continue
        for line in raw.decode("utf-8", errors="replace").splitlines():
            u = _wrap_domain(line)
            if u:
                urls.append(u)
    return urls


def fetch_blocklistproject() -> list[str]:
    sources = [
        "https://blocklistproject.github.io/Lists/phishing.txt",
        "https://blocklistproject.github.io/Lists/scam.txt",
        "https://blocklistproject.github.io/Lists/malware.txt",
        "https://blocklistproject.github.io/Lists/ransomware.txt",
        "https://blocklistproject.github.io/Lists/fraud.txt",
    ]
    urls: list[str] = []
    for src in sources:
        raw = _download(src)
        if not raw:
            continue
        for line in raw.decode("utf-8", errors="replace").splitlines():
            u = _wrap_domain(line)
            if u:
                urls.append(u)
    return urls


def fetch_spam404() -> list[str]:
    raw = _download("https://raw.githubusercontent.com/Spam404/lists/master/main-blacklist.txt")
    if not raw:
        return []
    out: list[str] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        u = _wrap_domain(line)
        if u:
            out.append(u)
    return out


def fetch_viriback() -> list[str]:
    raw = _download("https://tracker.viriback.com/dump.php")
    if not raw:
        return []
    text = raw.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header:
        return []
    url_idx = None
    for i, h in enumerate(header):
        if h.strip().lower() == "url":
            url_idx = i
            break
    if url_idx is None:
        url_idx = 1
    out: list[str] = []
    for row in reader:
        if len(row) > url_idx:
            u = _norm_url(row[url_idx])
            if u:
                out.append(u)
    return out


def fetch_botvrij() -> list[str]:
    raw = _download("https://www.botvrij.eu/data/ioclist.url.raw")
    if not raw:
        return []
    out: list[str] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        u = _norm_url(line.strip())
        if u:
            out.append(u)
    return out


def fetch_bigbl_hacked() -> list[str]:
    raw = _download(
        "https://raw.githubusercontent.com/mitchellkrogza/"
        "The-Big-List-of-Hacked-Malware-Web-Sites/master/hacked-domains.list"
    )
    if not raw:
        return []
    out: list[str] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        u = _wrap_domain(line)
        if u:
            out.append(u)
    return out


# ============================================================================
# Dedupe
# ============================================================================
def load_existing_urls(path: Path) -> set[str]:
    if not path.exists():
        print(f"  [dedupe] {path.name} chưa tồn tại — fresh write")
        return set()
    print(f"  [dedupe] loading {path.name}...")
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
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
    print(f"  [dedupe] {len(seen):,} URL đã có trong {path.name}")
    return seen


# ============================================================================
# Source dispatcher
# ============================================================================
DEFAULT_OFF = {"pulsedive"}  # source mặc định bị skip, cần --include để bật


def build_sources(args, deadline: float | None) -> list[tuple[str, "callable"]]:
    sources: list[tuple[str, callable]] = []
    include = set(args.include or [])

    otx_key       = os.environ.get("OTX_API_KEY", "").strip()
    urlscan_key   = os.environ.get("URLSCAN_API_KEY", "").strip()
    pulsedive_key = os.environ.get("PULSEDIVE_API_KEY", "").strip()

    def _maybe_add(name: str, fn_factory, env_var_msg: str | None = None, key_ok: bool = True):
        if name in args.skip:
            print(f"  [skip] {name} (--skip)")
            return
        if name in DEFAULT_OFF and name not in include:
            print(f"  [skip] {name} (default OFF — dùng --include {name} để bật)")
            return
        if not key_ok:
            print(f"  [skip] {name} ({env_var_msg})")
            return
        sources.append((name, fn_factory()))

    # Tier A
    _maybe_add("otx",
               lambda: lambda: fetch_otx(otx_key, args.otx_max_pages, deadline),
               "env OTX_API_KEY chưa set", key_ok=bool(otx_key))
    _maybe_add("urlscan",
               lambda: lambda: fetch_urlscan(urlscan_key, args.urlscan_max_pages, deadline),
               "env URLSCAN_API_KEY chưa set", key_ok=bool(urlscan_key))
    _maybe_add("pulsedive",
               lambda: lambda: fetch_pulsedive(pulsedive_key, args.pulsedive_max_offset, deadline),
               "env PULSEDIVE_API_KEY chưa set", key_ok=bool(pulsedive_key))

    # Tier B
    tier_b: list[tuple[str, callable]] = [
        ("phishdb_domains",  fetch_phishdb_domains),
        ("blocklistproject", fetch_blocklistproject),
        ("spam404",          fetch_spam404),
        ("viriback",         fetch_viriback),
        ("botvrij",          fetch_botvrij),
        ("bigbl_hacked",     fetch_bigbl_hacked),
    ]
    for name, fn in tier_b:
        if name in args.skip:
            print(f"  [skip] {name} (--skip)")
            continue
        sources.append((name, fn))

    return sources


def _worker(name: str, fn) -> tuple[str, list[str], str | None, float]:
    t0 = time.time()
    try:
        urls = fn()
        return (name, urls, None, time.time() - t0)
    except Exception as e:
        return (name, [], f"{type(e).__name__}: {e}", time.time() - t0)


# ============================================================================
# Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch extra malicious URLs (auth + no-auth, REAL only). Parallel workers + incremental write."
    )
    p.add_argument("--skip", nargs="*", default=[],
                   help="Skip source theo tên: otx, urlscan, pulsedive, "
                        "phishdb_domains, blocklistproject, spam404, viriback, "
                        "botvrij, bigbl_hacked")
    p.add_argument("--include", nargs="*", default=[],
                   help="Bật các source mặc định OFF (hiện tại: pulsedive).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--source-timeout", type=int, default=1800,
                   help="Per-source deadline (giây). Default 1800 (30 phút). 0 = không giới hạn.")
    p.add_argument("--otx-max-pages", type=int, default=500)
    p.add_argument("--urlscan-max-pages", type=int, default=950)
    p.add_argument("--pulsedive-max-offset", type=int, default=20000)
    p.add_argument("--no-dedupe", action="store_true",
                   help="Không dedupe vs file output; ghi toàn bộ URL fetched.")
    p.add_argument("--separate", action="store_true",
                   help="Ghi ra urls-extra-all.csv riêng (không append urls-all.csv).")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = OUT_SEPARATE if args.separate else OUT_ALL
    deadline = (time.time() + args.source_timeout) if args.source_timeout > 0 else None

    print("=" * 72)
    print("Fetch malicious URLs — EXTRA sources (parallel + incremental write)")
    print("=" * 72)
    print(f"Output         : {out_path}  ({'WRITE' if args.separate else 'APPEND'})")
    print(f"Workers        : {args.workers}")
    print(f"Source timeout : {args.source_timeout}s ({args.source_timeout/60:.1f} min)" if args.source_timeout else "Source timeout : OFF")
    print(f"Skip           : {sorted(args.skip) if args.skip else '(none)'}")
    print(f"Include        : {sorted(args.include) if args.include else '(none)'}")
    print(f"Dedupe         : {'OFF (--no-dedupe)' if args.no_dedupe else f'vs {out_path.name}'}")
    print()

    existing = set() if args.no_dedupe else load_existing_urls(out_path)
    sources = build_sources(args, deadline)
    print(f"\nActive sources ({len(sources)}): {[n for n, _ in sources]}\n")

    if not sources:
        sys.exit("No active sources — kiểm tra env vars hoặc --skip/--include.")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen_this_run: set[str] = set()
    per_source: dict[str, dict] = {}
    total_new = 0
    total_vn = 0

    # ---- Open output file UPFRONT for incremental write ----
    file_existed = out_path.exists() and out_path.stat().st_size > 0
    mode = "a" if (not args.separate and file_existed) else "w"
    write_header = mode == "w" or not file_existed

    out_f = open(out_path, mode, encoding="utf-8", newline="")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(["url", "source", "fetched_at", "is_vn_target"])
        out_f.flush()
    write_lock = Lock()

    workers = min(args.workers, len(sources))
    print(f"[run] {workers} workers, mode={mode}, deadline={'+{:.0f}s'.format(args.source_timeout) if deadline else 'none'}\n")

    ex = ThreadPoolExecutor(max_workers=workers)
    interrupted = False
    try:
        future_to_name = {ex.submit(_worker, name, fn): name for name, fn in sources}
        with tqdm(total=len(future_to_name), desc="sources done", unit="src") as pbar:
            for fut in as_completed(future_to_name):
                name, urls, err, elapsed = fut.result()
                if err:
                    per_source[name] = {"raw": 0, "new": 0, "vn": 0,
                                        "status": f"error: {err}"}
                    tqdm.write(f"  [done] {name:<18} FAIL: {err}  ({elapsed:.1f}s)")
                    pbar.update(1)
                    continue

                # Dedupe + accumulate rows for THIS source only
                new_rows: list[tuple[str, str, str, int]] = []
                vn = 0
                for u in urls:
                    if not u or u in existing or u in seen_this_run:
                        continue
                    seen_this_run.add(u)
                    vn_flag = is_vn_target(u)
                    new_rows.append((u, name, fetched_at, int(vn_flag)))
                    if vn_flag:
                        vn += 1

                # Flush immediately
                if new_rows:
                    with write_lock:
                        writer.writerows(new_rows)
                        out_f.flush()
                added = len(new_rows)
                total_new += added
                total_vn += vn
                per_source[name] = {"raw": len(urls), "new": added, "vn": vn,
                                    "status": f"OK in {elapsed:.1f}s"}
                tqdm.write(
                    f"  [done] {name:<18} raw={len(urls):>8,}  "
                    f"new={added:>8,}  vn={vn:>5,}  ({elapsed:.1f}s)  → flushed"
                )
                pbar.update(1)
    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("\n[interrupt] Ctrl+C — đã ghi kết quả các source hoàn tất, force-exit để bỏ qua worker treo...")
    finally:
        with write_lock:
            try:
                out_f.flush()
                out_f.close()
            except Exception:
                pass
        # Cancel các future chưa start; running futures sẽ bị "bỏ rơi" (thread vẫn tồn tại
        # nhưng main process exit). Trên Windows ThreadPoolExecutor là daemon-like.
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            ex.shutdown(wait=False)  # Python < 3.9 fallback

    # ---- Summary ----
    lines = [
        "=" * 72,
        f"EXTRA fetch summary — {fetched_at}" + ("  [INTERRUPTED]" if interrupted else ""),
        "=" * 72,
        f"NEW URLs ghi xuống file  : {total_new:,}",
        f"  VN-targeted            : {total_vn:,}  ({100*total_vn/max(total_new,1):.2f}%)",
        f"Baseline ({out_path.name}) : {len(existing):,}",
        f"Combined now (estimate)  : {len(existing) + total_new:,}",
        f"Workers used             : {workers}",
        f"Source timeout           : {args.source_timeout}s",
        "",
        f"{'Source':<22} {'Raw':>10} {'New':>10} {'VN':>6}  Status",
        "-" * 72,
    ]
    for name, info in per_source.items():
        lines.append(
            f"{name:<22} {info['raw']:>10,} {info['new']:>10,} "
            f"{info['vn']:>6,}  {info['status']}"
        )
    # Sources chưa start hoặc treo
    completed_names = set(per_source.keys())
    for name, _ in sources:
        if name not in completed_names:
            lines.append(f"{name:<22} {'-':>10} {'-':>10} {'-':>6}  CANCELLED/HUNG")
    lines += ["", f"Output : {out_path}", ""]
    summary = "\n".join(lines)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    print("\n" + summary)

    if interrupted:
        # Force exit để main thread không chờ workers treo (Pulsedive stuck trong _session.get)
        os._exit(1)


if __name__ == "__main__":
    main()

"""
Fetch malicious / phishing URLs from ALL free, no-auth public sources.

Không lọc theo quốc gia ở bước fetch — kéo toàn bộ về.
Nhưng có cột `is_vn_target` (bool) đánh dấu URL nhắm mục tiêu Việt Nam,
dựa trên TLD .vn hoặc match brand/keyword Việt Nam phổ biến.

Output:
    dataset/vn-malicious/urls-all.csv          # toàn bộ URL từ mọi nguồn
    dataset/vn-malicious/urls-vn-target.csv    # subset chỉ VN-target
    dataset/vn-malicious/sources-summary.txt   # đếm theo nguồn

Usage:
    python "scripts/fetch-malicious-all.py"
    python "scripts/fetch-malicious-all.py" --skip urlhaus phishing_database
    python "scripts/fetch-malicious-all.py" --timeout 120

Nguồn (đều free, không cần API key):
    1.  URLhaus (abuse.ch)              — malware/C2 URLs (ZIP-CSV)
    2.  ThreatFox (abuse.ch)            — IOC URLs (CSV)
    3.  OpenPhish                       — phishing community (~500 latest)
    4.  Phishing.Database ACTIVE        — phishing URLs đang live
    5.  Phishing.Database INACTIVE      — phishing URLs historical (RẤT LỚN, 5-10M)
    6.  Phishing.Database NEW-today     — phishing mới trong ngày
    7.  Phishing.Army                   — DOMAIN -> wrap https://
    8.  tweetfeed.live                  — IOC từ Twitter (1 tháng gần)
    9.  DigitalSide.it OSINT            — daily latest URLs
    10. CertPL hole                     — Polish CERT phishing DOMAIN
    11. Cybercrime Tracker              — botnet C2 URLs
    12. Hagezi TIF                      — aggregated threat intel DOMAIN
    13. Malsilo                         — curated malicious URL feed
"""

import argparse
import csv
import gzip
import io
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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
OUT_VN       = OUT_DIR / "urls-vn-target.csv"
OUT_SUMMARY  = OUT_DIR / "sources-summary.txt"

# ============================================================================
# Sources
# ============================================================================
SOURCES = {
    "urlhaus": {
        # /downloads/csv/ trả về **ZIP** chứa csv.txt (full DB).
        "url":  "https://urlhaus.abuse.ch/downloads/csv/",
        "kind": "urlhaus_csv",
    },
    "threatfox": {
        # ThreatFox (abuse.ch) — IOC, full export (CSV trong ZIP).
        "url":  "https://threatfox.abuse.ch/export/csv/full/",
        "kind": "threatfox_csv",
    },
    "openphish": {
        "url":  "https://openphish.com/feed.txt",
        "kind": "txt_lines",
    },
    "phishing_database": {
        # mitchellkrogza/Phishing.Database — bản ACTIVE.
        "url":  "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
        "kind": "txt_lines",
    },
    "phishing_database_inactive": {
        # Historical phishing URLs (rất lớn, có thể 5-10M URLs).
        "url":  "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-INACTIVE.txt",
        "kind": "txt_lines",
    },
    "phishing_database_new_today": {
        # Phishing URLs phát hiện trong ngày — bổ sung độ tươi.
        "url":  "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-today.txt",
        "kind": "txt_lines",
    },
    "phishing_army": {
        "url":  "https://phishing.army/download/phishing_army_blocklist_extended.txt",
        "kind": "domain_list",
    },
    "tweetfeed": {
        "url":  "https://api.tweetfeed.live/v1/month/url",
        "kind": "tweetfeed_json",
    },
    "digitalside": {
        # OSINT.digitalside.it — daily latest URLs (small but fresh).
        "url":  "https://osint.digitalside.it/Threat-Intel/lists/latesturls.txt",
        "kind": "txt_lines",
    },
    "certpl": {
        # Polish CERT (CERT.PL) hole.cert.pl — DOMAIN list lớn.
        "url":  "https://hole.cert.pl/domains/v2/domains.txt",
        "kind": "domain_list",
    },
    "cybercrime_tracker": {
        # cybercrime-tracker.net — botnet C2 URLs.
        "url":  "http://cybercrime-tracker.net/all.php",
        "kind": "txt_lines",
    },
    "hagezi_tif": {
        # Hagezi Threat Intelligence Feed — aggregated DOMAIN, high quality.
        "url":  "https://raw.githubusercontent.com/hagezi/dns-blocklists/main/domains/tif.txt",
        "kind": "domain_list",
    },
    "malsilo": {
        # Malsilo — curated malicious URL feed.
        "url":  "https://malsilo.gitlab.io/feeds/dumps/url_list.txt",
        "kind": "txt_lines",
    },
}

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
    """True nếu URL có dấu hiệu nhắm mục tiêu VN (TLD .vn HOẶC brand VN trong URL)."""
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
# Fetch helpers
# ============================================================================
HEADERS = {
    "User-Agent": "secURLity-research/1.0 (+research; contact: maiphuhai123@gmail.com)",
    "Accept": "*/*",
}


def http_get(url: str, timeout: int) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
    r.raise_for_status()
    return r.content


def _maybe_unzip_first(raw: bytes) -> str:
    """Nếu raw là ZIP -> extract file đầu tiên; nếu không -> decode trực tiếp."""
    import zipfile
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                return f.read().decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _parse_urlhaus_like_csv(text: str, url_col_idx: int) -> list[str]:
    """
    URLhaus & ThreatFox dùng CSV với:
      - dòng comment bắt đầu bằng '#' (chứa dấu " không cân -> phải lọc TRƯỚC csv.reader)
      - field được quote bằng " và phân cách bằng dấu phẩy
    """
    cleaned = "\n".join(
        line for line in text.splitlines() if line and not line.startswith("#")
    )
    urls = []
    reader = csv.reader(io.StringIO(cleaned))
    for row in reader:
        if len(row) > url_col_idx:
            u = row[url_col_idx].strip().strip('"')
            if u and u.lower().startswith(("http://", "https://")):
                urls.append(u)
    return urls


def fetch_urlhaus(raw: bytes):
    """URLhaus full DB — ZIP chứa csv.txt; col 2 = url."""
    text = _maybe_unzip_first(raw)
    return _parse_urlhaus_like_csv(text, url_col_idx=2)


def fetch_threatfox(raw: bytes):
    """
    ThreatFox full export — ZIP chứa full.csv.
    Cols: first_seen_utc, ioc_id, ioc_value, ioc_type, threat_type, ...
    Chỉ lấy ioc_type='url'.
    """
    text = _maybe_unzip_first(raw)
    cleaned = "\n".join(
        line for line in text.splitlines() if line and not line.startswith("#")
    )
    urls = []
    reader = csv.reader(io.StringIO(cleaned))
    for row in reader:
        if len(row) < 4:
            continue
        ioc_type = row[3].strip().strip('"').lower()
        if ioc_type != "url":
            continue
        u = row[2].strip().strip('"')
        if u and u.lower().startswith(("http://", "https://")):
            urls.append(u)
    return urls


def fetch_txt_lines(raw: bytes):
    text = raw.decode("utf-8", errors="replace")
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def fetch_domain_list(raw: bytes):
    """Phishing.Army là DOMAIN list -> wrap https:// để có URL đầy đủ."""
    text = raw.decode("utf-8", errors="replace")
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        # Phishing.Army đôi khi có format "0.0.0.0 domain" — strip prefix nếu có
        parts = s.split()
        domain = parts[-1].strip()
        if domain and "." in domain:
            out.append(f"https://{domain}/")
    return out


def fetch_tweetfeed_json(raw: bytes):
    """tweetfeed.live trả về JSON list of objects: {value, type, ...}."""
    import json
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        v = item.get("value") or item.get("url")
        if v and isinstance(v, str):
            out.append(v.strip())
    return out


KIND_DISPATCH = {
    "urlhaus_csv":    fetch_urlhaus,
    "threatfox_csv":  fetch_threatfox,
    "txt_lines":      fetch_txt_lines,
    "domain_list":    fetch_domain_list,
    "tweetfeed_json": fetch_tweetfeed_json,
}


# ============================================================================
# Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Fetch malicious URLs from all free public sources.")
    p.add_argument("--skip", nargs="*", default=[],
                   help=f"Bỏ qua nguồn (chọn từ: {', '.join(SOURCES.keys())})")
    p.add_argument("--timeout", type=int, default=300,
                   help="HTTP timeout per source (s). Default 300s vì INACTIVE list lớn.")
    return p.parse_args()


def main():
    args = parse_args()
    skip = set(args.skip)

    print("=" * 70)
    print("Fetch malicious URLs — ALL free public sources")
    print("=" * 70)
    print(f"Output dir   : {OUT_DIR}")
    print(f"Sources      : {len(SOURCES)} ({len(SOURCES) - len(skip)} active)")
    print(f"Skip         : {sorted(skip) if skip else '(none)'}")
    print(f"Timeout      : {args.timeout}s\n")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen: set[str] = set()           # dedup global
    rows: list[tuple[str, str, str, bool]] = []  # (url, source, fetched_at, is_vn_target)
    per_source: dict[str, dict] = {}

    for name, meta in SOURCES.items():
        if name in skip:
            print(f"[skip] {name}")
            per_source[name] = {"raw": 0, "added": 0, "vn": 0, "status": "skipped"}
            continue

        url = meta["url"]
        kind = meta["kind"]
        print(f"[fetch] {name:20s} <- {url}")
        t0 = time.time()
        try:
            raw = http_get(url, timeout=args.timeout)
            urls = KIND_DISPATCH[kind](raw)
        except Exception as e:
            print(f"  FAIL ({type(e).__name__}): {e}")
            per_source[name] = {"raw": 0, "added": 0, "vn": 0,
                                "status": f"error: {type(e).__name__}: {e}"}
            continue

        n_raw = len(urls)
        added = 0
        vn = 0
        for u in urls:
            if not u or u in seen:
                continue
            seen.add(u)
            vn_flag = is_vn_target(u)
            rows.append((u, name, fetched_at, vn_flag))
            added += 1
            if vn_flag:
                vn += 1

        per_source[name] = {
            "raw": n_raw, "added": added, "vn": vn,
            "status": f"OK in {time.time() - t0:.1f}s"
        }
        print(f"  raw={n_raw:>8,}  new={added:>8,}  vn-target={vn:>6,}  ({time.time() - t0:.1f}s)\n")

    # ---- Write CSVs ----
    print("\n[write] CSVs...")
    with open(OUT_ALL, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "source", "fetched_at", "is_vn_target"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], int(r[3])])

    with open(OUT_VN, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "source", "fetched_at"])
        for r in rows:
            if r[3]:
                w.writerow([r[0], r[1], r[2]])

    # ---- Summary ----
    total = len(rows)
    total_vn = sum(1 for r in rows if r[3])
    lines = [
        "=" * 70,
        f"Summary — fetched at {fetched_at}",
        "=" * 70,
        f"Total unique URLs : {total:,}",
        f"VN-targeted       : {total_vn:,}  ({100 * total_vn / max(total, 1):.2f}%)",
        "",
        f"{'Source':<22} {'Raw':>10} {'Added':>10} {'VN':>8}  Status",
        "-" * 70,
    ]
    for name, info in per_source.items():
        lines.append(
            f"{name:<22} {info['raw']:>10,} {info['added']:>10,} {info['vn']:>8,}  {info['status']}"
        )
    lines += ["", f"Output:", f"  {OUT_ALL}", f"  {OUT_VN}", ""]

    summary = "\n".join(lines)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()

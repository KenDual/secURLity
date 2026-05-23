"""
EDA cho `dataset/dataset-4-final.csv` — phục vụ Stage 0 của model 4.

Tập trung vào các quyết định blocking trong CNN-LSTM-final.md:
  1. MAX_LEN  → cần URL length distribution P50/75/90/95/99/99.5/99.9/99.99/max
  2. Vocab    → cần char distribution CASE-PRESERVED + frequency theo label
  3. Preserve case  → cần % URL có uppercase vs all-lowercase vs mixed
  4. Component split  → length của scheme/host/path/query/fragment riêng biệt
  5. Domain pool (cho Item 8 domain-split)  → unique registered domains theo label

Khác `2. eda 3.py`:
  - KHÔNG dùng RunningStats cho URL length — dùng Counter (length histogram)
    để có EXACT percentile thay vì sample-based estimate.
  - KHÔNG lowercase URL trước khi đếm char (eda 3 mặc định lower).
  - THÊM section VOCAB COVERAGE — so sánh current vocab (51 char) vs
    proposed vocab (case-preserved + special chars) qua <UNK> rate.
  - THÊM section REGISTERED DOMAIN POOL — đếm số domain duy nhất, top
    domains theo label, distribution URLs/domain.
  - Bỏ classify_url() và phần suspicious-pattern detection — không cần cho
    việc chốt MAX_LEN/vocab. Giữ scheme/TLD/IP/VN sections cơ bản.

Usage (PowerShell):
    .\\venv\\Scripts\\Activate.ps1
    python "scripts/2. eda 4.py"
    python "scripts/2. eda 4.py" --max-rows 1000000   # nhanh, test
    python "scripts/2. eda 4.py" --output "dataset/eda-result 4.txt"
"""

import argparse
import csv
import math
import re
import string as _string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlsplit

# Tăng csv field size để chịu URL dài bất thường
csv.field_size_limit(10_000_000)

# Windows console mặc định cp1252 không print được ký tự Unicode (→, ↗, ...).
# Force UTF-8 cho cả stdout + stderr (Python 3.7+).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass


# ============================================================================
# Constants
# ============================================================================

# Vocab HIỆN TẠI (model 1/2/3) — để so sánh với vocab đề xuất
CURRENT_VOCAB_CHARS = set(_string.ascii_lowercase + _string.digits + "/:.-_?=&#@%+~")

# Vocab ĐỀ XUẤT (model 4) — case-preserved + thêm special chars phổ biến
# (danh sách bổ sung sẽ được CONFIRM lại sau khi xem CHAR DISTRIBUTION)
PROPOSED_VOCAB_CHARS = set(
    _string.ascii_letters       # 52 letters (case-preserved)
    + _string.digits            # 10 digits
    + "/:.-_?=&#@%+~"           # 13 special từ vocab cũ
    + ",;()[]{}!*$|<>'\" "     # 14 special bổ sung
)

# Common TLDs (gồm VN sub-TLDs) — dùng để fallback registered-domain extraction
# khi tldextract không có sẵn.
KNOWN_TLDS = sorted({
    ".com.vn", ".net.vn", ".edu.vn", ".gov.vn", ".org.vn", ".ac.vn",
    ".co.uk", ".co.jp", ".co.in", ".co.nz", ".com.au", ".com.br", ".com.cn",
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


# ============================================================================
# Streaming utilities
# ============================================================================

class LengthHistogram:
    """Exact length distribution — counter trên giá trị int.
    Mỗi URL/component dài tối đa vài nghìn ký tự → counter rất nhỏ (RAM ok).
    Cho phép tính EXACT percentile thay vì sample-based estimate.
    """
    __slots__ = ("counts", "total", "sum", "sum_sq", "min", "max")

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()
        self.total = 0
        self.sum = 0
        self.sum_sq = 0
        self.min: Optional[int] = None
        self.max: Optional[int] = None

    def update(self, value: int) -> None:
        self.counts[value] += 1
        self.total += 1
        self.sum += value
        self.sum_sq += value * value
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value

    @property
    def mean(self) -> float:
        return self.sum / self.total if self.total else 0.0

    @property
    def std(self) -> float:
        if self.total < 2:
            return 0.0
        var = (self.sum_sq / self.total) - self.mean ** 2
        return math.sqrt(max(var, 0.0))

    def percentile(self, p: float) -> Optional[int]:
        """Exact p-th percentile (p in [0, 100]). O(unique_lengths log unique_lengths)."""
        if self.total == 0:
            return None
        target = (p / 100.0) * self.total
        running = 0
        for length in sorted(self.counts):
            running += self.counts[length]
            if running >= target:
                return length
        return self.max


# ============================================================================
# URL parsing helpers
# ============================================================================

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


def get_registered_domain(host: str, tld: str) -> str:
    """eTLD+1 approximation. VD host='m.vnexpress.net', tld='.net' → 'vnexpress.net'.
    Không đẹp như tldextract+PSL nhưng đủ dùng cho counting (không cần exact).
    """
    if not host or host == "<none>" or tld == "<none>":
        return host or "<none>"
    if is_ip(host):
        return host  # IP coi như "domain" riêng
    # Trim TLD, lấy label cuối còn lại + TLD
    tld_clean = tld.lstrip(".")
    host_no_tld = host[: -(len(tld_clean) + 1)] if host.endswith("." + tld_clean) else host
    if not host_no_tld or host_no_tld == host:
        return host
    last_label = host_no_tld.rsplit(".", 1)[-1]
    return f"{last_label}.{tld_clean}"


def parse_url(raw_url: str):
    """Returns (url, scheme, host, path, query, fragment, port)."""
    url = raw_url.strip()
    if not url:
        return "", "", "", "", "", "", None
    try:
        if "://" not in url:
            parts = urlsplit("http://" + url)
            scheme = ""
        else:
            parts = urlsplit(url)
            scheme = parts.scheme.lower()
        host, port = split_host_port(parts.netloc.lower())
        return (url, scheme, host, parts.path or "", parts.query or "",
                parts.fragment or "", port)
    except ValueError:
        # URL malformed (vd IPv6 bracket sai) — trả về fallback
        return url, "", "", "", "", "", None


def case_class(s: str) -> str:
    """'lower' nếu toàn lowercase, 'upper' nếu toàn uppercase, 'mixed' nếu cả 2,
    'none' nếu không có chữ cái."""
    has_l = any("a" <= c <= "z" for c in s)
    has_u = any("A" <= c <= "Z" for c in s)
    if has_l and has_u:
        return "mixed"
    if has_l:
        return "lower"
    if has_u:
        return "upper"
    return "none"


# ============================================================================
# Reporting helpers
# ============================================================================

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


def print_lengths(hist: LengthHistogram, _print, label: str = "") -> None:
    if label:
        _print(f"\n  [{label}]")
    if hist.total == 0:
        _print("    (no data)")
        return
    _print(f"    count        : {hist.total:,}")
    _print(f"    min / max    : {hist.min} / {hist.max}")
    _print(f"    mean / std   : {hist.mean:.2f} / {hist.std:.2f}")
    for p in (50, 75, 90, 95, 99, 99.5, 99.9, 99.99):
        v = hist.percentile(p)
        _print(f"    P{p:<6}      : {v}")


def print_counter(counter: Counter, _print, top_n: int = 20,
                  total: Optional[int] = None, label: str = "") -> None:
    if label:
        _print(f"\n{label}")
    den = total or sum(counter.values())
    for key, val in counter.most_common(top_n):
        pct = fmt_pct(val, den)
        _print(f"  {str(key)!r:>30} : {val:>12,}  ({pct})")


# ============================================================================
# Main EDA loop
# ============================================================================

def run_eda(csv_path: str, max_rows: Optional[int], output_path: Optional[str]) -> None:
    start = time.time()
    _print, _close = make_printer(output_path)

    p = Path(csv_path)
    if not p.exists():
        _print(f"[ERROR] CSV not found: {p}")
        _close()
        sys.exit(1)

    size_mb = p.stat().st_size / 1024 ** 2
    _print(f"EDA dataset 4 (balanced) — {p}")
    _print(f"File size: {size_mb:,.1f} MB")
    if max_rows:
        _print(f"max-rows  : {max_rows:,}  (limit cho test, KHÔNG dùng cho final EDA)")

    # ---- Counters ----
    total = 0
    missing = 0
    invalid_label = 0
    label_counts = Counter()

    # URL length (tổng + per label)
    url_len = LengthHistogram()
    url_len_by_label = defaultdict(LengthHistogram)
    # Component lengths
    host_len = LengthHistogram()
    path_len = LengthHistogram()
    query_len = LengthHistogram()
    fragment_len = LengthHistogram()
    scheme_len = LengthHistogram()

    # Char distribution (case-preserved!)
    char_counts = Counter()                                # total
    char_counts_by_label = defaultdict(Counter)            # per label

    # Case stats
    case_counts_by_label = defaultdict(Counter)            # label → {lower, upper, mixed, none}

    # Vocab coverage
    unk_current_total = 0   # số char không thuộc CURRENT_VOCAB_CHARS
    unk_proposed_total = 0  # số char không thuộc PROPOSED_VOCAB_CHARS
    chars_total = 0

    # Scheme, TLD, IP
    scheme_counts = Counter()
    scheme_by_label = defaultdict(Counter)
    tld_counts = Counter()
    tld_by_label = defaultdict(Counter)
    ip_host_by_label = Counter()
    port_counts = Counter()

    # Registered domain pool (cho Item 8)
    domain_label_counts = defaultdict(Counter)   # domain → Counter({0: n, 1: m})
    # → biết mỗi domain có bao nhiêu URL benign vs malicious

    # VN targeting
    vn_tld_by_label = Counter()
    vn_brand_by_label = Counter()
    vn_either_by_label = Counter()

    # ---- Stream CSV ----
    progress_every = 500_000
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "url" not in reader.fieldnames or "label" not in reader.fieldnames:
            _print(f"[ERROR] CSV phải có columns 'url' và 'label'. "
                   f"Found: {reader.fieldnames}")
            _close()
            sys.exit(1)

        for row in reader:
            if max_rows and total >= max_rows:
                break

            total += 1
            raw_url = row.get("url", "")
            raw_label = row.get("label", "")

            if not raw_url:
                missing += 1
                continue

            try:
                label = int(raw_label)
            except (ValueError, TypeError):
                invalid_label += 1
                continue
            if label not in (0, 1):
                invalid_label += 1
                continue
            label_counts[label] += 1

            # ---- URL length ----
            n = len(raw_url)
            url_len.update(n)
            url_len_by_label[label].update(n)

            # ---- Char distribution + vocab coverage (CASE-PRESERVED) ----
            for c in raw_url:
                char_counts[c] += 1
                char_counts_by_label[label][c] += 1
                chars_total += 1
                if c not in CURRENT_VOCAB_CHARS:
                    unk_current_total += 1
                if c not in PROPOSED_VOCAB_CHARS:
                    unk_proposed_total += 1

            # ---- Case class ----
            case_counts_by_label[label][case_class(raw_url)] += 1

            # ---- Parse ----
            url, scheme, host, path, query, fragment, port = parse_url(raw_url)
            scheme_counts[scheme or "<none>"] += 1
            scheme_by_label[label][scheme or "<none>"] += 1
            scheme_len.update(len(scheme))
            host_len.update(len(host))
            path_len.update(len(path))
            query_len.update(len(query))
            fragment_len.update(len(fragment))

            if port:
                port_counts[port] += 1

            # ---- IP / TLD / Registered Domain ----
            if is_ip(host):
                ip_host_by_label[label] += 1
                tld = "<ip>"
                reg_domain = host
            else:
                tld = get_tld(host) if host else "<none>"
                reg_domain = get_registered_domain(host, tld) if host else "<none>"
            tld_counts[tld] += 1
            tld_by_label[label][tld] += 1
            domain_label_counts[reg_domain][label] += 1

            # ---- VN targeting ----
            has_vn_tld = bool(host and host.endswith(".vn"))
            has_vn_brand = bool(VN_BRAND_RE.search(raw_url))
            if has_vn_tld:
                vn_tld_by_label[label] += 1
            if has_vn_brand:
                vn_brand_by_label[label] += 1
            if has_vn_tld or has_vn_brand or VN_TLD_RE.search(raw_url):
                vn_either_by_label[label] += 1

            if total % progress_every == 0:
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0
                _print(f"  ... {total:>12,} rows  ({rate:,.0f}/s, "
                       f"elapsed {elapsed:.1f}s)")

    elapsed = time.time() - start

    # ============================================================================
    # REPORT
    # ============================================================================

    section("OVERVIEW", _print)
    _print(f"Total rows processed   : {total:,}")
    _print(f"Missing URL            : {missing:,}")
    _print(f"Invalid label          : {invalid_label:,}")
    valid_total = sum(label_counts.values())
    _print(f"Valid rows             : {valid_total:,}")
    _print(f"Label=0 (benign)       : {label_counts[0]:,}  "
           f"({fmt_pct(label_counts[0], valid_total)})")
    _print(f"Label=1 (malicious)    : {label_counts[1]:,}  "
           f"({fmt_pct(label_counts[1], valid_total)})")
    _print(f"Elapsed                : {elapsed:.1f}s "
           f"({total / max(elapsed,1):,.0f} rows/s)")

    # ----------------------------------------------------------------------------
    section("URL LENGTH DISTRIBUTION  (decision: MAX_LEN)", _print)
    _print("\n[ALL]")
    print_lengths(url_len, _print)
    _print("\n[Label=0 — benign]")
    print_lengths(url_len_by_label[0], _print)
    _print("\n[Label=1 — malicious]")
    print_lengths(url_len_by_label[1], _print)

    # Recommendation
    p99_all = url_len.percentile(99)
    p995_all = url_len.percentile(99.5)
    p999_all = url_len.percentile(99.9)
    _print("")
    _print(f">>> Suggested MAX_LEN candidates:")
    _print(f"      cover P99    ({p99_all} chars) → truncate ~1.0% URLs")
    _print(f"      cover P99.5  ({p995_all} chars) → truncate ~0.5% URLs")
    _print(f"      cover P99.9  ({p999_all} chars) → truncate ~0.1% URLs")
    _print(f"    Compute rule-of-thumb: round up to next multiple of 32 hoặc 64.")

    # ----------------------------------------------------------------------------
    section("COMPONENT LENGTHS", _print)
    for name, hist in [
        ("scheme  ", scheme_len),
        ("host    ", host_len),
        ("path    ", path_len),
        ("query   ", query_len),
        ("fragment", fragment_len),
    ]:
        _print(f"\n  [{name.strip()}]")
        print_lengths(hist, _print)

    # ----------------------------------------------------------------------------
    section("CHAR DISTRIBUTION  (CASE-PRESERVED — decision: vocab)", _print)
    _print(f"\nTotal characters processed: {chars_total:,}")
    _print(f"Unique chars seen          : {len(char_counts):,}")

    _print("\n[Top 100 chars — ALL]")
    for c, cnt in char_counts.most_common(100):
        repr_c = repr(c) if c not in (" ",) or True else c
        _print(f"  {repr_c:>6} : {cnt:>14,}  ({fmt_pct(cnt, chars_total)})")

    # Chars unique to mal vs benign — useful signal
    chars_benign = set(char_counts_by_label[0])
    chars_mal = set(char_counts_by_label[1])
    only_mal = chars_mal - chars_benign
    only_benign = chars_benign - chars_mal
    _print(f"\nChars chỉ xuất hiện trong MALICIOUS: {len(only_mal)}")
    if only_mal:
        _print(f"  {sorted(only_mal)[:50]}")
    _print(f"Chars chỉ xuất hiện trong BENIGN   : {len(only_benign)}")
    if only_benign:
        _print(f"  {sorted(only_benign)[:50]}")

    # ----------------------------------------------------------------------------
    section("CASE STATS  (decision: preserve case?)", _print)
    for label in (0, 1):
        counts = case_counts_by_label[label]
        n = sum(counts.values())
        _print(f"\n[Label={label}]  (n={n:,})")
        for k in ("lower", "upper", "mixed", "none"):
            v = counts.get(k, 0)
            _print(f"  {k:<10} : {v:>14,}  ({fmt_pct(v, n)})")

    _print("\n>>> Nếu 'mixed' chiếm tỉ lệ KHÁC BIỆT giữa label 0 và label 1,")
    _print("    preserve case sẽ giúp model phân biệt. Nếu cả 2 đều ~lowercase")
    _print("    thì preserve case tốn vocab mà không gain.")

    # ----------------------------------------------------------------------------
    section("VOCAB COVERAGE  (UNK rate)", _print)
    _print(f"\nVocab CURRENT (model 1/2/3, lowercase only): {len(CURRENT_VOCAB_CHARS)} chars")
    _print(f"  → UNK chars total : {unk_current_total:,}")
    _print(f"  → UNK rate        : {fmt_pct(unk_current_total, chars_total)}")
    _print(f"\nVocab PROPOSED (case-preserved + extended specials): "
           f"{len(PROPOSED_VOCAB_CHARS)} chars")
    _print(f"  → UNK chars total : {unk_proposed_total:,}")
    _print(f"  → UNK rate        : {fmt_pct(unk_proposed_total, chars_total)}")
    _print("\n>>> Target: UNK rate < 0.5% với vocab proposed.")
    _print("    Nếu cao hơn, xem CHAR DISTRIBUTION ở trên để thêm chars thiếu.")

    # ----------------------------------------------------------------------------
    section("SCHEME DISTRIBUTION", _print)
    print_counter(scheme_counts, _print, top_n=10, total=valid_total)
    for label in (0, 1):
        _print(f"\n[Label={label}]")
        print_counter(scheme_by_label[label], _print, top_n=10,
                      total=sum(scheme_by_label[label].values()))

    # ----------------------------------------------------------------------------
    section("TLD DISTRIBUTION", _print)
    print_counter(tld_counts, _print, top_n=50, total=valid_total,
                  label="[ALL — top 50]")
    for label in (0, 1):
        _print(f"\n[Label={label} — top 20]")
        print_counter(tld_by_label[label], _print, top_n=20,
                      total=sum(tld_by_label[label].values()))

    # ----------------------------------------------------------------------------
    section("IP-BASED HOSTS  &  PORTS", _print)
    _print(f"\nIP-based host (per label):")
    for label in (0, 1):
        n_lbl = label_counts[label]
        _print(f"  Label={label}: {ip_host_by_label[label]:,}  "
               f"({fmt_pct(ip_host_by_label[label], n_lbl)})")
    _print("\n[Top 15 ports]")
    print_counter(port_counts, _print, top_n=15)

    # ----------------------------------------------------------------------------
    section("REGISTERED DOMAIN POOL  (cho Item 8 — domain-split)", _print)
    n_domains = len(domain_label_counts)
    _print(f"\nUnique registered domains: {n_domains:,}")

    # Count domains theo label-majority
    dom_only_benign = 0
    dom_only_mal = 0
    dom_mixed = 0
    urls_in_mixed = 0
    for dom, lbl_cnt in domain_label_counts.items():
        has_b = lbl_cnt.get(0, 0) > 0
        has_m = lbl_cnt.get(1, 0) > 0
        if has_b and has_m:
            dom_mixed += 1
            urls_in_mixed += sum(lbl_cnt.values())
        elif has_b:
            dom_only_benign += 1
        elif has_m:
            dom_only_mal += 1
    _print(f"  Domain chỉ benign  : {dom_only_benign:,}  "
           f"({fmt_pct(dom_only_benign, n_domains)})")
    _print(f"  Domain chỉ mal     : {dom_only_mal:,}  "
           f"({fmt_pct(dom_only_mal, n_domains)})")
    _print(f"  Domain MIXED (cả 2): {dom_mixed:,}  "
           f"({fmt_pct(dom_mixed, n_domains)})  "
           f"→ chứa {urls_in_mixed:,} URLs")
    _print("\n>>> Domain mixed sẽ cần xử lý cẩn thận khi split theo domain.")
    _print("    Gợi ý: gán label theo majority, hoặc loại bỏ nếu < threshold.")

    # URLs/domain distribution
    urls_per_domain = Counter()
    for dom, lbl_cnt in domain_label_counts.items():
        urls_per_domain[sum(lbl_cnt.values())] += 1
    # Bucket thành các nhóm log-scale
    buckets = [
        (1, 1, "1 URL"),
        (2, 5, "2-5 URLs"),
        (6, 20, "6-20"),
        (21, 100, "21-100"),
        (101, 1_000, "101-1K"),
        (1_001, 10_000, "1K-10K"),
        (10_001, 100_000, "10K-100K"),
        (100_001, 10**12, ">100K"),
    ]
    _print("\n[URL count per domain — distribution]")
    for lo, hi, name in buckets:
        c = sum(v for k, v in urls_per_domain.items() if lo <= k <= hi)
        _print(f"  {name:>12} : {c:>10,} domains  ({fmt_pct(c, n_domains)})")

    # Top domains theo URL count
    top_domains = sorted(domain_label_counts.items(),
                         key=lambda kv: -sum(kv[1].values()))[:30]
    _print("\n[Top 30 domains theo URL count]")
    _print(f"  {'domain':>40}  {'#URLs':>10}  {'#benign':>10}  {'#mal':>10}  majority")
    for dom, lbl_cnt in top_domains:
        nb = lbl_cnt.get(0, 0)
        nm = lbl_cnt.get(1, 0)
        n = nb + nm
        maj = "benign" if nb > nm else ("mal" if nm > nb else "tie")
        _print(f"  {dom[:40]:>40}  {n:>10,}  {nb:>10,}  {nm:>10,}  {maj}")

    # ----------------------------------------------------------------------------
    section("VIETNAMESE TARGETING", _print)
    for label in (0, 1):
        n_lbl = label_counts[label]
        _print(f"\n[Label={label}]  total={n_lbl:,}")
        _print(f"  .vn TLD               : {vn_tld_by_label[label]:>12,}  "
               f"({fmt_pct(vn_tld_by_label[label], n_lbl)})")
        _print(f"  VN brand keyword      : {vn_brand_by_label[label]:>12,}  "
               f"({fmt_pct(vn_brand_by_label[label], n_lbl)})")
        _print(f"  TLD OR brand (either) : {vn_either_by_label[label]:>12,}  "
               f"({fmt_pct(vn_either_by_label[label], n_lbl)})")

    # ----------------------------------------------------------------------------
    section("SUMMARY — quyết định cho model 4", _print)
    _print("")
    _print(f"  MAX_LEN candidate    :  P99={p99_all}, P99.5={p995_all}, P99.9={p999_all}")
    _print(f"  Vocab proposed size  :  {len(PROPOSED_VOCAB_CHARS)} chars")
    _print(f"  UNK rate (proposed)  :  {fmt_pct(unk_proposed_total, chars_total)}  "
           "(target <0.5%)")
    _print(f"  Unique reg-domains   :  {n_domains:,}  "
           "→ feasible cho domain-split (Item 8)")
    _print(f"  Mixed-label domains  :  {dom_mixed:,} "
           f"({fmt_pct(dom_mixed, n_domains)})")
    _print("")
    _print(f"  Total elapsed        :  {elapsed:.1f}s "
           f"({total / max(elapsed,1):,.0f} rows/s)")

    _close()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=r"D:\! secURLity\dataset\dataset-4-final.csv",
                    help="CSV dataset path")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Limit rows (cho test nhanh). Bỏ option = full dataset.")
    ap.add_argument("--output", default=r"D:\! secURLity\dataset\eda-result 4.txt",
                    help="Output file path (None = stdout only)")
    args = ap.parse_args()

    run_eda(args.path, args.max_rows, args.output)


if __name__ == "__main__":
    main()

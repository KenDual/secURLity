"""
Preprocess dataset 4 -> tokenize URLs to int32 sequences, split train/val/test.

Sản phẩm Stage 1 của model 4 (xem CNN-LSTM-final.md).

Decisions từ Stage 0 EDA:
  MAX_LEN     = 256   (P99.5=217 → 256 covers ~P99.7, multiple of 64)
  vocab_size  = 94    (PAD=0 + UNK=1 + 92 case-preserved chars)
  case        = PRESERVED  (mixed-case ratio L0=11.83% vs L1=5.50% có signal)

Hai phiên bản split (Item 8 — domain-honest evaluation):

  RANDOM split  (in-distribution metric — quen thuộc, giống model 1/2/3)
    - Stratified theo label
    - Tỉ lệ 80/10/10 chính xác trên row count
    - Output:  data/processed/model_4/random/{train,val,test}_{X,y}.npy

  DOMAIN split  (out-of-distribution metric — F1 "thật" cho production)
    - Mỗi registered domain CHỈ thuộc về MỘT split
    - Greedy bin-packing theo URL count (vì benign concentrated trong ~580 domains
      với top 5 chiếm ~4M URLs — random domain split sẽ phá tỉ lệ 80/10/10)
    - Tỉ lệ 80/10/10 trên row count (approximation, không chính xác do bin-pack)
    - Mixed-label domains (96 domains, 4.66M URLs) đi nguyên block theo majority
    - Output:  data/processed/model_4/domain/{train,val,test}_{X,y}.npy

Shared artifacts:
  data/processed/model_4/vocab.json       (94 entries)
  data/processed/model_4/metadata.json    (config + split sizes + label counts)
  data/processed/model_4/<mode>/train.csv val.csv test.csv  (gốc URL + label
      cho lexical-feature extractor downstream và debug)

Memory profile (RTX 3060 Ti host, 32GB RAM):
  - Load CSV vào pandas        ~ 3-4 GB
  - Domain extraction (regex)  ~ 1-2 GB string column
  - Encoded master KHÔNG load — ghi thẳng vào memmap NPY của từng split
  - Peak RAM ước tính           ~ 6-8 GB

Disk profile:
  - train_X.npy  : ~15.74M × 256 × 4 = 16.1 GB  (mỗi mode)
  - val_X.npy    : ~ 1.97M × 256 × 4 =  2.0 GB
  - test_X.npy   : ~ 1.97M × 256 × 4 =  2.0 GB
  - 2 modes × 3 splits = 6 NPYs ≈ 40 GB total
  - + 6 CSV files copy (gốc URL + label)        ≈ 3 GB
  - Tổng ước tính              ≈ 43 GB

Usage (PowerShell):
    .\\venv\\Scripts\\Activate.ps1

    # Mặc định: chạy cả 2 mode
    python "scripts/3. preprocess-data 4.py"

    # Chỉ random
    python "scripts/3. preprocess-data 4.py" --mode random

    # Chỉ domain
    python "scripts/3. preprocess-data 4.py" --mode domain
"""

import argparse
import csv
import json
import string as _string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 stdout cho Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass


# ============================================================================
# Config (chốt từ Stage 0 EDA)
# ============================================================================

PROJECT_ROOT = Path(r"D:\! secURLity")
INPUT_CSV    = PROJECT_ROOT / "dataset" / "dataset-4-final.csv"
OUT_ROOT     = PROJECT_ROOT / "data" / "processed" / "model_4"

MAX_LEN      = 256
PAD_IDX      = 0
UNK_IDX      = 1
SEED         = 42

SPLIT_RATIOS = (0.80, 0.10, 0.10)   # train, val, test
SPLIT_NAMES  = ("train", "val", "test")

# 92 chars vocab — chốt từ EDA `eda-result 4.txt` (PROPOSED_VOCAB_CHARS)
VOCAB_CHARS = sorted(
    set(_string.ascii_letters       # 52 (a-z + A-Z)
        + _string.digits            # 10 (0-9)
        + "/:.-_?=&#@%+~"           # 13 từ vocab cũ
        + ",;()[]{}!*$|<>'\" ")     # 17 thêm vào (gồm space)
)
assert len(VOCAB_CHARS) == 92, f"Expected 92 chars in vocab, got {len(VOCAB_CHARS)}"


# ============================================================================
# Vocab + encoding
# ============================================================================

def build_vocab() -> dict[str, int]:
    """Returns: {char -> index}. PAD=0, UNK=1, các char khác bắt đầu từ 2."""
    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for i, c in enumerate(VOCAB_CHARS):
        vocab[c] = i + 2
    return vocab


def encode_url(url: str, char_to_idx: dict, max_len: int = MAX_LEN) -> np.ndarray:
    """url -> int32[MAX_LEN], padded with PAD_IDX, truncated to max_len.

    CASE PRESERVED — KHÔNG lowercase url trước khi encode (item 2).
    """
    out = np.full(max_len, PAD_IDX, dtype=np.int32)
    n = min(len(url), max_len)
    for i in range(n):
        out[i] = char_to_idx.get(url[i], UNK_IDX)
    return out


def encode_chunk(urls, char_to_idx: dict, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode 1 chunk URLs -> int32[len(urls), max_len]. Vectorized inner."""
    n = len(urls)
    out = np.full((n, max_len), PAD_IDX, dtype=np.int32)
    for i, url in enumerate(urls):
        m = min(len(url), max_len)
        for j in range(m):
            out[i, j] = char_to_idx.get(url[j], UNK_IDX)
    return out


# ============================================================================
# Registered domain extraction (cho domain-split)
# ============================================================================
# Same logic as `2. eda 4.py`. Manual eTLD+1 — không cần tldextract / PSL.

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


def _is_ip(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or not (0 <= int(p) <= 255):
            return False
    return True


def extract_reg_domain(url: str) -> str:
    """Approximate eTLD+1. Trả về '<unknown>' nếu không parse được."""
    if not url:
        return "<unknown>"
    s = url
    # Strip scheme
    if "://" in s:
        s = s.split("://", 1)[1]
    # Strip path/query/fragment
    for sep in ("/", "?", "#"):
        if sep in s:
            s = s.split(sep, 1)[0]
    # Strip user info
    if "@" in s:
        s = s.rsplit("@", 1)[1]
    # Strip port
    if ":" in s:
        s = s.rsplit(":", 1)[0]
    host = s.lower().strip()
    if not host:
        return "<unknown>"
    if _is_ip(host):
        return host
    # Match against known TLDs
    for tld in KNOWN_TLDS:
        if host.endswith(tld):
            tld_clean = tld.lstrip(".")
            host_no_tld = host[: -(len(tld_clean) + 1)] if host.endswith("." + tld_clean) else ""
            if not host_no_tld:
                return host
            last_label = host_no_tld.rsplit(".", 1)[-1]
            return f"{last_label}.{tld_clean}"
    # Fallback: last 2 labels
    if "." in host:
        labels = host.rsplit(".", 2)
        return ".".join(labels[-2:])
    return host


# ============================================================================
# Split logic
# ============================================================================

def random_split_stratified(labels: np.ndarray, ratios=SPLIT_RATIOS,
                            seed=SEED) -> np.ndarray:
    """Stratified random split theo label. Trả về array int8 (0=train, 1=val, 2=test)."""
    n = len(labels)
    assign = np.full(n, -1, dtype=np.int8)
    rng = np.random.default_rng(seed)

    for lbl in np.unique(labels):
        idx_lbl = np.where(labels == lbl)[0]
        rng.shuffle(idx_lbl)
        n_lbl = len(idx_lbl)
        n_train = int(ratios[0] * n_lbl)
        n_val   = int(ratios[1] * n_lbl)
        # test = remainder
        assign[idx_lbl[:n_train]] = 0
        assign[idx_lbl[n_train : n_train + n_val]] = 1
        assign[idx_lbl[n_train + n_val:]] = 2

    assert (assign != -1).all(), "Có row chưa được assign"
    return assign


def domain_split_balanced(domains: pd.Series, labels: np.ndarray,
                          ratios=SPLIT_RATIOS, seed=SEED) -> np.ndarray:
    """Stratified-per-label domain split + greedy bin-packing theo URL count.

    Mỗi registered domain CHỈ thuộc về 1 split (domain-honest).
    Để KHÔNG bị skew label ratio giữa các split (problem khi naive greedy:
    top benign domains nhét hết vào train, val/test toàn small mal domains),
    ta:
      1. Phân loại mỗi domain theo MAJORITY label (0=benign-majority, 1=mal-majority).
         Mixed domains (96 domain với 4.66M URL) đa số là benign-majority → group 0.
      2. Trong mỗi nhóm majority, greedy bin-pack 80/10/10 theo URL count.
         - Sort domain trong nhóm theo URL count desc (top benign domain trước,
           shuffle nhỏ để tie-break)
         - Mỗi domain → split có deficit LỚN NHẤT (URL count target - current).
      3. Kết quả: train/val/test đều có ~80/20 benign-vs-mal vì 2 nhóm được
         chia tỉ lệ riêng rồi merge.

    Trả về array int8 (0=train, 1=val, 2=test).
    """
    n = len(domains)
    rng = np.random.default_rng(seed)

    print(f"  Building domain stats from {n:,} rows...", flush=True)
    t0 = time.time()
    df_dom = pd.DataFrame({"domain": domains.values, "label": labels})
    grouped = df_dom.groupby("domain", sort=False)["label"]
    n_mal = grouped.sum().astype(np.int64)              # mal count per domain
    n_total = grouped.size().astype(np.int64)           # total URLs per domain
    n_ben = n_total - n_mal
    is_mal_majority = (n_mal > n_ben).astype(np.int8)   # 1 if mal-majority
    print(f"    domain stats           : {len(n_total):,} domains  "
          f"({time.time()-t0:.1f}s)")

    # Greedy bin-pack riêng cho từng nhóm majority
    domain_to_split: dict[str, int] = {}
    counts_per_group_split = {0: [0, 0, 0], 1: [0, 0, 0]}

    for maj in (0, 1):
        # Convert sang numpy object array (tránh UserWarning của
        # rng.shuffle với pandas StringArray)
        domains_in_grp = np.asarray(
            is_mal_majority[is_mal_majority == maj].index, dtype=object)
        if len(domains_in_grp) == 0:
            continue
        # Shuffle để tie-break
        rng.shuffle(domains_in_grp)
        # Sort URL count desc (numpy argsort + stable preserves shuffle order trong tie)
        sizes = n_total.loc[domains_in_grp].values
        order = np.argsort(-sizes, kind="stable")
        domains_sorted = domains_in_grp[order]
        sizes_sorted = sizes[order]

        total_urls = int(sizes_sorted.sum())
        targets = np.array([r * total_urls for r in ratios], dtype=np.float64)
        counts = np.zeros(3, dtype=np.float64)

        t0 = time.time()
        for d, cnt in zip(domains_sorted, sizes_sorted):
            deficits = targets - counts
            best = int(np.argmax(deficits))
            domain_to_split[d] = best
            counts[best] += int(cnt)
        counts_per_group_split[maj] = counts.tolist()
        maj_name = "mal-majority " if maj else "benign-majority"
        print(f"    pack {maj_name}: {len(domains_sorted):,} domains, "
              f"{int(total_urls):,} URLs  → "
              f"train={int(counts[0]):,} val={int(counts[1]):,} "
              f"test={int(counts[2]):,}  ({time.time()-t0:.1f}s)")

    # Map back to row level
    t0 = time.time()
    assign = domains.map(domain_to_split).values.astype(np.int8)
    print(f"    map back to rows       : {time.time()-t0:.1f}s")

    if (assign == -1).any():
        raise RuntimeError("Có domain không assign được — bug stratified greedy")

    return assign


# ============================================================================
# Save splits
# ============================================================================

def save_split(out_dir: Path, urls: list[str], labels: np.ndarray,
               char_to_idx: dict, chunk_size: int = 50_000) -> dict:
    """Encode URLs in chunks và ghi thẳng vào X_{split}.npy (memmap).

    Trả về dict {split_name: count} để metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(urls)
    print(f"    Writing {n:,} URLs -> X.npy + y.npy ...", flush=True)

    # np.lib.format.open_memmap tạo .npy hợp lệ và mmap-write
    X_mm = np.lib.format.open_memmap(
        out_dir / "X.npy", mode="w+", dtype=np.int32, shape=(n, MAX_LEN)
    )

    t0 = time.time()
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_urls = urls[chunk_start:chunk_end]
        X_mm[chunk_start:chunk_end] = encode_chunk(chunk_urls, char_to_idx)
        if (chunk_start // chunk_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = chunk_end / max(elapsed, 0.001)
            eta = (n - chunk_end) / max(rate, 1)
            print(f"      {chunk_end:>12,} / {n:,}  "
                  f"({rate:,.0f}/s, eta {eta:.0f}s)", flush=True)
    X_mm.flush()
    del X_mm
    np.save(out_dir / "y.npy", labels.astype(np.int8))

    return {"count": n,
            "label_0": int((labels == 0).sum()),
            "label_1": int((labels == 1).sum())}


def save_split_csv(out_dir: Path, urls: list[str], labels: np.ndarray) -> None:
    """Lưu URLs + labels dạng QUOTE_ALL CSV để lexical-feature extractor đọc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "split.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["url", "label"])
        for u, y in zip(urls, labels):
            writer.writerow([u, int(y)])


# ============================================================================
# Stats reporter
# ============================================================================

def print_split_stats(name: str, assign: np.ndarray, labels: np.ndarray,
                      domains: pd.Series = None) -> None:
    n = len(labels)
    print(f"\n  [{name} split]")
    for s_idx, s_name in enumerate(SPLIT_NAMES):
        mask = assign == s_idx
        n_s = int(mask.sum())
        n_b = int((labels[mask] == 0).sum())
        n_m = int((labels[mask] == 1).sum())
        ratio = n_s / n * 100
        b_pct = (n_b / n_s * 100) if n_s else 0
        m_pct = (n_m / n_s * 100) if n_s else 0
        line = (f"    {s_name:<5} : {n_s:>12,}  ({ratio:5.2f}%)   "
                f"benign={n_b:>11,} ({b_pct:5.2f}%)  "
                f"mal={n_m:>11,} ({m_pct:5.2f}%)")
        print(line)
        if domains is not None:
            n_d = domains[mask].nunique()
            print(f"            unique domains = {n_d:,}")

    # Verify domain-split: 0 overlap
    if domains is not None:
        d_sets = []
        for s_idx in range(3):
            d_sets.append(set(domains[assign == s_idx].unique()))
        overlap_01 = len(d_sets[0] & d_sets[1])
        overlap_02 = len(d_sets[0] & d_sets[2])
        overlap_12 = len(d_sets[1] & d_sets[2])
        print(f"    domain overlap  train∩val={overlap_01}  "
              f"train∩test={overlap_02}  val∩test={overlap_12}  "
              f"(target = 0)")


# ============================================================================
# Main
# ============================================================================

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("random", "domain", "both"),
                    default="both", help="Split mode (default: both)")
    ap.add_argument("--input", default=str(INPUT_CSV),
                    help="Input CSV path")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Limit rows (cho smoke test)")
    args = ap.parse_args(argv)

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load CSV
    # ----------------------------------------------------------------
    t_global = time.time()
    print(f"[1/5] Load CSV : {csv_path}")
    print(f"      size     : {csv_path.stat().st_size / 1024**2:,.1f} MB")
    t0 = time.time()
    df = pd.read_csv(csv_path, dtype={"url": "string", "label": "int8"},
                     nrows=args.max_rows)
    print(f"      rows     : {len(df):,}  (loaded in {time.time()-t0:.1f}s)")

    # Strip whitespace/null safeguard
    df = df.dropna(subset=["url", "label"]).reset_index(drop=True)
    df["url"] = df["url"].astype(str)
    print(f"      after drop NaN: {len(df):,}")

    labels = df["label"].values.astype(np.int8)
    urls = df["url"].tolist()  # ~3-4 GB RAM nhưng giúp encode nhanh hơn
    print(f"      labels: benign={int((labels==0).sum()):,}  "
          f"mal={int((labels==1).sum()):,}")

    # ----------------------------------------------------------------
    # 2. Build + save vocab
    # ----------------------------------------------------------------
    print(f"\n[2/5] Build vocab ({len(VOCAB_CHARS)} chars + PAD + UNK)")
    vocab = build_vocab()
    with (OUT_ROOT / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"      vocab_size = {len(vocab)}")
    print(f"      saved      : {OUT_ROOT / 'vocab.json'}")

    # ----------------------------------------------------------------
    # 3. Compute split assignments
    # ----------------------------------------------------------------
    print(f"\n[3/5] Compute split assignments  (mode={args.mode})")
    assign_random = None
    assign_domain = None
    domains_series = None

    if args.mode in ("random", "both"):
        t0 = time.time()
        assign_random = random_split_stratified(labels)
        print(f"      RANDOM split  : done in {time.time()-t0:.1f}s")
        print_split_stats("random", assign_random, labels)

    if args.mode in ("domain", "both"):
        # Extract registered_domain cho mọi URL
        t0 = time.time()
        print(f"      Extracting reg_domain (1 lần dùng cho cả split + stats)...")
        domains_series = pd.Series([extract_reg_domain(u) for u in urls], dtype="string")
        print(f"        unique reg_domains: {domains_series.nunique():,}  "
              f"(in {time.time()-t0:.1f}s)")

        t0 = time.time()
        assign_domain = domain_split_balanced(domains_series, labels)
        print(f"      DOMAIN split  : done in {time.time()-t0:.1f}s")
        print_split_stats("domain", assign_domain, labels, domains_series)

    # ----------------------------------------------------------------
    # 4. Encode + write split NPYs
    # ----------------------------------------------------------------
    print(f"\n[4/5] Encode URLs + write NPY files")

    metadata: dict = {
        "max_len": MAX_LEN,
        "vocab_size": len(vocab),
        "pad_idx": PAD_IDX,
        "unk_idx": UNK_IDX,
        "seed": SEED,
        "split_ratios": list(SPLIT_RATIOS),
        "source_csv": str(csv_path),
        "case_preserved": True,
        "n_total": len(df),
        "modes": {},
    }

    for mode_name, assign in [("random", assign_random), ("domain", assign_domain)]:
        if assign is None:
            continue
        mode_dir = OUT_ROOT / mode_name
        mode_meta = {}
        print(f"\n  Mode: {mode_name}")
        for s_idx, s_name in enumerate(SPLIT_NAMES):
            mask = assign == s_idx
            split_urls = [u for u, m in zip(urls, mask) if m]
            split_labels = labels[mask]
            split_dir = mode_dir / s_name
            print(f"\n    [{mode_name}/{s_name}]")

            # Save CSV (for downstream lexical extraction + debug)
            save_split_csv(split_dir, split_urls, split_labels)

            # Encode + save NPY
            stats = save_split(split_dir, split_urls, split_labels, vocab)
            mode_meta[s_name] = stats

        # Compute pos_weight cho BCE loss
        n_pos = mode_meta["train"]["label_1"]
        n_neg = mode_meta["train"]["label_0"]
        pos_weight = n_neg / max(n_pos, 1)
        mode_meta["pos_weight"] = round(pos_weight, 4)

        metadata["modes"][mode_name] = mode_meta

    # ----------------------------------------------------------------
    # 5. Save metadata
    # ----------------------------------------------------------------
    with (OUT_ROOT / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_global
    print(f"\n[5/5] Done in {elapsed/60:.1f} min")
    print(f"      Output root : {OUT_ROOT}")
    print(f"      metadata    : {OUT_ROOT / 'metadata.json'}")


if __name__ == "__main__":
    main()

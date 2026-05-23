"""
Cân bằng dataset 3 theo tỉ lệ 80/20 (benign/malicious).

Quy trình:
  1. Đọc dataset-mal-3.csv -> giữ toàn bộ N_mal rows.
  2. Đọc dataset-benign-3.csv -> SHUFFLE GLOBAL (seed=42) vì file đang
     sort (VN trên, intl dưới) -> sample 4*N_mal rows -> đảm bảo lấy
     đều cả VN lẫn intl.
  3. Concat benign+mal -> shuffle 1 lần nữa -> ghi QUOTE_ALL.

Output:
    dataset/dataset-3-balanced.csv   (80% benign / 20% malicious)
"""

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"D:\! secURLity")
BENIGN_CSV   = PROJECT_ROOT / "dataset" / "dataset-benign-3.csv"
MAL_CSV      = PROJECT_ROOT / "dataset" / "dataset-mal-3.csv"
OUT_CSV      = PROJECT_ROOT / "dataset" / "dataset-3-balanced.csv"

SEED         = 42
TARGET_RATIO = 4   # benign : mal = 4 : 1  (i.e. 80/20)


def main() -> None:
    t0 = time.time()

    # ---- mal ----------------------------------------------------------------
    print(f"[1/4] Read malicious : {MAL_CSV.name}")
    mal = pd.read_csv(MAL_CSV, dtype={"url": "string", "label": "int8"})
    n_mal = len(mal)
    print(f"      rows = {n_mal:,}")

    # ---- benign (shuffle global + sample) -----------------------------------
    n_benign_target = TARGET_RATIO * n_mal
    print(f"[2/4] Read benign    : {BENIGN_CSV.name}")
    benign = pd.read_csv(BENIGN_CSV, dtype={"url": "string", "label": "int8"})
    print(f"      rows = {len(benign):,}")

    if len(benign) < n_benign_target:
        raise RuntimeError(
            f"benign chỉ có {len(benign):,} rows, cần {n_benign_target:,} "
            f"để giữ tỉ lệ 80/20. Giảm TARGET_RATIO hoặc thu thêm benign."
        )

    print(f"[3/4] Shuffle global benign + sample {n_benign_target:,} rows "
          f"(seed={SEED})")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(benign))[:n_benign_target]
    idx.sort()  # sort lại để .iloc chạy nhanh hơn (cache friendly)
    benign = benign.iloc[idx].reset_index(drop=True)

    # ---- concat + shuffle final + write ------------------------------------
    print(f"[4/4] Concat + final shuffle -> {OUT_CSV.name}")
    out = pd.concat([benign, mal], ignore_index=True)
    # final shuffle để benign/mal trộn đều trong CSV
    out = out.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)

    # sanity check
    counts = out["label"].value_counts().to_dict()
    n_total = len(out)
    print(f"      Total rows  : {n_total:,}")
    print(f"      Benign (0)  : {counts.get(0, 0):,}  ({counts.get(0, 0)/n_total*100:.2f}%)")
    print(f"      Malicious(1): {counts.get(1, 0):,}  ({counts.get(1, 0)/n_total*100:.2f}%)")

    out.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL,
               encoding="utf-8", lineterminator="\n")

    size_mb = OUT_CSV.stat().st_size / 1024**2
    print(f"\nDone in {time.time()-t0:.1f}s.")
    print(f"Output: {OUT_CSV}  ({size_mb:,.1f} MB)")


if __name__ == "__main__":
    main()

"""
Preprocess URLs (model 3 — VN real): Split data + Encode to numpy arrays.
Run AFTER `scripts/2. prepare-dataset 3.py`.

Input  : dataset/dataset 3 (vn).csv  (~15M rows quoted, 12.5M benign + 2.6M malicious)
Output : data/processed/model_3/

Khác model 2:
  - Đầu vào QUOTED (mọi field bọc bằng "). pandas đọc default quoting=
    csv.QUOTE_MINIMAL nên xử lý transparent — không cần option đặc biệt.
  - Class imbalance ~86/14 (benign/malicious) -> pos_weight ~6, dùng khi
    train BCEWithLogitsLoss (giống pattern model 2).

MAX_LEN = 100 (đồng bộ model 1 + model 2). EDA model 3 có thể khác,
chạy lại nếu muốn tinh chỉnh sau.
"""

import csv
import json
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
PROJECT_ROOT = Path(r"D:\! secURLity")
DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset 3 (vn).csv"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed" / "model_3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters (đồng bộ model 1 + 2)
MAX_LEN = 100
SEED = 42

print("=" * 70)
print("PREPROCESSING PIPELINE (model 3 — VN real URLs)")
print("=" * 70)
print(f"Input  : {DATASET_PATH}")
print(f"Output : {OUTPUT_DIR}")
print(f"MAX_LEN: {MAX_LEN}")

# Step 1: Load & Clean
print("\n[1/5] Loading dataset (quoted CSV)...")
# quoting=csv.QUOTE_MINIMAL là default của pd.read_csv — đọc tốt cả file QUOTE_ALL.
# dtype=str để giữ nguyên label dạng string trước khi coerce sang int.
df = pd.read_csv(DATASET_PATH, dtype={"url": "string", "label": "string"},
                 quoting=csv.QUOTE_MINIMAL)
df.columns = df.columns.str.strip()

# Coerce label sang int; row nào không phải '0'/'1' sẽ NaN -> drop
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df["url"] = df["url"].str.strip().str.lower()
df = df.dropna(subset=["url", "label"])
df = df[df["url"] != ""].reset_index(drop=True)
df["label"] = df["label"].astype(np.int32)

print(f"   Loaded {len(df):,} URLs")
print(f"   Label distribution: {dict(df['label'].value_counts().sort_index())}")

# Step 2: Stratified Split (80/10/10)
print("\n[2/5] Splitting train/val/test (80/10/10)...")
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
)
print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Save splits as quoted CSV (giữ format giống input để consistent)
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, quoting=csv.QUOTE_ALL)
val_df.to_csv(OUTPUT_DIR / "val.csv",     index=False, quoting=csv.QUOTE_ALL)
test_df.to_csv(OUTPUT_DIR / "test.csv",   index=False, quoting=csv.QUOTE_ALL)

# Step 3: Build Vocabulary (đồng bộ model 1 + 2)
print("\n[3/5] Building character vocabulary...")
CHARS = string.ascii_lowercase + string.digits + "/:.-_?=&#@%+~"
vocab = {"<PAD>": 0, "<UNK>": 1}
vocab.update({c: i + 2 for i, c in enumerate(CHARS)})
VOCAB_SIZE = len(vocab)

print(f"   Vocabulary size: {VOCAB_SIZE}")
print(f"   Sample chars: {list(vocab.keys())[:10]}...")

with open(OUTPUT_DIR / "vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

# Step 4: Encode URLs to numpy arrays
def encode_batch(urls, vocab, max_len):
    """Vectorized encoding (faster than loop)."""
    unk_idx = vocab["<UNK>"]
    pad_idx = vocab["<PAD>"]
    encoded = np.full((len(urls), max_len), pad_idx, dtype=np.int32)
    for i, url in enumerate(urls):
        url_truncated = url[:max_len]
        for j, char in enumerate(url_truncated):
            encoded[i, j] = vocab.get(char, unk_idx)
    return encoded


print("\n[4/5] Encoding URLs to numpy arrays...")
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"   Encoding {name} set ({len(split_df):,} URLs)...")
    X = encode_batch(split_df["url"].tolist(), vocab, MAX_LEN)
    y = split_df["label"].values.astype(np.int32)
    np.save(OUTPUT_DIR / f"{name}_X.npy", X)
    np.save(OUTPUT_DIR / f"{name}_y.npy", y)
    print(f"   Saved {name}_X.npy shape: {X.shape}")

# Step 5: Save metadata
print("\n[5/5] Saving metadata...")
n_train_neg = int((train_df["label"] == 0).sum())
n_train_pos = int((train_df["label"] == 1).sum())
metadata = {
    "vocab_size": VOCAB_SIZE,
    "max_len": MAX_LEN,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "label_counts": {
        "train": {int(k): int(v) for k, v in train_df["label"].value_counts().sort_index().items()},
        "val":   {int(k): int(v) for k, v in val_df["label"].value_counts().sort_index().items()},
        "test":  {int(k): int(v) for k, v in test_df["label"].value_counts().sort_index().items()},
    },
    "pos_weight": float(n_train_neg / max(n_train_pos, 1)),
    "source_csv": str(DATASET_PATH),
    "csv_quoting": "QUOTE_ALL",
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print(f"\nFiles saved to: {OUTPUT_DIR}")
print(f"pos_weight (for BCEWithLogitsLoss) = {metadata['pos_weight']:.4f}")
print("\nNext step: copy `scripts/4. train-model_2.py` -> `4. train-model_3.py`,")
print("           sửa PROJECT_ROOT subdir 'model_2' -> 'model_3', train.")

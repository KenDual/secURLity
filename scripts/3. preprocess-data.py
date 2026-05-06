"""
Preprocess URLs: Split data + Encode to numpy arrays
Run this ONCE before training.
Expected time: ~5 minutes for 10M URLs
"""

import string
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# Paths
PROJECT_ROOT = Path(r"D:\! secURLity")
DATASET_PATH = PROJECT_ROOT / "dataset" / "urls_synthetic_10m.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
MAX_LEN = 100  # EDA shows P99=83, so 100 is enough
SEED = 42

print("=" * 70)
print("PREPROCESSING PIPELINE")
print("=" * 70)

# Step 1: Load & Clean
print("\n[1/5] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

# Handle both 'URL' and 'url' column names
url_col = 'URL' if 'URL' in df.columns else 'url'
df = df.rename(columns={url_col: 'url'})

df['url'] = df['url'].str.strip().str.lower()
df = df.dropna(subset=['url', 'label'])
df = df[df['url'] != ''].reset_index(drop=True)

print(f"   Loaded {len(df):,} URLs")
print(f"   Label distribution: {dict(df['label'].value_counts().sort_index())}")

# Step 2: Stratified Split
print("\n[2/5] Splitting train/val/test (80/10/10)...")
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED
)

print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Save splits as CSV (for reference)
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Step 3: Build Vocabulary from Training Set ONLY
print("\n[3/5] Building character vocabulary from training set...")
CHARS = string.ascii_lowercase + string.digits + "/:.-_?=&#@%+~"
vocab = {'<PAD>': 0, '<UNK>': 1}
vocab.update({c: i+2 for i, c in enumerate(CHARS)})
VOCAB_SIZE = len(vocab)

print(f"   Vocabulary size: {VOCAB_SIZE}")
print(f"   Sample chars: {list(vocab.keys())[:10]}...")

# Save vocab
with open(OUTPUT_DIR / "vocab.json", 'w') as f:
    json.dump(vocab, f, indent=2)

# Step 4: Encode URLs to numpy arrays
def encode_batch(urls, vocab, max_len):
    """Vectorized encoding (faster than loop)"""
    unk_idx = vocab['<UNK>']
    pad_idx = vocab['<PAD>']
    
    encoded = np.full((len(urls), max_len), pad_idx, dtype=np.int32)
    
    for i, url in enumerate(urls):
        url_truncated = url[:max_len]
        for j, char in enumerate(url_truncated):
            encoded[i, j] = vocab.get(char, unk_idx)
    
    return encoded

print("\n[4/5] Encoding URLs to numpy arrays...")

# Encode each split
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"   Encoding {name} set ({len(split_df):,} URLs)...")
    
    X = encode_batch(split_df['url'].tolist(), vocab, MAX_LEN)
    y = split_df['label'].values.astype(np.int32)
    
    np.save(OUTPUT_DIR / f"{name}_X.npy", X)
    np.save(OUTPUT_DIR / f"{name}_y.npy", y)
    
    print(f"   ✓ Saved {name}_X.npy shape: {X.shape}")

# Step 5: Save metadata
print("\n[5/5] Saving metadata...")
metadata = {
    "vocab_size": VOCAB_SIZE,
    "max_len": MAX_LEN,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "label_counts": {
        "train": {int(k): int(v) for k, v in train_df['label'].value_counts().sort_index().items()},
        "val": {int(k): int(v) for k, v in val_df['label'].value_counts().sort_index().items()},
        "test": {int(k): int(v) for k, v in test_df['label'].value_counts().sort_index().items()}
    },
    "pos_weight": float(len(train_df[train_df['label']==0]) / len(train_df[train_df['label']==1]))
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print(f"\nFiles saved to: {OUTPUT_DIR}")
print("\nNext step: Run train_fast.py")
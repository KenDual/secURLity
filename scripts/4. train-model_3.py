"""
Fast CNN-LSTM Training with Mixed Precision (model 3 — VN real URLs)
Uses pre-encoded numpy arrays from `3. preprocess-data 3.py`.

Input  : data/processed/model_3/{train,val,test}_{X,y}.npy + metadata.json
Output : models/cnn_lstm_best_3.pt, models/results_3.json

Tối ưu tốc độ trên RTX 3060 Ti (8GB VRAM, 4864 CUDA cores, 152 Tensor cores):
  - **AMP (Automatic Mixed Precision)** với FP16: ~1.7-2x throughput,
    halve VRAM cho activations -> tăng được batch size.
  - **TF32** cho FP32 matmul còn lại (Ampere SM_86 native support).
  - **mmap_mode='r'** khi load .npy: dataset 3 ~6GB train_X — không load
    full vào RAM, page-on-demand. Cũng giải quyết OOM ở test phase mà
    CLAUDE.md note (model 2 issue: persistent_workers + 3 .npy fork blowup).
  - **BATCH_SIZE=4096** (gấp 2 model 2): AMP chiếm ~1/2 memory, 8GB VRAM
    đủ thoải mái. Throughput cao hơn vì giảm per-iter overhead.
  - **set_to_none=True** cho zero_grad: nhanh hơn so với zero tensors.
  - **non_blocking=True** cho .to(device): overlap H2D copy với compute.

Nếu OOM, hạ BATCH_SIZE xuống 2048 hoặc 3072.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(r"D:\! secURLity")
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_3"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = MODELS_DIR / "cnn_lstm_best_3.pt"
RESULTS_PATH = MODELS_DIR / "results_3.json"

# ============================================================================
# Hyperparameters
# ============================================================================
BATCH_SIZE = 4096          # AMP + 8GB VRAM cho phép gấp đôi model 2
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
EMBED_DIM = 64             # multiple of 8 -> tensor core friendly
CONV_CHANNELS = 128        # multiple of 8
LSTM_HIDDEN = 128          # multiple of 8
DROPOUT = 0.4
NUM_WORKERS = 4
USE_AMP = True             # FP16 mixed precision

# ============================================================================
# Device + Ampere optimizations
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Capability: SM_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")

    # cuDNN auto-tuner cho conv/LSTM kernels
    torch.backends.cudnn.benchmark = True

    # TF32 cho matmul/conv FP32 còn lại (Ampere+) — gần như free speedup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"AMP      : {USE_AMP} (FP16)")
    print(f"TF32     : enabled (matmul + cudnn)")
else:
    USE_AMP = False  # AMP chỉ có ý nghĩa trên CUDA
    print("WARNING: CUDA not available — sẽ chạy CPU rất chậm")

# ============================================================================
# Load metadata
# ============================================================================
with open(DATA_DIR / "metadata.json") as f:
    meta = json.load(f)

print(f"\nDataset: Train={meta['train_size']:,} | Val={meta['val_size']:,} | Test={meta['test_size']:,}")
print(f"Vocab size           : {meta['vocab_size']}")
print(f"Class imbalance ratio: {meta['pos_weight']:.2f}  (= N_neg / N_pos)")
print(f"BATCH_SIZE           : {BATCH_SIZE}")


# ============================================================================
# Dataset (memory-mapped — không load full .npy vào RAM)
# ============================================================================
class PreEncodedDataset(Dataset):
    """Read .npy bằng mmap_mode='r' để tránh RAM blowup khi multi-worker fork.
    Trên dataset 3 (~6GB train_X), load full sẽ chiếm RAM nhiều và mỗi DataLoader
    worker fork sẽ copy-on-write -> 4 workers x 6GB = OOM trên máy 16GB.
    mmap chia sẻ page cache OS, không nhân lên."""

    def __init__(self, X_path, y_path):
        self.X = np.load(X_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # torch.tensor(...) tự copy -> writable, tránh warning về readonly memmap
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ============================================================================
# Model (giống model 2)
# ============================================================================
class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, conv_channels, lstm_hidden, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            conv_channels, lstm_hidden,
            num_layers=1, batch_first=True,
            bidirectional=True, dropout=0,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.embed(x)              # [B, L, E]
        x = x.permute(0, 2, 1)         # [B, E, L]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)         # [B, L', C]
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)  # [B, 2H]
        return self.fc(h).squeeze(1)


# ============================================================================
# Train / Eval
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for X, y in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for X, y in tqdm(loader, desc="Evaluating", leave=False):
        X = X.to(device, non_blocking=True)
        if USE_AMP:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(X)
        else:
            logits = model(X)
        # sigmoid + threshold trên FP32 cho ổn định
        preds = (torch.sigmoid(logits.float()) > 0.5).cpu().int().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.int().numpy())
    return np.array(all_labels), np.array(all_preds)


# ============================================================================
# MAIN (must be wrapped for Windows multiprocessing)
# ============================================================================
if __name__ == "__main__":
    print("\nLoading datasets (mmap)...")
    train_dataset = PreEncodedDataset(DATA_DIR / "train_X.npy", DATA_DIR / "train_y.npy")
    val_dataset   = PreEncodedDataset(DATA_DIR / "val_X.npy",   DATA_DIR / "val_y.npy")
    test_dataset  = PreEncodedDataset(DATA_DIR / "test_X.npy",  DATA_DIR / "test_y.npy")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    # ---- Model + loss + optimizer + AMP scaler ----
    model = CNNLSTM(
        vocab_size=meta["vocab_size"],
        embed_dim=EMBED_DIM,
        conv_channels=CONV_CHANNELS,
        lstm_hidden=LSTM_HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    pos_weight = torch.tensor([meta["pos_weight"]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device="cuda") if USE_AMP else None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params         : {n_params:,}")

    print("\n" + "=" * 70)
    print("TRAINING START (model 3)")
    print("=" * 70)

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_labels, val_preds = evaluate(model, val_loader)
        val_f1 = f1_score(val_labels, val_preds)

        print(f"Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_f1": best_f1,
            }, CKPT_PATH)
            print(f"New best model saved (F1={best_f1:.4f}) -> {CKPT_PATH.name}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ---- Test ----
    print("\n" + "=" * 70)
    print("TESTING (model 3)")
    print("=" * 70)

    checkpoint = torch.load(CKPT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (Val F1={checkpoint['best_f1']:.4f})")

    test_labels, test_preds = evaluate(model, test_loader)

    print("\n" + classification_report(
        test_labels, test_preds,
        target_names=["Benign", "Malicious"],
        digits=4,
    ))

    results = {
        "model": "model_3",
        "data_dir": str(DATA_DIR),
        "checkpoint": str(CKPT_PATH),
        "best_epoch": checkpoint["epoch"],
        "best_val_f1": float(checkpoint["best_f1"]),
        "amp": USE_AMP,
        "batch_size": BATCH_SIZE,
        "test_report": classification_report(test_labels, test_preds, output_dict=True),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("Training complete!")

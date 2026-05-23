"""
Train Hybrid CNN-LSTM cho model 4 (Stage 2 + 3 của CNN-LSTM-final.md).

Architecture (Items 4, 5):

    Sequence path (Item 5):
        Embedding(94, 64)
        → Conv1d(64→128, k=3) + ReLU + MaxPool(2)
        → Conv1d(128→128, k=5) + ReLU + MaxPool(2)
        → Stacked BiLSTM (2 layers, hidden=128, dropout=0.2)
        → AttentionPool (with pad-aware mask)
        → (B, 256)

    Feature path (Item 4):
        Linear(30→64) + ReLU + Dropout(0.2)
        → Linear(64→32) + ReLU
        → (B, 32)

    Hybrid head:
        concat → (B, 288)
        → Linear(288→64) + ReLU + Dropout(0.4)
        → Linear(64→1) → logit

Training (Items 9, 10):
    - Optimizer       : AdamW (weight_decay=0.01)
    - Scheduler       : LinearWarmup (5% steps) → CosineAnnealingLR (eta_min=1e-5)
    - scheduler.step() per BATCH (không phải per-epoch)
    - Loss            : BCE with logits + pos_weight (từ metadata) + label_smoothing=0.05
    - AMP (FP16)      : torch.amp autocast + GradScaler
    - TF32            : matmul + cudnn
    - Early stop      : patience=5 trên val F1
    - Checkpoint:
        models/cnn_lstm_best_4_<mode>.pt  (best val F1)
        models/cnn_lstm_last_4_<mode>.pt  (mỗi epoch, để --resume)

Quan trọng (theo CLAUDE.md Known issues #1):
    - KHÔNG load test DataLoader trong train script — chạy `5. eval-model_4.py`
      riêng để eval test (Windows DataLoader workers crash sau nhiều epoch).
    - `persistent_workers=False` để tránh handle leak trên Windows.

Usage:
    .\\venv\\Scripts\\Activate.ps1

    # Random split — in-distribution metric
    python "scripts/4. train-model_4.py" --split-mode random

    # Domain split — out-of-distribution metric (con số "thật")
    python "scripts/4. train-model_4.py" --split-mode domain

    # Resume nếu bị crash
    python "scripts/4. train-model_4.py" --split-mode random --resume

    # Dry-run (chỉ build model, in param count, không train)
    python "scripts/4. train-model_4.py" --split-mode random --dry-run
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

# Force UTF-8 stdout cho Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT   = Path(r"D:\! secURLity")
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "model_4"
MODELS_ROOT    = PROJECT_ROOT / "models"
MODELS_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Hyperparameters (có thể override qua CLI)
# ============================================================================

# Batch size — MAX_LEN=256 + stacked BiLSTM nặng hơn model 3 (MAX_LEN=100,
# single-layer). Start với 2048; nếu GPU OOM thì giảm về 1024 hoặc 512.
DEFAULT_BATCH_SIZE = 2048
DEFAULT_EPOCHS     = 30
DEFAULT_LR         = 1e-3
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.05      # 5% steps đầu warmup
LR_MIN             = 1e-5      # cosine floor
LABEL_SMOOTHING    = 0.05      # Item 10
PATIENCE           = 5
DEFAULT_NUM_WORKERS = 4
SEED = 42

# Architecture (tensor-core-friendly — multiples of 8)
EMBED_DIM     = 64
CONV_CHANNELS = 128
LSTM_HIDDEN   = 128
LSTM_LAYERS   = 2          # Item 5
LSTM_DROPOUT  = 0.2        # inter-layer
FEAT_HIDDEN   = 64
FEAT_OUT      = 32
FEAT_DROPOUT  = 0.2
HEAD_HIDDEN   = 64
HEAD_DROPOUT  = 0.4

LOG_EVERY_N_BATCHES = 100


# ============================================================================
# Dataset
# ============================================================================

class URLDataset(Dataset):
    """Trả về (x_seq, x_feat, y) — phục vụ hybrid model.

    LAZY mmap pattern (cần thiết trên Windows):
      `__init__` CHỈ lưu file paths + length. Không mở mmap ở parent process
      vì:
        - Windows DataLoader dùng `spawn`, pickle Dataset gửi qua pipe đến
          worker → numpy.memmap KHÔNG serialize được (`OSError [Errno 22]
          Invalid argument when serializing numpy.memmap state`).
        - Lazy load: mỗi worker tự mmap lần đầu __getitem__ chạy, mỗi file
          mở 1 lần per-process → vẫn 0 copy 16GB vào RAM.
    """

    def __init__(self, X_path: Path, y_path: Path, feat_path: Path):
        self.X_path = str(X_path)
        self.y_path = str(y_path)
        self.feat_path = str(feat_path)
        # Đọc shape nhanh từ .npy header (KHÔNG load array)
        # np.load(mmap_mode="r") chỉ memory-map header + return ndarray view
        self._len = int(np.load(self.X_path, mmap_mode="r").shape[0])
        # mmap handles — chỉ open trong worker khi __getitem__ chạy lần đầu
        self._X = None
        self._y = None
        self._feat = None

    def _ensure_open(self) -> None:
        if self._X is None:
            self._X    = np.load(self.X_path,    mmap_mode="r")
            self._y    = np.load(self.y_path,    mmap_mode="r")
            self._feat = np.load(self.feat_path, mmap_mode="r")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx):
        self._ensure_open()
        return (
            torch.from_numpy(self._X[idx].astype(np.int64)),
            torch.from_numpy(self._feat[idx].astype(np.float32)),
            torch.tensor(float(self._y[idx]), dtype=torch.float32),
        )

    def __getstate__(self):
        """Loại bỏ mmap khỏi pickle state — workers tự reload."""
        state = self.__dict__.copy()
        state["_X"] = None
        state["_y"] = None
        state["_feat"] = None
        return state


# ============================================================================
# Model
# ============================================================================

class AttentionPool(nn.Module):
    """Pad-aware attention pool. Mask=False vị trí PAD."""

    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, H), mask: (B, L) bool — True = giữ, False = PAD
        scores = self.attn(x).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)  # (B, L)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)


class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM + lexical features."""

    def __init__(self, vocab_size: int, n_features: int, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx

        # Sequence path
        self.embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_idx)
        self.conv1 = nn.Conv1d(EMBED_DIM, CONV_CHANNELS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(CONV_CHANNELS, CONV_CHANNELS, kernel_size=5, padding=2)
        # MaxPool1d(2) twice → seq_len / 4
        self.lstm = nn.LSTM(
            input_size=CONV_CHANNELS,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        lstm_out_dim = 2 * LSTM_HIDDEN  # bidirectional
        self.attn_pool = AttentionPool(lstm_out_dim)

        # Feature path (Item 4)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, FEAT_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(FEAT_DROPOUT),
            nn.Linear(FEAT_HIDDEN, FEAT_OUT),
            nn.ReLU(inplace=True),
        )

        # Head
        head_in = lstm_out_dim + FEAT_OUT  # 256 + 32 = 288
        self.head = nn.Sequential(
            nn.Linear(head_in, HEAD_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, L) int64
        # x_feat: (B, n_features) float32

        # Pad mask cho attention (downsample qua 2 max-pool /2)
        pad_mask = (x_seq != self.pad_idx).float().unsqueeze(1)  # (B, 1, L)

        # Sequence path
        x = self.embed(x_seq)              # (B, L, E)
        x = x.transpose(1, 2)              # (B, E, L)
        x = F.relu(self.conv1(x))          # (B, C, L)
        x = F.max_pool1d(x, 2)             # (B, C, L/2)
        pad_mask = F.max_pool1d(pad_mask, 2)
        x = F.relu(self.conv2(x))          # (B, C, L/2)
        x = F.max_pool1d(x, 2)             # (B, C, L/4)
        pad_mask = F.max_pool1d(pad_mask, 2)
        x = x.transpose(1, 2)              # (B, L/4, C)

        # LSTM cần FP32 cho stability với AMP — autocast tự cast input nếu cần
        x, _ = self.lstm(x)                # (B, L/4, 2*H)

        # Attention pool với mask
        pad_mask_bool = pad_mask.squeeze(1).bool()  # (B, L/4)
        h_seq = self.attn_pool(x, mask=pad_mask_bool)  # (B, 2*H)

        # Feature path
        h_feat = self.feat_mlp(x_feat)  # (B, FEAT_OUT)

        # Concat + head
        h = torch.cat([h_seq, h_feat], dim=1)  # (B, 288)
        logit = self.head(h).squeeze(-1)       # (B,)
        return logit


# ============================================================================
# Loss (Item 10 — Label smoothing)
# ============================================================================

def bce_with_smoothing(logits: torch.Tensor, targets: torch.Tensor,
                       pos_weight: torch.Tensor,
                       smoothing: float = LABEL_SMOOTHING) -> torch.Tensor:
    """BCE with logits + pos_weight + label smoothing.

    targets ∈ {0, 1} → smoothed to {smoothing, 1 - smoothing}.
    Reason (Item 10): tránh model overconfident, calibrate probability tốt hơn
    cho 4-tier risk score downstream.
    """
    targets_smooth = targets * (1.0 - 2.0 * smoothing) + smoothing
    return F.binary_cross_entropy_with_logits(
        logits, targets_smooth, pos_weight=pos_weight
    )


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, threshold: float = 0.5):
    model.eval()
    all_probs = []
    all_y = []
    for x_seq, x_feat, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        x_feat = x_feat.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(x_seq, x_feat)
        # Sigmoid trên FP32 để tránh overflow ở extreme logits
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.numpy())

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_y).astype(int)
    y_pred = (probs >= threshold).astype(int)

    metrics = {
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, probs)),
        "pr_auc":    float(average_precision_score(y_true, probs)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]
    return metrics, probs, y_true


# ============================================================================
# Helpers
# ============================================================================

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def print_model_summary(model: CNNLSTM) -> None:
    total = count_params(model)
    print(f"\n  Param count by component:")
    for name, sub in [
        ("embed     ", model.embed),
        ("conv1     ", model.conv1),
        ("conv2     ", model.conv2),
        ("lstm      ", model.lstm),
        ("attn_pool ", model.attn_pool),
        ("feat_mlp  ", model.feat_mlp),
        ("head      ", model.head),
    ]:
        p = count_params(sub)
        pct = p / total * 100
        print(f"    {name}: {p:>10,}  ({pct:5.1f}%)")
    print(f"    {'TOTAL     '}: {total:>10,}  ({total/1e6:.2f}M)")


def smoke_test_forward(model: CNNLSTM, vocab_size: int, n_features: int,
                       max_len: int, device: torch.device) -> None:
    """Forward + backward dummy batch để kiểm tra shape + AMP OK."""
    print(f"\n  [smoke forward] B=4, L={max_len}, n_feat={n_features}")
    model.train()
    x_seq = torch.randint(0, vocab_size, (4, max_len), dtype=torch.long, device=device)
    x_feat = torch.randn(4, n_features, device=device)
    y = torch.randint(0, 2, (4,), dtype=torch.float32, device=device)

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    pos_w = torch.tensor(1.0, device=device)

    with autocast(device_type=device.type, enabled=use_amp):
        logits = model(x_seq, x_feat)
        loss = bce_with_smoothing(logits, y, pos_w)
    scaler.scale(loss).backward()
    print(f"  forward OK: logits.shape={tuple(logits.shape)}, "
          f"loss={loss.item():.4f}")


# ============================================================================
# Main
# ============================================================================

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-mode", choices=("random", "domain"), required=True,
                    help="Mode để train (preprocess đã tạo cả 2)")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    ap.add_argument("--resume", action="store_true",
                    help="Resume từ models/cnn_lstm_last_4_<mode>.pt")
    ap.add_argument("--no-amp", action="store_true",
                    help="Tắt AMP (FP16) — chỉ để debug NaN")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chỉ build model + in summary, không train")
    args = ap.parse_args(argv)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # TF32 (residual FP32 matmul)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # --------------------------------------------------------------------
    # Load metadata + feat_stats
    # --------------------------------------------------------------------
    metadata_path = PROCESSED_ROOT / "metadata.json"
    if not metadata_path.exists():
        print(f"[ERROR] {metadata_path} không tồn tại. Chạy `3. preprocess-data 4.py` trước.")
        sys.exit(2)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    MAX_LEN    = int(metadata["max_len"])
    VOCAB_SIZE = int(metadata["vocab_size"])
    PAD_IDX    = int(metadata["pad_idx"])
    mode_meta = metadata["modes"].get(args.split_mode)
    if mode_meta is None:
        print(f"[ERROR] Mode '{args.split_mode}' không có trong metadata.json")
        sys.exit(2)
    pos_weight = float(mode_meta["pos_weight"])

    feat_stats_path = PROCESSED_ROOT / args.split_mode / "feat_stats.json"
    if not feat_stats_path.exists():
        print(f"[ERROR] {feat_stats_path} không tồn tại. Chạy "
              f"`4-feat. extract-lexical_4.py` trước.")
        sys.exit(2)
    feat_stats = json.loads(feat_stats_path.read_text(encoding="utf-8"))
    n_features = int(feat_stats["n_features"])

    print(f"==================================================================")
    print(f"Train model 4  ·  split-mode = {args.split_mode}")
    print(f"==================================================================")
    print(f"  MAX_LEN         = {MAX_LEN}")
    print(f"  VOCAB_SIZE      = {VOCAB_SIZE}")
    print(f"  PAD_IDX         = {PAD_IDX}")
    print(f"  n_features      = {n_features}")
    print(f"  pos_weight      = {pos_weight:.4f}  (= benign/mal ratio TRAIN)")
    print(f"  label_smoothing = {LABEL_SMOOTHING}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  epochs (max)    = {args.epochs}")
    print(f"  lr (peak)       = {args.lr}")
    print(f"  weight_decay    = {WEIGHT_DECAY}")
    print(f"  warmup ratio    = {WARMUP_RATIO}")
    print(f"  patience        = {args.patience}")
    print(f"  num_workers     = {args.num_workers}")

    # --------------------------------------------------------------------
    # Build model
    # --------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  device          = {device}")
    if device.type == "cuda":
        print(f"  GPU             = {torch.cuda.get_device_name(0)}")
        print(f"  GPU mem total   = {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model = CNNLSTM(VOCAB_SIZE, n_features, pad_idx=PAD_IDX).to(device)
    print_model_summary(model)

    # Smoke forward
    smoke_test_forward(model, VOCAB_SIZE, n_features, MAX_LEN, device)

    if args.dry_run:
        print("\n  [DRY-RUN] Build + smoke OK. Exit.")
        return

    # --------------------------------------------------------------------
    # Build datasets (lazy mmap — chỉ store paths, mở trong workers)
    # --------------------------------------------------------------------
    mode_root = PROCESSED_ROOT / args.split_mode

    def split_paths(split: str):
        d = mode_root / split
        for fname in ("X.npy", "y.npy", "feat.npy"):
            p = d / fname
            if not p.exists():
                print(f"[ERROR] Thiếu {p}. Chạy lại preprocess + lexical extract.")
                sys.exit(2)
        return d / "X.npy", d / "y.npy", d / "feat.npy"

    print(f"\n  Building datasets (lazy mmap — workers tự mở) ...")
    t0 = time.time()
    train_ds = URLDataset(*split_paths("train"))
    val_ds   = URLDataset(*split_paths("val"))
    print(f"    train: N={len(train_ds):,}")
    print(f"    val  : N={len(val_ds):,}")
    print(f"    ready in {time.time()-t0:.1f}s")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        # persistent_workers=False — tránh handle leak Windows (CLAUDE.md #1)
        persistent_workers=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    n_train_batches = len(train_loader)
    print(f"    train batches = {n_train_batches:,}")

    # --------------------------------------------------------------------
    # Optimizer + Scheduler (Item 9)
    # --------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    total_steps  = args.epochs * n_train_batches
    warmup_steps = max(int(WARMUP_RATIO * total_steps), 1)
    print(f"\n  Scheduler: warmup({warmup_steps}) → cosine({total_steps-warmup_steps})")
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                            total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=total_steps - warmup_steps,
                                     eta_min=LR_MIN)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    use_amp = (not args.no_amp) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    pos_w = torch.tensor(pos_weight, device=device, dtype=torch.float32)

    # --------------------------------------------------------------------
    # Checkpoint paths
    # --------------------------------------------------------------------
    best_ckpt_path = MODELS_ROOT / f"cnn_lstm_best_4_{args.split_mode}.pt"
    last_ckpt_path = MODELS_ROOT / f"cnn_lstm_last_4_{args.split_mode}.pt"

    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    if args.resume and last_ckpt_path.exists():
        print(f"\n  Resuming from {last_ckpt_path.name}")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = float(ckpt.get("best_f1", 0.0))
        patience_counter = int(ckpt.get("patience_counter", 0))
        print(f"    Resumed: epoch {start_epoch}, best_f1={best_f1:.4f}, "
              f"patience {patience_counter}/{args.patience}")
    elif args.resume:
        print(f"  [WARN] --resume nhưng {last_ckpt_path.name} không tồn tại — "
              f"train từ đầu.")

    # --------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------
    print(f"\n==================================================================")
    print(f"Start training")
    print(f"==================================================================")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_epoch = time.time()
        running_loss = 0.0
        running_count = 0
        t_log = time.time()
        seen_log = 0

        for i, (x_seq, x_feat, y) in enumerate(train_loader):
            x_seq  = x_seq.to(device,  non_blocking=True)
            x_feat = x_feat.to(device, non_blocking=True)
            y      = y.to(device,      non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(x_seq, x_feat)
                loss = bce_with_smoothing(logits, y, pos_w,
                                          smoothing=LABEL_SMOOTHING)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # per-batch (Item 9.4)

            running_loss  += loss.item() * x_seq.size(0)
            running_count += x_seq.size(0)
            seen_log      += x_seq.size(0)

            if (i + 1) % LOG_EVERY_N_BATCHES == 0:
                dt = time.time() - t_log
                rate = seen_log / max(dt, 1e-9)
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  ep {epoch+1:02d} [{i+1:>5}/{n_train_batches}]  "
                      f"loss={loss.item():.4f}  "
                      f"lr={lr_now:.2e}  "
                      f"{rate:>6,.0f} URLs/s",
                      flush=True)
                t_log = time.time()
                seen_log = 0

        avg_loss = running_loss / max(running_count, 1)
        train_time = time.time() - t_epoch

        # ----- Validation -----
        t_val = time.time()
        val_metrics, val_probs, val_y_true = evaluate(model, val_loader,
                                                      device, use_amp=use_amp)
        val_time = time.time() - t_val

        print(f"\n  === Epoch {epoch+1}/{args.epochs} === "
              f"({train_time:.0f}s train + {val_time:.0f}s val)")
        print(f"    train loss   = {avg_loss:.4f}")
        print(f"    val F1       = {val_metrics['f1']:.4f}")
        print(f"    val P / R    = {val_metrics['precision']:.4f} / "
              f"{val_metrics['recall']:.4f}")
        print(f"    val ROC-AUC  = {val_metrics['roc_auc']:.4f}")
        print(f"    val PR-AUC   = {val_metrics['pr_auc']:.4f}")
        cm = val_metrics["confusion_matrix"]
        print(f"    val ConfMat  = TN={cm[0][0]:>9,}  FP={cm[0][1]:>9,}  "
              f"FN={cm[1][0]:>9,}  TP={cm[1][1]:>9,}")

        # Calibration sanity (Item 10.4)
        hist_buckets = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        hist, _ = np.histogram(val_probs, bins=hist_buckets)
        print(f"    val prob hist (bins {hist_buckets}):")
        print(f"      {hist.tolist()}  (target: spread, NOT U-shape)")

        # ----- Checkpoint -----
        improved = val_metrics["f1"] > best_f1 + 1e-6
        if improved:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "val_metrics": val_metrics,
            "train_loss": avg_loss,
            "best_f1": best_f1,
            "patience_counter": patience_counter,
            "config": {
                "max_len": MAX_LEN,
                "vocab_size": VOCAB_SIZE,
                "pad_idx": PAD_IDX,
                "n_features": n_features,
                "split_mode": args.split_mode,
                "label_smoothing": LABEL_SMOOTHING,
                "pos_weight": pos_weight,
                "embed_dim": EMBED_DIM,
                "conv_channels": CONV_CHANNELS,
                "lstm_hidden": LSTM_HIDDEN,
                "lstm_layers": LSTM_LAYERS,
                "feat_out": FEAT_OUT,
                "head_hidden": HEAD_HIDDEN,
            },
        }
        torch.save(ckpt, last_ckpt_path)

        if improved:
            torch.save(ckpt, best_ckpt_path)
            print(f"    ⭐ New best F1={best_f1:.4f} → saved "
                  f"{best_ckpt_path.name}")
        else:
            print(f"    No improvement (patience {patience_counter}/"
                  f"{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n  ⏹  Early stopping at epoch {epoch+1}.")
                break

    # --------------------------------------------------------------------
    print(f"\n==================================================================")
    print(f"  DONE. Best val F1 = {best_f1:.4f}")
    print(f"  Best ckpt: {best_ckpt_path}")
    print(f"  Last ckpt: {last_ckpt_path}")
    print(f"==================================================================")
    print(f"\n  Next: chạy `5. eval-model_4.py --split-mode {args.split_mode}` ")
    print(f"  để eval TEST set (KHÔNG eval ở đây — tránh Windows DataLoader")
    print(f"  crash sau nhiều epoch, xem CLAUDE.md Known issues #1).")


if __name__ == "__main__":
    main()

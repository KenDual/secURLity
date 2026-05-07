"""
Synthetic Malicious URL Generator (Char-RNN)
=============================================

Học pattern từ ~413k malicious URL thực (dataset/malicious_dataset.csv) và sinh
ra 5-6 triệu URL độc hại synthetic giữ được đặc tính cấu trúc của dữ liệu thực.

Kiến trúc: Embedding -> 2-layer LSTM -> Dropout -> Linear (softmax over chars).

Pipeline khớp downstream CNN-LSTM (`scripts/3. preprocess-data.py`):
- URLs được lowercase trước khi train Char-RNN (giống preprocess).
- Vocab cố định = ALLOWED_CHARS của CNN-LSTM + {PAD, SOS, EOS} (52 tokens).
- MAX_URL_LEN=100 (P99=83 theo EDA; preprocess cũng cắt ở 100).
=> URL synthetic sinh ra chỉ chứa ký tự hợp lệ → preprocess không tạo UNK noise.

Pipeline:
    python "scripts/1. url-generator (model 2).py" --mode train
    python "scripts/1. url-generator (model 2).py" --mode generate --count 6000000 --temperature 0.8
    python "scripts/1. url-generator (model 2).py" --mode both   --count 100000 --temperature 0.8

Chọn Tem tầm 0.8 vì tem mức đó cho ra URL đủ đa dạng và cũng sát với thực tế. okela nhé             o(*￣▽￣*)ブ
"""

import argparse
import csv
import json
import math
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\! secURLity")
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATASET_DIR / "malicious_dataset.csv"
CKPT_PATH = MODELS_DIR / "char_rnn_url_generator.pt"
VOCAB_PATH = MODELS_DIR / "char_rnn_vocab.json"
DEFAULT_OUTPUT = DATASET_DIR / "synthetic_malicious_urls.csv"

# Special tokens (single, unused-in-URL chars so tokenization is unambiguous)
PAD = "\x00"
SOS = "\x01"   # start of sequence
EOS = "\x02"   # end of sequence

# Khớp 1-1 với CNN-LSTM vocab (`scripts/3. preprocess-data.py:63`)
# 26 + 10 + 13 = 49 chars. Char-RNN vocab = PAD + SOS + EOS + 49 = 52 tokens.
ALLOWED_CHARS = string.ascii_lowercase + string.digits + "/:.-_?=&#@%+~"
ALLOWED_SET = set(ALLOWED_CHARS)

MAX_URL_LEN = 100          # khớp MAX_LEN của preprocess (P99=83)
MAX_SEQ_LEN = MAX_URL_LEN + 2   # +SOS, +EOS

# Hyperparameters (tuned for ~413k URLs / generation 5-6M scale)
# Char-RNN chạy LSTM trên FULL seq (khác CNN-LSTM có MaxPool downsample)
# nên model phải gọn hơn để đạt tốc độ tương đương.
EMBED_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 1             # 1 layer đủ cho vocab 52, char-level pattern đơn giản
DROPOUT = 0.2
BATCH_SIZE = 4096          # GTX 1660 SUPER 6GB: model nhỏ hơn → batch lớn hơn được
EPOCHS = 15                # bù lại model nhẹ → cần thêm vài epoch để hội tụ
LR = 2e-3

# Generation defaults
GEN_BATCH_SIZE = 4096      # sinh song song trên GPU
WRITE_CHUNK = 100_000      # flush CSV mỗi 100k dòng

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# A. Data Processing
# ---------------------------------------------------------------------------
def preprocess_data(csv_path: Path):
    """Đọc CSV malicious URLs, lowercase + filter về ALLOWED_CHARS, encode numpy.

    Vocab cố định khớp CNN-LSTM downstream để URL synthetic sinh ra không
    sinh ký tự sẽ bị map về <UNK> ở pipeline preprocess.
    """
    print(f"[preprocess] Reading {csv_path}")
    df = pd.read_csv(csv_path, usecols=["url"])
    urls = df["url"].dropna().astype(str).str.strip().str.lower()
    urls = urls[urls.str.len() > 0].tolist()
    print(f"[preprocess] Loaded {len(urls):,} URLs (lowercased)")

    # Filter ký tự về ALLOWED_SET + truncate. Drop URL còn rỗng / quá ngắn.
    cleaned = []
    dropped = 0
    for u in urls:
        u = "".join(ch for ch in u if ch in ALLOWED_SET)
        if len(u) < 6:
            dropped += 1
            continue
        cleaned.append(u[:MAX_URL_LEN])
    print(f"[preprocess] After char-filter: {len(cleaned):,} URLs (dropped {dropped:,})")

    # Vocab cố định: [PAD, SOS, EOS] + ALLOWED_CHARS
    itos = [PAD, SOS, EOS] + list(ALLOWED_CHARS)
    stoi = {ch: i for i, ch in enumerate(itos)}
    vocab_size = len(itos)
    print(f"[preprocess] Vocab size: {vocab_size} (fixed, matches CNN-LSTM)")

    # Encode song song với numpy
    pad_id = stoi[PAD]
    sos_id = stoi[SOS]
    eos_id = stoi[EOS]

    encoded = np.full((len(cleaned), MAX_SEQ_LEN), pad_id, dtype=np.int16)
    for i, u in enumerate(cleaned):
        encoded[i, 0] = sos_id
        for j, ch in enumerate(u):
            encoded[i, j + 1] = stoi[ch]
        encoded[i, len(u) + 1] = eos_id

    # Save vocab cho generate mode
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({"itos": itos, "stoi": stoi}, f, ensure_ascii=False)
    print(f"[preprocess] Vocab saved to {VOCAB_PATH}")

    return encoded, itos, stoi


# ---------------------------------------------------------------------------
# B. Model
# ---------------------------------------------------------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)            # (B, L, E)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)              # (B, L, V)
        return logits, hidden


def build_model(vocab_size: int) -> CharRNN:
    model = CharRNN(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Params: {n_params/1e6:.2f}M | device={device}")
    return model


# ---------------------------------------------------------------------------
# C. Training
# ---------------------------------------------------------------------------
def _build_bucketed_batches(lengths_np: np.ndarray, batch_size: int):
    """Sort indices by length, chia thành batch consecutive, shuffle batch order.

    Mỗi batch chứa URL có độ dài tương tự nhau → trim padding hiệu quả.
    Trả về list[(idx_tensor, max_len_in_batch)] đã sẵn sàng cho training loop.
    """
    order = np.argsort(lengths_np, kind="stable")  # ascending
    n = len(order)
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = order[start:end]
        max_len = int(lengths_np[idx].max())
        batches.append((idx, max_len))
    return batches


def train_model(encoded: np.ndarray, vocab_size: int, epochs: int = EPOCHS):
    """Tối ưu cho GTX 1660 SUPER: load all data lên GPU + length bucketing.

    - Bỏ DataLoader (num_workers overhead trên Windows).
    - Mỗi batch chỉ pad đến max length THỰC trong batch (không phải MAX_SEQ_LEN).
    - Loss/token accumulate trên GPU, .item() chỉ 1 lần/epoch.
    """
    pad_id = 0
    n_total = len(encoded)
    n_val = max(1000, n_total // 20)
    n_train = n_total - n_val

    # Random 95/5 split (deterministic)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # Tính length thực (số token != PAD) cho bucketing
    lengths = (encoded != pad_id).sum(axis=1).astype(np.int32)
    train_lengths = lengths[train_idx]
    val_lengths = lengths[val_idx]

    # Move all data lên GPU một lần (int16 ≈ 84MB cho 413k * 102)
    train_data = torch.from_numpy(encoded[train_idx]).to(device)
    val_data = torch.from_numpy(encoded[val_idx]).to(device)
    print(f"[train] Train={n_train:,} | Val={n_val:,} | "
          f"Avg len: train={train_lengths.mean():.1f}, val={val_lengths.mean():.1f}")
    if device.type == "cuda":
        print(f"[train] GPU mem after load: "
              f"{torch.cuda.memory_allocated()/1e6:.0f} MB")

    # Pre-compute val batches once (val không shuffle)
    val_batches = _build_bucketed_batches(val_lengths, BATCH_SIZE * 2)

    model = build_model(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # khớp train-model.py
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_val = math.inf
    for epoch in range(1, epochs + 1):
        # ---- Train
        model.train()
        train_batches = _build_bucketed_batches(train_lengths, BATCH_SIZE)
        rng.shuffle(train_batches)  # shuffle batch order

        loss_sum = torch.zeros(1, device=device)
        tok_sum = torch.zeros(1, device=device)
        pbar = tqdm(train_batches, desc=f"Epoch {epoch}/{epochs}")
        for bi, (idx_np, max_len) in enumerate(pbar):
            # Slice batch trên GPU, trim đến max_len thực + cast int16->int64
            idx = torch.from_numpy(idx_np).to(device)
            batch = train_data.index_select(0, idx)[:, :max_len].long()
            x = batch[:, :-1]
            y = batch[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                n_tok = (y != pad_id).sum()
                loss_sum += loss.detach() * n_tok
                tok_sum += n_tok

            if bi % 50 == 0:
                pbar.set_postfix(loss=f"{(loss_sum/tok_sum.clamp(min=1)).item():.4f}")

        train_loss = (loss_sum / tok_sum.clamp(min=1)).item()

        # ---- Val
        model.eval()
        v_loss = torch.zeros(1, device=device)
        v_tok = torch.zeros(1, device=device)
        with torch.no_grad():
            for idx_np, max_len in val_batches:
                idx = torch.from_numpy(idx_np).to(device)
                batch = val_data.index_select(0, idx)[:, :max_len].long()
                x = batch[:, :-1]
                y = batch[:, 1:]
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                n_tok = (y != pad_id).sum()
                v_loss += loss * n_tok
                v_tok += n_tok
        val_loss = (v_loss / v_tok.clamp(min=1)).item()
        ppl = math.exp(min(20, val_loss))
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ppl={ppl:.2f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "epoch": epoch,
                "val_loss": val_loss,
            }, CKPT_PATH)
            print(f"[checkpoint] Saved best to {CKPT_PATH} (val_loss={val_loss:.4f})")

    print(f"[train] Done. Best val_loss={best_val:.4f}")


# ---------------------------------------------------------------------------
# D. Generation
# ---------------------------------------------------------------------------
def _load_for_inference():
    """Load vocab + model checkpoint. Trả về (model, itos, stoi)."""
    if not VOCAB_PATH.exists() or not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Cần train trước. Thiếu {VOCAB_PATH} hoặc {CKPT_PATH}."
        )
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    itos = vocab["itos"]
    stoi = {ch: i for i, ch in enumerate(itos)}

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = CharRNN(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=0.0,  # không dropout khi inference
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[generate] Loaded checkpoint epoch={ckpt.get('epoch','?')} "
          f"val_loss={ckpt.get('val_loss','?'):.4f}")
    return model, itos, stoi


@torch.no_grad()
def _generate_batch(model: CharRNN, stoi: dict, itos: list,
                    batch_size: int, temperature: float, prefix: str,
                    max_len: int = MAX_URL_LEN) -> list:
    """Sinh `batch_size` URL song song.

    1) Khởi tạo hidden = None, feed prefix qua model -> warm hidden state.
    2) Lặp: lấy logits ký tự cuối, chia temperature, sampling đa thức.
    3) Dừng khi tất cả seq emit EOS hoặc đạt max_len.
    """
    pad_id = stoi[PAD]
    sos_id = stoi[SOS]
    eos_id = stoi[EOS]

    # Warm-up: feed SOS + prefix chars để có hidden state ban đầu
    warm_tokens = [sos_id] + [stoi[c] for c in prefix if c in stoi]
    warm = torch.tensor([warm_tokens] * batch_size, dtype=torch.long, device=device)
    logits, hidden = model(warm)
    last_logits = logits[:, -1, :]  # (B, V)

    # Lưu các ký tự đã sinh (chưa kể prefix). prefix sẽ được prepend khi decode.
    generated = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    cur_logits = last_logits
    for _ in range(max_len - len(prefix)):
        # Cấm sinh PAD và SOS
        cur_logits[:, pad_id] = -float("inf")
        cur_logits[:, sos_id] = -float("inf")

        # Temperature sampling
        scaled = cur_logits / max(1e-6, temperature)
        probs = F.softmax(scaled, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

        # Với seq đã finished, ép về PAD để không append nữa
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)

        # Update finished mask
        just_finished = (next_tok == eos_id) & (~finished)
        finished = finished | just_finished | (next_tok == eos_id)

        # Lưu char (CPU)
        next_cpu = next_tok.detach().cpu().tolist()
        for i, t in enumerate(next_cpu):
            if t != pad_id and t != eos_id:
                generated[i].append(t)

        if finished.all():
            break

        # Step tiếp: feed token vừa sinh
        step_in = next_tok.unsqueeze(1)  # (B, 1)
        logits, hidden = model(step_in, hidden)
        cur_logits = logits[:, -1, :]

    # Decode -> string, prepend prefix
    urls = []
    for toks in generated:
        s = "".join(itos[t] for t in toks)
        urls.append(prefix + s)
    return urls


def generate_urls(count: int, temperature: float = 0.8,
                  prefix: str = "", prefix_mix: bool = True,
                  output_path: Path = DEFAULT_OUTPUT,
                  dedup_against_input: bool = True,
                  gen_batch_size: int = GEN_BATCH_SIZE):
    """Sinh `count` URL malicious synthetic, ghi streaming vào CSV.

    prefix_mix=True (mặc định): random-pick "https://" / "http://" / "" theo
    tỉ lệ ~85/13/2 để khớp distribution thực (87% HTTPS theo APWG Q4 2024).
    """
    model, itos, stoi = _load_for_inference()

    # Set seen từ dataset gốc để dedup
    seen = set()
    if dedup_against_input and INPUT_CSV.exists():
        print(f"[generate] Loading original URLs for dedup...")
        df = pd.read_csv(INPUT_CSV, usecols=["url"])
        seen = set(df["url"].dropna().astype(str).str.strip().tolist())
        print(f"[generate] Dedup set: {len(seen):,} URLs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[generate] Target={count:,} | T={temperature} | batch={gen_batch_size}")
    print(f"[generate] Output -> {output_path}")

    written = 0
    duplicates = 0
    invalid = 0
    buffer = []
    t0 = time.time()

    rng = np.random.default_rng(0xC0FFEE)
    PREFIX_CHOICES = ["https://", "http://", ""]
    PREFIX_PROBS = [0.85, 0.13, 0.02]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])

        pbar = tqdm(total=count, desc="Generating", unit="url")
        while written < count:
            # Chọn prefix cho batch (đồng nhất trong 1 batch để tận dụng warm-up chung)
            if prefix:
                cur_prefix = prefix
            elif prefix_mix:
                cur_prefix = str(rng.choice(PREFIX_CHOICES, p=PREFIX_PROBS))
            else:
                cur_prefix = ""

            urls = _generate_batch(
                model, stoi, itos,
                batch_size=gen_batch_size,
                temperature=temperature,
                prefix=cur_prefix,
            )

            for u in urls:
                if written >= count:
                    break
                # Quality filters cơ bản
                if len(u) < 6 or " " in u or "\n" in u or "\r" in u:
                    invalid += 1
                    continue
                if u in seen:
                    duplicates += 1
                    continue
                seen.add(u)
                buffer.append((u, 1))
                written += 1
                pbar.update(1)

            # Flush buffer định kỳ
            if len(buffer) >= WRITE_CHUNK:
                writer.writerows(buffer)
                buffer.clear()

        if buffer:
            writer.writerows(buffer)

        pbar.close()

    elapsed = time.time() - t0
    rate = written / max(1e-6, elapsed)
    print(f"[generate] Done. Written={written:,} | dup_skipped={duplicates:,} "
          f"| invalid_skipped={invalid:,} | {elapsed/60:.1f} min ({rate:.0f} url/s)")


# ---------------------------------------------------------------------------
# E. Main / CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Char-RNN synthetic malicious URL generator")
    parser.add_argument("--mode", choices=["train", "generate", "both"], default="both")
    parser.add_argument("--count", type=int, default=6_000_000,
                        help="Số URL synthetic cần sinh (mode generate/both)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature sampling (0.5 ổn định, 1.0 đa dạng)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Cố định prefix (vd 'https://'). Trống = mix theo phân bố.")
    parser.add_argument("--no-prefix-mix", action="store_true",
                        help="Tắt mix prefix HTTP/HTTPS, để model tự sinh từ SOS.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--gen-batch", type=int, default=GEN_BATCH_SIZE)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-dedup", action="store_true",
                        help="Bỏ qua dedup so với dataset gốc (tăng tốc).")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        encoded, itos, _ = preprocess_data(INPUT_CSV)
        train_model(encoded, vocab_size=len(itos), epochs=args.epochs)

    if args.mode in ("generate", "both"):
        generate_urls(
            count=args.count,
            temperature=args.temperature,
            prefix=args.prefix,
            prefix_mix=not args.no_prefix_mix,
            output_path=Path(args.output),
            dedup_against_input=not args.no_dedup,
            gen_batch_size=args.gen_batch,
        )


if __name__ == "__main__":
    main()

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import argparse

# === Paths (must match training) ===
PROJECT_ROOT = Path(r"D:\! secURLity")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "cnn_lstm_best (model 1).pt"
METADATA_PATH = DATA_DIR / "metadata.json"
VOCAB_PATH = DATA_DIR / "vocab.json"

# === Hyperparameters (must match training) ===
MAX_LEN = 100
EMBED_DIM = 64
CONV_CHANNELS = 128
LSTM_HIDDEN = 128
DROPOUT = 0.4
PAD_IDX = 0


def risk_level(score: float) -> str:
    pct = score * 100
    if pct <= 25: return "Low Risk"
    if pct <= 50: return "Caution"
    if pct <= 75: return "Suspicious"
    return "High Risk"


class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, conv_channels, lstm_hidden, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=0)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h).squeeze(1)


def load_vocab(path: Path):
    """Load vocab.json - supports dict {char: int} or list [char, char, ...]."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        char2idx = {k: int(v) for k, v in data.items()}
    elif isinstance(data, list):
        char2idx = {c: i for i, c in enumerate(data)}
    else:
        raise ValueError(f"Unsupported vocab format: {type(data)}")
    unk_idx = None
    for k in ('<UNK>', '<unk>', 'UNK', 'unk', '<OOV>', '[UNK]'):
        if k in char2idx:
            unk_idx = char2idx[k]
            break
    return char2idx, unk_idx


def encode_url(url: str, char2idx: dict, unk_idx, max_len: int = MAX_LEN) -> np.ndarray:
    url = url.strip().lower()
    fallback = unk_idx if unk_idx is not None else PAD_IDX
    encoded = [char2idx.get(c, fallback) for c in url[:max_len]]
    if len(encoded) < max_len:
        encoded += [PAD_IDX] * (max_len - len(encoded))
    return np.array(encoded, dtype=np.int64)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    char2idx, unk_idx = load_vocab(VOCAB_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    print("=== Diagnostics ===")
    print(f"Device         : {device}")
    print(f"Vocab size     : {len(char2idx)} (metadata: {meta['vocab_size']})")
    print(f"PAD index      : {PAD_IDX}")
    print(f"UNK index      : {unk_idx if unk_idx is not None else 'NOT FOUND -> fallback to PAD'}")
    print(f"First 10 vocab : {list(char2idx.items())[:10]}")
    if len(char2idx) != meta['vocab_size']:
        print("WARNING: vocab size mismatch with metadata.json")

    model = CNNLSTM(meta['vocab_size'], EMBED_DIM, CONV_CHANNELS, LSTM_HIDDEN, DROPOUT).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Checkpoint     : epoch {checkpoint['epoch']}, Val F1={checkpoint['best_f1']:.4f}")
    print("===================\n")

    @torch.no_grad()
    def predict(url: str) -> float:
        x = torch.from_numpy(encode_url(url, char2idx, unk_idx)).unsqueeze(0).to(device)
        return torch.sigmoid(model(x)).item()

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('urls', nargs='*', help='URLs to scan')
        parser.add_argument('--file', '-f', help='Path to .txt file (one URL per line)')
        args = parser.parse_args()

        urls_to_scan = []
        if args.file:
            with open(args.file, encoding='utf-8') as f:
                urls_to_scan = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(urls_to_scan)} URLs from {args.file}\n")
        elif args.urls:
            urls_to_scan = args.urls

        if urls_to_scan:
            for url in urls_to_scan:
                s = predict(url)
                print(f"[{risk_level(s):12s}] {s*100:6.2f}%  {url}")
        else:
            print("Interactive mode. Enter URL (or 'exit'):\n")
            while True:
                try:
                    url = input("URL > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print(); break
                if not url or url.lower() in ('exit', 'quit'):
                    break
                s = predict(url)
                print(f"Score: {s*100:6.2f}%  ->  {risk_level(s)}\n")


if __name__ == '__main__':
    main()
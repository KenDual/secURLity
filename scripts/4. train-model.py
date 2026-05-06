"""
Fast CNN-LSTM Training with Mixed Precision
Uses pre-encoded numpy arrays from preprocess.py
Expected: ~2-3 minutes per epoch (vs 1 hour with old script)
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(r"D:\! secURLity")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 2048
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
EMBED_DIM = 64
CONV_CHANNELS = 128
LSTM_HIDDEN = 128
DROPOUT = 0.4
NUM_WORKERS = 4

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# Load metadata
with open(DATA_DIR / "metadata.json") as f:
    meta = json.load(f)

print(f"\nDataset: Train={meta['train_size']:,} | Val={meta['val_size']:,} | Test={meta['test_size']:,}")
print(f"Vocab size: {meta['vocab_size']}")
print(f"Class imbalance ratio: {meta['pos_weight']:.2f}")

# Dataset class for pre-encoded data
class PreEncodedDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).long(),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

# Model
class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, conv_channels, lstm_hidden, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # CNN layers
        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        # LSTM
        self.lstm = nn.LSTM(
            conv_channels, lstm_hidden,
            num_layers=1, batch_first=True,
            bidirectional=True, dropout=0
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Embedding
        x = self.embed(x)  # [B, L, E]
        
        # CNN
        x = x.permute(0, 2, 1)  # [B, E, L]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # LSTM
        x = x.permute(0, 2, 1)  # [B, L', C]
        _, (h, _) = self.lstm(x)
        
        # Concat bidirectional hidden states
        h = torch.cat([h[-2], h[-1]], dim=1)  # [B, 2H]
        
        # Classifier
        return self.fc(h).squeeze(1)

# DataLoaders
print("\nLoading datasets...")
train_dataset = PreEncodedDataset(DATA_DIR / "train_X.npy", DATA_DIR / "train_y.npy")
val_dataset = PreEncodedDataset(DATA_DIR / "val_X.npy", DATA_DIR / "val_y.npy")
test_dataset = PreEncodedDataset(DATA_DIR / "test_X.npy", DATA_DIR / "test_y.npy")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)

# Model setup
model = CNNLSTM(
    vocab_size=meta['vocab_size'],
    embed_dim=EMBED_DIM,
    conv_channels=CONV_CHANNELS,
    lstm_hidden=LSTM_HIDDEN,
    dropout=DROPOUT
).to(device)

# Loss with pos_weight for imbalanced data
pos_weight = torch.tensor([meta['pos_weight']]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training function
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader.dataset)

# Validation function
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for X, y in tqdm(loader, desc="Evaluating", leave=False):
        X = X.to(device)
        logits = model(X)
        preds = (torch.sigmoid(logits) > 0.5).cpu().int().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.int().numpy())
    return np.array(all_labels), np.array(all_preds)


# === MAIN EXECUTION - Must be wrapped for Windows ===
if __name__ == '__main__':
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_labels, val_preds = evaluate(model, val_loader)
        val_f1 = f1_score(val_labels, val_preds)
        
        print(f"Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, MODELS_DIR / "cnn_lstm_best.pt")
            print(f"✓ New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model and test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    checkpoint = torch.load(MODELS_DIR / "cnn_lstm_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (Val F1={checkpoint['best_f1']:.4f})")
    
    test_labels, test_preds = evaluate(model, test_loader)
    
    print("\n" + classification_report(
        test_labels, test_preds,
        target_names=['Benign', 'Malicious'],
        digits=4
    ))
    
    # Save final results
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_val_f1': float(checkpoint['best_f1']),
        'test_report': classification_report(test_labels, test_preds, output_dict=True)
    }
    
    with open(MODELS_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {MODELS_DIR / 'results.json'}")
    print("Training complete!")
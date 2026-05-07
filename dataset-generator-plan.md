# Synthetic Malicious URL Generator — Design Document

Module: `scripts/1. url-generator (model 2).py`
Mục tiêu: học pattern từ ~413k malicious URL thật (`dataset/malicious_dataset.csv`)
và sinh thêm ~5-6 triệu URL độc hại synthetic dùng cho training CNN-LSTM
trong dự án **secURLity**.

---

## Phần 1 — Ý tưởng

### 1.1. Bối cảnh & động cơ

Pipeline phát hiện URL độc hại của secURLity yêu cầu một dataset cân bằng quy mô
~10 triệu URL (85% benign / 15% malicious). Trên thực tế chỉ thu thập được
**413,319 malicious URLs thật** (URLhaus, PhishTank, OpenPhish), còn cách đích
~1.1 triệu mẫu — quá ít để CNN-LSTM khái quát hoá tốt.

**Lựa chọn giải pháp.** Có ba hướng tiếp cận để tăng số lượng:

| Hướng | Ưu | Nhược |
|---|---|---|
| Rule-based (regex template) | Nhanh, kiểm soát được | Pattern đơn điệu, dễ bị model học shortcut |
| GAN / Diffusion text | Đa dạng, sát phân phối | Quá nặng cho char-level URL, training khó hội tụ |
| **Char-RNN (LSTM) — đã chọn** | **Học pattern char-level trực tiếp từ data thật, sinh nhanh, đủ đa dạng** | Cần điều chỉnh temperature để cân bằng chất lượng |

### 1.2. Vì sao Char-RNN phù hợp với URL

URL **không có ngữ pháp tự nhiên** như văn bản → không cần BPE/word tokenizer.
Đặc trưng malicious lại nằm chính ở **mức ký tự**:

- **DGA domain**: chuỗi ký tự ngẫu nhiên có entropy cao (`hidok4f8zl.firebaseapp.com`).
- **Typosquatting**: thay 1-2 ký tự brand thật (`paypaletroi.it`, `www.paypal.cgi.google.com`).
- **IP-based C2/botnet**: `http://104.95.253.227:15089/mozi.m`.
- **Hosting abuse**: subdomain ngẫu nhiên trên platform thật (`firebaseapp.com`,
  `web.app`, `weeblysite.com`, `pastebin.com/raw/...`).
- **CMS compromise**: query phổ biến `?option=com_content&view=article&id=X&itemid=Y`
  của Joomla bị chiếm.

Char-RNN học toàn bộ tín hiệu này tự động từ data, không cần feature engineering.

### 1.3. Triết lý thiết kế

**(a) "Match downstream" — output sẵn sàng dùng.**
Generator được thiết kế để URL sinh ra **không cần preprocess lại** trước khi
đưa vào CNN-LSTM. Cụ thể:
- Vocab ký tự khớp 1-1 với `scripts/3. preprocess-data.py`.
- URL được lowercase + lọc về tập ký tự CNN-LSTM hỗ trợ.
- Độ dài cắt ở 100 chars (khớp `MAX_LEN` của preprocess).

→ Loại bỏ hoàn toàn UNK noise khi đưa qua pipeline downstream.

**(b) Diversity có kiểm soát — Temperature sampling.**
Không dùng argmax (sẽ tạo collapse vào pattern lặp); thay bằng multinomial
sampling từ phân phối đã chia temperature *T*. Thử nghiệm A/B trên 5k mẫu:

| T | Đặc tính | Kết luận |
|---|---|---|
| 0.5 | URL bảo thủ, ~80% theo template `www.X.com/index.php`, thiếu đa dạng | Loại — model dễ overfit shortcut |
| **0.8** | **Cân bằng: domain ngắn lẫn URL dài có path/query, IP-based, TLD đa dạng** | **Chọn dùng** |
| 1.0 | Quá noisy: TLD bịa (`.cor`, `.bur`), URL malformed, char salad | Loại — gây hại training |

**(c) Phân phối real-world.**
Prefix scheme được mix theo APWG Q4 2024: 85% `https://`, 13% `http://`,
2% trống (để model tự sinh). Tham chiếu nhất quán với `scripts/1. url-generator.py`
(URL generator cũ).

**(d) Chống ô nhiễm dữ liệu.**
Dedup so với 413k URL gốc trong quá trình sinh để tránh leakage giữa training
data thật và synthetic.

### 1.4. Kết quả thực tế

Sinh **1,000,000 URL synthetic** (T=0.8) trong file
`dataset/synth_T08 (RNN).csv`:

- 0 duplicates so với gốc.
- 0.23% URL có dạng malformed nhỏ (`..` trong host) — chấp nhận được.
- Phân phối scheme/TLD/length sát real-world.
- Pattern học được rất sát thực: Joomla compromise, Mozi botnet thật,
  hosting abuse, DGA, typosquatting (xem mục 2.7).

---

## Phần 2 — Kỹ thuật

### 2.1. Pipeline 5 hàm chính

| Hàm | Vai trò |
|---|---|
| `preprocess_data(csv_path)` | Load CSV → lowercase → filter ký tự → encode int16 (N×102) |
| `build_model(vocab_size)` | Khởi tạo CharRNN (Embedding → LSTM → Dropout → Linear) |
| `train_model(encoded, …)` | On-GPU bucketed training, save best checkpoint |
| `generate_urls(count, temperature, …)` | Batch parallel temperature sampling, streaming write CSV |
| `main()` | CLI orchestration: `--mode train\|generate\|both` |

### 2.2. Vocab & encoding

```python
ALLOWED_CHARS = string.ascii_lowercase + string.digits + "/:.-_?=&#@%+~"  # 49
itos = [PAD, SOS, EOS] + list(ALLOWED_CHARS)                              # 52
```

- Khớp 1-1 với `CHARS` trong `scripts/3. preprocess-data.py:63`.
- `MAX_URL_LEN = 100` (P99 length từ EDA = 83), `MAX_SEQ_LEN = 102` (+SOS, +EOS).
- Encoded shape: `(413k, 102)` int16 ≈ 84 MB → vừa thoải mái GPU memory.
- URL sau filter < 6 chars bị drop.

### 2.3. Model

```
Embedding(vocab=52, dim=64, padding_idx=0)
  └─> LSTM(input=64, hidden=256, num_layers=1, batch_first=True)
       └─> Dropout(p=0.2)
            └─> Linear(256, 52)
```

~330k parameters. Single-layer LSTM đủ năng lực cho vocab 52 (so với CNN-LSTM
downstream có thêm Conv1D + MaxPool downsample seq).

### 2.4. Training objective

- **Loss**: Cross-Entropy với `ignore_index=PAD` (id=0) để không tính loss trên padding.
- **Teacher forcing**: input = full sequence, target = sequence shift-by-one.
- **Optimizer**: Adam, `lr=2e-3` (không scheduler, không grad clip — khớp pattern
  của `scripts/4. train-model.py` đã hoạt động ổn).
- **Epochs**: 15. **Batch size**: 4096.

### 2.5. Tối ưu hiệu năng (quan trọng cho GTX 1660 SUPER 6GB)

**Bottleneck ban đầu:** ~7.5 phút/epoch, GPU util chỉ 25%.

| Tối ưu | Cơ chế | Speedup |
|---|---|---|
| **On-GPU data loading** | Load toàn bộ `train_data` (84 MB int16) lên GPU một lần. Bỏ `DataLoader` (multi-process IPC trên Windows quá nặng — đây là bottleneck #1). | 2-3x |
| **Length bucketing** | Sort URLs theo length thực; mỗi batch chỉ pad đến `max_len_in_batch` thay vì `MAX_SEQ_LEN=102`. URL trung bình ~30-50 chars → tiết kiệm ~50% LSTM compute. | 1.5-2x |
| **Disable AMP / mixed precision** | GTX 1660 SUPER (Turing TU116) **không có Tensor Cores** → FP16 không tăng tốc, GradScaler còn add overhead. Quay về FP32 plain. | 1.3x |
| **Model gọn** | LSTM 2 layer × 384h → 1 layer × 256h. Dropout 0.3 → 0.2. | 3-4x |
| **GPU-side loss accumulator** | `loss_sum`, `tok_sum` là tensor trên GPU; `.item()` chỉ gọi 1 lần / 50 batch (cho pbar). Giảm CPU↔GPU sync. | 1.1x |
| `cudnn.benchmark = True` | cuDNN auto-tune kernel cho input shapes lặp lại. | 1.05x |

**Kết quả:** 7.5 phút/epoch → **~2 phút/epoch** (~4x). Total 15 epochs ≈ 30 phút.

Hàm `_build_bucketed_batches(lengths, batch_size)` chịu trách nhiệm phần
bucketing: trả về `list[(idx_array, max_len_in_batch)]` precomputed, training
loop chỉ cần slice `train_data.index_select(0, idx)[:, :max_len].long()`.

### 2.6. Sinh dữ liệu (`generate_urls`)

**Cơ chế batch parallel** (`_generate_batch`):

1. **Warm-up hidden state**: feed `[SOS] + prefix_chars` qua model → lấy
   `hidden_state` ban đầu cho cả batch.
2. **Loop sampling** (max `MAX_URL_LEN - len(prefix)` step):
   - Mask `PAD` và `SOS` ra khỏi logits (gán `-inf`) để không bao giờ sample được.
   - `scaled = logits / T` → `softmax` → `torch.multinomial(probs, 1)`.
   - Sequence nào đã emit `EOS` thì tiếp theo bị ép về `PAD` (không append nữa).
   - Dừng sớm khi `finished.all()`.
3. **Decode**: ghép char ids thành string, prepend `prefix`.

**Tham số sinh:**

```
GEN_BATCH_SIZE = 4096    # 4096 URLs sinh đồng thời trên GPU
WRITE_CHUNK    = 100_000 # flush CSV mỗi 100k rows (streaming, không tràn RAM)
PREFIX_PROBS   = [0.85, 0.13, 0.02]  # https / http / empty (APWG Q4 2024)
```

**Quality filter sau sinh:**
- `len(url) < 6` → drop.
- chứa space/newline → drop.
- trùng với 413k URL gốc (load vào `set` ngay từ đầu) → drop.

CSV được mở 1 lần với `csv.writer`, ghi streaming theo chunk; không bao giờ
giữ toàn bộ output trong RAM → có thể sinh tới 6M+ URL trên máy 8GB RAM.

### 2.7. Verification trên 1M URL đã sinh

**Statistical health**

| Metric | Giá trị |
|---|---|
| Rows | 1,000,000 |
| Duplicates (so với gốc) | 0 |
| HTTP / HTTPS | 12.4% / 87.6% (target 13/87) |
| Length min/median/P95/P99/max | 7 / 41 / 98 / 100 / 100 |
| Double-dot trong host | 0.23% |
| No-dot-in-host | 1.05% |
| IP-based URLs | 1.23% |
| TLD top | com 56.6%, net 4.6%, app 3.7%, de, br, ru, org, jp, info |

**Pattern model học được (qualitative)**

| Pattern | Ví dụ sinh ra |
|---|---|
| Joomla CMS compromise | `…/index.php?option=com_content&view=article&id=49&itemid=78` |
| Mozi botnet (thật) | `http://104.95.253.227:15089/mozi.m` |
| Phishing trên hosting abuse | `https://maildinshaakckjnrfz.web.app/`, `att-10785.weebly.com` |
| Typosquatting brand combo | `www.paypal.cgi.google.com/auto/ag/net.php` |
| DGA domain | `https://3id1659dc5994ccea6d3e13d5d4c7.jcg.com/` |
| C2 IP + port | `http://177.80.102.228:8070/tmpftp/td/contact` |
| Malware download path | `…/files/788976.exe`, `…/uploads/38015.exe` |

### 2.8. CLI usage

```powershell
# Train
python "scripts/1. url-generator (model 2).py" --mode train --epochs 15

# Sinh 1M URL với T=0.8
python "scripts/1. url-generator (model 2).py" --mode generate `
    --count 1000000 --temperature 0.8 `
    --output "dataset/synth_T08 (RNN).csv"

# Other flags: --prefix, --no-prefix-mix, --no-dedup, --gen-batch
```

### 2.9. Artifacts

| Path | Nội dung |
|---|---|
| `models/char_rnn_url_generator.pt` | Best checkpoint (state_dict + config) |
| `models/char_rnn_vocab.json` | `{itos, stoi}` để generate độc lập sau train |
| `dataset/synth_T08 (RNN).csv` | 1M URL synthetic, format `url,label=1` |
script để lọc (filter) dữ liệu sau khi sinh không?
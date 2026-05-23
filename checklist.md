# secURLity — Technical Checklist

> **Focus hiện tại**: Model 4 (CNN-LSTM + XGBoost + SGDClassifier) trên dataset thật 19.68M URLs.  
> Models 1/2/3 (synthetic/VN-only) đã deprecated — giữ artifact trên disk nhưng không maintain.

---

## 0. Environment

- [x] Python venv (`./venv/`) với GPU support (CUDA trên RTX 3060 Ti, 8GB VRAM).
- [x] Dependencies cơ bản: `torch`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `requests`, `beautifulsoup4`, `tldextract`.
- [x] XGBoost stack: `xgboost`, `pyarrow`.
- [x] ONNX stack: `onnx`, `onnxruntime`, `onnxscript`.
- [x] SGD/HTML stack: `polars`, `joblib`, `scipy`.
- [x] SHAP (cho XGBoost inference `--explain`): `pip install shap`.
- [x] Python 3.14 (lazy-mmap pattern cần thiết vì Windows `multiprocessing.spawn`).

---

## 1. Data Collection

### 1.1 Benign URLs — `scripts/crawl-benign-urls.py`

- [x] Pipeline 3 tầng per seed: **crt.sh** → **sitemap** → **BFS crawler fallback**.
  - crt.sh: enumerate subdomain qua Certificate Transparency logs, giới hạn 30 subdomain/seed.
  - Sitemap: `robots.txt` + default paths, parse `urlset`/`sitemapindex` lồng + `.xml.gz` (regex `<loc>`, không cần lxml).
  - BFS: respect `robots.txt`, per-host rate-limit 1s/req, depth-limited, scope eTLD+1.
- [x] `ThreadPoolExecutor` (--workers=8 default) — song song nhiều seed, mỗi seed host riêng nên không block nhau.
- [x] **Resume mode** (`--resume`): load `processed_seeds.txt` + preload URL từ CSV cũ vào `global_seen` để dedupe.
- [x] Thread-safe: `_rate_lock`, `_csv_lock`, `_seen_lock`, `_processed_lock`.
- [x] Output: `dataset/intl-benign/urls.csv` (columns: `url, seed, source, depth`).
- [x] Seeds: `dataset/intl-benign/seeds.txt` (international, multi-language).

### 1.2 Malicious URLs — `scripts/fetch-malicious-url.py`

- [x] 13 nguồn free no-auth: URLhaus, ThreatFox, OpenPhish, Phishing.Database (ACTIVE/INACTIVE/NEW), Phishing.Army, tweetfeed.live, DigitalSide.it, CertPL, Cybercrime Tracker, Hagezi TIF, Malsilo.
- [x] Format handlers: `urlhaus_csv` (ZIP-CSV), `threatfox_csv` (ZIP-CSV), `txt_lines`, `domain_list` (wrap `https://`), `tweetfeed_json`.
- [x] Flag `is_vn_target` (bool): TLD `.vn` OR VN brand/keyword regex (ngân hàng, ví điện tử, ecommerce, telecom VN...).
- [x] Output: `dataset/vn-malicious/urls-all.csv` + `urls-vn-target.csv` + `sources-summary.txt`.

### 1.3 Extra Malicious URLs — `scripts/fetch-malicious-extra.py`

- [x] Parallel `ThreadPoolExecutor` — mỗi source 1 worker, incremental write (Ctrl+C chỉ mất source đang chạy).
- [x] **Per-source deadline** (--source-timeout=1800s): tự cắt khi vượt deadline, không loop vô hạn.
- [x] **Max retry** 3 lần khi 429/5xx.
- [x] Tier A (API key từ env var): OTX AlienVault, URLScan.io, Pulsedive (default OFF, dùng `--include pulsedive` để bật).
- [x] Tier B (no-auth): phishdb_domains, blocklistproject, spam404, viriback, botvrij, bigbl_hacked.
- [x] Append + dedupe vào `urls-all.csv`.

### 1.4 Dataset balancing — `scripts/balance-dataset-3.py`

- [x] Shuffle toàn bộ benign URL từ `urls-benign-global.csv` + `urls-benign-vn.csv`.
- [x] Sample 4×N_mal benign → balance 80/20.
- [x] Lưu `dataset/dataset-4-final.csv` với `csv.QUOTE_ALL` (URL có `,`/`"` không phá CSV).
- [x] Final: **19,678,135 URL** (15,742,507 benign + 3,935,628 malicious).

---

## 2. EDA — `scripts/2. eda 4.py`

- [x] Streaming read (constant memory) qua generator.
- [x] Chốt `MAX_LEN=256` (P99=195, P99.5=217 → 256 cover ~P99.7, multiple of 64).
- [x] Chốt `vocab_size=94` (PAD + UNK + 92 case-preserved chars; UNK rate = 0.03%).
- [x] Xác nhận **preserve case**: mixed-case ratio benign=11.83% vs mal=5.50% → signal có nghĩa.
- [x] Xác nhận TLD distribution: `.vn` = 19.85% benign vs 0.18% mal → `tld_is_vn` là feature cực mạnh.
- [x] Phát hiện 101 chars CHỈ xuất hiện trong mal (control bytes, `<>#"`) → thêm `has_control_chars` feature.
- [x] Output: `dataset/eda-result 4.txt`.

---

## 3. Model 1 — CNN-LSTM

### 3.1 Preprocessing — `scripts/3. preprocess-data 4.py`

- [x] Load `dataset/dataset-4-final.csv` (QUOTE_ALL, 19.68M rows).
- [x] **KHÔNG lowercase** URL — case là signal.
- [x] Char encode với vocab 94, `MAX_LEN=256`, pad về 0 (PAD index).
- [x] Tạo **2 split modes trong 1 lần chạy**:
  - **random**: stratified 80/10/10 theo label → `data/processed/model_4/random/`
  - **domain**: greedy bin-pack domains 80/10/10 (zero domain overlap) → `data/processed/model_4/domain/`
- [x] Verify: `test_domains ∩ train_domains = ∅` (confirmed trong script).
- [x] Lưu `vocab.json`, `metadata.json` (shared), `split.csv` (QUOTE_ALL), `X.npy`, `y.npy` per split per mode.

### 3.2 Lexical feature extraction — `scripts/4-feat. extract-lexical_4.py`

- [x] 30 features theo thứ tự cố định (phải match `FEATURE_NAMES` tuyệt đối khi inference).
- [x] Normalization: `log1p` cho 18 count/length features → standardize toàn bộ bằng TRAIN-only mean/std.
- [x] Lưu `feat_stats.json` per mode (dùng tại inference để re-apply cùng transform).
- [x] Lưu `feat.npy` per split per mode.
- [x] Cả 2 modes (random + domain) trong 1 lần chạy.

### 3.3 Architecture — `scripts/4. train-model_4.py`

```
CNNLSTM class:
  Embedding(94, 64)
  → Conv1d(64→128, kernel=3) + ReLU + Conv1d(128→128, kernel=3) + ReLU + MaxPool1d(2)
  → BiLSTM × 2 (hidden=128 mỗi chiều → output=256) + dropout=0.3
  → AttentionPool (pad-aware, chuẩn hóa softmax, expose weights)
  → Char path output: 256-dim

  Lexical MLP: Linear(30→64) + ReLU + Linear(64→32) + ReLU
  → Lexical path output: 32-dim

  Hybrid head: concat(256+32=288) → Linear(288→64) + ReLU + Linear(64→1)
```

Total params: **795,106** (~0.80 M)  
Tensor-core-friendly: embed=64, conv=128, lstm=128 (all multiples of 8)

### 3.4 Training config

- [x] `AdamW(lr=1e-3, weight_decay=0.01)` + `LinearLR (5% warmup) → CosineAnnealingLR (eta_min=1e-5)`, **per-batch** `scheduler.step()`.
- [x] `BCEWithLogitsLoss(pos_weight=4.0)` + label smoothing ε=0.05 (targets: {0.05, 0.95}).
- [x] Batch 2048, AMP FP16 (`torch.amp.autocast` + `GradScaler`), TF32 (`matmul.allow_tf32`, `set_float32_matmul_precision("high")`).
- [x] **Lazy-mmap pattern** (`URLDataset`): `__init__` chỉ lưu path, `__getitem__` mở mmap lần đầu per worker, `__getstate__` clear mmap fields trước pickle. (Giải pháp cho Windows `OSError: [Errno 22]` với DataLoader multiprocessing.)
- [x] Checkpoint lưu `model_state_dict` + `optimizer_state_dict` + `scheduler_state_dict` + `scaler_state_dict` + `config`.

### 3.5 Training results

| | Random split | Domain split |
|---|---|---|
| Best epoch | 25 (max 30) | **0** (early-stop sau epoch 1) |
| Val F1 | 0.99996 | 0.7049 |
| Val ROC-AUC | 0.99999 | 0.9851 |
| Val PR-AUC | 0.99997 | 0.9320 |
| Checkpoint | `cnn_lstm_best_4_random.pt` | `cnn_lstm_best_4_domain.pt` |

> Domain split overfit ngay epoch 1 vì zero domain overlap → train/val thấy hoàn toàn khác domains. Val F1=0.70 chỉ là số trên val hardest domains; Test F1=0.9888 xác nhận model vẫn generalize tốt.

### 3.6 Threshold tuning — `scripts/6. tune-threshold_4.py`

- [x] Sweep threshold 0.01→0.99 step 0.01 trên val set.
- [x] 3 operating points:
  - `default`: max F1
  - `safer`: threshold thấp nhất mà precision ≥ 0.99
  - `precise`: threshold cao nhất mà recall ≥ 0.99
- [x] Lưu `models/thresholds_4.json` per split-mode.
- [x] `predict-url_4.py` đọc via `resolve_threshold()`, chấp nhận `--mode {default|safer|precise}` + custom `--threshold X.XX`.

**Random split thresholds**: default=0.42, safer=0.18, precise=0.97  
**Domain split thresholds**: default=0.98, safer=0.99, precise=0.98 (collapse vì sweep step=0.01 quá thô ở vùng [0.98, 0.99])

### 3.7 Evaluation — `scripts/5. eval-model_4.py`

> Script riêng vì train script crash ở test phase (DataLoader worker spawn exhaustion sau 18+ epochs — Windows handle issue). Workaround: `num_workers=0`, chỉ load test split.

**Random split — Test set (1,967,816 URLs):**

| Mode | Threshold | F1 | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|
| default | 0.42 | **0.999926** | 0.999919 | 0.999934 | 32 | 26 |
| safer | 0.18 | 0.999902 | 0.999830 | 0.999975 | 67 | **10** |
| precise | 0.97 | 0.999886 | 0.999947 | 0.999825 | **21** | 69 |

Curve: **ROC-AUC = 0.999992** | **PR-AUC = 0.999969**

**Domain split — Test set (1,967,813 URLs):**

| Mode | Threshold | F1 | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|
| **default** | 0.98 | **0.9888** | 0.9845 | 0.9931 | 6,154 | 2,698 |
| safer | 0.99 | 0.4426 | 0.9915 | 0.2849 | 965 | 281,025 |
| precise | 0.98 | (= default) | — | — | — | — |

Curve: **ROC-AUC = 0.998894** | **PR-AUC = 0.987224**

### 3.8 Explainability — `predict-url_4.py --explain`

- [x] Monkey-patch `model.attn_pool.forward` để stash `_last_weights` mỗi forward (không sửa train script).
- [x] Upsample `L/4 → L` bằng `np.repeat(weights, 4)`, crop về `len(url_truncated)`.
- [x] ANSI 256-color terminal heatmap (gradient pale yellow → bright red qua 10 buckets).
- [x] Top-K char positions (--explain-top-k, default 5), dedupe by L/4 block.
- [x] Top-K lexical features theo `|z-score|`.
- [x] JSON mode (`--json`) strip ANSI, giữ raw weights array.
- [x] Latency < 50ms/URL (verified ~30-40ms warm).

### 3.9 ONNX Export — `scripts/8. export-onnx_4.py`

- [x] **Dùng `dynamo=False`** (legacy TorchScript exporter). Dynamo exporter torch 2.11 output 0.02 MB ONNX (mất weights) + hardcode reshape shape → phá dynamic batch với LSTM.
- [x] Wrapper `CNNLSTMWithAttention` expose `attention_weights` là output thứ 2 (shape `(B, 64)`).
- [x] 2 outputs ONNX: `logit` + `attention_weights`.
- [x] Sanity check: PyTorch vs ONNX FP32 max_diff = 4.77e-7 (PASS, target < 1e-4).
- [x] INT8 dynamic quant: `quantize_dynamic(..., weight_type=QuantType.QInt8)`.
- [x] Kết quả file:
  - `models/cnn_lstm_4_{random,domain}.onnx` — FP32, 3.04 MB
  - `models/cnn_lstm_4_{random,domain}_int8.onnx` — INT8, 0.79 MB (3.84x compression)

**CPU latency benchmark (1-thread, 1 URL):**

| Backend | ms/URL | URL/s | Speedup |
|---|---|---|---|
| PyTorch FP32 CPU | 2.543 | 393 | 1.00x baseline |
| **ONNX FP32 CPU** | **1.671** | **599** | **1.52x** ✅ |
| ONNX INT8 CPU | 6.595 | 152 | 0.39x ⚠️ (LSTM không có INT8 kernel) |

**Khuyến nghị production: ONNX FP32** (`predict-url_4_onnx.py`, default).

---

## 4. Model 2 — XGBoost

### 4.1 Feature extraction — `scripts/3. feature-extract_xgb_4.py`

- [x] 105 features theo thứ tự cố định trong `FEATURE_NAMES` (deterministic qua probe URL).
- [x] Stateless features — không dùng train statistics → không có leakage.
- [x] Multiprocessing `Pool` + chunked processing (CHUNK_SIZE=4000, N_WORKERS=cpu_count-1).
- [x] Đọc từ `data/processed/model_4/random/` (split.csv QUOTE_ALL format).
- [x] Output: `data/processed/xgboost_4/{train,val,test}_{X,y}_xgb.npy`, `xgb_4_feature_cols.json`, `xgb_4_feature_meta.json`.

**Nhóm features chính:**
- Structural (12): url_len, host_len, path_len, path_depth, is_bare_hostname, path_to_url_ratio...
- Char composition (10): digit_count/density, hyphen_count, dot_count, at_count, pct_count...
- Case/charset (6): mixed_case_ratio, has_upper, has_control_chars, has_non_ascii, has_punycode...
- Host-specific (9): host_digit_density, longest_digit_run, longest_consonant_run, host_vowel_ratio...
- Entropy (3): entropy_url, entropy_host, entropy_path (Shannon)
- Scheme (3): is_http, is_https, is_ip_host
- TLD one-hot (53): Top 50 + tld_other + tld_missing + tld_label_count
- TLD buckets (3): tld_is_vn, tld_is_suspicious, tld_is_common
- Phishing keywords (2): num_phishing_keywords, has_phishing_keyword
- File extension (3): has_malware_ext, has_cdn_ext, has_any_ext

### 4.2 Training — `scripts/4. train-xgb_4.py`

- [x] `XGBClassifier` với `tree_method="hist"`, `device="cuda"` (RTX 3060 Ti).
- [x] `scale_pos_weight=4.0` (đọc từ `xgb_4_feature_meta.json`).
- [x] Early stopping 50 rounds trên val ROC-AUC.
- [x] Hyperparams: `max_depth=8, lr=0.05, n_estimators=2000, subsample=0.8, colsample_bytree=0.8`.
- [x] **Training time: 234.9s** trên CUDA, best_iteration=1991.
- [x] Lưu model: `models/xgb_url_4.ubj` (14.8 MB).
- [x] Lưu metrics: `models/xgb_url_4_metrics.json`.
- [x] Lưu feature importance (gain/weight/cover/total_gain/total_cover): `models/xgb_url_4_importance.json`.

**Val metrics**: ROC-AUC=0.999999 | PR-AUC=0.999995 | F1@0.5=0.999585

### 4.3 Evaluation — `scripts/5. eval-xgb_4.py`

- [x] Threshold tuning bằng `precision_recall_curve` (vectorized, nhanh hơn grid sweep).
- [x] 3 operating points: default (max F1), safer (precision≥0.99), precise (recall≥0.99).
- [x] Lưu `models/results_xgb_4.json` + `models/thresholds_xgb_4.json`.

**Test set (1,967,816 URLs):**

| Mode | Threshold | F1 | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|
| **default** | 0.400 | **0.999618** | 0.999418 | 0.999817 | 229 | 72 |
| safer | 0.002 | 0.994969 | 0.990001 | 0.999987 | 3,975 | 5 |
| precise | 0.998 | 0.994930 | 0.999908 | 0.990002 | 36 | 3,935 |

Curve: **ROC-AUC = 0.999999** | **PR-AUC = 0.999995**

### 4.4 Inference + SHAP — `scripts/predict-url_4_xgb.py`

- [x] Load `extract_features()` từ `3. feature-extract_xgb_4.py` qua `importlib` (filename có space + leading digit).
- [x] `resolve_threshold()` đọc `thresholds_xgb_4.json`, chấp nhận tên mode hoặc float.
- [x] `--explain` flag: SHAP `TreeExplainer`, sub-millisecond per URL.
- [x] SHAP output: top-K features theo `|shap_value|`, raw_value, direction (→MAL / →BEN).
- [x] SHAP values ở log-odds space: `sigmoid(expected_value + sum(shap)) ≈ prob`.
- [x] `--topk` (default 10): số features hiển thị.

---

## 5. Model 3 — SGDClassifier (HTML Content)

### 5.1 Dataset

- **Nguồn**: HuggingFace `phreshphish/phreshphish` — raw HTML pages từ benign và phishing websites thực.
- **Format**: parquet files (train-*.parquet, test-*.parquet), columns: `html`, `label` (benign/phish).
- **Location local**: `D:\phreshphish\data\`

### 5.2 Training — `scripts/4. train-model-SGDClassifier.py`

- [x] **HTML preprocessing** (polars vectorized):
  - Slice đến `HTML_MAX_CHARS×5`
  - `re.sub(r">\s+<", "><")` + `re.sub(r"\s+", " ")` + strip
  - Slice cuối về `HTML_MAX_CHARS=10,000`
- [x] `HashingVectorizer(analyzer="char", ngram_range=(3,4), n_features=2^20, norm="l2", lowercase=False)`.
- [x] **Cache vectorized output** bằng `joblib` (key = hash của vectorizer config + preprocess config) → tránh re-vectorize nhiều epoch.
- [x] **Prefetch pipeline**: `ProcessPoolExecutor` vectorize các file kế tiếp trong khi main process đang train batch hiện tại (depth=2).
- [x] `SGDClassifier(loss="log_loss", class_weight=balanced, partial_fit())` — online learning.
- [x] Lưu artifact (vectorizer_params + clf + metadata) vào `D:\phreshphish\models\SGDClassifier_2.joblib` (4.2 MB).

### 5.3 Kết quả

**Test set (168,060 samples):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| benign | 0.8150 | 0.9302 | **0.8688** |
| phish | 0.9003 | 0.7491 | **0.8178** |
| weighted avg | 0.8540 | 0.8474 | 0.8455 |

**Accuracy: 0.8474 | ROC-AUC: 0.9277**

### 5.4 Inference — `scripts/predict-rawHTML-SGDClassifier.py`

- [x] Load artifact: `vectorizer_params` → rebuild `HashingVectorizer`, load `clf`.
- [x] `normalize_html()` giữ nguyên preprocessing pipeline (bắt buộc nhất quán với train).
- [x] Single file mode: `--file page.html`.
- [x] Folder batch mode: `--folder ./pages [--recursive]` — yield per file, in summary cuối.
- [x] `predict_proba()` cho cả 2 classes.

---

## 6. Artifacts tổng hợp

### 6.1 Models

| File | Size | Mô tả |
|---|---|---|
| `models/cnn_lstm_best_4_random.pt` | 9.1 MB | CNN-LSTM PyTorch checkpoint (random split) |
| `models/cnn_lstm_best_4_domain.pt` | 9.1 MB | CNN-LSTM PyTorch checkpoint (domain split) — **production** |
| `models/cnn_lstm_4_random.onnx` | 3.04 MB | ONNX FP32 (random) |
| `models/cnn_lstm_4_domain.onnx` | 3.04 MB | ONNX FP32 (domain) — **production** |
| `models/cnn_lstm_4_random_int8.onnx` | 0.79 MB | ONNX INT8 (random) |
| `models/cnn_lstm_4_domain_int8.onnx` | 0.79 MB | ONNX INT8 (domain) |
| `models/xgb_url_4.ubj` | 14.8 MB | XGBoost model |
| `D:\phreshphish\models\SGDClassifier_2.joblib` | 4.2 MB | SGDClassifier HTML model |

### 6.2 Metrics & Thresholds

| File | Mô tả |
|---|---|
| `models/results_4.json` | CNN-LSTM training + val + test metrics per split-mode |
| `models/thresholds_4.json` | 3 operating points × 2 split-modes cho CNN-LSTM |
| `models/results_xgb_4.json` | XGBoost test metrics + operating points |
| `models/thresholds_xgb_4.json` | 3 operating points cho XGBoost |
| `models/xgb_url_4_metrics.json` | XGBoost training metrics |
| `models/xgb_url_4_importance.json` | Feature importance (gain/weight/cover) |
| `models/result_SGDClassifier.txt` | SGDClassifier classification report |

### 6.3 Processed data

| Path | Mô tả |
|---|---|
| `data/processed/model_4/vocab.json` | 94 chars, case-preserved |
| `data/processed/model_4/metadata.json` | max_len=256, n_total, pos_weight per mode |
| `data/processed/model_4/{random,domain}/feat_stats.json` | mean/std cho lexical normalize |
| `data/processed/model_4/{random,domain}/{train,val,test}/{X.npy, y.npy, feat.npy, split.csv}` | Preprocessed data |
| `data/processed/xgboost_4/xgb_4_feature_cols.json` | Tên 105 features (theo thứ tự) |
| `data/processed/xgboost_4/xgb_4_feature_meta.json` | TLD lists, pos_weight, notes |
| `data/processed/xgboost_4/{train,val,test}_{X,y}_xgb.npy` | XGBoost feature matrices |

---

## 7. Inference — Quick reference

```powershell
.\venv\Scripts\Activate.ps1

# CNN-LSTM — ONNX FP32 (recommended production)
python "scripts/predict-url_4_onnx.py" --split-mode domain "https://example.com"
python "scripts/predict-url_4_onnx.py" --split-mode domain --explain "https://example.com"
python "scripts/predict-url_4_onnx.py" --split-mode domain --mode safer "https://example.com"

# CNN-LSTM — PyTorch (có --explain đầy đủ, nhanh hơn trên GPU)
python "scripts/predict-url_4.py" --split-mode domain "https://example.com"
python "scripts/predict-url_4.py" --split-mode domain --explain "https://example.com"

# XGBoost
python "scripts/predict-url_4_xgb.py" "https://example.com"
python "scripts/predict-url_4_xgb.py" --explain --topk 15 "https://example.com"
python "scripts/predict-url_4_xgb.py" --thr safer "https://example.com"

# SGDClassifier HTML
python "scripts/predict-rawHTML-SGDClassifier.py" --file page.html
python "scripts/predict-rawHTML-SGDClassifier.py" --folder ./pages --recursive
```

---

## 8. Known Issues & Workarounds

| Issue | Severity | Workaround |
|---|---|---|
| Windows DataLoader + numpy.memmap → OSError [Errno 22] | High | Lazy-mmap pattern: `__init__` lưu path, `__getitem__` mở mmap lần đầu per worker |
| Train script crash ở test phase sau 18+ epochs (Windows handle exhaustion) | Medium | Chạy `5. eval-model_4.py` riêng (num_workers=0, chỉ load test) |
| `importlib` import module tên có space/digit (eval/tune scripts) | Low | `importlib.util.spec_from_file_location()` — default num_workers=0 trong eval/tune |
| PyTorch 2.11 dynamo ONNX exporter broken cho LSTM (0.02MB, mất weights) | High | Dùng `dynamo=False` (legacy TorchScript exporter) |
| ONNX INT8 chậm hơn FP32 3.9x (LSTM không có INT8 kernel trong onnxruntime CPU EP) | Medium | Dùng ONNX FP32 cho production; INT8 chỉ khi disk size critical |
| CNN-LSTM FP bare hostname (google.com → MAL ~99%) | Medium | `path_depth=0` lệch -3.04 std dưới TRAIN mean. Workaround: domain allowlist filter cho URL ngắn |
| Domain split threshold coarse (safer/precise collapse tại 0.98) | Low | Sweep step=0.01 không đủ mịn. Tương lai: sweep step=0.001 ở vùng [0.95, 0.999] |
| SGDClassifier HTML model F1 thấp hơn 2 model kia | Design | Model này analyze HTML content, không URL structure — complementary không competitive |

---

## 9. Kết quả tổng hợp

| Model | Split | Test F1 | ROC-AUC | PR-AUC | Latency | Status |
|---|---|---|---|---|---|---|
| CNN-LSTM | domain (production) | **0.9888** | 0.9989 | 0.9872 | 1.67 ms/URL | ✅ DONE |
| CNN-LSTM | random (in-dist) | 0.999926 | 0.999992 | 0.999969 | 1.67 ms/URL | ✅ DONE |
| XGBoost | random | **0.999618** | 0.999999 | 0.999995 | < 1 ms/URL | ✅ DONE |
| SGDClassifier | phreshphish | 0.847 (acc) | 0.9277 | — | — | ✅ DONE |
| **Ensemble** | — | TBD | TBD | TBD | TBD | 🔲 WIP |

---

## 10. Deprecated (Models 1/2/3)

Artifact vẫn tồn tại trên disk nhưng **không maintain**:
- `models/cnn_lstm_best_{1,2,3}.pt`, `models/results_{1,2,3}.json`, `models/xgb_url_2.ubj`
- `dataset/dataset {1,2,3}.csv`, `dataset/eda-result {1,2,3}.txt`
- `scripts/*_1.py`, `*_2.py`, `*_3.py` variants

Khác biệt chính với model 4: MAX_LEN=100, vocab=51, không có lexical features, không có attention export, dữ liệu synthetic (model 2) hoặc VN-only crawl (model 3).

---

## 11. Phase 2 — Ensemble (Chưa implement)

- [ ] Script `predict-ensemble.py`: load CNN-LSTM ONNX (domain) + XGBoost, predict cả 2, combine.
- [ ] Tune ensemble weights trên val set (hoặc dùng stacking meta-learner).
- [ ] Integrate SGDClassifier khi HTML content available (optional path).
- [ ] Unified CLI: `--url <URL> [--html page.html] [--explain]`.
- [ ] Unified explanation: merge attention heatmap + SHAP values.
- [ ] Benchmark ensemble F1 vs từng model đơn lẻ.

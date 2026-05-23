# secURLity — Kế hoạch dự án

> **Mục tiêu**: Xây dựng hệ thống phát hiện URL độc hại (malicious URL detection) đa model, dựa trên dữ liệu thật 100% thu thập từ internet, có thể giải thích được kết quả (explainable AI).

---

## 1. Vấn đề & Giải pháp

### Vấn đề
URL độc hại (phishing, malware C2, botnet...) ngày càng tinh vi, né tránh các blacklist đơn giản. Các giải pháp rule-based dễ bị bypass; model ML dùng dữ liệu synthetic thiếu tính đại diện cho môi trường thực tế.

### Giải pháp
Xây dựng hệ thống 3 tầng phân tích:
1. **Phân tích chuỗi URL** (char-level + lexical features) — hiểu cấu trúc và ngữ nghĩa của URL.
2. **Phân tích feature thủ công** (tabular ML) — các đặc trưng kỹ thuật từ URL có thể giải thích bằng domain knowledge.
3. **Phân tích nội dung HTML** (content-based) — dùng raw HTML page để phân biệt benign/phishing ngay cả khi URL trông bình thường.

Mỗi model phân tích từ một góc độ khác nhau. Kết hợp cả 3 (ensemble) cho kết quả cuối cùng.

---

## 2. Kiến trúc hệ thống

```
                     ┌─────────────────────────────┐
                     │          Input               │
                     │   URL (string) + HTML page   │
                     └──────┬─────────────┬─────────┘
                            │             │
              ┌─────────────▼──┐     ┌────▼──────────────────┐
              │   URL-based    │     │   Content-based        │
              │                │     │                        │
         ┌────▼────┐  ┌────────▼──┐  │  ┌──────────────────┐ │
         │  Model  │  │  Model 2  │  │  │     Model 3      │ │
         │    1    │  │  XGBoost  │  │  │  SGDClassifier   │ │
         │CNN-LSTM │  │ 105 feats │  │  │  HashingVec      │ │
         │+Lexical │  │ +SHAP     │  │  │  HTML char ngram │ │
         └────┬────┘  └──────┬────┘  │  └────────┬─────────┘ │
              │ Attention    │ SHAP  │            │ Prob      │
              │ heatmap      │ vals  │            │           │
              └──────────────┴───────┴────────────┘
                             │
                    ┌────────▼────────┐
                    │  Ensemble layer │  ← (WIP — Phase 2)
                    │ weighted voting │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final output   │
                    │ Benign / MAL    │
                    │ Risk level      │
                    │ Explanation     │
                    └─────────────────┘
```

---

## 3. Dataset

### 3.1 Tổng quan

| Thuộc tính | Chi tiết |
|---|---|
| **Tổng số URL** | 19,678,135 |
| **Tỉ lệ** | 80% benign / 20% malicious |
| **Nguồn gốc** | 100% dữ liệu thật từ internet |
| **File chính** | `dataset/dataset-4-final.csv` (QUOTE_ALL) |

### 3.2 Thu thập dữ liệu benign

Script: `scripts/crawl-benign-urls.py`

Pipeline thu thập theo thứ tự ưu tiên:
1. **crt.sh** — enumerate subdomain qua Certificate Transparency logs (tối đa 30 subdomain/seed)
2. **Sitemap** — đọc `robots.txt` + path mặc định, parse `urlset`/`sitemapindex` (bao gồm `.xml.gz` lồng nhau)
3. **BFS crawler** — fallback khi sitemap < threshold (100 URL), respect `robots.txt`, per-host rate-limit 1s/request, scope eTLD+1

Đặc điểm kỹ thuật:
- Chạy song song nhiều seed qua `ThreadPoolExecutor` (--workers, default 8)
- **Resume mode** (`--resume`): skip seed đã xử lý + preload URL đã có để dedupe
- Seeds: international multi-language (file `dataset/intl-benign/seeds.txt`)

Output: `dataset/intl-benign/urls.csv` (columns: `url, seed, source, depth`)

### 3.3 Thu thập dữ liệu malicious

**Script chính** (`scripts/fetch-malicious-url.py`) — 13 nguồn free, no-auth:

| Nguồn | Loại | Format |
|---|---|---|
| URLhaus (abuse.ch) | Malware/C2 URLs | ZIP-CSV |
| ThreatFox (abuse.ch) | IOC URLs | ZIP-CSV |
| OpenPhish | Phishing (~500 latest) | txt |
| Phishing.Database ACTIVE | Phishing đang live | txt |
| Phishing.Database INACTIVE | Historical (5-10M) | txt |
| Phishing.Database NEW-today | Phishing mới trong ngày | txt |
| Phishing.Army | Domain list → wrap https:// | txt |
| tweetfeed.live | IOC từ Twitter (1 tháng) | JSON |
| DigitalSide.it OSINT | Daily latest URLs | txt |
| CertPL hole.cert.pl | Polish CERT phishing domain | txt |
| Cybercrime Tracker | Botnet C2 URLs | txt |
| Hagezi TIF | Aggregated threat intel domain | txt |
| Malsilo | Curated malicious URL feed | txt |

**Script mở rộng** (`scripts/fetch-malicious-extra.py`) — song song, incremental:
- Tier A (API key): OTX AlienVault, URLScan.io, Pulsedive
- Tier B (no-auth): phishdb_domains, blocklistproject, spam404, viriback, botvrij, bigbl_hacked

Tất cả URL có flag `is_vn_target` (bool) dựa trên TLD `.vn` + VN brand/keyword regex.

### 3.4 Cân bằng dataset

Script: `scripts/balance-dataset-3.py`  
- Shuffle 40M benign → sample 4×N_mal → final 19.68M URL (80/20 ratio)
- Lưu với `csv.QUOTE_ALL` để URL có `,`/`"` không phá CSV format

---

## 4. Model 1 — CNN-LSTM (Model chính)

### 4.1 Kiến trúc

Hybrid architecture kết hợp 2 nhánh:

**Nhánh char-level sequence:**
```
URL string
  → Char encoding (MAX_LEN=256, vocab=94 ký tự, case-preserved)
  → Embedding(94, 64)
  → Conv1d(64→128) × 2 + MaxPool(2)
  → Stacked BiLSTM (2 layers, hidden=128)
  → AttentionPool (pad-aware, expose attention weights)
  → vector 256-dim
```

**Nhánh lexical features:**
```
URL string
  → 30 features thủ công
  → log1p transform (18 features count/length)
  → Standardize (mean/std từ TRAIN set)
  → Linear(30→64→32) MLP
  → vector 32-dim
```

**Hybrid head:**
```
concat(256 + 32) = 288-dim
  → Linear(288→64→1) + Sigmoid
  → probability P(malicious)
```

Tổng params: **795,106** (~0.80 M)

### 4.2 Vocab & Encoding

- `MAX_LEN = 256` (EDA P99=195, P99.5=217 → 256 cover ~P99.7, multiple of 64)
- `vocab_size = 94`: `<PAD>(0)` + `<UNK>(1)` + 52 letters (case-preserved!) + 10 digits + 30 special chars
- **KHÔNG lowercase** input URL — mixed-case ratio là signal (benign 11.83% vs mal 5.50%)
- UNK rate = 0.03%

### 4.3 Lexical features (30 features)

| Nhóm | Features |
|---|---|
| **Lengths (4)** | `url_length, host_length, path_length, query_length` |
| **Counts (14)** | `num_{dots, hyphens, underscores, slashes, question_marks, equals, amps, ats, percents, digits, subdomains}, subdomain_length_max, path_depth, query_param_count` |
| **Ratios (4)** | `digit_ratio, letter_ratio, vowel_ratio, special_char_ratio` |
| **Binary flags (7)** | `has_ip_host, has_port, is_https, tld_is_vn, tld_is_common, has_punycode, has_double_slash_in_path` |
| **Entropy (1)** | `host_entropy` (Shannon) |

Normalization: log1p cho 18 count/length features (heavy-tailed), sau đó standardize toàn bộ bằng TRAIN-only mean/std → lưu trong `data/processed/model_4/<mode>/feat_stats.json`.

### 4.4 Training

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| LR Schedule | LinearLR (5% warmup) → CosineAnnealingLR (eta_min=1e-5), per-batch |
| Loss | BCEWithLogitsLoss(pos_weight=4.0) + label smoothing ε=0.05 |
| Batch size | 2048 |
| Precision | AMP FP16 + GradScaler, TF32 enabled |
| Data loading | `mmap_mode='r'` trên numpy arrays (lazy-mmap pattern) |
| Hardware | RTX 3060 Ti 8GB |

### 4.5 Hai chế độ split

| | Random split | Domain split |
|---|---|---|
| **Logic** | Stratified 80/10/10 theo label | Greedy bin-pack 80/10/10 theo registered domain (zero overlap) |
| **Val F1** | 0.99996 (epoch 25) | 0.7049 (epoch 0 — overfit ngay sau 1 epoch) |
| **Test F1** | 0.999926 (thr=0.42) | **0.9888** (thr=0.98) |
| **Test ROC-AUC** | 0.999992 | 0.998894 |
| **Test PR-AUC** | 0.999969 | 0.987224 |
| **Production?** | ❌ Inflated (in-distribution) | ✅ **YES** — generalizes to new domains |
| **Checkpoint** | `models/cnn_lstm_best_4_random.pt` | `models/cnn_lstm_best_4_domain.pt` |

**Quy tắc production**: luôn dùng `--split-mode domain` cho inference thực tế.

### 4.6 Explainability — Attention-based (Option C)

**Quyết định**: Bỏ SHAP/DeepExplainer (1-15s/URL) → dùng **attention weights** built-in của `AttentionPool` (chi phí ~0, byproduct của forward pass).

Cách hoạt động:
- `AttentionPool` trả về weights shape `(B, L/4)` = `(B, 64)` → expose qua ONNX output thứ 2
- Upsample `np.repeat(weights, 4)` → map về character positions
- Hiển thị ANSI 256-color terminal heatmap (gradient pale yellow → bright red)
- Bổ sung: top-K lexical features theo `|z-score|` để cover nhánh lexical path

```powershell
python "scripts/predict-url_4.py" --explain --split-mode domain "https://example.com"
python "scripts/predict-url_4_onnx.py" --explain --split-mode domain "https://example.com"
```

### 4.7 Deployment — ONNX

| Backend | Latency (ms/URL) | URL/s | Ghi chú |
|---|---|---|---|
| PyTorch FP32 CPU | 2.543 | 393 | Baseline |
| **ONNX FP32 CPU** | **1.671** | **599** | **Recommended production** |
| ONNX INT8 CPU | 6.595 | 152 | Chậm hơn FP32! (LSTM không có INT8 kernel) |

- ONNX FP32: 3.04 MB | INT8: 0.79 MB (3.84x compression)
- Exporter: `dynamo=False` (legacy TorchScript) — dynamo exporter trong torch 2.11 bị broken cho LSTM

**Known limitation**: `https://google.com` (bare hostname, no path) bị FP ~99% vì `path_depth=0` lệch -3.04 std dưới TRAIN mean. Workaround: dùng domain allowlist filter cho URL ngắn.

---

## 5. Model 2 — XGBoost

### 5.1 Mục tiêu

Model tabular bổ sung cho CNN-LSTM: học từ **features kỹ thuật có thể diễn giải** (interpretable features), đặc biệt tốt với các signal không phải sequence như TLD, IP host, entropy, charset anomaly.

### 5.2 Features (105 features)

Script: `scripts/3. feature-extract_xgb_4.py`

| Nhóm | Số features | Ví dụ |
|---|---|---|
| **Structural** | 12 | `url_len, host_len, path_len, path_depth, is_bare_hostname, path_to_url_ratio` |
| **Char composition** | 10 | `digit_count, digit_density, hyphen_count, dot_count, at_count, pct_count, amp_count` |
| **Case/charset** | 6 | `mixed_case_ratio, has_upper, has_control_chars, has_non_ascii, non_ascii_ratio, has_punycode` |
| **Host-specific** | 9 | `host_digit_density, host_hyphen_count, host_longest_digit_run, host_vowel_ratio, host_entropy` |
| **Entropy** | 3 | `entropy_url, entropy_host, entropy_path` |
| **Scheme** | 3 | `is_http, is_https, is_ip_host` |
| **TLD one-hot** | 53 | Top 50 TLDs + `tld_other, tld_missing, tld_label_count` |
| **TLD buckets** | 3 | `tld_is_vn, tld_is_suspicious, tld_is_common` |
| **Phishing keywords** | 2 | `num_phishing_keywords, has_phishing_keyword` |
| **File extension** | 3 | `has_malware_ext, has_cdn_ext, has_any_ext` |

Lưu ý thiết kế:
- Tất cả feature đều **stateless** (không dùng statistics train set) → không có train/test leakage
- Bỏ SCAM_BAIT/C2_PATHS/BRANDS/BENIGN_WORDS (synthetic-era artifacts từ model 2 cũ)
- TOP_TLDS rebuilt từ EDA model 4 (bao gồm cả benign-heavy: `.vn, .gov.uk` và mal-heavy: `.xyz, .top, .click, .sbs`)
- `.vn` TLD = 19.85% benign vs 0.18% mal → `tld_is_vn` là signal cực mạnh

### 5.3 Training

| Tham số | Giá trị |
|---|---|
| Algorithm | XGBoost `binary:logistic`, `tree_method=hist` |
| Hardware | CUDA (RTX 3060 Ti) |
| max_depth | 8 |
| learning_rate | 0.05 |
| n_estimators | 2000 (ES tại vòng 1991) |
| subsample / colsample_bytree | 0.8 / 0.8 |
| scale_pos_weight | 4.0 (80/20 dataset) |
| early_stopping_rounds | 50 (eval trên val ROC-AUC) |
| Training time | 234.9s trên CUDA |

### 5.4 Kết quả

**Val set (1.97M URLs):**
- ROC-AUC = 0.999999 | PR-AUC = 0.999995 | F1@0.5 = 0.9996

**Test set (1.97M URLs):**
| Mode | Threshold | F1 | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|
| **default** | 0.400 | 0.999618 | 0.999418 | 0.999817 | 229 | 72 |
| **safer** | 0.002 | 0.994969 | 0.990001 | 0.999987 | 3,975 | 5 |
| **precise** | 0.998 | 0.994930 | 0.999908 | 0.990002 | 36 | 3,935 |

**Curve metrics: ROC-AUC = 0.999999 | PR-AUC = 0.999995**

Model: `models/xgb_url_4.ubj` (14.8 MB)

### 5.5 Explainability — SHAP

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_row)
```

- SHAP `TreeExplainer` — sub-millisecond per URL (cây ensemble, tính exact không approximate)
- Output: top-K features theo `|SHAP value|`, signed contribution (positive → push MAL, negative → push BEN)
- SHAP values ở log-odds space: `sigmoid(expected_value + sum(shap_values)) ≈ probability`

```powershell
python "scripts/predict-url_4_xgb.py" --explain --topk 10 "https://example.com"
```

---

## 6. Model 3 — SGDClassifier (HTML Content)

### 6.1 Mục tiêu

Phân tích **nội dung HTML thô** của trang web, không phải URL. URL có thể trông hoàn toàn bình thường nhưng nội dung HTML phishing kit thì khác biệt rõ rệt.

### 6.2 Dataset

**Nguồn**: HuggingFace — [`phreshphish/phreshphish`](https://huggingface.co/datasets/phreshphish/phreshphish)
- Raw HTML files từ benign và malicious/phishing websites thực tế
- Format: parquet files (train-*.parquet, test-*.parquet)
- Labels: `benign` / `phish`
- Vị trí local: `D:\phreshphish\data\`

### 6.3 Kỹ thuật

**Vectorization**: `HashingVectorizer`
- `analyzer="char"` (character n-grams)
- `ngram_range=(3, 4)` — char 3-grams và 4-grams
- `n_features=2^20` (~1M buckets)
- `norm="l2"`, `lowercase=False` (case-sensitive — quan trọng vì HTML có case signal)
- HTML preprocessing: normalize whitespace, giới hạn `HTML_MAX_CHARS=10,000`

**Classifier**: `SGDClassifier`
- `loss="log_loss"` (logistic regression online)
- `class_weight="balanced"`
- `partial_fit()` — online learning, xử lý từng batch parquet file
- Cache vectorized output bằng joblib để tránh re-vectorize

### 6.4 Kết quả

**Test set (168,060 samples):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| **benign** | 0.8150 | 0.9302 | 0.8688 |
| **phish** | 0.9003 | 0.7491 | 0.8178 |
| **macro avg** | 0.8577 | 0.8396 | 0.8433 |
| **weighted avg** | 0.8540 | 0.8474 | 0.8455 |

**Accuracy**: 0.8474 | **ROC-AUC**: 0.9277

Model: `D:\phreshphish\models\SGDClassifier_2.joblib` (4.2 MB)

### 6.5 Inference

```powershell
# Single HTML file
python "scripts/predict-rawHTML-SGDClassifier.py" --file page.html

# Thư mục chứa nhiều file HTML
python "scripts/predict-rawHTML-SGDClassifier.py" --folder ./pages --recursive
```

---

## 7. Ensemble (Phase 2 — Chưa implement)

### 7.1 Vấn đề cần giải quyết

Ba model phân tích từ góc độ khác nhau, mỗi model có điểm mạnh/yếu:

| Model | Điểm mạnh | Điểm yếu |
|---|---|---|
| CNN-LSTM | Hiểu pattern char-level, attention explanation | FP trên bare hostname, cần URL string |
| XGBoost | Features diễn giải được, SHAP exact, nhanh | Không hiểu context sequence |
| SGDClassifier | Phân tích content HTML thực tế | Cần fetch HTML (latency cao), F1 thấp hơn |

### 7.2 Chiến lược ensemble (dự kiến)

**Input combination:**
- URL-based: CNN-LSTM prob + XGBoost prob → weighted average hoặc meta-learner
- Content-based: SGD prob → chỉ dùng khi HTML page available

**Voting scheme (dự kiến):**
```
final_prob = w1 * p_cnn_lstm + w2 * p_xgb    # URL-only mode
final_prob = w1 * p_cnn_lstm + w2 * p_xgb + w3 * p_sgd  # Full mode (có HTML)
```

Weights sẽ được tune trên val set. Cũng có thể dùng stacking (LightGBM/LogReg meta-learner).

### 7.3 Roadmap implement

- [ ] Script ensemble URL-only: load CNN-LSTM ONNX + XGBoost, predict cả 2, weighted average
- [ ] Script ensemble full: + SGDClassifier khi HTML available
- [ ] Tune weights trên val set
- [ ] Unified CLI: `predict-ensemble.py --url <URL> [--html page.html]`
- [ ] Unified explanation output: merge attention heatmap + SHAP values + (HTML features)

---

## 8. Pipeline tổng thể

```
Data Collection
  ├── crawl-benign-urls.py      → dataset/intl-benign/urls.csv
  ├── fetch-malicious-url.py    → dataset/vn-malicious/urls-all.csv
  └── fetch-malicious-extra.py  → (append vào urls-all.csv)
        ↓
  balance-dataset-3.py          → dataset/dataset-4-final.csv (19.68M, QUOTE_ALL)

CNN-LSTM Pipeline (Model 1)
  ├── 2. eda 4.py               → dataset/eda-result 4.txt
  ├── 3. preprocess-data 4.py   → data/processed/model_4/{random,domain}/
  ├── 4-feat. extract-lexical_4.py → feat.npy + feat_stats.json per mode
  ├── 4. train-model_4.py       → models/cnn_lstm_best_4_{random,domain}.pt
  ├── 6. tune-threshold_4.py    → models/thresholds_4.json
  ├── 5. eval-model_4.py        → models/results_4.json
  └── 8. export-onnx_4.py       → models/cnn_lstm_4_*.onnx (FP32 + INT8)

XGBoost Pipeline (Model 2)
  ├── 3. feature-extract_xgb_4.py → data/processed/xgboost_4/
  ├── 4. train-xgb_4.py         → models/xgb_url_4.ubj
  └── 5. eval-xgb_4.py          → models/results_xgb_4.json + thresholds_xgb_4.json

SGDClassifier Pipeline (Model 3)
  └── 4. train-model-SGDClassifier.py → D:\phreshphish\models\SGDClassifier_2.joblib

Inference
  ├── predict-url_4.py          → CNN-LSTM PyTorch (+ --explain attention)
  ├── predict-url_4_onnx.py     → CNN-LSTM ONNX (+ --explain attention)
  ├── predict-url_4_xgb.py      → XGBoost (+ --explain SHAP)
  └── predict-rawHTML-SGDClassifier.py → SGDClassifier HTML

[Phase 2]
  └── predict-ensemble.py       → Combine tất cả 3 models → final decision
```

---

## 9. Kết quả tổng hợp

| Model | Test F1 | ROC-AUC | PR-AUC | Latency | Explanation |
|---|---|---|---|---|---|
| **CNN-LSTM (domain split)** | **0.9888** | 0.9989 | 0.9872 | 1.67 ms/URL (ONNX) | Attention heatmap + lexical z-scores |
| **CNN-LSTM (random split)** | 0.999926 | 0.999992 | 0.999969 | 1.67 ms/URL | (in-distribution, không dùng production) |
| **XGBoost** | 0.999618 | 0.999999 | 0.999995 | < 1 ms/URL | SHAP values (exact) |
| **SGDClassifier (HTML)** | 0.847 (accuracy) | 0.9277 | — | depends on HTML fetch | — |

> **Production**: CNN-LSTM domain split + XGBoost kết hợp cho URL-based detection. SGDClassifier là tầng bổ sung khi có HTML content.

---

## 10. Tech Stack

| Component | Công nghệ |
|---|---|
| **Deep Learning** | PyTorch 2.11, CUDA/AMP/TF32 |
| **Deployment DL** | ONNX Runtime (FP32 + INT8 dynamic quant) |
| **Gradient Boosting** | XGBoost ≥ 2.0 (CUDA `tree_method=hist`) |
| **Content Model** | scikit-learn SGDClassifier + HashingVectorizer |
| **Explainability** | Attention weights (built-in) + SHAP TreeExplainer |
| **Data** | numpy memmap, pandas, polars (SGD pipeline) |
| **URL parsing** | tldextract (PSL snapshot, offline) |
| **Web crawling** | requests + BeautifulSoup4 + urllib.robotparser |
| **Serialization** | ONNX, XGBoost UBJ, joblib |
| **Environment** | Python 3.14, venv, Windows 10 |

---

## 11. Quyết định quan trọng đã chốt

1. **100% real data** — Không dùng synthetic URL generator (Char-RNN từ model 2 cũ). Dữ liệu thật = model generalize tốt hơn.
2. **Domain split** cho production — Random split inflated (F1 0.99996 vs 0.9888 thực). Domain split mới phản ánh khả năng tổng quát hóa thực sự.
3. **Attention thay SHAP** cho CNN-LSTM — SHAP/DeepExplainer 1-15s/URL không dùng được real-time. Attention weights free từ forward pass.
4. **ONNX FP32 thay INT8** cho production CNN-LSTM — LSTM không có INT8 kernel trong onnxruntime CPU EP; INT8 chậm hơn FP32 3.9x (6.6ms vs 1.7ms).
5. **Case-preserved encoding** — Mixed-case ratio là signal có thống kê (benign 11.83% vs mal 5.50%).
6. **ONNX legacy exporter** (`dynamo=False`) — Dynamo exporter trong torch 2.11 output 0.02 MB ONNX (mất weights).
7. **Lazy-mmap pattern cho DataLoader** — Windows không thể pickle memmap qua `multiprocessing.spawn`. Giải pháp: `__init__` chỉ lưu path, `__getitem__` mở mmap lần đầu per worker.

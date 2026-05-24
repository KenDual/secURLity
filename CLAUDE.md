# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**secURLity** là hệ thống phát hiện URL độc hại (malicious URL detection) kết hợp **3 model AI** phân tích từ các góc độ khác nhau, huấn luyện trên **19,678,135 URL thật** (80/20 benign/malicious, không synthetic).

### 3 Models — tổng quan nhanh

| Model | Approach | Explainability | Production inference |
|---|---|---|---|
| **CNN-LSTM** (chính) | Char-level sequence + 30 lexical features | Attention heatmap (built-in, ~0 cost) | ONNX FP32 1.67 ms/URL |
| **XGBoost** | 105 tabular features từ URL | SHAP TreeExplainer (exact, sub-ms) | < 1 ms/URL (UBJ file) |
| **SGDClassifier** | Raw HTML content (char 3-4 grams) | — | Cần fetch HTML page trước |

**Ensemble** giữa 3 model: **chưa implement** (Phase 2). Hiện tại mỗi model chạy độc lập.

**Reference docs:**
- `PLAN.md` — Kiến trúc, ý tưởng, chiến lược tổng thể (đọc trước nếu mới vào dự án)
- `checklist.md` — Technical checklist chi tiết + metrics thực tế
- `CNN-LSTM-final.md` — Implementation log + decision history đầy đủ cho CNN-LSTM

---

## Model 1 — CNN-LSTM

Hybrid architecture kết hợp char-level sequence path + lexical feature path:
- Char path: `Embedding(94, 64)` → `Conv1d(64→128)` × 2 + MaxPool → `Stacked BiLSTM (2L, h=128)` → `AttentionPool` (pad-aware)
- Lexical path: 30 engineered features → `Linear(30→64→32)` MLP
- Hybrid head: `concat(256 + 32)` → `Linear(288→64→1)` → logit
- 795,106 params (~0.80 M), trained on 19,678,135 URLs

Explainability: **attention-based (Option C), NOT SHAP**. Attention weights exposed as 2nd ONNX output.

### Two split modes (random vs domain)

| | Random split | Domain split |
|---|---|---|
| Split logic | stratified 80/10/10 by label | greedy bin-pack 80/10/10 by registered domain (zero overlap) |
| Val F1 | 0.99996 (epoch 25) | 0.7049 (epoch 0 — overfit immediately) |
| **Test F1** | **0.999926** (thr=0.42) | **0.9888** (thr=0.98) |
| Test ROC-AUC | 0.999992 | 0.998894 |
| Test PR-AUC | 0.999969 | 0.987224 |
| Production use? | No — in-distribution, inflated | **YES** — generalizes to new domains |
| Checkpoint | `models/cnn_lstm_best_4_random.pt` | `models/cnn_lstm_best_4_domain.pt` |
| ONNX (FP32) | `models/cnn_lstm_4_random.onnx` | `models/cnn_lstm_4_domain.onnx` |
| ONNX (INT8) | `models/cnn_lstm_4_random_int8.onnx` | `models/cnn_lstm_4_domain_int8.onnx` |

**Production rule**: always use `--split-mode domain`. Random split kept for comparison only.

---

## Model 2 — XGBoost

105 tabular features extracted from URL string. Trained on same 19.68M URL dataset (random split).

**Test results**: ROC-AUC = 0.999999 | PR-AUC = 0.999995 | F1 (default thr=0.40) = **0.999618**  
**Model file**: `models/xgb_url_4.ubj` (14.8 MB)  
**Thresholds**: `models/thresholds_xgb_4.json` (default=0.40, safer=0.002, precise=0.998)

Explainability: **SHAP TreeExplainer** (exact, sub-millisecond per URL — appropriate for tree ensembles).  
Do NOT confuse: SHAP is correct for XGBoost. SHAP was dropped for CNN-LSTM only (too slow there).

---

## Model 3 — SGDClassifier (HTML Content)

Classifies raw HTML page content using `HashingVectorizer` (char 3-4 grams, 2^20 features) + `SGDClassifier` trained on HuggingFace dataset `phreshphish/phreshphish`.

**Test results**: accuracy = 0.8474 | ROC-AUC = 0.9277 | F1(phish) = 0.8178  
**Model file**: `D:\phreshphish\models\SGDClassifier_2.joblib` (4.2 MB) — **different root than project**  
**Dataset local path**: `D:\phreshphish\data\` (parquet files, NOT in `D:\! secURLity\`)

This model requires fetching the HTML page first; it complements URL-based models for content-level analysis.

---

## Full Pipeline

```powershell
.\venv\Scripts\Activate.ps1

# =====================================================================
# DATA COLLECTION (one-time)
# =====================================================================

# Crawl benign URLs (sitemap + BFS crawler + crt.sh subdomain)
python "scripts/crawl-benign-urls.py"                         # seeds: dataset/intl-benign/seeds.txt
python "scripts/crawl-benign-urls.py" --resume                # resume interrupted run

# Fetch malicious URLs (13 free public threat feeds)
python "scripts/fetch-malicious-url.py"
python "scripts/fetch-malicious-extra.py"                     # extra sources (Tier A/B), parallel

# Balance to 80/20 → dataset/dataset-4-final.csv (19.68M, QUOTE_ALL)
python "scripts/balance-dataset-3.py"

# =====================================================================
# CNN-LSTM PIPELINE (Model 1)
# =====================================================================

# EDA → dataset/eda-result 4.txt
python "scripts/2. eda 4.py"

# Preprocess: BOTH random + domain modes in 1 run
python "scripts/3. preprocess-data 4.py"

# Extract 30 lexical features, normalize from TRAIN stats
python "scripts/4-feat. extract-lexical_4.py"                 # both modes by default

# Train (per split-mode)
python "scripts/4. train-model_4.py" --split-mode random      # ~2-3h on 3060 Ti
python "scripts/4. train-model_4.py" --split-mode domain      # overfits epoch 1, fast

# Threshold tuning + eval
python "scripts/6. tune-threshold_4.py" --split-mode random
python "scripts/6. tune-threshold_4.py" --split-mode domain
python "scripts/5. eval-model_4.py" --split-mode random       # writes models/results_4.json
python "scripts/5. eval-model_4.py" --split-mode domain

# ONNX export (FP32 + INT8, 2 outputs: logit + attention_weights)
python "scripts/8. export-onnx_4.py" --split-mode random
python "scripts/8. export-onnx_4.py" --split-mode domain

# Inference
python "scripts/predict-url_4.py" --split-mode domain "https://example.com"
python "scripts/predict-url_4.py" --split-mode domain --explain "https://example.com"
python "scripts/predict-url_4_onnx.py" --split-mode domain "https://example.com"   # 1.67ms/URL
python "scripts/predict-url_4_onnx.py" --split-mode domain --explain "https://example.com"

# =====================================================================
# XGBOOST PIPELINE (Model 2)
# =====================================================================

# Extract 105 features (multiprocessing, reads from model_4/random splits)
python "scripts/3. feature-extract_xgb_4.py"

# Train (CUDA, early stopping, ~235s)
python "scripts/4. train-xgb_4.py"

# Eval + threshold tuning → results_xgb_4.json + thresholds_xgb_4.json
python "scripts/5. eval-xgb_4.py"

# Inference (+ SHAP explanation)
python "scripts/predict-url_4_xgb.py" "https://example.com"
python "scripts/predict-url_4_xgb.py" --explain --topk 10 "https://example.com"
python "scripts/predict-url_4_xgb.py" --thr safer "https://example.com"

# =====================================================================
# SGDCLASSIFIER PIPELINE (Model 3) — paths under D:\phreshphish\
# =====================================================================

# Train (dataset from D:\phreshphish\data\, model saved to D:\phreshphish\models\)
python "scripts/4. train-model-SGDClassifier.py"

# Inference (takes .html file or folder)
python "scripts/predict-rawHTML-SGDClassifier.py" --file page.html
python "scripts/predict-rawHTML-SGDClassifier.py" --folder ./pages --recursive
```

---

## Directory layout (production-relevant)

```
D:\! secURLity\
├── PLAN.md                                  # Project plan, architecture, strategy
├── checklist.md                             # Technical checklist + all metrics
├── CNN-LSTM-final.md                        # CNN-LSTM implementation log + decisions
├── CLAUDE.md                                # This file
├── venv\                                    # Python 3.14 + PyTorch 2.11 + onnxruntime
│
├── dataset\
│   ├── dataset-4-final.csv                  # 19.68M URLs (80/20, QUOTE_ALL)
│   ├── eda-result 4.txt                     # EDA output
│   └── intl-benign\
│       ├── seeds.txt                        # Crawl seeds (international)
│       └── urls.csv                         # Crawled benign URLs
│
├── data\processed\
│   ├── model_4\
│   │   ├── vocab.json                       # 94 chars, case-preserved
│   │   ├── metadata.json                    # max_len=256, pos_weight per mode
│   │   ├── random\
│   │   │   ├── feat_stats.json              # mean/std for lexical normalization
│   │   │   └── {train,val,test}\{X.npy, y.npy, feat.npy, split.csv}
│   │   └── domain\                          # same structure
│   └── xgboost_4\
│       ├── xgb_4_feature_cols.json          # 105 feature names (ordered)
│       ├── xgb_4_feature_meta.json          # TLD lists, pos_weight, notes
│       └── {train,val,test}_{X,y}_xgb.npy  # XGBoost feature matrices
│
├── models\
│   ├── cnn_lstm_best_4_random.pt            # CNN-LSTM checkpoint (random)
│   ├── cnn_lstm_best_4_domain.pt            # CNN-LSTM checkpoint (domain) — PRODUCTION
│   ├── cnn_lstm_4_{random,domain}.onnx      # ONNX FP32 (2 outputs: logit + attention)
│   ├── cnn_lstm_4_{random,domain}_int8.onnx # ONNX INT8 (3.84x smaller, slower on CPU)
│   ├── xgb_url_4.ubj                        # XGBoost model (14.8 MB)
│   ├── results_4.json                       # CNN-LSTM train+test metrics per split-mode
│   ├── thresholds_4.json                    # CNN-LSTM operating points per split-mode
│   ├── results_xgb_4.json                   # XGBoost test metrics + operating points
│   ├── thresholds_xgb_4.json               # XGBoost operating points
│   ├── xgb_url_4_metrics.json              # XGBoost training metrics
│   ├── xgb_url_4_importance.json           # Feature importance (gain/weight/cover)
│   └── result_SGDClassifier.txt            # SGDClassifier classification report
│
├── scripts\
│   ├── crawl-benign-urls.py                # Sitemap + BFS crawler + crt.sh
│   ├── fetch-malicious-url.py              # 13 threat feeds (no-auth)
│   ├── fetch-malicious-extra.py            # Extra sources, parallel, incremental
│   ├── balance-dataset-3.py               # 80/20 balance → dataset-4-final.csv
│   ├── 2. eda 4.py                         # EDA (streaming, constant memory)
│   ├── 3. preprocess-data 4.py             # Both random+domain splits in 1 run
│   ├── 4-feat. extract-lexical_4.py        # 30 lexical features, log1p+std from TRAIN
│   ├── 4. train-model_4.py                 # AdamW + cosine + LS + AMP + lazy-mmap
│   ├── 5. eval-model_4.py                  # Test eval (num_workers=0)
│   ├── 6. tune-threshold_4.py              # Sweep 0.01-0.99, 3 operating points
│   ├── 8. export-onnx_4.py                 # ONNX FP32+INT8, dynamo=False
│   ├── predict-url_4.py                    # CNN-LSTM PyTorch inference + --explain
│   ├── predict-url_4_onnx.py               # CNN-LSTM ONNX inference + --explain
│   ├── 3. feature-extract_xgb_4.py         # 105 XGBoost features (multiprocessing)
│   ├── 4. train-xgb_4.py                   # XGBoost CUDA training
│   ├── 5. eval-xgb_4.py                    # XGBoost test eval + threshold tuning
│   ├── predict-url_4_xgb.py               # XGBoost inference + SHAP --explain
│   ├── 4. train-model-SGDClassifier.py     # SGDClassifier HTML online learning
│   └── predict-rawHTML-SGDClassifier.py    # SGDClassifier inference (--file/--folder)

D:\phreshphish\                              # SGDClassifier dataset & model (SEPARATE ROOT)
├── data\                                   # train-*.parquet, test-*.parquet (HuggingFace)
├── models\SGDClassifier_2.joblib           # Trained model (4.2 MB)
└── cache\                                  # Vectorized cache keyed by config hash
```

---

## Key technical details

### CNN-LSTM — Char-level encoding
- `MAX_LEN = 256` (EDA P99.5=217 → 256 covers ~P99.7, multiple of 64)
- `vocab_size = 94`: `<PAD>(0)` + `<UNK>(1)` + 52 letters (case-preserved!) + 10 digits + 30 special chars
- **Do NOT lowercase URL** — case is signal (mixed-case ratio: benign 11.83% vs mal 5.50%)
- Always read `MAX_LEN`, `vocab_size`, `pad_idx` from `metadata.json` or checkpoint `config` — never hardcode

### CNN-LSTM — Lexical features (30, strict order)
Order must match `FEATURE_NAMES` in `4-feat. extract-lexical_4.py`:
- Lengths (4): `url_length, host_length, path_length, query_length`
- Counts (14): `num_{dots, hyphens, underscores, slashes, question_marks, equals, amps, ats, percents, digits, subdomains}, subdomain_length_max, path_depth, query_param_count`
- Ratios (4): `digit_ratio, letter_ratio, vowel_ratio, special_char_ratio`
- Binary flags (7): `has_ip_host, has_port, is_https, tld_is_vn, tld_is_common, has_punycode, has_double_slash_in_path`
- Entropy (1): `host_entropy` (Shannon)

Normalization: log1p for 18 count/length features, then standardize all with TRAIN-only mean/std from `feat_stats.json`. Apply same transform at inference.

### CNN-LSTM — Training
- `AdamW(lr=1e-3, weight_decay=0.01)` + `LinearLR (5% warmup) → CosineAnnealingLR (eta_min=1e-5)` per-batch
- `BCEWithLogitsLoss(pos_weight=4.0)` + label smoothing ε=0.05
- Batch 2048, AMP FP16 + GradScaler, TF32 enabled
- `mmap_mode='r'` on train arrays — peak RAM ~6-8 GB

### CNN-LSTM — Threshold tuning
3 operating points in `models/thresholds_4.json` per split-mode:
- `default`: max F1 | `safer`: lowest thr where precision ≥ 0.99 | `precise`: highest thr where recall ≥ 0.99

Domain split: `safer`/`precise` collapse at thr=0.98 (sweep step 0.01 too coarse) — use `default` for production.

### CNN-LSTM — Explainability (attention-based, Option C — NOT SHAP)
SHAP dropped for CNN-LSTM (1-15s/URL). Built-in `AttentionPool` weights are free.

- **PyTorch**: monkey-patch `model.attn_pool.forward` → stash `_last_weights`. Zero training impact.
- **ONNX**: wrapper `CNNLSTMWithAttention` exposes `attention_weights` as 2nd output (shape `(B, 64)`).

Upsample: `np.repeat(weights, 4)` then crop to URL length (each position covers 4 chars after 2× MaxPool).  
Also shows top-K lexical features by `|z-score|` from `feat_stats.json` — covers lexical path.

### CNN-LSTM — ONNX deployment
- **Use `dynamo=False`** (legacy TorchScript exporter). Dynamo exporter in torch 2.11 produces 0.02 MB ONNX (loses weights) with hardcoded reshape → breaks dynamic batch on LSTM.
- CPU latency (1-thread, 1 URL): PyTorch 2.54 ms | **ONNX FP32 1.67 ms (recommended)** | ONNX INT8 6.60 ms (slower — LSTM has no INT8 kernel in onnxruntime CPU EP)
- INT8: 0.79 MB (3.84x vs FP32's 3.04 MB), F1 drop ≤ 0.05 pp — use only when disk size critical

### XGBoost — Features (105, strict order)
Order defined by `FEATURE_NAMES` in `3. feature-extract_xgb_4.py` (deterministic via probe URL).  
All features are stateless — no train statistics used → no leakage across splits.

Key feature groups:
- Structural (12): `url_len, host_len, path_depth, is_bare_hostname, path_to_url_ratio, ...`
- Case/charset (6): `mixed_case_ratio, has_upper, has_control_chars, has_non_ascii, has_punycode, ...`
- TLD one-hot (50) + aggregate buckets (3): `tld_is_vn` (19.85% benign vs 0.18% mal — strong signal), `tld_is_suspicious`, `tld_is_common`
- Entropy (3): `entropy_url, entropy_host, entropy_path`
- Phishing keywords (2), file extensions (3), host-specific (9), ...

Feature columns saved in `data/processed/xgboost_4/xgb_4_feature_cols.json`. Inference must use same `extract_features()` function from `3. feature-extract_xgb_4.py` (loaded via `importlib`).

### XGBoost — SHAP explainability
SHAP IS appropriate for XGBoost (not CNN-LSTM). `TreeExplainer` computes exact Shapley values sub-millisecond.

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_row)  # shape (1, 105), in log-odds space
```

Positive SHAP value → push toward MAL; negative → push toward BEN.  
`sigmoid(expected_value + sum(shap_values)) ≈ model probability`.

### SGDClassifier — HTML preprocessing (must stay in sync)
The `normalize_html()` function in `predict-rawHTML-SGDClassifier.py` **must be identical** to the preprocessing in `4. train-model-SGDClassifier.py`. Any change to one must be applied to both:

```python
html = html[:max_chars * 5]
html = re.sub(r">\s+<", "><", html)
html = re.sub(r"\s+", " ", html)
html = html.strip()[:max_chars]   # max_chars = 10000
```

Vectorizer params (also must match): `analyzer="char", ngram_range=(3,4), n_features=2**20, norm="l2", lowercase=False`.

### Risk scoring (all models)
All predict scripts output a 4-tier risk level from raw probability (independent of binary decision):
- 0–25% → **Low Risk** | 26–50% → **Caution** | 51–75% → **Suspicious** | 76–100% → **High Risk**

`[ben + High Risk]` means: model output >75% but below threshold → manual review candidate.

---

## Hardcoded paths

`PROJECT_ROOT = Path(r"D:\! secURLity")` appears in:
- CNN-LSTM: `2. eda 4.py`, `3. preprocess-data 4.py`, `4-feat. extract-lexical_4.py`, `4. train-model_4.py`, `5. eval-model_4.py`, `6. tune-threshold_4.py`, `8. export-onnx_4.py`, `predict-url_4.py`, `predict-url_4_onnx.py`
- XGBoost: `3. feature-extract_xgb_4.py`, `4. train-xgb_4.py`, `5. eval-xgb_4.py`, `predict-url_4_xgb.py`
- Data crawlers: `crawl-benign-urls.py`, `fetch-malicious-url.py`, `fetch-malicious-extra.py`

**SGDClassifier uses a DIFFERENT root**: `D:\phreshphish\` (hardcoded in `4. train-model-SGDClassifier.py` and `predict-rawHTML-SGDClassifier.py`). Update separately if moved.

---

## Known issues / lessons

1. **Windows DataLoader + numpy.memmap → OSError [Errno 22]**: Cannot pickle memmap through `multiprocessing.spawn`. `URLDataset` uses lazy-mmap: `__init__` stores paths only, `__getitem__` opens mmap on first call per worker, `__getstate__` clears mmap fields before pickle.

2. **Train script crashes at test phase** (Windows handle exhaustion after 18+ epochs of DataLoader workers). Workaround: train script never loads test split. Run `5. eval-model_4.py` separately (`num_workers=0`, test split only).

3. **importlib import for awkward filenames**: `5. eval-model_4.py` and `6. tune-threshold_4.py` import `CNNLSTM`/`URLDataset` from `4. train-model_4.py` via `importlib` (filename has space + leading digit). Similarly, `predict-url_4_xgb.py` imports `extract_features` from `3. feature-extract_xgb_4.py`. Worker processes can't find dynamically-loaded modules → these scripts default to `num_workers=0`.

4. **PyTorch 2.11 ONNX dynamo exporter broken for LSTM**: 0.02 MB output, hardcoded reshape. Always use `dynamo=False`.

5. **ONNX INT8 slower than FP32 on LSTM**: No INT8 kernel in onnxruntime CPU EP for LSTM ops → dequantize overhead. Use ONNX FP32 for production latency.

6. **CNN-LSTM false-positive on bare hostnames**: `https://google.com` → MAL ~99%. `path_depth=0` is -3.04 std below TRAIN mean (threat-feed URLs often have no path; crawled benign URLs typically do). Workaround: domain allowlist filter before invoking model for very-short URLs.

7. **Domain split threshold coarse resolution**: `safer`/`precise` collapse at thr=0.98 because sweep step=0.01 is too coarse for [0.98, 0.99] range. Future improvement: sweep step=0.001 in that region.

8. **SGDClassifier paths under different root**: `D:\phreshphish\` — don't confuse with `D:\! secURLity\`. Dataset (parquet), model (joblib), and cache are all under the phreshphish root.

---

## When making changes

### CNN-LSTM changes
- Default to `--split-mode domain` for all production/inference recommendations.
- Read `MAX_LEN`, `vocab_size`, `pad_idx` from `metadata.json` or checkpoint `config` — never hardcode.
- Architecture changes: update `CNNLSTM` in `4. train-model_4.py` AND `CNNLSTMWithAttention` in `8. export-onnx_4.py` — they must replicate the same forward pass.
- Predict script changes: `encode_url`, `normalize_features`, `resolve_threshold`, attention helpers live in `predict-url_4.py` only — `predict-url_4_onnx.py` imports them via importlib.
- Do NOT re-introduce SHAP for CNN-LSTM. Decision documented in `CNN-LSTM-final.md` Item 12 (Option C: attention-based).
- Do NOT add `matplotlib` unless explicitly requested — scripts gracefully skip plot generation.

### XGBoost changes
- Feature order is **fixed** by `FEATURE_NAMES` in `3. feature-extract_xgb_4.py`. Adding/removing features requires re-running the full XGBoost pipeline (extract → train → eval).
- Always load `extract_features()` from `3. feature-extract_xgb_4.py` via importlib at inference — never duplicate the function.
- SHAP is correct for XGBoost. Use `shap.TreeExplainer` (not DeepExplainer or KernelSHAP).
- `predict-url_4_xgb.py` uses `thresholds_xgb_4.json` (not `thresholds_4.json` which is CNN-LSTM).

### SGDClassifier changes
- Any change to `normalize_html()` or vectorizer params MUST be applied to both `4. train-model-SGDClassifier.py` and `predict-rawHTML-SGDClassifier.py`. Mismatch = silent accuracy regression.
- Paths are under `D:\phreshphish\` — update both scripts if moved.
- `partial_fit()` requires `classes=np.array([0, 1])` on every call.

### Ensemble (Phase 2 — not yet implemented)
- When implementing: load CNN-LSTM ONNX domain + XGBoost, combine probabilities.
- SGDClassifier is optional path (requires HTML fetch, higher latency).
- Tune ensemble weights on val set (not test set).

---

## Legacy models (1/2/3) — deprecated

Artifacts kept on disk but **not maintained**:
- `models/cnn_lstm_best_{1,2,3}.pt`, `models/results_{1,2,3}.json`, `models/xgb_url_2.ubj`
- `dataset/dataset {1,2,3}.csv`, `dataset/eda-result {1,2,3}.txt`
- `scripts/*_1.py`, `*_2.py`, `*_3.py` variants

Differ from current: `MAX_LEN=100`, `vocab_size=51`, no lexical features, no attention export, synthetic data (model 2) or VN-only crawl (model 3). Do not mix with model 4 artifacts.

---

## Results at a glance

| Model | Test F1 | ROC-AUC | PR-AUC | Note |
|---|---|---|---|---|
| CNN-LSTM (domain — production) | **0.9888** | 0.998894 | 0.987224 | thr=0.98 |
| CNN-LSTM (random — reference) | 0.999926 | 0.999992 | 0.999969 | thr=0.42, in-distribution |
| XGBoost | **0.999618** | 0.999999 | 0.999995 | thr=0.40, random split |
| SGDClassifier (HTML) | 0.8178 (F1 phish) | 0.9277 | — | accuracy=0.847 |
| **Ensemble** | TBD | TBD | TBD | Phase 2 — not implemented |

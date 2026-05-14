# secURLity — Project Checklist

Hiện đang focus: **Model 3 — VN real-world** (`dataset/dataset 3 (vn).csv`, ~15M URLs, 12.5M benign crawl + 2.6M malicious từ public threat feeds). Artifact ở `data/processed/model_3/` và `models/*_3.*`. Model 2 (`dataset 2.csv`, 10M Char-RNN + rule-based) vẫn được maintain song song để so sánh.

## 0. Environment

- [x] Set up Python venv (`./venv/`) with GPU support (CUDA on GTX 1660 SUPER, 6.4GB VRAM).
- [x] Install dependencies: `torch`, `scikit-learn`, `pandas`, `numpy`, `tqdm`.
- [x] Install XGBoost stack: `pip install xgboost tldextract pyarrow` *(tldextract dùng cho feature extraction; pyarrow optional nhưng tăng tốc IO).* SHAP và TensorBoard cài sau, ở mục 9.

## 1. Data generation (model 2)

- [x] Train Char-RNN URL generator trên `dataset/malicious_dataset.csv` (413k URLhaus rows). Checkpoint: `models/char_rnn_url_generator (model 2).pt`, vocab: `models/char_rnn_vocab (model 2).json`.
- [x] Generate 3M malicious URLs từ Char-RNN → `dataset/dataset-malicious 2.csv`.
- [x] Generate 7M benign URLs từ wordlist generator → `dataset/dataset-benign 2.csv`.
- [x] Merge benign + malicious → `dataset/dataset 2.csv` (~535MB, 10M rows). Columns: `url`, `label` (0=benign, 1=malicious).

## 2. Exploratory data analysis

- [x] `scripts/2. eda 1.py` — EDA cho dataset 1.
- [x] `scripts/2. eda 2.py` — EDA cho dataset 2 (constant-memory streaming, ~25 phút cho 10M rows). Kết quả: `dataset/eda-result 2.txt`.
- [x] EDA xác nhận label distribution 70/30 (benign/malicious), URL length P99=100 → chốt `MAX_LEN=100`.
- [x] EDA xác nhận leakage signals (Char-RNN artifacts, fixed benign domain pools) → ghi nhận để cảnh giác về metric.

## 3. Preprocessing (CNN-LSTM)

- [x] `scripts/3. preprocess-data 2.py`:
  - [x] Load `dataset/dataset 2.csv`, coerce label về numeric (drop row label rác).
  - [x] Lowercase + strip URLs.
  - [x] Stratified 80/10/10 train/val/test split (seed=42).
  - [x] Build char vocab 51 tokens (`<PAD>`, `<UNK>` + 49 chars).
  - [x] Encode URLs về numpy `(N, 100)` int32, pad bằng PAD index.
  - [x] Save artifacts vào `data/processed/model_2/`: `train/val/test_{X,y}.npy`, `vocab.json`, `metadata.json`, splits `.csv`.

## 4. CNN-LSTM training

- [x] `scripts/4. train-model_2.py` — architecture: `Embed(51, 64) → Conv1d×2 → MaxPool → BiLSTM(128) → FC`.
- [x] Loss: `BCEWithLogitsLoss(pos_weight=2.33)`.
- [x] Adam `lr=1e-3`, `BATCH_SIZE=2048`, early stopping `patience=5`.
- [x] Train chạy đến epoch 14 thì early stop. Best epoch = 9, Val F1 = **0.9993**.
- [x] Checkpoint: `models/cnn_lstm_best_2.pt`.

## 5. CNN-LSTM evaluation

- [x] Eval trên test set (1M URLs):
  - Accuracy = **0.9996**
  - Precision/Recall/F1 (Malicious) = **0.9994 / 0.9992 / 0.9993**
  - ROC-AUC = **1.0000**
  - Confusion matrix: TN=699,816 / FP=184 / FN=231 / TP=299,769
- [x] Save kết quả: `models/results_2.json`.
- [ ] **Benchmark trên URL thực ngoài dataset** (URLhaus recent, PhishTank, Alexa top-1M) để confirm real-world performance — chưa làm.

## 6. Inference

- [x] `scripts/predict-url_2.py` — CNN-LSTM:
  - [x] Load `models/cnn_lstm_best_2.pt` + `data/processed/model_2/vocab.json`.
  - [x] Encode single URL (lowercase, pad/truncate đến 100 chars).
  - [x] Output: malicious score (0-1) + risk level (Low/Caution/Suspicious/High).
  - [x] CLI mode: single URL, multiple URLs, `--file urls.txt`, interactive.
- [x] `scripts/predict-xgb_2.py` — XGBoost:
  - [x] Load `models/xgb_url_2.ubj` + `data/processed/xgboost/xgb_feature_cols.json`.
  - [x] Reuse `extract_features()` từ `scripts/5. feature-extract_2.py` (dynamic import vì tên file có space) → đảm bảo featurization tại inference khớp tuyệt đối với train.
  - [x] Cross-check thứ tự cột giữa file extract và `xgb_feature_cols.json` để bắt drift.
  - [x] Inference trên CPU (`set_param({device: cpu})`) — single-row predict overhead-bound, GPU không lợi.
  - [x] Dùng `booster.best_iteration` để cắt đúng số tree tốt nhất từ early stopping.
  - [x] Output + CLI giống `predict-url_2.py` (single, multi, `--file`, interactive); batch mode in thêm throughput URL/s.

## 7. XGBoost

### 7.1 Feature engineering — `scripts/5. feature-extract_2.py`

**Triết lý:** Feature có ý nghĩa **real-world** (DGA detection, phishing heuristic, lexical anomaly), KHÔNG memorization dataset cụ thể. Vì vậy:
- **Include:** entropy, vowel ratio, longest digit/consonant run trong host → bắt gián tiếp Char-RNN artifacts mà vẫn generic cho real DGA.
- **Exclude:** known-domain pools (`host_in_ecommerce_pool`, `host_in_news_pool`,...) — đó chỉ là memorization của benign generator, vô giá trị real-world.
- **Brand mention** tách 2 góc:
  - `brand_in_registered_domain` (legit — google.com) → benign signal.
  - `brand_in_path_or_sub_only` (paypal trong path nhưng host khác brand) → phishing signal.

**Feature groups (~75 cột):**
- [x] **Structural (10):** `url_len`, `host_len`, `path_len`, `query_len`, `fragment_len`, `path_depth` (số segment), `num_query_params`, `has_query`, `has_fragment`, `has_port`.
- [x] **Char composition — whole URL (9):** `digit_count_url`, `digit_density_url`, `hyphen_count_url`, `dot_count_url`, `at_count`, `pct_count`, `amp_count`, `eq_count`, `qmark_count`.
- [x] **Host-specific (10):** `host_digit_count/density`, `host_hyphen_count`, `host_dot_count`, `host_num_subdomains`, `host_starts_with_www`, `host_longest_digit_run`, `host_longest_consonant_run`, `host_vowel_ratio`, `host_letter_count`.
- [x] **Entropy (3):** `entropy_url`, `entropy_host`, `entropy_path`.
- [x] **Scheme & host shape (3):** `is_http`, `is_https`, `is_ip_host`.
- [x] **TLD (53):** top-50 one-hot từ EDA + `tld_other` + `tld_missing` + `tld_label_count`. Parse bằng `tldextract` để chính xác multi-part suffix (`.co.uk`, `.com.vn`).
- [x] **Lexical heuristic flags (12):** `num_suspicious_keywords`, `has_suspicious_keyword`, `has_scam_bait`, `has_c2_path`, `num_c2_path_hits`, `brand_in_registered_domain`, `brand_in_path_or_sub_only`, `has_malware_ext`, `has_cdn_ext`, `has_any_ext`, `num_benign_words`, `has_benign_word`. Word pools đồng bộ với `scripts/2. eda 2.py`.

**Pipeline:**
- [x] Đọc các split đã được lowercase từ `data/processed/model_2/{train,val,test}.csv` (đã được `3. preprocess-data 2.py` tạo).
- [x] Mỗi feature đều stateless / không dùng statistics của tập train → KHÔNG có leakage giữa các split.
- [x] Lưu artifact (đã chuyển sang folder riêng `data/processed/xgboost/`):
  - `data/processed/xgboost/{train,val,test}_X_xgb.npy` (float32, shape `(N, F)`).
  - `data/processed/xgboost/{train,val,test}_y_xgb.npy` (int8).
  - `data/processed/xgboost/xgb_feature_cols.json` (tên cột, để SHAP plot đẹp).
  - `data/processed/xgboost/xgb_feature_meta.json` (`pos_weight`, top TLDs).

### 7.2 Training — `scripts/6. train-xgb_2.py`

- [x] `scale_pos_weight = neg/pos` từ training set (read từ `xgb_feature_meta.json`, ≈ 2.33).
- [x] Train với `xgb.train()` API, `tree_method=hist`, **`device=cuda`** (GPU, XGBoost ≥ 2.0).
- [x] Hyperparameters mặc định: `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=1.0`, `n_estimators=2000`.
- [x] Early stopping `EARLY_STOPPING_ROUNDS=50` trên val PR-AUC (`aucpr` — nhạy với class imbalance hơn ROC-AUC).
- [x] Evaluate trên test: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, classification report. **Val aucpr → ~0.9999** (train/val gap ~0.0001 → không overfit; xảy ra do dataset 2 có leakage signals, không phải bug).
- [x] Lưu model: `models/xgb_url_2.ubj`.
- [x] Lưu kết quả + top 30 feature importance (gain): `models/results_xgb_2.json`.

## 8. Hybrid inference & risk scoring (chưa làm)

- [ ] Inference script load `models/cnn_lstm_best_2.pt` + `models/xgb_url_2.ubj`, encode URL theo cả 2 cách (char-encoded `(1,100)` cho CNN-LSTM, feature vector `(1,F)` cho XGBoost — tái sử dụng hàm `extract_features()` từ `scripts/5. feature-extract_2.py`).
- [ ] Output: `score_cnn_lstm`, `score_xgb`, `risk_cnn_lstm`, `risk_xgb`.
- [ ] Apply shared risk thresholds: 0-25 Low / 26-50 Caution / 51-75 Suspicious / 76-100 High.
- [ ] (Optional) Aggregate strategy (mean / max / weighted) → final score + final risk.

## 9. Explainability (SHAP — chưa làm)

- [ ] `pip install shap matplotlib`.
- [ ] SHAP `DeepExplainer`/`GradientExplainer` cho CNN-LSTM trên 200 test samples → character-level attribution plots.
- [ ] SHAP `TreeExplainer` cho XGBoost trên 200 test samples (đọc `xgb_feature_cols.json` để label đúng cột) → feature importance bar + beeswarm summary plots.
- [ ] Save plots vào `reports/` (folder chưa tồn tại, cần `mkdir`).

## 11. Model 3 — Vietnamese real-world CNN-LSTM (current focus)

> Train CNN-LSTM trên dataset URL **thực** từ web VN (không synthetic) + malicious
> từ public threat feeds. Pipeline gần hoàn tất, chỉ còn benchmark real-world.

### 11.1 Data collection — benign

- [x] Khởi tạo seed list VN-targeted: `dataset/vn-crawl/seeds.txt`.
- [x] Pipeline `scripts/crawl-vn-urls.py`: sitemap-first + BFS crawler fallback + crt.sh subdomain.
  - crt.sh enumerate subdomain qua Certificate Transparency logs (giới hạn 30/seed).
  - Sitemap đọc `robots.txt` + path mặc định; parse `urlset` & `sitemapindex` lồng + `.xml.gz` (regex `<loc>`, không phụ thuộc lxml).
  - BFS crawler fallback khi sitemap < threshold (default 100). Respect `robots.txt`, per-host rate 1s/req, scope eTLD+1.
  - **Resume mode** (`--resume`): skip seed trong `processed_seeds.txt` + preload URL từ CSV cũ vào `global_seen`. Auto-bootstrap state file từ cột `seed` nếu chưa tồn tại.
- [x] Round 1 (~50 seeds): ~3M URLs.
- [x] Round 2: mở rộng `seeds.txt` lên 275 seeds (news, ecommerce, banks, fintech, edu, gov, legal/doc DBs, Wikipedia VN, online learning, ...). Final: **~12.5M URLs** → `dataset/urls-benign-3.csv`.

### 11.2 Data collection — malicious

- [x] `scripts/fetch-malicious-all.py`: pull từ 6 nguồn free no-auth, tag `is_vn_target` (boolean dựa trên `.vn` TLD + VN brand regex), không lọc theo geo.
  - Working: URLhaus (ZIP-CSV), ThreatFox (ZIP-CSV), OpenPhish, Phishing.Database ACTIVE/INACTIVE/NEW-today, Phishing.Army (domain → wrap `https://`), tweetfeed.live, DigitalSide.it, CertPL, cybercrime-tracker, Hagezi TIF, Malsilo.
  - Dead: PhishStats (404), Chongluadao (DNS fail) → bỏ.
  - Total ~2.6M URLs sau dedup → `dataset/urls-malicious-3.csv`.
  - Output thêm: `dataset/vn-malicious/urls-all.csv` (raw, có flag), `urls-vn-target.csv` (subset VN), `sources-summary.txt`.
- Note: estimated upper bound cho free no-auth malicious feeds là ~2-3M unique. Để vượt cần PhishTank/AlienVault OTX API key, hoặc Char-RNN augment (không làm — preserve real-data integrity).

### 11.3 Prepare + Preprocess

- [x] `scripts/2. prepare-dataset 3.py`: stream-read `urls-benign-3.csv` + `urls-malicious-3.csv`, dedup global, re-quote bằng `csv.QUOTE_ALL` để URL có `,` / `"` không phá CSV → `dataset/dataset 3 (vn).csv` (~15M rows quoted, ~1.1GB).
- [x] `scripts/3. preprocess-data 3.py`: `pd.read_csv` với `quoting=csv.QUOTE_MINIMAL` (default đọc quoted CSV transparent), stratified 80/10/10 split, char-encode `MAX_LEN=100`, vocab=51 (giống model 1+2) → `data/processed/model_3/`. Metadata lưu thêm `source_csv`, `csv_quoting`. Splits CSV save với `QUOTE_ALL`.

### 11.4 EDA — `scripts/2. eda 3.py`

- [x] Adapted từ eda 2 nhưng:
  - **VIETNAMESE TARGETING** section mới: count `.vn` TLD + VN brand match per label, VN sub-TLD breakdown (`.com.vn`/`.edu.vn`/`.gov.vn`/...).
  - VN sub-TLDs thêm vào `KNOWN_TLDS`.
  - **Bỏ** `BENIGN_WORDS` + section `BENIGN WORD COVERAGE` (mirror generator wordlist English, vô nghĩa cho real VN data dùng `tin-tuc`, `bai-viet`, ...).
  - **Rename** `KNOWN DOMAIN POOLS` → `ANCHOR DOMAINS (sanity check)` + note "không phải coverage report".
  - **`classify_url()` bỏ 5 nhánh benign theo pool** — pool quá nhỏ so với long-tail real, để mặc định mostly `unknown`.
  - Fix `parse_url` `try/except` cho Python 3.14 IPv6 ValueError.
  - Compiled regex unions cho keyword sets + domain pools (~15% speedup).
- [ ] Performance: pure-Python streaming max ~6-7k rows/s = ~40 phút cho 15M rows. Polars rewrite chưa làm (~30-50x speedup tiềm năng).

### 11.5 Train + Eval

- [x] `scripts/4. train-model_3.py`: kiến trúc giống model 2, tối ưu cho **RTX 3060 Ti 8GB**:
  - AMP (FP16) via `torch.amp.autocast` + `GradScaler` (~1.7-2x throughput).
  - TF32 cho matmul FP32 còn lại.
  - `BATCH_SIZE=4096` (gấp đôi model 2 vì AMP giảm activation memory).
  - `mmap_mode='r'` khi `np.load` → tránh RAM blowup + multi-worker fork issue.
  - `set_to_none=True` + `non_blocking=True` cho zero_grad + transfers.
  - Save thêm `scaler_state_dict` cho resumable training.
- [x] Training chạy đến epoch 18 (early-stop, patience=5). Best epoch **13**, **Val F1=0.9999**. Checkpoint `models/cnn_lstm_best_3.pt`.
- [x] Train script crash ở test phase với `OSError WinError 1114` (DLL init fail trong DataLoader worker spawn). Đây là pattern đã document trong CLAUDE.md cho model 2 (chỉ manifest khác — system thrashing sau 18 epoch).
- [x] **Formal workaround**: `scripts/5. eval-model_3.py` — chỉ load test set, `num_workers=0`, tính thêm ROC-AUC + PR-AUC + confusion matrix, ghi `models/results_3.json`.
- [x] Chạy `5. eval-model_3.py` để confirm test F1, ghi `results_3.json`.
- [x] `scripts/predict-url_3.py`: clone của `predict-url_2.py`, chỉ đổi path `model_2 → model_3`.

### 11.6 Real-world benchmark (chưa làm)

- [ ] Benchmark trên URL real-world **chưa thấy bao giờ** (recent URLhaus dump, recent PhishTank, freshly crawled VN sites). Val F1=0.9999 có thể bị inflated bởi feed-level shortcuts (HTTPS rate skew, TLD skew, IP-host skew).
- [ ] Quyết định có cần Feature extract + XGBoost cho model 3 không (chỉ làm nếu CNN-LSTM real-world performance không đủ).

## 10. Documentation & cleanup

- [x] `CLAUDE.md` — updated to reflect dual-pipeline (model_1 vs model_2) state.
- [x] `checklist.md` — this file.
- [ ] Write final report (research project deliverable) — chưa làm.
- [ ] Document Char-RNN generator (script source code có thể đã bị xoá; checkpoint còn nhưng cần re-derive training script nếu reproduce).
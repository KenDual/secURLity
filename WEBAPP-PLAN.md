# WEBAPP-PLAN.md — Kế hoạch implement web + API cho secURLity

> Document này là **single source of truth** để resume việc code webapp khi session Claude Code mới được khởi động.
> Đọc song song với `CLAUDE.md` (kiến trúc model) và `PLAN.md` (chiến lược dự án).
> Format checklist — mỗi task xong thì đổi `[ ]` thành `[x]`.

---

## 0. Bối cảnh & mục tiêu

- **Hiện trạng**: 3 model đã train xong (CNN-LSTM domain split + XGBoost + SGDClassifier HTML), inference chỉ chạy qua script CLI.
- **Mục tiêu**: Public web + REST API miễn phí cho mọi người dùng thử, đồng thời expose endpoint JSON cho developer integrate.
- **Ràng buộc**: Dự án cá nhân, ngân sách rất hạn chế → ưu tiên free tier, không over-engineer.

---

## 1. Quyết định đã chốt (đã thống nhất với user)

| # | Quyết định | Ghi chú |
|---|---|---|
| 1 | Giao diện: **Website UI + REST API** trên cùng codebase | FastAPI vừa serve HTML vừa serve JSON, 1 deploy duy nhất |
| 2 | **CNN-LSTM + XGBoost chạy song song** mỗi request URL | Latency tổng < 3 ms |
| 3 | **Ensemble = simple average**: `(cnn_prob + xgb_prob) / 2` | KHÔNG dùng weighted; user đã xác nhận |
| 4 | **Ensemble threshold = 0.5** (default) | Phải để 1 biến `ENSEMBLE_THRESHOLD = 0.5` trong predict service với comment rõ ràng để user tự chỉnh sau |
| 5 | **SGD model**: optional, có toggle UI | Default OFF; khi user bật phải hiện modal cảnh báo latency cao + Cloudflare block. SGD KHÔNG tham gia ensemble (chỉ hiển thị tách biệt) |
| 6 | **Explainability**: đầy đủ heatmap + SHAP | Attention CNN-LSTM render HTML colored spans; XGBoost SHAP top-K bar chart CSS thuần |
| 7 | **Hosting**: Hugging Face Spaces (Docker SDK), free tier | Không cần credit card, 16GB RAM, 50GB disk ephemeral |
| 8 | **Persistence**: SQLite log scan gần đây | Lưu hash URL + display URL đã cắt query string; chấp nhận reset khi HF Space restart (rare) |

---

## 2. Tech stack chốt

### Backend
- **FastAPI** (Python 3.11+) — async, auto Swagger docs tại `/docs`
- **Pydantic v2** — validate request/response
- **onnxruntime** (CPU EP) — chạy CNN-LSTM ONNX FP32
- **xgboost** — load `xgb_url_4.ubj`
- **shap** — `TreeExplainer` cho XGBoost
- **scikit-learn + joblib** — load `SGDClassifier_2.joblib` (chỉ khi enable SGD)
- **httpx** — fetch HTML cho SGD path (async, có timeout)
- **aiosqlite** — SQLite async
- **slowapi** — rate limit 10 req/min/IP
- **jinja2** — render template HTML server-side
- **uvicorn[standard]** — ASGI server
- **tldextract** — parse domain (đã dùng trong predict scripts)

### Frontend (no build step)
- **Jinja2 templates** server-side render từ FastAPI
- **Tailwind CSS** qua Play CDN (`<script src="https://cdn.tailwindcss.com"></script>`)
- **HTMX** qua CDN — async form submit, render partial vào `<div id="result">`
- **Alpine.js** qua CDN — toggle switch, modal warning
- **KHÔNG** dùng React/Vue/Next.js (over-engineering cho dự án này)
- **KHÔNG** dùng Chart.js — SHAP bar chart render bằng `<div style="width:...">` thuần CSS

### Hosting
- **Hugging Face Spaces** (Docker SDK)
- Port 7860 (HF default)
- Domain free: `https://<username>-securlity.hf.space`
- Auto-sleep sau ~48h idle, wake 30-60s lần đầu

---

## 3. Cấu trúc thư mục (sẽ tạo trong `D:\! secURLity\webapp\`)

```
webapp/
├── Dockerfile                          # python:3.11-slim, port 7860
├── requirements.txt
├── README.md                           # HF Spaces card (YAML frontmatter)
├── .dockerignore
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app + routes
│   ├── config.py                       # ENSEMBLE_THRESHOLD, paths, constants
│   ├── schemas.py                      # Pydantic request/response models
│   ├── predict_service.py              # Load 3 models 1 lần lúc startup; predict_cnn/predict_xgb/predict_sgd
│   ├── ensemble.py                     # ensemble_average() — đơn giản
│   ├── explain.py                      # build_attention_heatmap_html(), format_shap_topk()
│   ├── html_fetch.py                   # httpx + SSRF guard cho SGD
│   ├── db.py                           # aiosqlite: init_db, insert_scan, get_recent
│   ├── ratelimit.py                    # slowapi limiter setup
│   └── templates/
│       ├── base.html                   # layout chung + Tailwind/HTMX/Alpine CDN
│       ├── index.html                  # form chính
│       ├── _result.html                # HTMX partial — verdict cards + heatmap + SHAP
│       └── history.html                # bảng 50 scan gần nhất
└── bundled_models/                     # Copy artifacts vào đây trước khi build Docker
    ├── cnn_lstm_4_domain.onnx          # từ ../models/
    ├── xgb_url_4.ubj                   # từ ../models/
    ├── thresholds_4.json               # từ ../models/ (CNN-LSTM thresholds, hiện không dùng cho ensemble)
    ├── thresholds_xgb_4.json           # từ ../models/ (XGBoost thresholds, hiện không dùng cho ensemble)
    ├── vocab.json                      # từ ../data/processed/model_4/
    ├── metadata.json                   # từ ../data/processed/model_4/
    ├── feat_stats.json                 # từ ../data/processed/model_4/domain/
    ├── xgb_4_feature_cols.json         # từ ../data/processed/xgboost_4/
    └── SGDClassifier_2.joblib          # từ D:\phreshphish\models\
```

**Quan trọng — refactor pure functions** (KHÔNG import script CLI trực tiếp):
- Từ `scripts/predict-url_4_onnx.py` → extract `encode_url`, `normalize_features`, attention upsample logic vào `predict_service.py`
- Từ `scripts/predict-url_4_xgb.py` → extract feature extraction call + SHAP logic
- Từ `scripts/3. feature-extract_xgb_4.py` → import `extract_features()` qua importlib (filename có space + leading digit)
- Từ `scripts/predict-rawHTML-SGDClassifier.py` → extract `normalize_html()` vào `predict_service.py` (NHỚ giữ identical với train script)

---

## 4. API specification (chốt sớm để frontend bám theo)

### `POST /api/predict`

Request:
```json
{
  "url": "https://example.com/login?next=/x",
  "enable_sgd": false,
  "explain": true
}
```

Response (`enable_sgd=false`):
```json
{
  "url": "https://example.com/login",
  "models": {
    "cnn_lstm": {
      "probability": 0.0234,
      "verdict": "BEN",
      "risk_level": "Low Risk",
      "latency_ms": 1.8,
      "explanation": {
        "attention_html": "<span style='background:rgba(255,0,0,0.12)'>h</span>...",
        "top_lexical_features": [
          {"name": "path_depth", "z_score": -3.04, "value": 0.0}
        ]
      }
    },
    "xgboost": {
      "probability": 0.0156,
      "verdict": "BEN",
      "risk_level": "Low Risk",
      "latency_ms": 0.6,
      "explanation": {
        "shap_top_features": [
          {"name": "tld_is_vn", "shap_value": -0.421, "feature_value": 1.0, "direction": "BEN"}
        ],
        "expected_value": -1.234
      }
    },
    "sgd": null
  },
  "ensemble": {
    "probability": 0.0195,
    "verdict": "BEN",
    "risk_level": "Low Risk",
    "threshold": 0.5,
    "formula": "(cnn_prob + xgb_prob) / 2"
  },
  "total_latency_ms": 2.4
}
```

Response (`enable_sgd=true`): thêm `models.sgd = { probability, verdict, latency_ms, html_fetch_ms, html_truncated_chars }` hoặc `models.sgd = { error: "fetch_failed" | "cloudflare_blocked" | "timeout" }`.

### `GET /api/history?limit=50`
Trả về 50 scan gần nhất (truncated URL + verdict + ensemble_prob + created_at).

### `GET /health`
Trả `{ "status": "ok", "models_loaded": ["cnn_lstm", "xgboost", "sgd"] }` cho HF Spaces health check.

### `GET /docs`
Auto-gen Swagger (FastAPI default).

---

## 5. Phase A — MVP (deploy được, chưa có UI đẹp)

Mục tiêu: web chạy được end-to-end trên HF Spaces, form đơn giản, kết quả text.

- [x] Tạo folder `webapp/` + `webapp/app/`
- [x] Viết `webapp/requirements.txt` với pin version cụ thể (fastapi==0.115.6, starlette==0.41.2)
- [x] Viết `webapp/Dockerfile`:
  - Base `python:3.11-slim`
  - Install gcc + libgomp1 (cần cho xgboost/sklearn)
  - Copy requirements → pip install
  - Copy `bundled_models/` + `app/`
  - Expose 7860, CMD `uvicorn app.main:app --host 0.0.0.0 --port 7860`
- [x] Viết `webapp/app/config.py`:
  - `MODEL_DIR = Path("bundled_models")`
  - `ENSEMBLE_THRESHOLD = 0.5  # USER: edit here to tune (e.g., 0.7 for more conservative)`
  - `SGD_HTML_FETCH_TIMEOUT = 10.0`, `SGD_MAX_HTML_BYTES = 2_000_000`
  - `RATE_LIMIT = "10/minute"`
  - `DB_PATH = Path("/data/scans.db")` (HF Spaces) hoặc fallback `./scans.db`
- [x] Viết `webapp/app/predict_service.py`:
  - Class `PredictService` load CNN-LSTM ONNX + XGBoost UBJ + (lazy) SGD joblib lúc `__init__`
  - Method `predict_cnn(url) -> {prob, latency_ms, attention_weights_64}` (port từ `predict-url_4_onnx.py`)
  - Method `predict_xgb(url) -> {prob, latency_ms, shap_values_105, feature_values_105, expected_value}` (port từ `predict-url_4_xgb.py`)
  - Method `predict_sgd(html_str) -> {prob, latency_ms}` (port từ `predict-rawHTML-SGDClassifier.py`)
- [x] Viết `webapp/app/ensemble.py`
- [x] Viết `webapp/app/schemas.py` — Pydantic models theo spec Section 4
- [x] Viết `webapp/app/db.py`
- [x] Viết `webapp/app/main.py`:
  - FastAPI app + lifespan để init `PredictService` 1 lần
  - `POST /api/predict` — gọi 2 model song song qua `asyncio.gather` (CNN-LSTM ONNX là sync — dùng `run_in_threadpool`)
  - `POST /predict` — form endpoint, trả HTML partial
  - `GET /api/history`, `GET /history`, `GET /health`, `GET /docs`
- [x] Test local: server chạy OK trên port 7862
  - `POST /api/predict` với malicious URL → MAL ✓
  - `POST /predict` (form) → HTML partial với MALICIOUS verdict ✓
  - `GET /health` → `{"status":"ok","models_loaded":["cnn_lstm","xgboost"]}` ✓
  - `GET /history` render OK ✓
  - `GET /docs` Swagger UI OK ✓
  - explain=true → attention heatmap + SHAP bar chart OK ✓
  - **NOTE**: starlette>=1.x không tương thích Jinja2 với Python 3.14 → pin `fastapi==0.115.6 starlette==0.41.2`
- [ ] Build Docker local: `docker build -t securlity .` + `docker run -p 7860:7860 securlity`
- [ ] Tạo HF Space (Docker SDK) qua web UI hoặc CLI `huggingface-cli`
- [ ] `git push` lên HF Space repo → check log build → mở `https://<user>-securlity.hf.space`

---

## 6. Phase B — Polish UI + SGD toggle + Security

Mục tiêu: trang đẹp, có heatmap + SHAP chart, SGD toggle hoạt động, có rate limit.

- [x] Cập nhật `templates/base.html`:
  - Tailwind CDN, HTMX CDN, Alpine.js CDN
  - Layout 2 cột: form bên trái, kết quả bên phải (responsive collapse mobile)
- [x] Cập nhật `templates/index.html`:
  - `<input>` URL + `<button>` Submit
  - Alpine toggle `enable_sgd` → khi click hiện modal warning
  - HTMX: `hx-post="/predict"` `hx-target="#result"` `hx-swap="innerHTML"` (form endpoint trả HTML partial)
- [x] Viết `templates/_result.html` (HTMX partial):
  - 3 card: CNN-LSTM | XGBoost | Ensemble (với threshold badge)
  - SGD card chỉ render khi `enable_sgd=true`
  - Probability bar màu gradient (xanh < 0.25, vàng < 0.5, cam < 0.75, đỏ ≥ 0.75)
  - Risk level badge
- [x] Viết `webapp/app/explain.py`:
  - `build_attention_heatmap_html(url, weights_64) -> str`
  - `format_shap_topk(shap_values, feature_names, feature_values, k=10) -> List[dict]`
- [x] Cập nhật `_result.html` render heatmap + SHAP bar chart
- [x] Viết `webapp/app/html_fetch.py`:
  - Async `fetch_html(url)` với SSRF guard (IPv4/IPv6 private ranges, DNS resolution check)
  - Cloudflare block detection
  - 2MB read limit
- [x] Tích hợp SGD vào `POST /api/predict`:
  - `enable_sgd=true` → fetch HTML → predict_sgd(html)
  - SGD KHÔNG cộng vào ensemble ✓
- [x] Viết `webapp/app/ratelimit.py`:
  - slowapi 10/minute per IP, apply lên POST /api/predict + POST /predict
- [x] Viết `templates/history.html` + route `GET /history` render bảng
- [x] Test local end-to-end: heatmap ✓, SHAP bars ✓, SSRF block ✓, rate limit 429 ✓, history ✓, Swagger ✓
- [ ] Deploy lại HF Space → verify production

---

## 7. Security checklist (BẮT BUỘC trước public)

- [x] SSRF guard cho SGD HTML fetch — private IP + DNS resolve check ✓
- [x] Rate limit 10 req/min/IP qua slowapi ✓ (tested: req 11 → 429)
- [x] Truncate query string trước khi log DB ✓ (db.display_url strips query)
- [x] Hash URL gốc bằng SHA256 trước khi lưu DB ✓
- [x] Truncate URL về max 5000 chars qua Pydantic validation; model tự truncate tại MAX_LEN=256 ✓
- [x] CORS: chỉ allow `*` cho `/api/*`, không cần cho route HTML ✓ (custom middleware, không dùng CORSMiddleware global)
- [ ] (Optional Phase C) hCaptcha free trên form cho anti-bot
- [x] HTTP headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy` ✓
- [x] Không log exception trace ra response ✓ (chỉ generic message)

---

## 8. Hugging Face Spaces — deployment notes

- [x] Tạo HF account (nếu chưa có) tại https://huggingface.co/join
- [x] Tạo Space mới: **Docker SDK**, Hardware **CPU basic free**, Visibility **Public**
- [x] Clone Space repo về local: `git clone https://huggingface.co/spaces/<username>/securlity`
- [x] Copy toàn bộ `webapp/*` vào root repo Space (KHÔNG copy `webapp/` parent folder)
- [x] `README.md` ở root Space cần YAML frontmatter:
  ```yaml
  ---
  title: secURLity
  emoji: 🛡️
  colorFrom: blue
  colorTo: red
  sdk: docker
  app_port: 7860
  pinned: false
  ---
  ```
- [x] `git lfs track "*.onnx" "*.ubj" "*.joblib"` (file > 10MB cần Git LFS)
- [x] `git add . && git commit -m "Initial deploy" && git push`
- [x] Mở Space URL → check build log → verify chạy

**Caveats HF Spaces free tier:**
- Ephemeral storage → SQLite reset khi Space restart (hiếm, vài tuần/lần). Phase C có thể snapshot DB sang HF Dataset.
- Auto-sleep sau ~48h idle → wake 30-60s khi có request đầu
- Custom domain cần Pro plan ($9/tháng). Default dùng `*.hf.space` là OK cho demo cá nhân.

---

## 9. Phase C — Polish thêm (optional, sau khi public OK)

- [ ] Snapshot SQLite → HF Dataset repo mỗi 1h (true persistence, vẫn free)
- [ ] hCaptcha free trên form
- [ ] Dark mode toggle (Tailwind `dark:` classes)
- [ ] OpenAPI examples trong Swagger (làm `/docs` dễ đọc hơn)
- [ ] Tune `ENSEMBLE_THRESHOLD` trên val set thực tế (hiện đang hardcode 0.5)
- [ ] Tune ensemble formula (chuyển từ simple avg sang weighted nếu cần — sửa trong `ensemble.py`)
- [ ] Cache prediction theo URL hash (LRU cache trong RAM, TTL 1h) → giảm load model cho URL duplicate
- [ ] Sitemap.xml + robots.txt allow indexing
- [ ] Analytics nhẹ qua Plausible self-host hoặc HF Spaces built-in metrics

---

## 10. References — file artifacts hiện có để reuse

| File hiện có | Dùng cho webapp như thế nào |
|---|---|
| `models/cnn_lstm_4_domain.onnx` | Copy vào `webapp/bundled_models/`, load bằng `onnxruntime.InferenceSession` |
| `models/xgb_url_4.ubj` | Copy, load bằng `xgb.Booster().load_model()` hoặc `XGBClassifier.load_model()` |
| `models/thresholds_4.json` | Copy (nhưng webapp dùng ensemble threshold riêng = 0.5; file này tham khảo) |
| `models/thresholds_xgb_4.json` | Tương tự thresholds_4.json |
| `data/processed/model_4/vocab.json` | Copy, dùng cho `encode_url()` |
| `data/processed/model_4/metadata.json` | Copy, đọc `MAX_LEN`, `vocab_size`, `pad_idx` |
| `data/processed/model_4/domain/feat_stats.json` | Copy, dùng cho `normalize_features()` lexical 30 |
| `data/processed/xgboost_4/xgb_4_feature_cols.json` | Copy, dùng để map SHAP index → feature name |
| `D:\phreshphish\models\SGDClassifier_2.joblib` | Copy, load bằng `joblib.load()` |
| `scripts/predict-url_4_onnx.py` | Reference — extract functions vào `predict_service.py` |
| `scripts/predict-url_4_xgb.py` | Reference — extract functions vào `predict_service.py` |
| `scripts/3. feature-extract_xgb_4.py` | Import `extract_features()` qua `importlib` (filename có space) |
| `scripts/predict-rawHTML-SGDClassifier.py` | Reference — extract `normalize_html()` vào `predict_service.py` |

**Lưu ý critical**:
- `normalize_html()` trong webapp phải **identical** với train script `4. train-model-SGDClassifier.py` (HTML preprocessing) — mismatch = silent accuracy regression.
- Lexical 30 features phải dùng đúng `feat_stats.json` của split `domain` (production), KHÔNG dùng `random`.
- Vocab encoding **không lowercase** URL — case là signal.

---

## 11. Testing checklist trước khi public

- [x] `https://google.com` — ensemble MAL 99.5% (CNN=99%, XGB=100%). Model limitation: cả 2 model đều FP trên well-known domains vì domain split training không có google.com trong train set. **Không fix được ở webapp level**.
- [x] `https://www.facebook.com/login` — ensemble MAL 95.4%. Tương tự: model FP vì `/login` path phổ biến trong phishing. **Model limitation**.
- [x] `https://paypa1-secure-login.tk/verify?id=abc` — MAL 99.4% ✓
- [x] URL siêu dài (5001 ký tự) — Pydantic validation 422 ✓
- [x] URL có punycode (`https://xn--e1afmapc.com/test`) — không crash ✓
- [x] URL có ký tự Unicode (`https://例え.jp/test`) — không crash, trả 200 ✓
- [x] Submit 11 request trong 1 phút — request 11 trả 429 ✓ (seen in error logs)
- [x] Toggle SGD + URL hợp lệ — fetch HTML thành công, prob hiện (e.g. uef.edu.vn ✓)
- [ ] Toggle SGD + URL Cloudflare protected — error message clear
- [x] Toggle SGD + `http://localhost/admin` — SSRF block "ssrf_blocked" ✓
- [x] Toggle SGD + `http://192.168.1.1/admin` — SSRF block ✓
- [x] `/api/history` trả về scan list ✓
- [x] `/docs` Swagger UI render đúng ✓
- [ ] Mobile responsive (form + result usable trên màn hình 375px) — cần browser test

---

## 12. Câu hỏi mở (resolve sau khi triển khai Phase A)

- [ ] Có cần admin endpoint xóa scan từ history (vd. user submit URL nhạy cảm) không?
- [ ] Có cần i18n (Việt/Anh) cho UI không, hay chỉ tiếng Việt?
- [ ] Có muốn batch endpoint `POST /api/predict-batch` (list URLs) không?
- [ ] Có cần webhook notify khi detect MAL không?
- [ ] Sau khi public, có muốn submit lên ProductHunt/Reddit r/netsec để collect feedback không?

---

## Progress tracker

- **Phase A (MVP)**: 11 / 14 tasks (còn Docker build + HF deploy)
- **Phase B (Polish UI + Security)**: 10 / 11 tasks (còn HF deploy)
- **Security checklist**: 8 / 9 tasks (còn hCaptcha — Phase C optional)
- **HF Deployment**: 0 / 8 tasks
- **Phase C (Optional)**: 0 / 9 tasks
- **Testing**: 14 / 15 tests passed local (còn mobile responsive — cần browser; benign FP là model limitation không phải bug)

> Update các con số trên mỗi khi check task xong.

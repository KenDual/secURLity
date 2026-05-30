"""FastAPI application — secURLity URL malware detection web + API."""
import asyncio
import hashlib
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from . import config, db
from .ensemble import ensemble_average, risk_level
from .explain import build_attention_heatmap_html
from .html_fetch import fetch_html
from .predict_service import PredictService
from .ratelimit import limiter
from .schemas import (
    CNNExplanation, EnsembleResult, LexicalFeature, ModelResult, ModelSet,
    PredictRequest, PredictResponse, SHAPFeature, SGDResult, XGBExplanation,
)

# ── Globals ───────────────────────────────────────────────────────────────────

_service: Optional[PredictService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    await db.init_db()
    _service = PredictService(model_dir=config.MODEL_DIR)
    yield


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="secURLity API",
    description=(
        "Malicious URL detection using CNN-LSTM + XGBoost ensemble "
        "trained on 19.68M real-world URLs."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory=str(config._APP_DIR / "templates"))


# ── CORS (API routes only) + Security headers ─────────────────────────────────

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Handle OPTIONS preflight for /api/* before routing
    if request.method == "OPTIONS" and request.url.path.startswith("/api/"):
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "600",
            },
        )

    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    if request.url.path.startswith("/api/"):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

    return resp


# ── Shared prediction logic ───────────────────────────────────────────────────

async def _run_prediction(url: str, enable_sgd: bool, explain: bool) -> dict:
    """Core prediction logic shared by form endpoint and JSON API."""
    url = url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="URL cannot be empty")
    if len(url) > 5000:
        raise HTTPException(status_code=422, detail="URL too long (max 5000 chars)")

    # CNN-LSTM + XGBoost in parallel thread pool
    cnn_task = run_in_threadpool(_service.predict_cnn, url, explain)
    xgb_task = run_in_threadpool(_service.predict_xgb, url, explain)
    cnn_raw, xgb_raw = await asyncio.gather(cnn_task, xgb_task)

    # Ensemble
    ens_prob = ensemble_average(cnn_raw["probability"], xgb_raw["probability"])

    # SGD (optional, NOT part of ensemble)
    sgd_result: Optional[dict] = None
    if enable_sgd:
        html_content, fetch_ms, fetch_err = await fetch_html(url)
        if html_content:
            sgd_raw = await run_in_threadpool(_service.predict_sgd, html_content)
            sgd_prob = sgd_raw["probability"]
            sgd_result = {
                "probability": sgd_prob,
                "verdict": "MAL" if sgd_prob >= 0.5 else "BEN",
                "risk_level": risk_level(sgd_prob),
                "latency_ms": sgd_raw["latency_ms"],
                "html_fetch_ms": round(fetch_ms, 1),
                "error": None,
            }
        else:
            sgd_result = {"error": fetch_err or "fetch_failed"}

    # Build individual model verdicts (use 0.5 threshold for display)
    cnn_prob = cnn_raw["probability"]
    xgb_prob = xgb_raw["probability"]

    cnn_explanation: Optional[dict] = None
    if explain and "attention_weights" in cnn_raw:
        attn_html = build_attention_heatmap_html(
            url[:256], cnn_raw.get("attention_per_char", [])
        )
        cnn_explanation = {
            "attention_weights": cnn_raw.get("attention_weights", []),
            "attention_per_char": cnn_raw.get("attention_per_char", []),
            "top_lexical": cnn_raw.get("top_lexical", []),
            "attention_html": attn_html,
        }

    xgb_explanation: Optional[dict] = None
    if explain and "shap_top_features" in xgb_raw:
        raw_shap = xgb_raw.get("shap_top_features", [])
        max_abs = max((abs(f["shap_value"]) for f in raw_shap), default=1.0)
        for f in raw_shap:
            f["bar_width_pct"] = round(abs(f["shap_value"]) / max(max_abs, 1e-9) * 100, 1)
        xgb_explanation = {
            "shap_top_features": raw_shap,
            "expected_value": xgb_raw.get("expected_value", 0.0),
        }

    total_ms = round(cnn_raw["latency_ms"] + xgb_raw["latency_ms"], 1)

    result = {
        "url": db.display_url(url),
        "models": {
            "cnn_lstm": {
                "probability": cnn_prob,
                "verdict": "MAL" if cnn_prob >= 0.5 else "BEN",
                "risk_level": risk_level(cnn_prob),
                "latency_ms": cnn_raw["latency_ms"],
                "explanation": cnn_explanation,
            },
            "xgboost": {
                "probability": xgb_prob,
                "verdict": "MAL" if xgb_prob >= 0.5 else "BEN",
                "risk_level": risk_level(xgb_prob),
                "latency_ms": xgb_raw["latency_ms"],
                "explanation": xgb_explanation,
            },
            "sgd": sgd_result,
        },
        "ensemble": {
            "probability": ens_prob,
            "verdict": "MAL" if ens_prob >= config.ENSEMBLE_THRESHOLD else "BEN",
            "risk_level": risk_level(ens_prob),
            "threshold": config.ENSEMBLE_THRESHOLD,
            "formula": "(cnn_prob + xgb_prob) / 2",
        },
        "total_latency_ms": total_ms,
    }

    # Log to DB (non-blocking; failures don't affect response)
    try:
        sgd_prob_log = (
            sgd_result.get("probability") if sgd_result and not sgd_result.get("error") else None
        )
        await db.insert_scan(
            url_hash=hashlib.sha256(url.encode()).hexdigest(),
            url_display=db.display_url(url),
            cnn_prob=cnn_prob,
            xgb_prob=xgb_prob,
            ensemble_prob=ens_prob,
            ensemble_verdict=result["ensemble"]["verdict"],
            sgd_enabled=int(enable_sgd),
            sgd_prob=sgd_prob_log,
            latency_ms=int(total_ms),
        )
    except Exception:
        pass

    return result


# ── Routes — HTML ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
@limiter.limit(config.RATE_LIMIT)
async def predict_form(
    request: Request,
    url: str = Form(...),
    enable_sgd: Optional[str] = Form(None),
    explain: Optional[str] = Form(None),
):
    enable_sgd_bool = enable_sgd is not None
    explain_bool = explain is not None
    try:
        result = await _run_prediction(url, enable_sgd_bool, explain_bool)
    except HTTPException as exc:
        return templates.TemplateResponse(
            "_result.html",
            {"request": request, "error": exc.detail},
            status_code=exc.status_code,
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "_result.html",
            {"request": request, "error": "Prediction failed. Please try again."},
            status_code=500,
        )
    return templates.TemplateResponse("_result.html", {"request": request, **result})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    scans = await db.get_recent(limit=50)
    return templates.TemplateResponse("history.html", {"request": request, "scans": scans})


@app.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works(request: Request):
    return templates.TemplateResponse("how_it_works.html", {"request": request})


@app.get("/api-guide", response_class=HTMLResponse)
async def api_guide(request: Request):
    return templates.TemplateResponse("api_guide.html", {"request": request})


# ── Routes — JSON API ─────────────────────────────────────────────────────────

@app.post("/api/predict")
@limiter.limit(config.RATE_LIMIT)
async def predict_api(request: Request, req: PredictRequest):
    try:
        result = await _run_prediction(req.url, req.enable_sgd, req.explain)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal prediction error")
    return result


@app.get("/api/history")
async def api_history(limit: int = 50):
    return await db.get_recent(limit=min(limit, 100))


@app.get("/health")
async def health():
    models_ok = _service is not None
    return {
        "status": "ok" if models_ok else "degraded",
        "models_loaded": _service.loaded_models if _service else [],
        "ensemble_threshold": config.ENSEMBLE_THRESHOLD,
    }

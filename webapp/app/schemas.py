from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    url: str
    enable_sgd: bool = False
    explain: bool = False

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if len(v) > 5000:
            raise ValueError("URL too long (max 5000 characters)")
        return v


class LexicalFeature(BaseModel):
    name: str
    z_score: float
    value: float


class SHAPFeature(BaseModel):
    name: str
    shap_value: float
    feature_value: float
    direction: str  # "MAL" | "BEN"


class CNNExplanation(BaseModel):
    attention_weights: list[float]       # 64 raw weights (before upsample)
    attention_per_char: list[float]      # per-character weights (len = URL length)
    top_lexical: list[LexicalFeature]
    attention_html: Optional[str] = None  # rendered in Phase B


class XGBExplanation(BaseModel):
    shap_top_features: list[SHAPFeature]
    expected_value: float


class ModelResult(BaseModel):
    probability: float
    verdict: str        # "MAL" | "BEN"
    risk_level: str
    latency_ms: float
    explanation: Optional[CNNExplanation | XGBExplanation] = None


class SGDResult(BaseModel):
    probability: Optional[float] = None
    verdict: Optional[str] = None
    risk_level: Optional[str] = None
    latency_ms: Optional[float] = None
    html_fetch_ms: Optional[float] = None
    error: Optional[str] = None


class EnsembleResult(BaseModel):
    probability: float
    verdict: str
    risk_level: str
    threshold: float
    formula: str


class ModelSet(BaseModel):
    cnn_lstm: ModelResult
    xgboost: ModelResult
    sgd: Optional[SGDResult] = None


class PredictResponse(BaseModel):
    url: str            # cleaned (scheme://host + path[:30], query stripped)
    models: ModelSet
    ensemble: EnsembleResult
    total_latency_ms: float


class ScanRecord(BaseModel):
    id: int
    url_display: str
    cnn_prob: Optional[float]
    xgb_prob: Optional[float]
    ensemble_prob: Optional[float]
    ensemble_verdict: Optional[str]
    sgd_enabled: int
    latency_ms: Optional[int]
    created_at: str

"""
Model loader + inference service for all 3 models.

CNN-LSTM  : ONNX FP32 (domain split), vocab + feat_stats from bundled_models/
XGBoost   : UBJ model + SHAP TreeExplainer (initialised once at startup)
SGDClassifier : lazy-loaded on first SGD request

normalize_html() MUST stay identical to 4. train-model-SGDClassifier.py.
"""
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .scripts.lexical_features import extract_features as _extract_lexical
from .scripts.xgb_features import extract_features as _extract_xgb_dict, FEATURE_NAMES as _XGB_COLS


# ── HTML normalizer (must match train script exactly) ─────────────────────────

def _normalize_html(html: str, max_chars: int) -> str:
    if html is None:
        return ""
    html = str(html)
    html = html[:max_chars * 5]
    html = re.sub(r">\s+<", "><", html)
    html = re.sub(r"\s+", " ", html)
    html = html.strip()
    return html[:max_chars]


# ── Risk tier ─────────────────────────────────────────────────────────────────

def risk_level(prob: float) -> str:
    pct = prob * 100
    if pct <= 25:
        return "Low Risk"
    if pct <= 50:
        return "Caution"
    if pct <= 75:
        return "Suspicious"
    return "High Risk"


# ── PredictService ────────────────────────────────────────────────────────────

class PredictService:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._load_cnn_lstm()
        self._load_xgboost()
        # SGD loaded lazily
        self._sgd_model = None
        self._sgd_vectorizer = None
        self._sgd_max_chars: int = 10000

    # ── CNN-LSTM ──────────────────────────────────────────────────────────────

    def _load_cnn_lstm(self) -> None:
        import onnxruntime as ort

        onnx_path = self.model_dir / "cnn_lstm_4_domain.onnx"
        vocab_path = self.model_dir / "vocab.json"
        metadata_path = self.model_dir / "metadata.json"
        feat_stats_path = self.model_dir / "feat_stats.json"

        for p in (onnx_path, vocab_path, metadata_path, feat_stats_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing model artifact: {p}")

        # Vocab
        with vocab_path.open(encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._char2idx: dict[str, int] = {k: int(v) for k, v in vocab_data.items()}
        self._pad_idx = int(self._char2idx.get("<PAD>", 0))
        self._unk_idx = int(self._char2idx.get("<UNK>", 1))

        # Metadata
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._max_len = int(meta["max_len"])

        # Feat stats (domain split)
        with feat_stats_path.open(encoding="utf-8") as f:
            stats = json.load(f)
        self._lexical_names: list[str] = list(stats["feature_names"])
        self._lexical_log1p: set[str] = set(stats["log1p_features"])
        self._lexical_mean = np.array(stats["mean"], dtype=np.float32)
        self._lexical_std = np.array(stats["std"], dtype=np.float32)

        # ONNX session (1 thread — optimal for single-URL latency on CPU)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._cnn_sess = ort.InferenceSession(
            str(onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        output_names = [o.name for o in self._cnn_sess.get_outputs()]
        self._cnn_outputs = output_names
        self._has_attention = "attention_weights" in output_names

    def _encode_url(self, url: str) -> np.ndarray:
        """Char-level encode — KHÔNG lowercase (case is signal)."""
        url = url.strip()
        seq = [self._char2idx.get(c, self._unk_idx) for c in url[:self._max_len]]
        if len(seq) < self._max_len:
            seq.extend([self._pad_idx] * (self._max_len - len(seq)))
        return np.array(seq, dtype=np.int64)

    def _normalize_lexical(self, raw: np.ndarray) -> np.ndarray:
        """Apply log1p for count/length features, then standardize."""
        arr = raw.astype(np.float32, copy=True)
        log1p_mask = np.array(
            [name in self._lexical_log1p for name in self._lexical_names], dtype=bool
        )
        arr[:, log1p_mask] = np.log1p(arr[:, log1p_mask])
        arr = (arr - self._lexical_mean) / self._lexical_std
        # Clip instead of raising on NaN/Inf (defensive for unusual URLs)
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
        return arr

    def predict_cnn(self, url: str, explain: bool = False) -> dict:
        t0 = time.perf_counter()
        url_stripped = url.strip()[:self._max_len * 2]  # soft truncation before encode

        x_seq = self._encode_url(url_stripped)[None, :]        # (1, max_len)
        raw_feat = _extract_lexical(url_stripped).reshape(1, -1)  # (1, 30)
        feat_norm = self._normalize_lexical(raw_feat)           # (1, 30)

        outs = self._cnn_sess.run(None, {"x_seq": x_seq, "x_feat": feat_norm})
        logit = float(outs[0].flatten()[0])
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        result: dict = {"probability": prob, "latency_ms": latency_ms}

        if explain and self._has_attention:
            attn_idx = self._cnn_outputs.index("attention_weights")
            attn_lq = outs[attn_idx][0]        # (L/4,) = (64,)
            url_trunc = url_stripped[:self._max_len]
            attn_per_char = np.repeat(attn_lq, 4)[:self._max_len][:len(url_trunc)]

            result["attention_weights"] = attn_lq.tolist()
            result["attention_per_char"] = attn_per_char.tolist()

            feat_flat = feat_norm.flatten()
            raw_flat = raw_feat.flatten()
            top_idx = np.argsort(-np.abs(feat_flat))[:5]
            result["top_lexical"] = [
                {
                    "name": self._lexical_names[i],
                    "z_score": float(feat_flat[i]),
                    "value": float(raw_flat[i]),
                }
                for i in top_idx
            ]

        return result

    # ── XGBoost ───────────────────────────────────────────────────────────────

    def _load_xgboost(self) -> None:
        import xgboost as xgb
        import shap

        model_path = self.model_dir / "xgb_url_4.ubj"
        cols_path = self.model_dir / "xgb_4_feature_cols.json"

        for p in (model_path, cols_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing model artifact: {p}")

        self._xgb_model = xgb.XGBClassifier()
        self._xgb_model.load_model(str(model_path))

        saved_cols = json.loads(cols_path.read_text(encoding="utf-8"))
        self._xgb_cols: list[str] = saved_cols

        # SHAP explainer created once — reused per request (read-only, thread-safe)
        self._shap_explainer = shap.TreeExplainer(self._xgb_model)

    def predict_xgb(self, url: str, explain: bool = False) -> dict:
        t0 = time.perf_counter()

        feats_dict = _extract_xgb_dict(url.strip())
        X_row = np.array(
            [[feats_dict.get(k, 0.0) for k in self._xgb_cols]], dtype=np.float32
        )
        prob = float(self._xgb_model.predict_proba(X_row)[0, 1])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        result: dict = {"probability": prob, "latency_ms": latency_ms}

        if explain:
            shap_vals = self._shap_explainer.shap_values(X_row)
            sv: np.ndarray
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            else:
                sv = shap_vals[0]

            expected = self._shap_explainer.expected_value
            if hasattr(expected, "__len__") and not isinstance(expected, (str, bytes)):
                expected = expected[1] if len(expected) > 1 else expected[0]
            expected_f = float(expected)

            top_idx = np.argsort(-np.abs(sv))[:10]
            result["shap_top_features"] = [
                {
                    "name": self._xgb_cols[i],
                    "shap_value": float(sv[i]),
                    "feature_value": float(X_row[0, i]),
                    "direction": "MAL" if sv[i] > 0 else "BEN",
                }
                for i in top_idx
            ]
            result["expected_value"] = expected_f

        return result

    # ── SGDClassifier (lazy) ──────────────────────────────────────────────────

    def _load_sgd(self) -> None:
        if self._sgd_model is not None:
            return
        import joblib
        from sklearn.feature_extraction.text import HashingVectorizer

        sgd_path = self.model_dir / "SGDClassifier_2.joblib"
        if not sgd_path.exists():
            raise FileNotFoundError(f"SGD model not found: {sgd_path}")

        artifact = joblib.load(str(sgd_path))
        self._sgd_vectorizer = HashingVectorizer(**artifact["vectorizer_params"])
        self._sgd_model = artifact["clf"]
        self._sgd_max_chars = artifact.get("html_max_chars", 10000)

    def predict_sgd(self, html: str) -> dict:
        self._load_sgd()
        t0 = time.perf_counter()

        normalized = _normalize_html(html, self._sgd_max_chars)
        X = self._sgd_vectorizer.transform([normalized])
        proba = self._sgd_model.predict_proba(X)[0]
        classes = list(self._sgd_model.classes_)
        try:
            phish_idx = classes.index(1)
        except ValueError:
            phish_idx = 1
        prob_phish = float(proba[phish_idx])

        return {
            "probability": prob_phish,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # ── Health check ──────────────────────────────────────────────────────────

    @property
    def loaded_models(self) -> list[str]:
        models = ["cnn_lstm", "xgboost"]
        if self._sgd_model is not None:
            models.append("sgd")
        return models

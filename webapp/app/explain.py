"""
Explainability rendering for Phase B UI.
- build_attention_heatmap_html: colored HTML spans per character
- format_shap_topk: sorted SHAP contributions with bar width %
"""
from html import escape
from typing import Optional

import numpy as np


def build_attention_heatmap_html(url: str, weights_per_char: list[float]) -> str:
    """
    Render URL as HTML with per-character background color based on attention weight.
    weights_per_char: len(url) floats (already upsampled & cropped).
    Returns HTML string of <span> elements.
    """
    if not weights_per_char or not url:
        return escape(url)

    w = np.array(weights_per_char, dtype=np.float32)
    w_max = float(w.max())
    if w_max <= 0:
        return escape(url)

    norm = w / w_max  # normalize to [0, 1]
    parts = []
    for ch, intensity in zip(url, norm):
        alpha = round(float(intensity) * 0.85, 2)
        if alpha < 0.05:
            parts.append(f"<span class='font-mono'>{escape(ch)}</span>")
        else:
            parts.append(
                f"<span class='font-mono' style='background:rgba(239,68,68,{alpha})'>"
                f"{escape(ch)}</span>"
            )
    return "".join(parts)


def format_shap_topk(
    shap_values: list[float],
    feature_names: list[str],
    feature_values: list[float],
    k: int = 10,
) -> list[dict]:
    """
    Return top-K SHAP features sorted by |shap| descending.
    Each item: {name, shap_value, feature_value, direction, bar_width_pct}
    """
    if not shap_values:
        return []

    sv = np.array(shap_values)
    abs_order = np.argsort(-np.abs(sv))[:k]
    max_abs = float(np.abs(sv[abs_order[0]])) if len(abs_order) > 0 else 1.0

    result = []
    for idx in abs_order:
        s = float(sv[idx])
        bar_pct = round(abs(s) / max(max_abs, 1e-9) * 100, 1)
        result.append({
            "name": feature_names[idx],
            "shap_value": s,
            "feature_value": float(feature_values[idx]),
            "direction": "MAL" if s > 0 else "BEN",
            "bar_width_pct": bar_pct,
        })
    return result

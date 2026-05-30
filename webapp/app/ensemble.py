def ensemble_average(cnn_prob: float, xgb_prob: float) -> float:
    """Simple average ensemble. SGD is NOT included — it is displayed separately."""
    return (cnn_prob + xgb_prob) / 2


def risk_level(prob: float) -> str:
    """4-tier risk labeling from raw probability (independent of binary threshold)."""
    pct = prob * 100
    if pct <= 25:
        return "Low Risk"
    if pct <= 50:
        return "Caution"
    if pct <= 75:
        return "Suspicious"
    return "High Risk"

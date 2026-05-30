---
title: secURLity
emoji: 🔒
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# secURLity — Malicious URL Detection

Detects malicious URLs using an ensemble of two AI models trained on **19.68 million real-world URLs** (80% benign / 20% malicious, no synthetic data).

## Models

| Model | Approach | Test F1 | Latency |
|---|---|---|---|
| **CNN-LSTM** | Char-level sequence + 30 lexical features | 0.9888 | ~1.7 ms |
| **XGBoost** | 105 tabular URL features | 0.9996 | < 1 ms |
| **Ensemble** | Simple average of both probabilities | — | < 3 ms total |

Optional: **SGDClassifier** (HTML content analysis, phishing-specific, requires page fetch).

## API

```bash
curl -X POST https://<username>-securlity.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "explain": true}'
```

Full API docs: `/docs`

## Rate Limit

10 requests / minute / IP.

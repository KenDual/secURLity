FROM python:3.11-slim

# libgomp1 required by XGBoost; gcc for compiling any C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bundled_models/ ./bundled_models/
COPY app/ ./app/

# HF Spaces mounts /data as persistent volume (ephemeral on free tier)
RUN mkdir -p /data

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "warning"]

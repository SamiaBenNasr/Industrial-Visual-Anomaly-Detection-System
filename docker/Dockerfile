# Stage 1 — base commune
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RICH_DISABLE=1 \
    TQDM_DISABLE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# Stage 2 — build des dépendances
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip cache purge || true

RUN find /usr/local -name '*.pyc' -delete \
 && find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true \
 && find /usr/local/lib/python3.11/site-packages -name 'tests' -type d -exec rm -rf {} + 2>/dev/null || true


ENV TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/hf \
    HF_HUB_OFFLINE=0
RUN mkdir -p /app/.cache/torch /app/.cache/hf \
 && python -c "\
from anomalib.models import Patchcore; \
Patchcore(); \
print('Poids telecharges dans /app/.cache')"


# Stage 3 — image finale
FROM base AS runtime

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/.cache/torch /app/.cache/torch
COPY --from=builder /app/.cache/hf /app/.cache/hf

ENV TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/hf \
    HF_HUB_OFFLINE=1

COPY endpoint_score.py ./endpoint_score.py
COPY server.py ./server.py

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
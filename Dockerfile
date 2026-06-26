# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — PatchCore Anomaly Detection (inference)
# Build :  docker build -t patchcore-inference:latest .
# Run   :  docker run --rm -p 8080:8080 patchcore-inference:latest
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1 — base avec dépendances lourdes (cacheable)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RICH_DISABLE=1 \
    TQDM_DISABLE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances système minimales pour OpenCV + anomalib
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libgl1-mesa-dri \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# ── Dépendances Python ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Stage 2 — image finale (sans outils de build)
FROM base AS runtime

WORKDIR /app

# Copier uniquement les scripts nécessaires à l'inférence
COPY score.py          ./score.py
COPY endpoint_score.py ./endpoint_score.py

# Port exposé (API FastAPI)
EXPOSE 8080

# ── Serveur HTTP minimal ──────────────────────────────────────────────────────
# On enveloppe endpoint_score.py dans une micro-API FastAPI
# pour pouvoir tester localement sans Azure ML.
COPY server.py ./server.py

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

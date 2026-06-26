"""
server.py — Micro-API FastAPI pour tester l'endpoint localement (hors Azure ML).

Démarre avec :
    uvicorn server:app --reload --port 8080

Ou via Docker :
    docker run --rm -p 8080:8080 -e AZUREML_MODEL_DIR=./outputs patchcore-inference:latest

POST /score
    Body : { "image_b64": "<base64>", "image_size": [256, 256] }
    Réponse : { "pred_score": float, "pred_label": int, "is_anomalous": bool }

GET /health
    Réponse : { "status": "ok" }
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import endpoint_score as scoring_module

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise le modèle au démarrage."""
    log.info("Loading model …")
    scoring_module.init()
    log.info("Model ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="PatchCore Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    image_b64:  str
    image_size: list[int] = [256, 256]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    import json
    raw = json.dumps(req.model_dump())
    result_str = scoring_module.run(raw)
    result = json.loads(result_str)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(content=result)

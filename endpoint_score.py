"""
endpoint_score.py 

Payload JSON attendu:
{
  "image_b64": "<base64-encoded PNG/JPEG>",
  "image_size": [256, 256]          // optionnel, défaut 256x256
}

Réponse JSON:
{
  "pred_score": 0.312,
  "pred_label": 0,
  "is_anomalous": false
}
"""

import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Globals initialisés dans init()
_engine = None
_model  = None
_ckpt   = None


def init() -> None:
    """Appelé une fois au démarrage — charge le modèle en mémoire."""
    global _engine, _model, _ckpt

    import os
    os.environ.setdefault("RICH_DISABLE", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    # Azure ML monte le modèle enregistré dans AZUREML_MODEL_DIR
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    ckpt_candidates = list(Path(model_dir).rglob("*.ckpt"))

    if not ckpt_candidates:
        raise FileNotFoundError(f"No .ckpt found under {model_dir}")

    _ckpt   = str(ckpt_candidates[0])
    _model  = Patchcore()
    _engine = Engine()

    log.info("Model loaded from: %s", _ckpt)


def run(raw_data: str) -> str:
    """Appelé à chaque requête POST."""
    from anomalib.data import PredictDataset
    from PIL import Image

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    # ── Décoder l'image ───────────────────────────────────────────────────────
    image_b64 = payload.get("image_b64")
    if not image_b64:
        return json.dumps({"error": "Missing 'image_b64' field"})

    try:
        image_bytes = base64.b64decode(image_b64)
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return json.dumps({"error": f"Cannot decode image: {e}"})

    image_size = tuple(payload.get("image_size", [256, 256]))

    # Sauvegarder temporairement (PredictDataset lit depuis le disque)
    # Use tempfile for cross-platform compatibility (Windows doesn't have /tmp)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    pil_image.save(tmp_path)

    # ── Inférence ─────────────────────────────────────────────────────────────
    try:
        dataset = PredictDataset(path=str(tmp_path), image_size=image_size)
        predictions = _engine.predict(
            model=_model,
            dataset=dataset,
            ckpt_path=_ckpt,
        )
    except Exception as e:
        log.exception("Prediction failed")
        return json.dumps({"error": str(e)})

    if not predictions:
        return json.dumps({"error": "No prediction returned"})

    pred = predictions[0]
    score = float(pred.pred_score.item())
    label = int(pred.pred_label.item()) if hasattr(pred.pred_label, "item") \
            else int(pred.pred_label)

    result = {
        "pred_score":   round(score, 6),
        "pred_label":   label,
        "is_anomalous": bool(label),
    }
    
    # Clean up temporary file
    try:
        tmp_path.unlink()
    except Exception as e:
        log.warning("Could not delete temp file: %s", e)
    
    return json.dumps(result)

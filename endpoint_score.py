"""
endpoint_score.py

Payload JSON attendu:
{
  "image_b64": "<base64-encoded PNG/JPEG>",
  "image_size": [256, 256]          // optionnel, defaut 256x256
}

Reponse JSON:
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
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Globals initialises dans init()
_engine = None
_model  = None
_ckpt   = None


def init() -> None:
    """Appele une fois au demarrage - charge le modele en memoire."""
    global _engine, _model, _ckpt

    import os
    os.environ.setdefault("RICH_DISABLE", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    ckpt_candidates = list(Path(model_dir).rglob("*.ckpt"))

    if not ckpt_candidates:
        raise FileNotFoundError(f"No .ckpt found under {model_dir}")

    _ckpt   = str(ckpt_candidates[0])
    _model  = Patchcore()
    _engine = Engine()

    log.info("Model loaded from: %s", _ckpt)


def _image_quality_checks(pil_image) -> dict:
    """Signal type 2: basic input data quality checks, logged as custom
    metrics. Catches corrupt uploads, wrong resolution cameras, all-black
    frames (lens cap / camera offline), etc. -- issues that would otherwise
    silently degrade predictions without ever showing up as an HTTP error."""
    arr = np.array(pil_image.convert("L"))  # grayscale for quick stats
    return {
        "width": pil_image.width,
        "height": pil_image.height,
        "mean_brightness": float(arr.mean()),
        "std_brightness": float(arr.std()),
        "is_near_blank": bool(arr.std() < 2.0),  # near-uniform image = likely bad capture
    }


def run(raw_data: str) -> str:
    """Appele a chaque requete POST."""
    from anomalib.data import PredictDataset
    from PIL import Image

    start = time.time()

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    image_b64 = payload.get("image_b64")
    if not image_b64:
        return json.dumps({"error": "Missing 'image_b64' field"})

    try:
        image_bytes = base64.b64decode(image_b64)
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return json.dumps({"error": f"Cannot decode image: {e}"})

    image_size = tuple(payload.get("image_size", [256, 256]))

    # ---- Signal 2: data quality checks, logged regardless of prediction outcome ----
    quality = _image_quality_checks(pil_image)
    if quality["is_near_blank"]:
        log.warning("INPUT_QUALITY_FLAG near_blank_image width=%s height=%s std=%s",
                    quality["width"], quality["height"], quality["std_brightness"])

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    pil_image.save(tmp_path)

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

    latency_ms = (time.time() - start) * 1000

    
    log.info(
        "PREDICTION_METRIC pred_score=%.6f pred_label=%d is_anomalous=%s "
        "latency_ms=%.1f mean_brightness=%.2f",
        score, label, bool(label), latency_ms, quality["mean_brightness"],
    )

    result = {
        "pred_score":   round(score, 6),
        "pred_label":   label,
        "is_anomalous": bool(label),
    }

    try:
        tmp_path.unlink()
    except Exception as e:
        log.warning("Could not delete temp file: %s", e)

    return json.dumps(result)
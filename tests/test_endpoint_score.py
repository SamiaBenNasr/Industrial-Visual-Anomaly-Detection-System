"""
Tests unitaires pour endpoint_score.py.

"""

import base64
import io
import json
import types

import pytest
from PIL import Image

import endpoint_score


def _make_test_image_b64(size=(64, 64), color=(255, 0, 0)) -> str:
    """Genere une vraie petite image PNG encodee en base64 (pas un mock:
    endpoint_score.run() doit reellement pouvoir la decoder)."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class FakePred:
    """Imite un objet prediction anomalib (pred_score.item(), pred_label)."""
    def __init__(self, score: float, label: int):
        self.pred_score = types.SimpleNamespace(item=lambda: score)
        self.pred_label = label


class FakeEngine:
    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, model, dataset, ckpt_path):
        return self._predictions


class BrokenEngine:
    """Simule un crash d'inference (GPU OOM, checkpoint corrompu, etc.)."""
    def predict(self, model, dataset, ckpt_path):
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def reset_globals():
    """Les globals du module sont partages entre tests -- on les remet a
    None apres chaque test pour eviter qu'un test pollue le suivant."""
    yield
    endpoint_score._engine = None
    endpoint_score._model = None
    endpoint_score._ckpt = None


def _set_fake_engine(predictions) -> None:
    endpoint_score._engine = FakeEngine(predictions)
    endpoint_score._model = object()
    endpoint_score._ckpt = "fake.ckpt"


# ── Validation du payload ────────────────────────────────────────────────

def test_run_invalid_json_returns_error():
    result = json.loads(endpoint_score.run("ceci n'est pas du json"))
    assert "error" in result
    assert "Invalid JSON" in result["error"]


def test_run_missing_image_b64_returns_error():
    result = json.loads(endpoint_score.run(json.dumps({})))
    assert "error" in result
    assert "image_b64" in result["error"]


def test_run_invalid_base64_returns_error():
    payload = json.dumps({"image_b64": "!!!pas-du-base64-valide!!!"})
    result = json.loads(endpoint_score.run(payload))
    assert "error" in result
    assert "decode" in result["error"].lower()


def test_run_valid_base64_but_not_an_image_returns_error():
    """Base64 syntaxiquement valide, mais les octets decodes ne forment
    pas une image (fichier corrompu, mauvais format, texte brut...).
    Chemin de code different de l'erreur de decodage base64 : ici
    base64.b64decode() reussit, c'est PIL.Image.open() qui echoue."""
    garbage_bytes = b"ceci n'est absolument pas une image PNG/JPEG valide"
    payload = json.dumps({"image_b64": base64.b64encode(garbage_bytes).decode("utf-8")})
    result = json.loads(endpoint_score.run(payload))
    assert "error" in result
    assert "decode" in result["error"].lower()


def test_run_empty_image_b64_returns_error():
    """Chaine vide : passe le `if not image_b64` (falsy) -> meme chemin
    que le champ manquant, mais bon a couvrir explicitement."""
    payload = json.dumps({"image_b64": ""})
    result = json.loads(endpoint_score.run(payload))
    assert "error" in result
    assert "image_b64" in result["error"]


def test_run_truncated_image_returns_error():
    """Image PNG valide au depart, mais tronquee (upload interrompu,
    payload coupe en cours de route) -- cas realiste en production."""
    img = Image.new("RGB", (64, 64), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    truncated_bytes = buf.getvalue()[: len(buf.getvalue()) // 2]

    payload = json.dumps({"image_b64": base64.b64encode(truncated_bytes).decode("utf-8")})
    result = json.loads(endpoint_score.run(payload))
    assert "error" in result


# ── Inference reussie ────────────────────────────────────────────────────

def test_run_success_normal_case():
    _set_fake_engine([FakePred(score=0.12, label=0)])

    payload = json.dumps({"image_b64": _make_test_image_b64()})
    result = json.loads(endpoint_score.run(payload))

    assert result["pred_label"] == 0
    assert result["is_anomalous"] is False
    assert result["pred_score"] == pytest.approx(0.12)


def test_run_success_anomalous_case():
    _set_fake_engine([FakePred(score=0.91, label=1)])

    payload = json.dumps({"image_b64": _make_test_image_b64()})
    result = json.loads(endpoint_score.run(payload))

    assert result["pred_label"] == 1
    assert result["is_anomalous"] is True


def test_run_respects_custom_image_size():
    _set_fake_engine([FakePred(score=0.05, label=0)])

    payload = json.dumps({
        "image_b64": _make_test_image_b64(size=(128, 128)),
        "image_size": [128, 128],
    })
    result = json.loads(endpoint_score.run(payload))
    assert "pred_score" in result


# ── Gestion des erreurs d'inference ──────────────────────────────────────

def test_run_no_predictions_returned():
    _set_fake_engine([])  # engine.predict() renvoie une liste vide

    payload = json.dumps({"image_b64": _make_test_image_b64()})
    result = json.loads(endpoint_score.run(payload))
    assert "error" in result


def test_run_prediction_raises_exception():
    endpoint_score._engine = BrokenEngine()
    endpoint_score._model = object()
    endpoint_score._ckpt = "fake.ckpt"

    payload = json.dumps({"image_b64": _make_test_image_b64()})
    result = json.loads(endpoint_score.run(payload))
    assert "error" in result
    assert "boom" in result["error"]
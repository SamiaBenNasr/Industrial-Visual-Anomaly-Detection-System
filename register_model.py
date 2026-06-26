"""
register_model.py — Enregistre le checkpoint PatchCore dans Azure ML Model Registry.

Usage:
    python register_model.py --ckpt PATH [--name STR] [--version STR]
                             [--category STR] [--metrics-file PATH]

Variables d'env requises (ou passées en args) :
    AML_SUBSCRIPTION_ID, AML_RESOURCE_GROUP, AML_WORKSPACE
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register PatchCore model in Azure ML")
    parser.add_argument("--ckpt",           required=True,
                        help="Chemin local vers le .ckpt ou dossier de weights")
    parser.add_argument("--name",           default="patchcore-anomaly-detector",
                        help="Nom du modèle dans le registry")
    parser.add_argument("--version",        default=None,
                        help="Version explicite (auto si absent)")
    parser.add_argument("--category",       default="bottle",
                        help="Catégorie MVTecAD — stockée comme tag")
    parser.add_argument("--metrics-file",   default=None,
                        help="Chemin vers metrics.json produit par train.py")
    parser.add_argument("--subscription",   default=os.getenv("AML_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AML_RESOURCE_GROUP"))
    parser.add_argument("--workspace",      default=os.getenv("AML_WORKSPACE"))
    return parser.parse_args()


def load_metrics(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        log.warning("metrics-file not found: %s", path)
        return {}
    return json.loads(p.read_text())


def main() -> None:
    args = parse_args()

    for var, val in [
        ("subscription",   args.subscription),
        ("resource-group", args.resource_group),
        ("workspace",      args.workspace),
    ]:
        if not val:
            log.error("Missing --%s (or env var AML_%s)", var, var.upper().replace("-", "_"))
            sys.exit(1)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import DefaultAzureCredential

    ml = MLClient(
        DefaultAzureCredential(),
        subscription_id=args.subscription,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    metrics = load_metrics(args.metrics_file)

    # Tags: metadata visible dans Azure ML Studio
    tags = {
        "framework":  "anomalib",
        "model_type": "patchcore",
        "category":   args.category,
        **{f"metric_{k}": str(round(float(v), 4)) for k, v in metrics.items()
           if isinstance(v, (int, float))},
    }

    model = Model(
        path=str(ckpt_path),
        name=args.name,
        version=args.version,                   # None → auto-increment
        type=AssetTypes.CUSTOM_MODEL,
        description=(
            f"PatchCore anomaly detector trained on MVTecAD/{args.category}. "
            f"Exported via anomalib."
        ),
        tags=tags,
    )

    log.info("Registering model '%s' from %s …", args.name, ckpt_path)
    registered = ml.models.create_or_update(model)

    log.info("✅ Registered: %s  version=%s", registered.name, registered.version)
    log.info("   ID : %s", registered.id)

    # Expose pour le step GitHub Actions suivant
    gha_output = os.getenv("GITHUB_OUTPUT")
    if gha_output:
        with open(gha_output, "a") as fh:
            fh.write(f"model_name={registered.name}\n")
            fh.write(f"model_version={registered.version}\n")
        log.info("GITHUB_OUTPUT written.")


if __name__ == "__main__":
    main()

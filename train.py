"""
train.py — Anomaly detection training with PatchCore + MVTecAD
Usage:
    python train.py [--category CATEGORY] [--data-root PATH] [--output-dir PATH]
                    [--epochs N] [--train-batch INT] [--eval-batch INT]
                    [--neighbors N] [--export-format {onnx,torch,openvino}]
"""

import argparse
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

os.environ.setdefault("RICH_DISABLE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import mlflow
import azureml.mlflow


def _flatten_metrics(metrics: dict) -> dict:
    """Ne garde que les valeurs numeriques loggables par mlflow.log_metrics."""
    flat = {}
    for k, v in metrics.items():
        try:
            flat[k] = float(v)
        except (TypeError, ValueError):
            log.warning("Metric '%s' non numerique, ignoree pour MLflow (%r)", k, v)
    return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchCore anomaly detector")
    parser.add_argument("--category",      default="bottle",     help="MVTecAD category")
    parser.add_argument("--data-root",     default="data/mvtec", help="Root folder of MVTecAD dataset")
    parser.add_argument("--output-dir",    default="outputs",    help="Where to save weights & metrics")
    parser.add_argument("--epochs",        type=int, default=1,  help="Max training epochs")
    parser.add_argument("--train-batch",   type=int, default=32)
    parser.add_argument("--eval-batch",    type=int, default=32)
    parser.add_argument("--neighbors",     type=int, default=6,  help="PatchCore num_neighbors")
    parser.add_argument("--export-format", default="onnx",
                        choices=["onnx", "torch", "openvino"])
    parser.add_argument("--run-name", default=None,
                        help="Nom du run MLflow (defaut: auto-genere)")
    parser.add_argument("--experiment-name", default="patchcore-anomaly-detection",
                        help="Nom de l'experiment MLflow/Azure ML")
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"),
                        help="Optionnel: connecte MLflow au workspace Azure ML (tracking distant, "
                             "sans compute). Omets ces 3 args pour rester en tracking local (./mlruns).")
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace-name", default=os.getenv("AZURE_WORKSPACE_NAME"))
    return parser.parse_args()


def _maybe_connect_azureml_tracking(args: argparse.Namespace) -> None:
    """Si les 3 identifiants workspace sont fournis, pointe MLflow vers le
    tracking server Azure ML -- les runs apparaissent alors dans
    ml.azure.com > Jobs > All experiments, MEME si l'entrainement tourne
    sur ce PC et pas sur un compute Azure ML. Aucun job/compute requis :
    c'est juste le tracking (ecriture de metriques/artefacts) qui vise
    le workspace au lieu d'un dossier local.
    Sans ces identifiants, mlflow reste en tracking local (./mlruns).
    """
    if not (args.subscription_id and args.resource_group and args.workspace_name):
        log.info("Pas d'identifiants Azure ML fournis -> tracking MLflow local (./mlruns).")
        return

    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name,
        )
        tracking_uri = ml_client.workspaces.get(args.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        log.info("MLflow connecte au workspace Azure ML '%s' (tracking distant, sans compute).",
                  args.workspace_name)
    except Exception as e:
        log.warning("Impossible de se connecter au tracking Azure ML (%s). "
                    "Fallback sur le tracking local (./mlruns).", e)


def image_item_collate(batch):
    """Collate a list of ImageItem into an ImageBatch.
    Attributs réels découverts : image, gt_label, gt_mask, image_path, mask_path,
    anomaly_map, pred_score, pred_mask, pred_label, explanation
    """
    import torch
    from anomalib.data.dataclasses.torch.image import ImageBatch

    images     = torch.stack([item.image    for item in batch])
    gt_labels  = torch.stack([item.gt_label for item in batch])
    image_path = [item.image_path for item in batch]

    gt_masks = None
    if batch[0].gt_mask is not None:
        try:
            gt_masks = torch.stack([item.gt_mask for item in batch])
        except Exception:
            gt_masks = None

    return ImageBatch(
        image=images,
        gt_label=gt_labels,
        gt_mask=gt_masks,
        image_path=image_path,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _maybe_connect_azureml_tracking(args)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        log.info("MLflow run_id: %s", run_id)

        mlflow.log_params({
            "category":      args.category,
            "epochs":        args.epochs,
            "train_batch":   args.train_batch,
            "eval_batch":    args.eval_batch,
            "neighbors":     args.neighbors,
            "export_format": args.export_format,
        })

        _train_and_evaluate(args, output_dir)

        # run_id.txt permet au job CI (train-and-register) de recuperer
        # le run sans parser les logs stdout.
        (output_dir / "run_id.txt").write_text(run_id)
        mlflow.log_artifact(str(output_dir / "run_id.txt"))


def _train_and_evaluate(args: argparse.Namespace, output_dir: Path) -> None:
    from anomalib.data.datasets.image.mvtecad import (
        MVTecADDataset,
        make_mvtec_ad_dataset,
        IMG_EXTENSIONS,
    )
    from anomalib.data.utils import Split
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from torch.utils.data import DataLoader

    export_map = {
        "onnx":     ExportType.ONNX,
        "torch":    ExportType.TORCH,
        "openvino": ExportType.OPENVINO,
    }

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info("Loading MVTecAD — category: %s", args.category)
    root_category = Path(args.data_root) / args.category

    # Fix bug anomalib 2.5.0 Windows : split column est str pas enum
    all_samples = make_mvtec_ad_dataset(
        root_category, split=None, extensions=IMG_EXTENSIONS
    )

    train_ds = MVTecADDataset(
        root=Path(args.data_root), category=args.category, split=Split.TRAIN
    )
    train_ds.samples = all_samples[all_samples["split"] == "train"].reset_index(drop=True)

    test_ds = MVTecADDataset(
        root=Path(args.data_root), category=args.category, split=Split.TEST
    )
    test_ds.samples = all_samples[all_samples["split"] == "test"].reset_index(drop=True)

    log.info("Train samples: %d", len(train_ds))
    log.info("Test  samples: %d", len(test_ds))

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=image_item_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=image_item_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Initialising PatchCore (num_neighbors=%d)", args.neighbors)
    model = Patchcore(num_neighbors=args.neighbors)

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = Engine(
        max_epochs=args.epochs,
        enable_progress_bar=False,
        default_root_dir=str(output_dir),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training …")
    engine.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )
    log.info("Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("Running evaluation …")
    test_results = engine.test(
        model=model,
        dataloaders=test_loader,
    )

    metrics: dict = test_results[0] if test_results else {}
    log.info("Test metrics: %s", metrics)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics saved → %s", metrics_path)

    mlflow.log_metrics(_flatten_metrics(metrics))
    mlflow.log_artifact(str(metrics_path))

    # ── Export ────────────────────────────────────────────────────────────────
    log.info("Exporting model as %s …", args.export_format.upper())
    engine.export(
        model=model,
        export_type=export_map[args.export_format],
        export_root=str(output_dir / "export"),
    )
    log.info("Export complete → %s/export", output_dir)

    # ── Checkpoint & artefacts export dans MLflow ────────────────────────────
    ckpt_candidates = list(output_dir.rglob("*.ckpt"))
    if ckpt_candidates:
        mlflow.log_artifact(str(ckpt_candidates[0]), artifact_path="checkpoint")
        log.info("Checkpoint logge dans MLflow: %s", ckpt_candidates[0])
    else:
        log.warning("Aucun .ckpt trouve sous %s, rien a logger comme checkpoint", output_dir)

    export_dir = output_dir / "export"
    if export_dir.exists():
        mlflow.log_artifacts(str(export_dir), artifact_path="export")


if __name__ == "__main__":
    main()
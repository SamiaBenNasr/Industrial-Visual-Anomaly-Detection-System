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
    return parser.parse_args()


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

    # ── Export ────────────────────────────────────────────────────────────────
    log.info("Exporting model as %s …", args.export_format.upper())
    engine.export(
        model=model,
        export_type=export_map[args.export_format],
        export_root=str(output_dir / "export"),
    )
    log.info("Export complete → %s/export", output_dir)


if __name__ == "__main__":
    main()
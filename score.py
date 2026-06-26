"""
score.py — Run PatchCore inference on one image or a folder of images.
Usage:
    python score.py --input PATH --ckpt PATH [--image-size WxH] [--output-dir PATH]
                    [--save-maps] [--threshold FLOAT]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

os.environ.setdefault("RICH_DISABLE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score images with a trained PatchCore model")
    parser.add_argument("--input",      required=True,
                        help="Path to an image file OR a directory of images")
    parser.add_argument("--ckpt",       required=True,
                        help="Path to the .ckpt checkpoint produced by train.py")
    parser.add_argument("--image-size", default="256x256",
                        help="HxW to resize images, e.g. 256x256")
    parser.add_argument("--output-dir", default="outputs/predictions",
                        help="Where to write predictions.json (and optional anomaly maps)")
    parser.add_argument("--save-maps",  action="store_true",
                        help="Save anomaly heatmaps as PNG files")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Override anomaly score threshold (optional)")
    return parser.parse_args()


def parse_image_size(spec: str) -> tuple[int, int]:
    """Parse '256x256' or '256,256' → (256, 256)."""
    for sep in ("x", ","):
        if sep in spec:
            w, h = spec.split(sep)
            return int(w.strip()), int(h.strip())
    s = int(spec)
    return s, s


def save_heatmap(prediction, out_path: Path) -> None:
    """Save the anomaly map as a PNG — only imported when --save-maps is set."""
    import matplotlib
    matplotlib.use("Agg")                   # non-interactive, safe in CI
    import matplotlib.pyplot as plt

    anomaly_map = prediction.anomaly_map.squeeze()
    score  = prediction.pred_score.item()
    label  = prediction.pred_label.item() if hasattr(prediction.pred_label, "item") \
             else prediction.pred_label

    fig, ax = plt.subplots()
    im = ax.imshow(anomaly_map, cmap="hot")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Label: {label} | Score: {score:.4f}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image_size = parse_image_size(args.image_size)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Lazy imports ──────────────────────────────────────────────────────────
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info("Loading images from: %s", args.input)
    dataset = PredictDataset(
        path=args.input,
        image_size=image_size,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    model  = Patchcore()
    engine = Engine()

    log.info("Running predictions (ckpt=%s) …", ckpt_path)
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
    )

    if not predictions:
        log.warning("No predictions returned.")
        sys.exit(0)

    # ── Collect results ───────────────────────────────────────────────────────
    results = []
    for i, pred in enumerate(predictions):
        score = float(pred.pred_score.item())
        label = int(pred.pred_label.item()) if hasattr(pred.pred_label, "item") \
                else int(pred.pred_label)

        is_anomalous = bool(label)
        if args.threshold is not None:
            is_anomalous = score >= args.threshold

        # Handle image_path being a list or string
        image_path_val = pred.image_path
        if isinstance(image_path_val, list) and len(image_path_val) > 0:
            image_path_val = image_path_val[0]
        
        entry = {
            "image_path":   str(image_path_val),
            "pred_score":   score,
            "pred_label":   label,
            "is_anomalous": is_anomalous,
        }
        results.append(entry)
        log.info("[%d] %s  score=%.4f  label=%d", i, Path(str(image_path_val)).name, score, label)

        if args.save_maps:
            # Handle image_path being a list or string
            img_path_for_name = str(image_path_val)
            map_path = output_dir / f"heatmap_{i:04d}_{Path(img_path_for_name).stem}.png"
            save_heatmap(pred, map_path)
            log.info("Heatmap saved → %s", map_path)

    # ── Persist JSON ──────────────────────────────────────────────────────────
    out_json = output_dir / "predictions.json"
    out_json.write_text(json.dumps(results, indent=2))
    log.info("Predictions saved → %s  (%d images)", out_json, len(results))

    anomalous = sum(r["is_anomalous"] for r in results)
    log.info("Summary: %d / %d anomalous", anomalous, len(results))


if __name__ == "__main__":
    main()

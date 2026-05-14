"""Benchmark YOLO detections against VisDrone DET validation labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.detector_wrapper import ObjectDetector
from perception.visdrone_benchmark import (
    aggregate_counts,
    class_groups_for_names,
    evaluate_image,
    parse_visdrone_annotation,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO detections on VisDrone DET images."
    )
    parser.add_argument(
        "dataset_root",
        help="Path to VisDrone2019-DET-val containing images/ and annotations/.",
    )
    parser.add_argument(
        "--model",
        default="yolo26x.pt",
        help="YOLO model name or path. Default: yolo26x.pt",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["people", "car"],
        choices=["people", "car", "vehicle"],
        help="Class groups to evaluate. Default: people car",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of images to evaluate. Default: 25",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample images instead of using sorted order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=540,
        help="Random seed used with --random-sample. Default: 540",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Detector confidence threshold. Default: 0.10",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size. Default: 1280",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU threshold for true positives. Default: 0.50",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Ignore ground-truth boxes smaller than this pixel area. Default: 0",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmarks/visdrone",
        help="Directory for benchmark outputs.",
    )
    return parser.parse_args()


def discover_images(
    dataset_root: Path,
    limit: int | None,
    random_sample: bool = False,
    seed: int = 540,
) -> List[Path]:
    image_dir = dataset_root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"VisDrone image directory not found: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit is not None:
        sample_size = min(limit, len(image_paths))
        if random_sample:
            rng = random.Random(seed)
            return sorted(rng.sample(image_paths, sample_size))
        return image_paths[:sample_size]

    if random_sample:
        rng = random.Random(seed)
        shuffled_paths = list(image_paths)
        rng.shuffle(shuffled_paths)
        return shuffled_paths

    return image_paths


def annotation_path_for_image(dataset_root: Path, image_path: Path) -> Path:
    return dataset_root / "annotations" / f"{image_path.stem}.txt"


def save_outputs(summary: Dict[str, Any], per_image_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows_path = output_dir / "per_image_metrics.csv"
    if per_image_rows:
        with rows_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

    print(f"Benchmark summary saved to: {summary_path}")
    print(f"Per-image metrics saved to: {rows_path}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_paths = discover_images(
        dataset_root,
        args.limit,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    if not image_paths:
        raise ValueError(f"No images found under {dataset_root / 'images'}")
    if args.limit is not None and len(image_paths) < args.limit:
        print(
            f"Requested {args.limit} images, but only {len(image_paths)} are available. "
            "Evaluating all available images."
        )

    class_groups = class_groups_for_names(args.classes)
    target_gt_labels = set()
    for group in class_groups.values():
        target_gt_labels.update(group["visdrone_labels"])

    try:
        detector = ObjectDetector(
            model_name_or_path=args.model,
            confidence_threshold=args.conf,
            image_size=args.imgsz,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not load model '{args.model}'. If this is a newer Ultralytics "
            "checkpoint name, upgrade ultralytics or pass the full path to the "
            "downloaded .pt file."
        ) from exc

    per_image_counts = []
    per_image_rows: List[Dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        annotation_path = annotation_path_for_image(dataset_root, image_path)
        ground_truth = parse_visdrone_annotation(
            annotation_path,
            target_labels=target_gt_labels,
            min_area=args.min_area,
        )
        detections = detector.detect(image_path)
        image_counts = evaluate_image(
            detections=detections,
            ground_truth=ground_truth,
            class_groups=class_groups,
            iou_threshold=args.iou,
        )
        per_image_counts.append(image_counts)

        for class_name, counts in image_counts.items():
            row = {
                "image": image_path.name,
                "class_group": class_name,
                **counts.to_metrics(),
            }
            per_image_rows.append(row)

        print(f"[{index}/{len(image_paths)}] evaluated {image_path.name}")

    aggregate = aggregate_counts(per_image_counts)
    metrics = {class_name: counts.to_metrics() for class_name, counts in aggregate.items()}
    summary = {
        "dataset_root": str(dataset_root),
        "model": args.model,
        "image_count": len(image_paths),
        "requested_limit": args.limit,
        "random_sample": args.random_sample,
        "random_seed": args.seed if args.random_sample else None,
        "classes": args.classes,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "image_size": args.imgsz,
        "min_ground_truth_area": args.min_area,
        "metrics": metrics,
        "notes": [
            "VisDrone has ground-truth labels for people and vehicle categories.",
            "YOLO generic COCO-style labels are mapped to VisDrone class groups for this lightweight context benchmark.",
            "This is not the official VisDrone challenge mAP protocol; it reports IoU-threshold precision/recall/F1.",
        ],
    }

    save_outputs(summary, per_image_rows, Path(args.output_dir))

    print("\nSummary")
    for class_name, class_metrics in metrics.items():
        print(
            f"{class_name}: "
            f"P={class_metrics['precision']:.3f} "
            f"R={class_metrics['recall']:.3f} "
            f"F1={class_metrics['f1']:.3f} "
            f"GT={class_metrics['ground_truth_count']} "
            f"Pred={class_metrics['prediction_count']}"
        )


if __name__ == "__main__":
    main()

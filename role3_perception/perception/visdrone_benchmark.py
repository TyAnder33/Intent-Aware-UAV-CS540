"""VisDrone detection benchmark helpers.

VisDrone DET annotations are CSV-like text files with one box per line:
`bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion`.
This module evaluates normalized detector outputs against those labels for a
small set of class groups such as people and cars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Detection = Dict[str, Any]

VISDRONE_CATEGORY_BY_ID = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

DEFAULT_CLASS_GROUPS = {
    "people": {
        "visdrone_labels": {"pedestrian", "people"},
        "prediction_labels": {"person"},
    },
    "car": {
        "visdrone_labels": {"car"},
        "prediction_labels": {"car"},
    },
    "vehicle": {
        "visdrone_labels": {"car", "van", "truck", "bus"},
        "prediction_labels": {"car", "truck", "bus"},
    },
}


@dataclass(frozen=True)
class GroundTruthObject:
    label: str
    bbox_xyxy: List[float]
    truncation: int
    occlusion: int


@dataclass
class ClassCounts:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ground_truth_count: int = 0
    prediction_count: int = 0

    def to_metrics(self) -> Dict[str, float]:
        precision = _safe_divide(self.true_positives, self.true_positives + self.false_positives)
        recall = _safe_divide(self.true_positives, self.true_positives + self.false_negatives)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        return {
            "ground_truth_count": self.ground_truth_count,
            "prediction_count": self.prediction_count,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def parse_visdrone_annotation(
    annotation_path: Path,
    target_labels: Optional[Iterable[str]] = None,
    min_area: float = 0.0,
) -> List[GroundTruthObject]:
    """Parse one VisDrone DET annotation file."""
    target_label_set = set(target_labels) if target_labels is not None else None
    if not annotation_path.is_file():
        return []

    objects: List[GroundTruthObject] = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue

        left, top, width, height = (float(value) for value in parts[:4])
        score = int(float(parts[4]))
        category_id = int(float(parts[5]))
        truncation = int(float(parts[6]))
        occlusion = int(float(parts[7]))

        if score == 0:
            continue

        label = VISDRONE_CATEGORY_BY_ID.get(category_id)
        if label is None:
            continue

        if target_label_set is not None and label not in target_label_set:
            continue

        area = max(0.0, width) * max(0.0, height)
        if area < min_area:
            continue

        objects.append(
            GroundTruthObject(
                label=label,
                bbox_xyxy=[left, top, left + width, top + height],
                truncation=truncation,
                occlusion=occlusion,
            )
        )

    return objects


def evaluate_image(
    detections: Sequence[Detection],
    ground_truth: Sequence[GroundTruthObject],
    class_groups: Dict[str, Dict[str, set]],
    iou_threshold: float = 0.5,
) -> Dict[str, ClassCounts]:
    """Evaluate one image and return per-class-group counts."""
    counts = {group_name: ClassCounts() for group_name in class_groups}

    for group_name, group in class_groups.items():
        group_gt = [
            item for item in ground_truth if item.label in group["visdrone_labels"]
        ]
        group_predictions = [
            item
            for item in detections
            if str(item.get("label", "")) in group["prediction_labels"]
        ]
        group_predictions = sorted(
            group_predictions,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )

        matched_gt: set[int] = set()
        true_positives = 0
        false_positives = 0

        for prediction in group_predictions:
            best_match_index, best_iou = _best_unmatched_gt(
                prediction.get("bbox_xyxy", []),
                group_gt,
                matched_gt,
            )
            if best_match_index is not None and best_iou >= iou_threshold:
                matched_gt.add(best_match_index)
                true_positives += 1
            else:
                false_positives += 1

        false_negatives = len(group_gt) - true_positives
        counts[group_name] = ClassCounts(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            ground_truth_count=len(group_gt),
            prediction_count=len(group_predictions),
        )

    return counts


def aggregate_counts(per_image_counts: Iterable[Dict[str, ClassCounts]]) -> Dict[str, ClassCounts]:
    aggregate: Dict[str, ClassCounts] = {}
    for image_counts in per_image_counts:
        for group_name, counts in image_counts.items():
            if group_name not in aggregate:
                aggregate[group_name] = ClassCounts()
            aggregate[group_name].true_positives += counts.true_positives
            aggregate[group_name].false_positives += counts.false_positives
            aggregate[group_name].false_negatives += counts.false_negatives
            aggregate[group_name].ground_truth_count += counts.ground_truth_count
            aggregate[group_name].prediction_count += counts.prediction_count
    return aggregate


def class_groups_for_names(class_names: Sequence[str]) -> Dict[str, Dict[str, set]]:
    missing = [name for name in class_names if name not in DEFAULT_CLASS_GROUPS]
    if missing:
        raise ValueError(f"Unsupported benchmark class group(s): {', '.join(missing)}")
    return {name: DEFAULT_CLASS_GROUPS[name] for name in class_names}


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - intersection_area

    return _safe_divide(intersection_area, union_area)


def _best_unmatched_gt(
    prediction_bbox: Sequence[float],
    ground_truth: Sequence[GroundTruthObject],
    matched_gt: set[int],
) -> Tuple[Optional[int], float]:
    best_index: Optional[int] = None
    best_iou = 0.0
    for index, gt_object in enumerate(ground_truth):
        if index in matched_gt:
            continue

        candidate_iou = iou_xyxy(prediction_bbox, gt_object.bbox_xyxy)
        if candidate_iou > best_iou:
            best_index = index
            best_iou = candidate_iou

    return best_index, best_iou


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

from pathlib import Path

from perception.visdrone_benchmark import (
    GroundTruthObject,
    class_groups_for_names,
    evaluate_image,
    iou_xyxy,
    parse_visdrone_annotation,
)


def test_parse_visdrone_annotation_filters_people_and_car(tmp_path: Path) -> None:
    annotation_path = tmp_path / "frame.txt"
    annotation_path.write_text(
        "\n".join(
            [
                "10,20,30,40,1,1,0,0",
                "50,60,20,20,1,4,0,0",
                "0,0,10,10,0,4,0,0",
                "100,100,10,10,1,10,0,0",
            ]
        ),
        encoding="utf-8",
    )

    objects = parse_visdrone_annotation(
        annotation_path,
        target_labels={"pedestrian", "people", "car"},
    )

    assert [obj.label for obj in objects] == ["pedestrian", "car"]
    assert objects[0].bbox_xyxy == [10.0, 20.0, 40.0, 60.0]
    assert objects[1].bbox_xyxy == [50.0, 60.0, 70.0, 80.0]


def test_iou_xyxy() -> None:
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xyxy([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0


def test_evaluate_image_counts_matches() -> None:
    ground_truth = [
        GroundTruthObject(
            label="car",
            bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
            truncation=0,
            occlusion=0,
        )
    ]
    detections = [
        {"label": "car", "bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9},
        {"label": "car", "bbox_xyxy": [20.0, 20.0, 30.0, 30.0], "confidence": 0.8},
    ]

    counts = evaluate_image(
        detections=detections,
        ground_truth=ground_truth,
        class_groups=class_groups_for_names(["car"]),
        iou_threshold=0.5,
    )

    assert counts["car"].true_positives == 1
    assert counts["car"].false_positives == 1
    assert counts["car"].false_negatives == 0

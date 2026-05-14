# Role 3 Perception

This directory contains the perception component for the intent-aware UAV pipeline. It runs a pretrained YOLO detector on aerial images, converts detections into a structured scene representation, and serializes the result as a `BeliefState` JSON object for downstream intent generation.

The benchmark code also evaluates detector performance on VisDrone2019-DET validation images for the core aerial classes used in the paper: people and cars.

## What This Module Does

```text
Aerial image
-> YOLO detector
-> semantic mapper
-> BeliefState JSON
-> optional VisDrone benchmark metrics
```

Main files:

- `perception/detector_wrapper.py`: wraps a pretrained YOLO checkpoint and normalizes detections.
- `perception/semantic_mapper.py`: adds center, width, height, area, and coarse image-region fields.
- `perception/belief_state.py`: stores image metadata, detected objects, and optional drone metadata.
- `perception/pipeline.py`: exposes `build_belief_state(image)`.
- `scripts/demo_perception.py`: runs inference and saves JSON/annotated images.
- `scripts/benchmark_visdrone.py`: evaluates detections against VisDrone annotations.

## Environment Setup

Use Python 3.10 or 3.11 if possible. Python 3.12 can work, but PyTorch/NumPy compatibility issues are more common.

```bash
cd /path/to/Intent-Aware-UAV-CS540/role3_perception

python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.10` is not available, use your available Python executable:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` intentionally pins `numpy<2` because some PyTorch builds fail with NumPy 2.x during YOLO inference.

## Download VisDrone2019-DET Validation Data

The benchmark expects the original VisDrone DET validation folder with this structure:

```text
role3_perception/
  datasets/
    VisDrone2019-DET-val/
      images/
        *.jpg
      annotations/
        *.txt
```

Download the **Task 1: Object Detection in Images, valset** from the official VisDrone repository:

```text
https://github.com/VisDrone/VisDrone-Dataset
```

On that page, use:

```text
Download -> Task 1: Object Detection in Images -> VisDrone-DET dataset -> valset
```

The official page provides BaiduYun and Google Drive links for `VisDrone2019-DET-val`. After downloading, unzip it into `role3_perception/datasets/`.

Example:

```bash
mkdir -p datasets
unzip ~/Downloads/VisDrone2019-DET-val.zip -d datasets
```

After unzipping, verify:

```bash
ls datasets/VisDrone2019-DET-val/images | head
ls datasets/VisDrone2019-DET-val/annotations | head
```

## Model Checkpoint

The paper benchmark used `yolo26x.pt`, a large YOLO checkpoint loaded through the Ultralytics API. Large `.pt` files are not committed to the repository because they exceed normal GitHub file limits.

If your installed Ultralytics version supports YOLO26 checkpoints, this should auto-download on first use:

```bash
python scripts/benchmark_visdrone.py datasets/VisDrone2019-DET-val \
  --model yolo26x.pt \
  --classes people car \
  --limit 10 \
  --random-sample \
  --seed 540 \
  --conf 0.10 \
  --imgsz 1280
```

If auto-download fails, manually place the checkpoint somewhere outside Git tracking and pass its path:

```bash
python scripts/benchmark_visdrone.py datasets/VisDrone2019-DET-val \
  --model /path/to/yolo26x.pt \
  --classes people car \
  --limit 10 \
  --random-sample \
  --seed 540 \
  --conf 0.10 \
  --imgsz 1280
```

Using a different checkpoint such as `yolov8x.pt` or `yolov8n.pt` is fine for testing the code path, but the metrics will not reproduce the paper numbers.

## Reproduce The Paper Benchmark

The paper reports a lightweight detector benchmark on a deterministic 10-image random sample from `VisDrone2019-DET-val`.

Run from `role3_perception/`:

```bash
python scripts/benchmark_visdrone.py datasets/VisDrone2019-DET-val \
  --model /path/to/yolo26x.pt \
  --classes people car \
  --limit 10 \
  --random-sample \
  --seed 540 \
  --conf 0.10 \
  --imgsz 1280 \
  --iou 0.50
```

Expected summary from the reported run:

```text
people: P=0.923 R=0.135 F1=0.236 GT=266 Pred=39
car:    P=0.897 R=0.419 F1=0.571 GT=248 Pred=116
```

Outputs are written to:

```text
outputs/benchmarks/visdrone/summary.json
outputs/benchmarks/visdrone/per_image_metrics.csv
```

## Optional Larger Benchmark

For a more stable context benchmark, run a larger random sample:

```bash
python scripts/benchmark_visdrone.py datasets/VisDrone2019-DET-val \
  --model /path/to/yolo26x.pt \
  --classes people car \
  --limit 1000 \
  --random-sample \
  --seed 540 \
  --conf 0.10 \
  --imgsz 1280 \
  --iou 0.50
```

Note: the VisDrone validation split has fewer than 1000 images. If `--limit 1000` is requested, the script evaluates all available validation images and prints a warning.

## What The Benchmark Measures

This is a simple context benchmark, not the official VisDrone challenge metric.

It reports:

- true positives, false positives, and false negatives
- precision
- recall
- F1
- ground-truth object count
- prediction count

Matching rule:

- A predicted box is a true positive if it has IoU >= `0.50` with an unmatched ground-truth box from the same class group.
- Otherwise the prediction is a false positive.
- Unmatched ground-truth boxes are false negatives.

Class mapping:

- VisDrone `pedestrian` and `people` -> benchmark group `people`
- YOLO `person` -> benchmark group `people`
- VisDrone `car` -> benchmark group `car`
- YOLO `car` -> benchmark group `car`

This benchmark is useful for explaining detector behavior on aerial images. In the reported run, precision is high but recall is low, meaning YOLO26x is usually correct when it predicts people/cars, but misses many small distant objects in aerial scenes.

## Run Perception Inference On Images

To generate belief-state JSON for a directory of images:

```bash
python scripts/demo_perception.py datasets/VisDrone2019-DET-val/images \
  --model /path/to/yolo26x.pt \
  --limit 5 \
  --json \
  --save-vis
```

Outputs:

```text
outputs/json/
outputs/annotated/
```

To run on one specific image:

```bash
python scripts/demo_perception.py datasets/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg \
  --model /path/to/yolo26x.pt \
  --json \
  --save-vis
```

## Run Tests

```bash
pytest tests
```

The tests cover:

- detector output normalization with mocks
- semantic mapping geometry
- `BeliefState` helper methods
- perception pipeline orchestration
- VisDrone benchmark helper functions

## Notes For Reproducibility

- Do not commit datasets, generated outputs, or YOLO checkpoint files. They are ignored by `.gitignore`.
- The exact benchmark numbers depend on the YOLO checkpoint, Ultralytics version, confidence threshold, image size, random seed, and dataset split.
- For the paper results, use `yolo26x.pt`, `--conf 0.10`, `--imgsz 1280`, `--iou 0.50`, `--random-sample`, `--seed 540`, and `--limit 10`.

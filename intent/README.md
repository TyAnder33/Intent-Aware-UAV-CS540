# Intent Module

This module implements the intent-generation component of the Intent-Aware UAV Vision-Language Navigation pipeline.

It consumes structured perception outputs (BeliefState JSON) and produces ranked UAV action intents.

## Pipeline Overview

```text
Aerial Image
-> Perception (role3_perception)
-> BeliefState JSON
-> Intent Generation (this module)
-> Ranked UAV intents
```

## Module Structure

```text
intent/
├── rule_based/
│   ├── intent_rules.py
│   └── intent_scoring.py
├── vlm/
│   └── vlm_intent_generator.py
├── scripts/
│   ├── demo_intent_scoring.py
│   └── demo_vlm_intent_scoring.py
├── evaluation/
│   ├── create_vlm_eval_sheet.py
│   ├── compute_vlm_metrics.py
│   └── plot_vlm_results.py
└── README.md
```

## 1. Rule-Based Intent Generation

The rule-based method generates UAV intents using deterministic mappings from detected objects.

**Inputs:**

- Detected object labels (e.g., car, person, building)
- Detection confidence
- Object spatial position and size

**Outputs** — each image produces:

- Intent name
- Target
- Applicability score
- Feasibility score
- Final score
- Explanation

**Run** from project root:

```bash
python intent/scripts/demo_intent_scoring.py
```

Outputs are written under:

```text
intent/outputs/rule_based/
```

## 2. VLM-Based Intent Generation

The VLM-based method uses a Vision-Language Model (Qwen2.5-VL) to generate scene-aware intents.

**Inputs:**

- Original aerial image
- BeliefState JSON

**Outputs** — each image produces:

- 3 ranked intent candidates
- Applicability score (scene relevance)
- Feasibility score (execution practicality)
- Final score (average)

**Notes:**

- Scores are heuristic (model-generated), not calibrated probabilities
- Outputs are more flexible but less strictly grounded than rule-based

**Run:**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python intent/scripts/demo_vlm_intent_scoring.py
```

Outputs are written under:

```text
intent/outputs/vlm/
```

## Environment Setup

Recommended:

```bash
conda create -n intent_perception python=3.10 -y
conda activate intent_perception
```

Install dependencies:

```bash
pip install torch torchvision pillow matplotlib
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils
pip install modelscope
```

## Model Setup

The VLM model is not included in the repository due to size. Download locally using ModelScope:

```bash
mkdir -p intent/models

modelscope download \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir intent/models/qwen2_5_vl_3b
```

## Evaluation

### Rule-Based

Evaluate on larger datasets (30–50 images). Metrics:

- Number of intents per image
- Average score
- Coverage

### VLM (Proof-of-Concept)

Evaluated on a small subset due to GPU constraints. Metrics:

- Parse success rate
- Average top-1 score
- Perception overlap
- Scene relevance (manual)
- Feasibility (manual)

### Evaluation Workflow

Create evaluation sheet:

```bash
python intent/evaluation/create_vlm_eval_sheet.py
```

Fill in manually:

```text
scene_relevance_manual
feasibility_manual
```

Compute metrics:

```bash
python intent/evaluation/compute_vlm_metrics.py
```

Generate plots:

```bash
python intent/evaluation/plot_vlm_results.py
```

## Summary

- Rule-based method provides a fast, deterministic baseline
- VLM method provides flexible, scene-aware intent generation
- Combined, they demonstrate both reliability and semantic reasoning capabilities
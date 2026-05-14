# Intent Module

This module implements the intent-generation component of the Intent-Aware UAV Vision-Language Navigation pipeline.

It consumes structured perception outputs (BeliefState JSON) and produces ranked UAV action intents.

The module explores two complementary approaches:

- deterministic symbolic reasoning (rule-based)
- semantic scene reasoning using a Vision-Language Model (VLM)

The goal is not to train a new foundation model, but to study how UAV intent candidates can emerge from structured aerial perception outputs.

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
│   ├── demo_vlm_intent_scoring.py
│   ├── demo_vlm_single_image.py
│   └── run_vlm_one_by_one.sh
├── outputs/
│   ├── rule_based/
│   └── vlm/
├── evaluation/
│   ├── create_rule_based_eval_sheet.py
│   ├── compute_rule_based_metrics.py
│   ├── plot_rule_based_results.py
│   ├── create_vlm_eval_sheet.py
│   ├── compute_vlm_metrics.py
│   └── plot_vlm_results.py
└── README.md
```

## Experimental Goal

The purpose of this module is to evaluate whether UAV intents can be generated from aerial perception outputs using:

1. symbolic object-to-intent reasoning
2. semantic scene-level reasoning using a Vision-Language Model

The experiments compare:

- grounding quality
- semantic richness
- feasibility
- diversity
- structured parsing reliability

## Environment Setup

This module can be used either:

- independently as a standalone intent-generation component
- or as part of the full repository pipeline described in the root `README.md`

If running the full project pipeline, reuse the previously configured environment.

If running this module independently, a recommended environment is:

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

The VLM model is not included in the repository due to size constraints.

Download locally using ModelScope:

```bash
mkdir -p intent/models

modelscope download \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir intent/models/qwen2_5_vl_3b
```

Expected structure:

```text
intent/models/qwen2_5_vl_3b/
```

## Hardware Notes and GPU Constraints

The VLM experiments were performed on a consumer GPU with approximately 7–8 GB VRAM.

Because Qwen2.5-VL-3B is memory intensive:

- batch inference was unstable
- sequential image processing was used instead
- explicit CUDA cleanup was required between images

To reduce memory fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The repository includes:

```text
intent/scripts/run_vlm_one_by_one.sh
```

for lower-memory sequential execution.

## Input Requirements

This module expects perception outputs generated from:

```text
role3_perception/
```

Required inputs:

```text
role3_perception/outputs/json/
```

Each JSON file represents one structured BeliefState scene.

The VLM pipeline additionally requires the original aerial images.

## 1. Rule-Based Intent Generation

The rule-based method generates UAV intents using deterministic mappings from detected objects.

### Inputs

- detected object labels
- detection confidence
- spatial region information
- object size and location

### Outputs

Each image produces:

- intent name
- target
- applicability score
- feasibility score
- final score
- explanation

### Run

From repository root:

```bash
python intent/scripts/demo_intent_scoring.py
```

Outputs are written under:

```text
intent/outputs/rule_based/
```

### Characteristics

Advantages:

- deterministic
- interpretable
- strongly grounded in detections
- computationally inexpensive

Limitations:

- limited semantic diversity
- repetitive intent structures
- restricted scene understanding
- weak contextual reasoning

## 2. VLM-Based Intent Generation

The VLM-based method uses a Vision-Language Model (Qwen2.5-VL) to generate scene-aware UAV intents.

### Inputs

- original aerial image
- BeliefState JSON

### Outputs

Each image produces:

- 3 ranked intent candidates
- applicability score
- feasibility score
- final score
- textual explanation

### Run

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python intent/scripts/demo_vlm_intent_scoring.py
```

Outputs are written under:

```text
intent/outputs/vlm/
```

### Characteristics

Advantages:

- semantically richer intent generation
- broader scene understanding
- more diverse action proposals
- better contextual reasoning

Limitations:

- weaker grounding to detected entities
- occasional abstraction mismatch
- parse instability
- significantly higher GPU requirements

### Notes

- scores are heuristic model-generated values
- scores are not calibrated probabilities
- generated intents may reference scene semantics rather than exact detected labels

## Quick Reproduction Using Existing Outputs

The repository already includes generated output JSON files under:

```text
intent/outputs/rule_based/
intent/outputs/vlm/
```

This allows users to directly reproduce:

- evaluation metrics
- plots
- score distributions
- summary statistics

without rerunning expensive inference.

This is the recommended path for:

- presentation reproduction
- quick verification
- CPU-only environments
- environments without GPU support

### Reproduce Rule-Based Metrics and Plots

```bash
python intent/evaluation/compute_rule_based_metrics.py
python intent/evaluation/plot_rule_based_results.py
```

### Reproduce VLM Metrics and Plots

```bash
python intent/evaluation/compute_vlm_metrics.py
python intent/evaluation/plot_vlm_results.py
```

## Quick Reproduction Using Pre-Labeled Evaluation CSV Files

The repository also includes manually completed evaluation CSV files used in the reported experiments.

These files already contain the manually filled semantic evaluation fields:

```text
scene_relevance_manual
feasibility_manual
```

allowing users to skip the manual labeling stage and directly regenerate metrics and plots.

Included files:

```text
intent/evaluation/rule_based_eval_sheet.csv
intent/evaluation/vlm_eval_sheet.csv
```

This path is recommended for:

- reproducing the final reported results
- regenerating presentation plots
- verifying evaluation scripts
- avoiding repeated manual annotation

### Regenerate Rule-Based Results from Existing CSV

```bash
python intent/evaluation/compute_rule_based_metrics.py
python intent/evaluation/plot_rule_based_results.py
```

### Regenerate VLM Results from Existing CSV

```bash
python intent/evaluation/compute_vlm_metrics.py
python intent/evaluation/plot_vlm_results.py
```

## Full Experimental Reproduction Workflow

The following steps reproduce the full pipeline from raw perception outputs.

---

# Step 1 — Generate Intent Outputs

## Rule-Based

```bash
python intent/scripts/demo_intent_scoring.py
```

Outputs:

```text
intent/outputs/rule_based/
```

## VLM-Based

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python intent/scripts/demo_vlm_intent_scoring.py
```

Outputs:

```text
intent/outputs/vlm/
```

---

# Step 2 — Create Evaluation CSV

## Rule-Based

```bash
python intent/evaluation/create_rule_based_eval_sheet.py
```

## VLM-Based

```bash
python intent/evaluation/create_vlm_eval_sheet.py
```

Generated CSV files contain:

- image identifiers
- detected entities
- generated intents
- scores
- empty manual evaluation columns

---

# Step 3 — Manual Evaluation

Unlike object detection datasets, there is currently no standardized UAV-intent benchmark containing:

- aerial scene
- valid UAV intents
- feasibility annotations
- semantic relevance labels

Because of this, partial manual annotation is required.

The following columns must be filled manually:

```text
scene_relevance_manual
feasibility_manual
```

Suggested scoring:

```text
0.0 = incorrect / poor
0.5 = partially relevant
1.0 = strongly relevant
```

This evaluation is necessary for assessing:

- semantic correctness
- UAV applicability
- scene grounding quality
- practical feasibility

---

# Step 4 — Compute Metrics

## Rule-Based

```bash
python intent/evaluation/compute_rule_based_metrics.py
```

## VLM-Based

```bash
python intent/evaluation/compute_vlm_metrics.py
```

---

# Step 5 — Generate Plots

## Rule-Based

```bash
python intent/evaluation/plot_rule_based_results.py
```

## VLM-Based

```bash
python intent/evaluation/plot_vlm_results.py
```

Generated plots include:

- score distributions
- intent distributions
- scene relevance summaries
- feasibility summaries
- parse success statistics

## Parse Success Rate

Parse success rate measures whether generated VLM outputs can be successfully converted into valid structured JSON.

This metric evaluates structural reliability rather than semantic quality.

A high parse success rate indicates that downstream components can reliably consume generated outputs.

## Experimental Insights

Key observations from the experiments:

- Rule-based methods provide strong grounding and deterministic behavior but suffer from low diversity.
- VLM-based methods produce richer and more scene-aware intents but occasionally lose strict grounding to detected entities.
- The primary challenge is balancing semantic reasoning with perception grounding.
- Evaluation currently requires manual annotation due to the absence of standardized UAV-intent datasets.
- Sequential inference was significantly more stable than batch processing on limited-VRAM GPUs.

## Summary

- Rule-based intent generation provides a fast and interpretable baseline.
- VLM-based generation provides richer semantic reasoning and contextual scene understanding.
- Together, both approaches demonstrate complementary reasoning capabilities for future intent-aware UAV navigation systems.
- The repository is structured to support both lightweight reproduction and full experimental regeneration workflows.

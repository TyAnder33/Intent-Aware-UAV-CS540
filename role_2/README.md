# Role 2 — Evaluation, Methodology Alignment, and Ablation

This directory contains the **Role 2** deliverables for the Intent-Aware UAV VLN graduate project (COSC 540, Spring 2026). Role 2 owns the evaluation methodology that surrounds the two intent generators (rule-based and VLM) — it does *not* produce intents itself. Every script in this folder is a thin consumer of artefacts produced by `role3_perception/` and `intent/`, and every claim numbered in the IEEE write-up that is not a perception number or a per-image intent number is locked here.

```text
role3_perception/outputs/json/  ──┐
                                  ├──► role_2/  ──► diversity stats, baseline, ablation,
intent/outputs/{rule_based,vlm}/ ─┘                  IPA scaffolding, methodology tests
```

Nothing in `role_2/` overwrites files in `intent/` or `role3_perception/`; this folder is read-only with respect to upstream modules.

---

## 1. Directory Contents

```text
role_2/
├── README.md                        (this file)
├── week03_diversity_check.py        per-image diversity over rule + VLM outputs
├── week06_baseline.py               no-intent-layer baseline (consumes perception JSON)
├── week06_ipa_template.csv          manual IPA (intent-prediction-accuracy) scaffold
├── week07_ablation_diversity.py     paper Tables II/III diversity ablation
├── week09_final_checklist.md        end-of-project submission checklist
└── test_methodology_alignment.py    9-case unittest suite locking figures to code
```

The week-numbered prefix preserves the order in which each artefact was produced during the semester; it has no run-time meaning.

---

## 2. Pipeline Position

```text
                       ┌──────────────────────────────────────────────┐
 Perception ──► Belief │           Intent generation                  │
 (role3_perception)    │  ├─ intent/rule_based/  (deterministic)      │
                       │  └─ intent/vlm/         (Qwen2.5-VL-3B)      │
                       └─────────────┬────────────────────────────────┘
                                     │  intent_candidates[*]
                                     ▼
                       ┌──────────────────────────────────────────────┐
                       │  Role 2 — evaluation surface (this folder)   │
                       │  • diversity (week03)                        │
                       │  • baseline (week06)                         │
                       │  • IPA scaffold (week06_ipa_template.csv)    │
                       │  • ablation (week07)                         │
                       │  • methodology tests (test_methodology…)     │
                       └──────────────────────────────────────────────┘
```

Role 2 deliberately does not edit `intent/rule_based/`, `intent/vlm/`, or any perception code. The only writes performed by this folder land under `role_2/outputs/baseline/` (a per-image baseline JSON, see §4) and the manual IPA CSV.

---

## 3. Prerequisites

All Role 2 scripts run on plain CPU Python ≥ 3.10 with the standard library. No PyTorch, no GPU, no VLM checkpoint required — these scripts read JSON that the upstream modules already produced.

Before running anything here you should have:

1. perception belief states under `role3_perception/outputs/json/` (from `role3_perception/scripts/demo_perception.py`);
2. rule-based candidates under `intent/outputs/rule_based/` (from `intent/scripts/demo_intent_scoring.py`);
3. VLM candidates under `intent/outputs/vlm/` (from `intent/scripts/demo_vlm_intent_scoring.py`, GPU machine).

The repository ships with all three sets of cached outputs, so every Role 2 script can be executed on a fresh checkout without re-running perception or the VLM.

---

## 4. Scripts in Detail

All scripts are designed to be invoked from the **repository root** so that the relative paths in each module resolve correctly.

### 4.1 `week03_diversity_check.py` — diversity validation on the VLM pilot set

```powershell
python role_2\week03_diversity_check.py
```

For every file in `intent/outputs/vlm/` and `intent/outputs/rule_based/`, reports per image:

- `n_candidates`
- `unique intent labels`
- `unique targets`
- intent diversity ratio = `unique_intents / n_candidates`
- target diversity ratio = `unique_targets / n_candidates`

Dataset-level means are printed, and images that fall below the 0.5 diversity threshold are flagged. This is the operational definition of "diversity collapse" used in paper §V.D ("two nearly identical candidates provides no disambiguation value").

Outputs: terminal only — no files written.

### 4.2 `week06_baseline.py` — no-intent-layer baseline

```powershell
python role_2\week06_baseline.py
```

A trivial baseline that mimics what an end-to-end VLN system without the intent layer would emit: pick the highest-confidence detected entity in the belief state and produce a single default action for that entity class via the table:

```python
DEFAULT_ACTION = {
    "car":         "hover_over_vehicle",
    "truck":       "hover_over_vehicle",
    "bus":         "hover_over_vehicle",
    "motorcycle":  "hover_over_vehicle",
    "vehicle":     "hover_over_vehicle",
    "person":      "monitor_person_area",
    "bicycle":     "track_bicycle_movement",
    "umbrella":    "monitor_person_area",
}
```

No multi-hypothesis reasoning is performed; the detection confidence is reported directly as `final_score`. One JSON per image is written to `role_2/outputs/baseline/`. The folder is created automatically on first run.

This baseline supplies the "without intent layer" column in paper §V.G.

### 4.3 `week06_ipa_template.csv` — manual IPA annotation scaffold

A blank CSV with the header

```text
image,method,top1_intent,top1_target,top1_score,ipa_correct_0_or_1,annotator_initials,notes
```

used during weeks 6–7 to record the manual **intent-prediction-accuracy** annotations that feed paper §V.E (`scene_relevance_manual`, `feasibility_manual` ≈ 80.49 % / 85.37 % for the VLM column). The template is intentionally empty; populate it during the manual review pass and keep the populated copy outside the repo unless explicitly required for submission.

### 4.4 `week07_ablation_diversity.py` — diversity ablation (paper Tables II / III)

```powershell
python role_2\week07_ablation_diversity.py
```

Side-by-side comparison of hypothesis diversity for:

- **(a)** the rule-based generator (no diversity prompt; deterministic lookup),
- **(b)** the VLM generator (with the explicit "ensure diversity" constraint in `INTENT_SCHEMA`).

For each generator, prints:

- number of evaluated images,
- unique top-1 intent types (full count and top-10 view),
- top-10 most common distribution,
- mean per-image intent diversity ratio.

The script then asserts that the **top-10-view delta equals `+8`**, matching the paper's Tables II / III headline number (VLM top-10 = 10 unique types; rule-based top-10 = 2 unique types; `10 − 2 = +8`). The assertion is the locking mechanism that prevents the figure from drifting without the test failing.

### 4.5 `test_methodology_alignment.py` — 9-test methodology suite

```powershell
python role_2\test_methodology_alignment.py
```

A `unittest` suite that locks the *current* code in `intent/` and the *cached* JSON outputs to every numeric claim made in the paper. Nine cases:

| # | Test | What it locks |
|---|---|---|
| 1 | `test_rule_based_candidates_use_shipped_formula` | Rule scorer applies `round((s_a + s_f) / 2, 3)` on a synthetic belief state. |
| 2 | `test_vlm_source_has_filter_and_target_source` | VLM source must reference `valid_target_labels` and stamp `target_source = "belief_state"`. |
| 3 | `test_rule_based_outputs_match_shipped_formula` | Every cached rule-based candidate satisfies the shipped formula byte-for-byte. |
| 4 | `test_fig2_cached_rule_based_matches` | Fig. 2 / Fig. 3 sample reproduces `[0.672, 0.672, 0.662]`. |
| 5 | `test_fig4_cached_vlm_matches` | Fig. 4 / Fig. 5 sample reproduces `[0.85, 0.75, 0.65]` and the three paper-listed intent labels. |
| 6 | Fig. 7 distribution | `hover_over_vehicle = 28`, `monitor_person_area = 21`, 49 images, 2 unique types. |
| 7 | Fig. 10 distribution | Top-10 counts `[7, 5, 5, 3, 3, 3, 1, 1, 1, 1]`, 47 images, top-10-view delta `+8`. |
| 8 | `*_metrics.txt` lock | Fig. 8 percentages (100 / 76.60 / 80.49 / 85.37) and means (0.742 / 0.847) present in `intent/evaluation/*_metrics.txt`. |
| 9 | Rule mapping table | Core `vehicle` and `person` mappings used by Fig. 7 are present in `intent/rule_based/intent_rules.py`. |

A note on Eq. (1): the **rule-based code** uses equal-weight 3-decimal-place rounding, and every rule-based figure (Figs. 2, 3, 7, 8) was produced with that formula. The **VLM code** uses the 0.6 / 0.4 weighting described in paper Eq. (1). The test suite locks each code path to the figures it actually produced rather than asserting one formula across both — this is an intentional documentation-vs-code discrepancy flagged in [`week09_final_checklist.md`](week09_final_checklist.md) for R3/R4 (the paper authors) to resolve.

### 4.6 `week09_final_checklist.md`

The end-of-project submission checklist enumerating:

- which code paths match which figures (all green at submission time);
- the three known paper↔code discrepancies that are out of scope for Role 2 to alter (`Eq. (1)` weighting wording, bicycle example rule, Fig. 4 free-form targets predate the current target filter);
- the re-run order needed to refresh every cached artefact before submission;
- presentation / slide deliverable status.

Treat this file as the single source of truth for "is the project ready to submit" — every box ticked there corresponds to a passing test or a regenerable artefact in the rest of the repository.

---

## 5. Expected Output (Smoke Test)

From a fresh checkout with the shipped cached outputs, the four runnable scripts should produce roughly:

```text
$ python role_2\week03_diversity_check.py
[vlm]        47 images   mean intent_div 1.000   mean target_div 0.872   below-0.5 flagged: 0
[rule_based] 49 images   mean intent_div 1.000   mean target_div 0.667   below-0.5 flagged: 0

$ python role_2\week06_baseline.py
49 belief states processed, 49 baseline intents emitted
mean baseline final_score ≈ 0.66
output: role_2/outputs/baseline/*.json

$ python role_2\week07_ablation_diversity.py
=== Diversity Ablation (paper §V.D, Tables II/III) ===
Rule-based: 49 images
  unique top-1 intent types (full)    : 2
  top-10 most common (figure-aligned) : 2
  top-10 distribution: [('hover_over_vehicle', 28), ('monitor_person_area', 21)]
VLM       : 47 images
  unique top-1 intent types (full)    : 14
  top-10 most common (figure-aligned) : 10
  top-10 distribution: [('fly_over_park_pathway', 7), ('fly_over_people', 5), ...]
Delta unique top-1 intent types (top-10) : +8
Paper figure / Tables II vs III: 10 - 2 = +8

$ python role_2\test_methodology_alignment.py
......... (9 tests, OK)
```

If the `+8` assertion fails, or if the methodology suite reports any failure, do **not** edit the assertions — investigate whether `intent/outputs/` was regenerated with a changed formula or a different image set, and reconcile against [`week09_final_checklist.md`](week09_final_checklist.md).

---

## 6. Reproducing Paper Claims

The Role 2 → paper mapping is:

| Paper element | Locked by |
|---|---|
| §III.D Eq. (1) consistency | `test_methodology_alignment.py` (tests 1, 3) |
| §III.F VLM target-filter contract | `test_methodology_alignment.py` (test 2) |
| §V.D diversity collapse claim | `week03_diversity_check.py`, `week07_ablation_diversity.py` |
| §V.E manual IPA (scene relevance / feasibility) | `week06_ipa_template.csv` populated externally; metrics file consumed by test 8 |
| §V.G no-intent-layer baseline column | `week06_baseline.py` |
| Tables II / III diversity ablation (+8 delta) | `week07_ablation_diversity.py` (assert) and test 7 |
| Fig. 2 / Fig. 3 score triple `[0.672, 0.672, 0.662]` | test 4 |
| Fig. 4 / Fig. 5 score triple `[0.85, 0.75, 0.65]` + intents | test 5 |
| Fig. 7 rule-based distribution (28 / 21) | test 6 |
| Fig. 8 evaluation summary percentages | test 8 |
| Fig. 10 VLM top-10 distribution and counts | test 7 |

If any of those values change in the paper, the corresponding test will fail and must be updated *as part of the same commit* that updates the paper. This is the project's anti-drift contract.

---

## 7. Notes for Maintainers

- Role 2 scripts must remain pure consumers of `intent/outputs/` and `role3_perception/outputs/`. Do not add code here that generates intent candidates or modifies belief states — that work belongs in `intent/` and `role3_perception/` respectively.
- The methodology suite is intentionally permissive about the *VLM composite formula* (it does not assert 0.6 / 0.4 vs. equal-weight on the VLM code). Cached VLM JSON in `intent/outputs/vlm/` predates the current weighting and reports model-supplied scores; this is documented in the test docstring and in `week09_final_checklist.md`.
- The `role_2/outputs/baseline/` directory is regenerated end-to-end by `week06_baseline.py` and is safe to delete.
- For the IPA workflow, populate `week06_ipa_template.csv` locally and store the annotated copy outside the repository unless submission requires it.

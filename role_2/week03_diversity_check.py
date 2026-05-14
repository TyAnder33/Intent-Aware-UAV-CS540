"""Week 3 — Diversity validation on the VLM pilot set.

Reads every JSON file in `intent/outputs/vlm/` and reports, per image:
  - n_candidates
  - unique intent labels
  - unique targets
  - intent diversity ratio = unique_intents / n_candidates
  - target diversity ratio = unique_targets / n_candidates

Then prints dataset-level means and flags images that fall below the
0.5 diversity threshold (paper §V.D: "two nearly identical candidates
provides no disambiguation value").

Run from repo root:
    python role_2/week03_diversity_check.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VLM_DIR = PROJECT_ROOT / "intent" / "outputs" / "vlm"
RULE_DIR = PROJECT_ROOT / "intent" / "outputs" / "rule_based"


def diversity_for_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    cands = data.get("intent_candidates", [])
    if not cands:
        return None

    intents = [c.get("intent", "") for c in cands]
    targets = [c.get("target", "") for c in cands]
    n = len(cands)

    return {
        "n": n,
        "unique_intents": len(set(intents)),
        "unique_targets": len(set(targets)),
        "intent_div": len(set(intents)) / n,
        "target_div": len(set(targets)) / n,
    }


def summarize(label, directory):
    files = sorted(directory.glob("*.json"))
    if not files:
        print(f"[{label}] No files in {directory}")
        return

    rows = [diversity_for_file(p) for p in files]
    rows = [r for r in rows if r]

    n = len(rows)
    mean_intent = sum(r["intent_div"] for r in rows) / n
    mean_target = sum(r["target_div"] for r in rows) / n
    below = sum(1 for r in rows if r["intent_div"] < 0.5)

    print(f"\n[{label}] images={n}")
    print(f"  mean intent diversity: {mean_intent:.3f}")
    print(f"  mean target diversity: {mean_target:.3f}")
    print(f"  images with intent diversity < 0.5: {below}/{n}")


if __name__ == "__main__":
    summarize("VLM", VLM_DIR)
    summarize("Rule-based", RULE_DIR)

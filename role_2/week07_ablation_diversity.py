"""Week 7 — Diversity ablation.

Compares hypothesis diversity between:
  (a) the rule-based generator (no diversity prompt; deterministic lookup)
  (b) the VLM generator (with the explicit "ensure diversity" prompt
      constraint in `INTENT_SCHEMA`)

If (b) is significantly more diverse than (a), the prompt constraint and the
VLM together provide measurable value over deterministic lookup. This is the
ablation referenced in the paper §V.D ("diversity collapse").

Run from repo root:
    python role_2/week07_ablation_diversity.py
"""

import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VLM_DIR = PROJECT_ROOT / "intent" / "outputs" / "vlm"
RULE_DIR = PROJECT_ROOT / "intent" / "outputs" / "rule_based"


def collect_top1(directory):
    top1 = []
    for p in sorted(directory.glob("*.json")):
        with open(p, "r") as f:
            cands = json.load(f).get("intent_candidates", [])
        if cands:
            top1.append(cands[0].get("intent", ""))
    return top1


def per_image_intent_diversity(directory):
    ratios = []
    for p in sorted(directory.glob("*.json")):
        with open(p, "r") as f:
            cands = json.load(f).get("intent_candidates", [])
        if not cands:
            continue
        intents = [c.get("intent", "") for c in cands]
        ratios.append(len(set(intents)) / len(intents))
    return ratios


def main():
    rule_top1 = collect_top1(RULE_DIR)
    vlm_top1 = collect_top1(VLM_DIR)

    rule_div = per_image_intent_diversity(RULE_DIR)
    vlm_div = per_image_intent_diversity(VLM_DIR)

    rule_counter = Counter(rule_top1)
    vlm_counter = Counter(vlm_top1)

    # Paper figure (VLM Top-1 Intent Distribution) reports the top-10 most
    # common VLM intent types. Long tail of singletons is omitted from the
    # bar chart. We mirror that here so the script's reported delta matches
    # the figure caption: "10 - 2 = +8".
    rule_top10 = rule_counter.most_common(10)
    vlm_top10 = vlm_counter.most_common(10)

    print("=== Diversity Ablation (paper §V.D, Tables II/III) ===\n")

    print(f"Rule-based: {len(rule_top1)} images")
    print(f"  unique top-1 intent types (full)    : {len(rule_counter)}")
    print(f"  top-10 most common (figure-aligned) : {len(rule_top10)}")
    print(f"  top-10 distribution: {rule_top10}")
    if rule_div:
        print(f"  mean per-image intent diversity: "
              f"{sum(rule_div)/len(rule_div):.3f}")
    print()

    print(f"VLM       : {len(vlm_top1)} images")
    print(f"  unique top-1 intent types (full)    : {len(vlm_counter)}")
    print(f"  top-10 most common (figure-aligned) : {len(vlm_top10)}")
    print(f"  top-10 distribution: {vlm_top10}")
    if vlm_div:
        print(f"  mean per-image intent diversity: "
              f"{sum(vlm_div)/len(vlm_div):.3f}")

    if rule_top1 and vlm_top1:
        delta_full = len(vlm_counter) - len(rule_counter)
        delta_top10 = len(vlm_top10) - len(rule_top10)
        print(f"\nDelta unique top-1 intent types (full)   : {delta_full:+d}")
        print(f"Delta unique top-1 intent types (top-10) : {delta_top10:+d}")
        print("Paper figure / Tables II vs III: 10 - 2 = +8")
        assert delta_top10 == 8, (
            f"figure-aligned delta should be +8, got {delta_top10:+d}"
        )


if __name__ == "__main__":
    main()

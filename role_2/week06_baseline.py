"""Week 6 — No-intent-layer baseline.

A trivial baseline that mimics what an end-to-end VLN system would do
*without* the multi-hypothesis intent layer: pick the highest-confidence
detected entity and emit a single default action for that entity class.

This runs over the existing belief states in
`role3_perception/outputs/json/` and writes one JSON per image to
`role_2/outputs/baseline/`.

Compared head-to-head with `intent/outputs/rule_based/` and
`intent/outputs/vlm/` for the paper §V.G comparison table.

Run from repo root:
    python role_2/week06_baseline.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_DIR = PROJECT_ROOT / "role3_perception" / "outputs" / "json"
OUT_DIR = PROJECT_ROOT / "role_2" / "outputs" / "baseline"

DEFAULT_ACTION = {
    "car": "hover_over_vehicle",
    "truck": "hover_over_vehicle",
    "bus": "hover_over_vehicle",
    "motorcycle": "hover_over_vehicle",
    "vehicle": "hover_over_vehicle",
    "person": "monitor_person_area",
    "bicycle": "track_bicycle_movement",
    "umbrella": "monitor_person_area",
}


def baseline_intent(belief_state):
    objects = belief_state.get("objects", belief_state.get("detected_objects", []))
    if not objects:
        return None

    top = max(objects, key=lambda o: o.get("confidence", 0.0))
    label = top.get("label", "").lower()
    action = DEFAULT_ACTION.get(label, f"observe_{label}")

    # No multi-hypothesis reasoning, no separate s_a / s_f: report the
    # detection confidence directly as the final score (paper §V.G context).
    conf = float(top.get("confidence", 0.0))
    return {
        "intent": action,
        "target": label,
        "applicability_score": round(conf, 3),
        "feasibility_score": round(conf, 3),
        "final_score": round(conf, 2),
        "reason": f"baseline: highest-confidence detection was {label} (conf={conf:.2f})",
    }


def main():
    files = sorted(PERCEPTION_DIR.glob("*.json"))
    if not files:
        print(f"No perception JSON found in {PERCEPTION_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total, emitted, scores = 0, 0, []

    for p in files:
        with open(p, "r") as f:
            bs = json.load(f)

        cand = baseline_intent(bs)
        total += 1
        if cand:
            emitted += 1
            scores.append(cand["final_score"])

        out_path = OUT_DIR / p.name
        with open(out_path, "w") as f:
            json.dump({
                "source_image": bs.get("source_image", bs.get("image_path", "")),
                "intent_candidates": [cand] if cand else [],
                "method": "baseline_no_intent_layer",
            }, f, indent=2)

    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"Baseline run complete: {emitted}/{total} images produced an intent.")
    print(f"  mean top-1 score: {mean:.3f}")
    print(f"  outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()

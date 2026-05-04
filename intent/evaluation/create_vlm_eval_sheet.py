import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VLM_OUTPUT_DIR = PROJECT_ROOT / "intent" / "outputs" / "vlm"
PERCEPTION_JSON_DIR = PROJECT_ROOT / "role3_perception" / "outputs" / "json"
EVAL_CSV = PROJECT_ROOT / "intent" / "evaluation" / "vlm_eval_sheet.csv"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_detected_labels(perception_json):
    data = load_json(perception_json)
    objects = data.get("objects", data.get("detected_objects", []))
    labels = []

    for obj in objects:
        label = obj.get("label")
        if label:
            labels.append(label.lower())

    return sorted(set(labels))


def main():
    vlm_files = sorted(VLM_OUTPUT_DIR.glob("*.json"))

    rows = []

    for vlm_file in vlm_files:
        data = load_json(vlm_file)
        intents = data.get("intent_candidates", [])

        perception_file = PERCEPTION_JSON_DIR / vlm_file.name
        detected_labels = get_detected_labels(perception_file) if perception_file.exists() else []

        top1 = intents[0] if len(intents) > 0 else {}
        top2 = intents[1] if len(intents) > 1 else {}
        top3 = intents[2] if len(intents) > 2 else {}

        target_text = " ".join([
            str(top1.get("target", "")),
            str(top2.get("target", "")),
            str(top3.get("target", ""))
        ]).lower()

        perception_overlap = 0
        for label in detected_labels:
            if label in target_text:
                perception_overlap = 1
                break

        rows.append({
            "image": vlm_file.stem,
            "parse_success": 1,
            "num_intents": len(intents),
            "detected_labels": ";".join(detected_labels),

            "top1_intent": top1.get("intent", ""),
            "top1_target": top1.get("target", ""),
            "top1_score": top1.get("final_score", ""),

            "top2_intent": top2.get("intent", ""),
            "top2_target": top2.get("target", ""),
            "top2_score": top2.get("final_score", ""),

            "top3_intent": top3.get("intent", ""),
            "top3_target": top3.get("target", ""),
            "top3_score": top3.get("final_score", ""),

            "perception_overlap": perception_overlap,

            "scene_relevance_manual": "",
            "feasibility_manual": "",
            "notes": ""
        })

    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No VLM JSON outputs found.")
        return

    with open(EVAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved VLM evaluation sheet to: {EVAL_CSV}")


if __name__ == "__main__":
    main()
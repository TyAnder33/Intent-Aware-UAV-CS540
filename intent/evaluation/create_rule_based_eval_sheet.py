import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTENT_OUTPUT_DIR = PROJECT_ROOT / "intent" / "outputs" / "rule_based"
EVAL_CSV = PROJECT_ROOT / "intent" / "evaluation" / "rule_based_eval_sheet.csv"


def main():
    json_files = sorted(INTENT_OUTPUT_DIR.glob("*.json"))

    if not json_files:
        print(f"No intent JSON files found in: {INTENT_OUTPUT_DIR}")
        return

    rows = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        intents = data.get("intent_candidates", [])
        detected_entities = data.get("detected_entities", [])

        top1 = intents[0] if len(intents) > 0 else {}
        top2 = intents[1] if len(intents) > 1 else {}
        top3 = intents[2] if len(intents) > 2 else {}

        rows.append({
            "image": json_file.stem,
            "detected_entities": ";".join(detected_entities),
            "num_intents": len(intents),

            "top1_intent": top1.get("intent", ""),
            "top1_target": top1.get("target", ""),
            "top1_score": top1.get("final_score", ""),

            "top2_intent": top2.get("intent", ""),
            "top2_target": top2.get("target", ""),
            "top2_score": top2.get("final_score", ""),

            "top3_intent": top3.get("intent", ""),
            "top3_target": top3.get("target", ""),
            "top3_score": top3.get("final_score", ""),

            "has_detected_entities": 1 if detected_entities else 0,
            "has_generated_intents": 1 if intents else 0,

            "top1_reasonable": "",
            "top3_has_reasonable": "",
            "notes": ""
        })

    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(EVAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation sheet to: {EVAL_CSV}")
    print("Now manually fill: top1_reasonable, top3_has_reasonable, notes")


if __name__ == "__main__":
    main()
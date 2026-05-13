import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from intent.rule_based.intent_scoring import process_belief_json


PERCEPTION_OUTPUT_DIR = PROJECT_ROOT / "role3_perception" / "outputs" / "json"
INTENT_OUTPUT_DIR = PROJECT_ROOT / "intent" / "outputs" / "rule_based"


def main():
    json_files = sorted(PERCEPTION_OUTPUT_DIR.glob("*.json"))

    if not json_files:
        print(f"No perception JSON files found in: {PERCEPTION_OUTPUT_DIR}")
        print("Run role3_perception/scripts/demo_perception.py first.")
        return

    INTENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for json_file in json_files:
        output_file = INTENT_OUTPUT_DIR / json_file.name
        result = process_belief_json(json_file, output_file)

        print("\nImage:", json_file.name)
        print("Detected entities:", result["detected_entities"])

        top_intents = result["intent_candidates"][:3]

        if not top_intents:
            print("No intent candidates generated.")
            continue

        for i, intent in enumerate(top_intents, start=1):
            print(
                f"{i}. {intent['intent']} | "
                f"target={intent['target']} | "
                f"score={intent['final_score']}"
            )

    print(f"\nSaved rule-based intent outputs to: {INTENT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
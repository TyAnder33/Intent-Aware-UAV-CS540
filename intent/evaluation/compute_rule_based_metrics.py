import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "intent" / "evaluation" / "rule_based_eval_sheet.csv"
OUT_PATH = PROJECT_ROOT / "intent" / "evaluation" / "rule_based_metrics.txt"


def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


def main():
    with open(CSV_PATH, "r") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    top1_reasonable = sum(safe_float(r["top1_reasonable"]) for r in rows)
    top3_reasonable = sum(safe_float(r["top3_has_reasonable"]) for r in rows)

    avg_num_intents = sum(safe_float(r["num_intents"]) for r in rows) / total
    avg_top1_score = sum(safe_float(r["top1_score"]) for r in rows) / total

    top1_intents = [r["top1_intent"] for r in rows if r["top1_intent"]]
    top1_targets = [r["top1_target"] for r in rows if r["top1_target"]]

    intent_counter = Counter(top1_intents)
    target_counter = Counter(top1_targets)

    unique_top1_intents = len(intent_counter)
    unique_top1_targets = len(target_counter)

    most_common_intent, most_common_intent_count = intent_counter.most_common(1)[0]
    most_common_target, most_common_target_count = target_counter.most_common(1)[0]

    lines = []
    lines.append("Rule-Based Intent Evaluation Results")
    lines.append("=" * 45)
    lines.append(f"Total images: {total}")
    lines.append(f"Top-1 reasonable: {top1_reasonable:.1f} / {total} ({100 * top1_reasonable / total:.2f}%)")
    lines.append(f"Top-3 has reasonable intent: {top3_reasonable:.1f} / {total} ({100 * top3_reasonable / total:.2f}%)")
    lines.append(f"Average intents per image: {avg_num_intents:.2f}")
    lines.append(f"Average top-1 score: {avg_top1_score:.3f}")
    lines.append("")
    lines.append("Diversity / Concentration")
    lines.append("-" * 45)
    lines.append(f"Unique top-1 intent types: {unique_top1_intents}")
    lines.append(f"Unique top-1 target types: {unique_top1_targets}")
    lines.append(f"Most common top-1 intent: {most_common_intent} ({most_common_intent_count}/{total})")
    lines.append(f"Most common top-1 target: {most_common_target} ({most_common_target_count}/{total})")
    lines.append("")
    lines.append("Top-1 intent distribution:")
    for intent, count in intent_counter.most_common():
        lines.append(f"- {intent}: {count}")

    output = "\n".join(lines)
    print(output)

    with open(OUT_PATH, "w") as f:
        f.write(output)

    print(f"\nSaved metrics to: {OUT_PATH}")


if __name__ == "__main__":
    main()
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "intent" / "evaluation" / "rule_based_eval_sheet.csv"
OUT_DIR = PROJECT_ROOT / "intent" / "evaluation"


def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


with open(CSV_PATH, "r") as f:
    rows = list(csv.DictReader(f))

total = len(rows)

top1_reasonable = sum(safe_float(r["top1_reasonable"]) for r in rows)
top3_reasonable = sum(safe_float(r["top3_has_reasonable"]) for r in rows)

metrics = {
    "Top-1 Reasonable": 100 * top1_reasonable / total,
    "Top-3 Reasonable": 100 * top3_reasonable / total,
    "Generated Intents": 100 * sum(safe_float(r["has_generated_intents"]) for r in rows) / total,
}

plt.figure()
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 100)
plt.ylabel("Percentage (%)")
plt.title("Rule-Based Intent Evaluation Summary")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT_DIR / "rule_based_summary.png")

top1_scores = [safe_float(r["top1_score"]) for r in rows]

plt.figure()
plt.plot(top1_scores, marker="o")
plt.ylim(0, 1)
plt.xlabel("Image Index")
plt.ylabel("Top-1 Score")
plt.title("Rule-Based Top-1 Score per Image")
plt.tight_layout()
plt.savefig(OUT_DIR / "rule_based_scores.png")

intent_counts = Counter(r["top1_intent"] for r in rows if r["top1_intent"])
most_common = intent_counts.most_common(8)

labels = [x[0] for x in most_common]
values = [x[1] for x in most_common]

plt.figure()
plt.bar(labels, values)
plt.ylabel("Count")
plt.title("Top-1 Intent Distribution")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "rule_based_intent_distribution.png")

print("Saved plots:")
print(OUT_DIR / "rule_based_summary.png")
print(OUT_DIR / "rule_based_scores.png")
print(OUT_DIR / "rule_based_intent_distribution.png")
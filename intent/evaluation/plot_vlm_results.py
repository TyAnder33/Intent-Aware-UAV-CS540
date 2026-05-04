import csv
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "intent" / "evaluation" / "vlm_eval_sheet.csv"
OUTPUT_DIR = PROJECT_ROOT / "intent" / "evaluation"

rows = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total = len(rows)

def safe_int(x):
    try: return int(x)
    except: return 0

def safe_float(x):
    try: return float(x)
    except: return 0.0

# Metrics
parse_success = sum(safe_int(r["parse_success"]) for r in rows)
overlap = sum(safe_int(r["perception_overlap"]) for r in rows)

scene = sum(safe_int(r["scene_relevance_manual"]) for r in rows if r["scene_relevance_manual"])
feas = sum(safe_int(r["feasibility_manual"]) for r in rows if r["feasibility_manual"])

scene_total = len([r for r in rows if r["scene_relevance_manual"]])
feas_total = len([r for r in rows if r["feasibility_manual"]])

# Convert to %
metrics = {
    "Parse Success": parse_success / total * 100,
    "Perception Overlap": overlap / total * 100,
    "Scene Relevance": scene / scene_total * 100 if scene_total else 0,
    "Feasibility": feas / feas_total * 100 if feas_total else 0
}

# Plot 1: Bar chart
plt.figure()
plt.bar(metrics.keys(), metrics.values())
plt.title("VLM Evaluation Summary (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vlm_bar_chart.png")

# Plot 2: Score distribution
scores = [safe_float(r["top1_score"]) for r in rows if r["top1_score"]]

plt.figure()
plt.plot(scores, marker='o')
plt.title("Top-1 Score per Image")
plt.xlabel("Image Index")
plt.ylabel("Score")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vlm_scores.png")

print("Plots saved in:", OUTPUT_DIR)
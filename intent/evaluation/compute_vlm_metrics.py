import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_CSV = PROJECT_ROOT / "intent" / "evaluation" / "vlm_eval_sheet.csv"
METRICS_TXT = PROJECT_ROOT / "intent" / "evaluation" / "vlm_metrics.txt"


def safe_int(x):
    try:
        return int(x)
    except:
        return 0


def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


def pct(x, total):
    return 100.0 * x / total if total else 0.0


def main():
    if not EVAL_CSV.exists():
        print(f"CSV not found: {EVAL_CSV}")
        return

    with open(EVAL_CSV, "r") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    parse_success = sum(safe_int(r["parse_success"]) for r in rows)
    perception_overlap = sum(safe_int(r["perception_overlap"]) for r in rows)

    avg_intents = sum(safe_int(r["num_intents"]) for r in rows) / total if total else 0

    top1_scores = [safe_float(r["top1_score"]) for r in rows if r["top1_score"]]
    avg_top1_score = sum(top1_scores) / len(top1_scores) if top1_scores else 0

    labeled_scene = [r for r in rows if r["scene_relevance_manual"].strip() != ""]
    labeled_feasibility = [r for r in rows if r["feasibility_manual"].strip() != ""]

    scene_relevant = sum(safe_int(r["scene_relevance_manual"]) for r in labeled_scene)
    feasible = sum(safe_int(r["feasibility_manual"]) for r in labeled_feasibility)

    lines = []
    lines.append("VLM Intent Evaluation Results")
    lines.append("=" * 40)
    lines.append(f"Total images: {total}")
    lines.append(f"Parse success rate: {parse_success} / {total} ({pct(parse_success, total):.2f}%)")
    lines.append(f"Average intents per image: {avg_intents:.2f}")
    lines.append(f"Average top-1 score: {avg_top1_score:.3f}")
    lines.append(f"Perception overlap rate: {perception_overlap} / {total} ({pct(perception_overlap, total):.2f}%)")
    lines.append("")
    lines.append("Manual Evaluation")
    lines.append("-" * 40)
    lines.append(f"Scene relevance: {scene_relevant} / {len(labeled_scene)} ({pct(scene_relevant, len(labeled_scene)):.2f}%)")
    lines.append(f"Feasibility: {feasible} / {len(labeled_feasibility)} ({pct(feasible, len(labeled_feasibility)):.2f}%)")

    output = "\n".join(lines)
    print(output)

    with open(METRICS_TXT, "w") as f:
        f.write(output)

    print(f"\nSaved metrics to: {METRICS_TXT}")


if __name__ == "__main__":
    main()
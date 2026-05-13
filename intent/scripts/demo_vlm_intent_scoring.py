import json
import sys
from pathlib import Path
import gc
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from intent.vlm.vlm_intent_generator import load_model, generate_vlm_intents


PERCEPTION_JSON_DIR = PROJECT_ROOT / "role3_perception" / "outputs" / "json"
IMAGE_DIR = PROJECT_ROOT / "role3_perception" / "datasets" / "VisDrone2019-DET-val" / "images"
OUTPUT_DIR = PROJECT_ROOT / "intent" / "outputs" / "vlm"


def find_image_for_json(json_file):
    possible_extensions = [".jpg", ".jpeg", ".png"]

    for ext in possible_extensions:
        image_path = IMAGE_DIR / f"{json_file.stem}{ext}"
        if image_path.exists():
            return image_path

    return None


def main():
    json_files = sorted(PERCEPTION_JSON_DIR.glob("*.json"))

    if not json_files:
        print(f"No perception JSON files found in: {PERCEPTION_JSON_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading VLM model...")
    model, processor = load_model()
    print("Model loaded.")

    # Start small first because VLM inference is slower than rule-based.
    json_files = json_files[:50]

    for json_file in json_files:
        image_path = find_image_for_json(json_file)

        if image_path is None:
            print(f"Skipping {json_file.name}: matching image not found.")
            continue

        print(f"\nProcessing: {image_path.name}")

        try:
            result = generate_vlm_intents(
                image_path=image_path,
                belief_json_path=json_file,
                model=model,
                processor=processor
            )

            output_file = OUTPUT_DIR / json_file.name

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            print(f"Saved: {output_file}")

            for i, intent in enumerate(result.get("intent_candidates", [])[:3], start=1):
                print(
                    f"{i}. {intent.get('intent')} | "
                    f"target={intent.get('target')} | "
                    f"score={intent.get('final_score')}"
                )

        except Exception as e:
            print(f"Failed on {json_file.name}: {e}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    print(f"\nVLM outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
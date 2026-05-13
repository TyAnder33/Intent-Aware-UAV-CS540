# import json
# import sys
# import gc
# from pathlib import Path

# import torch

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# sys.path.append(str(PROJECT_ROOT))

# from intent.vlm.vlm_intent_generator import load_model, generate_vlm_intents


# PERCEPTION_JSON_DIR = PROJECT_ROOT / "role3_perception" / "outputs" / "json"
# IMAGE_DIR = PROJECT_ROOT / "role3_perception" / "datasets" / "VisDrone2019-DET-val" / "images"
# OUTPUT_DIR = PROJECT_ROOT / "intent" / "outputs" / "vlm"


# def find_image_for_json(json_file):
#     for ext in [".jpg", ".jpeg", ".png"]:
#         image_path = IMAGE_DIR / f"{json_file.stem}{ext}"
#         if image_path.exists():
#             return image_path
#     return None


# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python intent/scripts/demo_vlm_single_image.py <json_filename>")
#         sys.exit(1)

#     json_name = sys.argv[1]
#     json_file = PERCEPTION_JSON_DIR / json_name

#     if not json_file.exists():
#         print(f"JSON not found: {json_file}")
#         sys.exit(1)

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     output_file = OUTPUT_DIR / json_file.name

#     if output_file.exists():
#         print(f"Skipping {json_file.name}: output already exists.")
#         return

#     image_path = find_image_for_json(json_file)

#     if image_path is None:
#         print(f"Image not found for: {json_file.name}")
#         sys.exit(1)

#     print(f"Loading model for: {json_file.name}")
#     model, processor = load_model()

#     try:
#         result = generate_vlm_intents(
#             image_path=image_path,
#             belief_json_path=json_file,
#             model=model,
#             processor=processor
#         )

#         with open(output_file, "w") as f:
#             json.dump(result, f, indent=2)

#         print(f"Saved: {output_file}")

#         for i, intent in enumerate(result.get("intent_candidates", [])[:3], start=1):
#             print(
#                 f"{i}. {intent.get('intent')} | "
#                 f"target={intent.get('target')} | "
#                 f"score={intent.get('final_score')}"
#             )

#     finally:
#         del model
#         del processor
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.ipc_collect()


# if __name__ == "__main__":
#     main()


import json
import sys
from pathlib import Path
import gc
import torch
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from intent.vlm.vlm_intent_generator import load_model, generate_vlm_intents


PERCEPTION_JSON_DIR = PROJECT_ROOT / "role3_perception" / "outputs" / "json"
IMAGE_DIR = PROJECT_ROOT / "role3_perception" / "datasets" / "VisDrone2019-DET-val" / "images"
OUTPUT_DIR = PROJECT_ROOT / "intent" / "outputs" / "vlm"
FAIL_LOG = PROJECT_ROOT / "intent" / "outputs" / "vlm_failed_images.txt"


def find_image_for_json(json_file):
    for ext in [".jpg", ".jpeg", ".png"]:
        image_path = IMAGE_DIR / f"{json_file.stem}{ext}"
        if image_path.exists():
            return image_path
    return None


def main():
    json_files = sorted(PERCEPTION_JSON_DIR.glob("*.json"))[:50]

    if not json_files:
        print(f"No perception JSON files found in: {PERCEPTION_JSON_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    print("Loading VLM model...")
    model, processor = load_model()
    print("Model loaded.")

    with open(FAIL_LOG, "w") as fail_log:
        for idx, json_file in enumerate(json_files, start=1):
            image_path = find_image_for_json(json_file)
            output_file = OUTPUT_DIR / json_file.name

            print(f"\n[{idx}/50] Processing: {json_file.name}")

            if output_file.exists():
                print(f"Skipping existing output: {output_file}")
                success_count += 1
                continue

            if image_path is None:
                msg = f"{json_file.name}: matching image not found"
                print(f"Skipping {msg}")
                fail_log.write(msg + "\n")
                fail_count += 1
                continue

            try:
                result = generate_vlm_intents(
                    image_path=image_path,
                    belief_json_path=json_file,
                    model=model,
                    processor=processor
                )

                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)

                print(f"Saved: {output_file}")

                for i, intent in enumerate(result.get("intent_candidates", [])[:3], start=1):
                    print(
                        f"{i}. {intent.get('intent')} | "
                        f"target={intent.get('target')} | "
                        f"score={intent.get('final_score')}"
                    )

                success_count += 1

            except Exception as e:
                print(f"Failed on {json_file.name}: {e}")
                fail_log.write(f"{json_file.name}: {type(e).__name__}: {e}\n")
                fail_count += 1

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    print("\nVLM run complete.")
    print(f"Successful outputs: {success_count}")
    print(f"Failed images: {fail_count}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Failure log saved to: {FAIL_LOG}")


if __name__ == "__main__":
    main()
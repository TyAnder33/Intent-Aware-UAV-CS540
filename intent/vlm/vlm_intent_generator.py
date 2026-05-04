import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info



PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = str(PROJECT_ROOT / "intent" / "models" / "qwen2_5_vl_3b")


INTENT_SCHEMA = """
Return ONLY valid JSON.
Do not use markdown.
Do not use explanations outside JSON.
Use double quotes for all keys and string values.

Generate exactly 3 UAV intent candidates.

IMPORTANT:
- You must only use targets from the detected object labels in the belief state.
- Do not invent objects that are not listed in the belief state.
- If the image seems to contain something but it is not in the belief state, do not use it as a target.
- The target field must exactly match one of the detected labels.
- The intent should describe a UAV action that is feasible for that detected target.

Return this exact format:

{
  "intent_candidates": [
    {
      "intent": "short_uav_action_name",
      "target": "one_detected_label_from_belief_state",
      "target_source": "belief_state",
      "applicability_score": 0.0,
      "feasibility_score": 0.0,
      "final_score": 0.0,
      "reason": "short reason based on detected label, confidence, region, and image context"
    }
  ]
}
"""


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    return model, processor


def extract_json(text):
    """
    Extract JSON object from model output.
    If parsing fails, save the raw output for debugging.
    """

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in VLM output:\n{text}")

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception as e:
        debug_path = Path("intent/outputs/vlm_last_raw_output.txt")
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        with open(debug_path, "w") as f:
            f.write(text)

        raise ValueError(
            f"VLM returned invalid JSON. Raw output saved to: {debug_path}\n"
            f"Original parse error: {e}"
        )


def summarize_belief_state(belief_state):
    objects = belief_state.get("objects", belief_state.get("detected_objects", []))

    summary = []
    valid_labels = []

    for obj in objects:
        label = obj.get("label")

        if label is None:
            continue

        valid_labels.append(label)

        summary.append({
            "label": label,
            "confidence": obj.get("confidence"),
            "bbox_xyxy": obj.get("bbox_xyxy"),
            "area": obj.get("area"),
            "region_horizontal": obj.get("region_horizontal"),
            "region_vertical": obj.get("region_vertical")
        })

    return {
        "image_width": belief_state.get("image_width", belief_state.get("width")),
        "image_height": belief_state.get("image_height", belief_state.get("height")),
        "valid_target_labels": sorted(list(set(valid_labels))),
        "detected_objects": summary
    }

def filter_and_normalize_vlm_result(result, belief_summary):
    valid_labels = set(belief_summary.get("valid_target_labels", []))

    filtered = []

    for item in result.get("intent_candidates", []):
        target = item.get("target", "")

        if target not in valid_labels:
            continue

        applicability = float(item.get("applicability_score", 0.0))
        feasibility = float(item.get("feasibility_score", 0.0))

        applicability = max(0.0, min(1.0, applicability))
        feasibility = max(0.0, min(1.0, feasibility))

        item["applicability_score"] = round(applicability, 3)
        item["feasibility_score"] = round(feasibility, 3)
        item["final_score"] = round((applicability + feasibility) / 2, 3)
        item["target_source"] = "belief_state"

        filtered.append(item)

    filtered.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    return {
        "intent_candidates": filtered
    }

def generate_vlm_intents(image_path, belief_json_path, model, processor):
    with open(belief_json_path, "r") as f:
        belief_state = json.load(f)

    belief_summary = summarize_belief_state(belief_state)

    prompt = f"""
You are an intent-generation module for a UAV/drone.

Given:
1. The aerial image.
2. The perception belief state produced by an object detector.

Your task:
Generate possible UAV intents from the current scene and score each intent.

Definitions:
- applicability_score: how relevant the intent is to the visible scene.
- feasibility_score: how safe and executable the intent appears from the image and belief state.
- final_score: average of applicability_score and feasibility_score.

Belief state:
{json.dumps(belief_summary, indent=2)}

Valid target labels:
{belief_summary["valid_target_labels"]}

You must choose targets only from the valid target labels above.

{INTENT_SCHEMA}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                    "min_pixels": 128 * 28 * 28,
                    "max_pixels": 384 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    raw_result = extract_json(output_text)

    result = filter_and_normalize_vlm_result(raw_result, belief_summary)

    result["source_image"] = str(image_path)
    result["belief_state"] = str(belief_json_path)
    result["method"] = "vlm_qwen2_5_vl_3b_grounded"
    result["valid_target_labels"] = belief_summary["valid_target_labels"]

    return result
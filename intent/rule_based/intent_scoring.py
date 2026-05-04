import json
from pathlib import Path
from intent.rule_based.intent_rules import ENTITY_TO_INTENTS, COMBINATION_RULES


INTENT_PRIORITY = {
    "hover_over_street": 0.85,
    "follow_street_forward": 0.80,
    "align_with_street_direction": 0.75,

    "orbit_building": 0.75,
    "inspect_building_facade": 0.80,
    "approach_building": 0.65,
    "hover_near_building": 0.70,

    "hover_over_vehicle": 0.75,
    "follow_vehicle": 0.65,
    "inspect_vehicle_area": 0.75,
    "track_vehicle_along_street": 0.85,

    "hover_near_person": 0.55,
    "monitor_person_area": 0.70,

    "hover_near_tree": 0.55,
    "inspect_tree_area": 0.60,

    "hover_over_open_area": 0.90,
    "cross_open_area": 0.80,
    "descend_for_closer_inspection": 0.70,

    "gain_altitude_for_overview": 0.85,
    "observe_building_from_street_side": 0.75,
    "inspect_building_surroundings": 0.75,
}


def normalize_label(label):
    label = label.lower().strip()

    if label in ["car", "truck", "bus", "motorcycle"]:
        return "vehicle"

    if label in ["road"]:
        return "street"

    return label


def region_score(obj):
    horizontal = obj.get("region_horizontal", "")
    vertical = obj.get("region_vertical", "")

    score = 0.5

    if horizontal == "center":
        score += 0.25

    if vertical == "middle":
        score += 0.25
    elif vertical in ["top", "bottom"]:
        score += 0.10

    return min(score, 1.0)


def visibility_score(obj, image_width, image_height):
    image_area = max(image_width * image_height, 1)
    object_area = obj.get("area", 0)
    ratio = object_area / image_area

    # aerial objects are often small, so do not punish small objects too much
    if ratio >= 0.20:
        return 1.0
    if ratio >= 0.10:
        return 0.85
    if ratio >= 0.03:
        return 0.70
    if ratio >= 0.01:
        return 0.55
    return 0.40


def score_intent(intent, target_objects, image_width, image_height, source):
    if not target_objects:
        return 0.0, 0.0, "No visible target object found"

    avg_conf = sum(o.get("confidence", 0.0) for o in target_objects) / len(target_objects)
    avg_visibility = sum(visibility_score(o, image_width, image_height) for o in target_objects) / len(target_objects)
    avg_region = sum(region_score(o) for o in target_objects) / len(target_objects)
    priority = INTENT_PRIORITY.get(intent, 0.60)

    applicability_score = (
        0.50 * avg_conf +
        0.30 * avg_visibility +
        0.20 * avg_region
    )

    feasibility_score = (
        0.45 * avg_conf +
        0.20 * avg_visibility +
        0.15 * avg_region +
        0.20 * priority
    )

    final_score = (
        0.50 * applicability_score +
        0.50 * feasibility_score
    )

    reason = (
        f"Generated from {source}; target detected with confidence "
        f"{avg_conf:.2f}, visibility {avg_visibility:.2f}, region score {avg_region:.2f}."
    )

    return round(applicability_score, 3), round(feasibility_score, 3), reason


def generate_ranked_intents_from_belief_state(belief_state):
    image_width = belief_state.get("image_width", belief_state.get("width", 1))
    image_height = belief_state.get("image_height", belief_state.get("height", 1))

    objects = belief_state.get("objects", belief_state.get("detected_objects", []))

    normalized_objects = []
    for obj in objects:
        label = normalize_label(obj.get("label", ""))
        obj = dict(obj)
        obj["normalized_label"] = label
        normalized_objects.append(obj)

    entities = sorted(set(o["normalized_label"] for o in normalized_objects))
    candidates = []
    seen = set()

    for entity in entities:
        if entity in ENTITY_TO_INTENTS:
            target_objects = [o for o in normalized_objects if o["normalized_label"] == entity]

            for intent in ENTITY_TO_INTENTS[entity]:
                key = (intent, entity)
                if key in seen:
                    continue

                applicability, feasibility, reason = score_intent(
                    intent,
                    target_objects,
                    image_width,
                    image_height,
                    "entity_rule"
                )

                candidates.append({
                    "intent": intent,
                    "target": entity,
                    "source": "entity_rule",
                    "applicability_score": applicability,
                    "feasibility_score": feasibility,
                    "final_score": round((applicability + feasibility) / 2, 3),
                    "reason": reason
                })

                seen.add(key)

    entity_set = set(entities)

    for combo, intents in COMBINATION_RULES.items():
        if combo.issubset(entity_set):
            combo_name = "+".join(sorted(combo))
            target_objects = [
                o for o in normalized_objects
                if o["normalized_label"] in combo
            ]

            for intent in intents:
                key = (intent, combo_name)
                if key in seen:
                    continue

                applicability, feasibility, reason = score_intent(
                    intent,
                    target_objects,
                    image_width,
                    image_height,
                    "combination_rule"
                )

                candidates.append({
                    "intent": intent,
                    "target": combo_name,
                    "source": "combination_rule",
                    "applicability_score": applicability,
                    "feasibility_score": feasibility,
                    "final_score": round((applicability + feasibility) / 2, 3),
                    "reason": reason
                })

                seen.add(key)

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    return {
        "source_image": belief_state.get("source_image", belief_state.get("image_path", "")),
        "detected_entities": entities,
        "intent_candidates": candidates
    }


def process_belief_json(input_json_path, output_json_path):
    with open(input_json_path, "r") as f:
        belief_state = json.load(f)

    result = generate_ranked_intents_from_belief_state(belief_state)

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


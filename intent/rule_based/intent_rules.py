ENTITY_TO_INTENTS = {
    "building": [
        "orbit_building",
        "inspect_building_facade",
        "approach_building",
        "hover_near_building",
    ],
    "street": [
        "hover_over_street",
        "follow_street_forward",
        "align_with_street_direction",
    ],
    "road": [
        "hover_over_street",
        "follow_street_forward",
        "align_with_street_direction",
    ],
    "car": [
        "hover_over_vehicle",
        "follow_vehicle",
        "inspect_vehicle_area",
    ],
    "vehicle": [
        "hover_over_vehicle",
        "follow_vehicle",
        "inspect_vehicle_area",
    ],
    "truck": [
        "hover_over_vehicle",
        "follow_vehicle",
        "inspect_vehicle_area",
    ],
    "person": [
        "hover_near_person",
        "monitor_person_area",
    ],
    "tree": [
        "hover_near_tree",
        "inspect_tree_area",
    ],
    "open_area": [
        "hover_over_open_area",
        "cross_open_area",
        "descend_for_closer_inspection",
    ],
}

COMBINATION_RULES = {
    frozenset(["building", "street"]): [
        "gain_altitude_for_overview",
        "observe_building_from_street_side",
    ],
    frozenset(["road", "car"]): [
        "track_vehicle_along_street",
    ],
    frozenset(["street", "vehicle"]): [
        "track_vehicle_along_street",
    ],
    frozenset(["building", "car"]): [
        "inspect_building_surroundings",
    ],
}

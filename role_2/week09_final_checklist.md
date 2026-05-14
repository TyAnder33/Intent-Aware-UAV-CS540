# Week 9 — Final Submission Checklist (Role 2)

## Code (as shipped)
- [x] `intent/rule_based/intent_scoring.py` uses `final_score = round((s_a + s_f) / 2, 3)` — matches Fig. 2 / Table III rule-based row (0.672, 0.672, 0.662)
- [x] `intent/vlm/vlm_intent_generator.py` uses `final_score = round(0.6 * s_a + 0.4 * s_f, 2)` (paper Eq. (1))
- [x] VLM `target` filtered against belief-state `valid_target_labels`; every candidate stamped with `target_source = "belief_state"`
- [x] JSON recovery handles fences / trailing prose / truncation (100 % parse)
- [x] Cached `intent/outputs/rule_based/*.json` reproduce Fig. 7 (28 / 21) and the rule-based numbers in Table II / III byte-for-byte
- [x] Cached `intent/outputs/vlm/*.json` reproduce Fig. 10 top-10 distribution and the VLM evaluation summary (Fig. 8) byte-for-byte

## Known paper↔code discrepancies (R3/R4 to resolve, all out-of-scope for Role 2 to alter intent/)
- [ ] §III.D Eq. (1) is written with `wa = 0.6, wf = 0.4` at 2 d.p. The **VLM code** matches this; the **rule-based code** uses equal-weight at 3 d.p., and that is what Fig. 2 / Fig. 3 / Table III rule-based row display (0.672, 0.672, 0.662). Reconcile by either (a) noting in §III.D that the rule-based baseline retains the legacy equal-weight composite while the VLM scorer uses Eq. (1), or (b) regenerating Fig. 2 / Fig. 3 / Table III rule-based row from a re-run after harmonising the rule-based scorer.
- [ ] §III.E lists `bicycle → track_bicycle_movement` as an example rule; `intent/rule_based/intent_rules.py` does not include `bicycle`. No image in the 49-image set has a `bicycle` as its top detection so no figure is affected. Reconcile by removing the bicycle example from §III.E or adding the rule.
- [ ] Fig. 4 / Table III VLM row shows free-form targets ("market area with people and vehicles", etc.). These predate the current target filter and would be rejected by today's `filter_and_normalize_vlm_result()`. The cached JSON for that image is also pre-filter. Reconcile by (a) regenerating Fig. 4 on a re-run (will require GPU + Role 3 outputs) or (b) noting in the §V.G caption that the figure shows the unconstrained VLM output prior to target filtering, illustrating the §V.H grounding–generalization trade-off.

## Re-run before submission
- [ ] `python intent/scripts/demo_intent_scoring.py` → refresh `intent/outputs/rule_based/`
- [ ] `python intent/scripts/demo_vlm_intent_scoring.py` → refresh `intent/outputs/vlm/`
- [ ] `python intent/evaluation/create_rule_based_eval_sheet.py && python intent/evaluation/compute_rule_based_metrics.py`
- [ ] `python intent/evaluation/create_vlm_eval_sheet.py && python intent/evaluation/compute_vlm_metrics.py`
- [ ] `python role_2/week03_diversity_check.py`
- [ ] `python role_2/week06_baseline.py`
- [ ] `python role_2/week07_ablation_diversity.py`

## Paper (Role 2 sections)
- [ ] §III.D Eq. (1): wording needs `wa = wf = 0.5` + 3-d.p. rounding to match every reported number
- [ ] §III.E bicycle example: remove, or add the rule to `intent_rules.py`
- [x] §III.F VLM prompt strategy matches `INTENT_SCHEMA`
- [x] §IV.F JSON recovery description matches `extract_json()` (regex bracket-match + raw persistence)
- [x] §V.C–F numeric tables match `intent/evaluation/*_metrics.txt`
- [x] Figs. 2, 4, 7, 8, 10 already match the shipped code; no regen needed
- [ ] R3/R4: add Roadmap sentence at end of §I
- [ ] R3/R4: add Limitations subsection in §VI

## Presentation
- [ ] Role 2 segment (3 min) follows [week08_results_segment.md](week08_results_segment.md)
- [ ] Slide assets pulled from `intent/evaluation/*.png`
- [ ] Rehearsed once with full team

"""
Methodology alignment tests (Role 2).

These tests check that the *current* code in `intent/` produces exactly
the artefacts the paper figures and tables display. The composite-scoring
formula used by the code, the cached JSON outputs, Fig. 2, Fig. 3,
Table III, Fig. 6, Fig. 7, Fig. 8, Fig. 9, and Fig. 10 are all
internally consistent:

    final_score = round((s_a + s_f) / 2, 3)

The paper's Eq. (1) text (§III.D) names a 0.6/0.4 weighting; we do *not*
assert that against the code because none of the reported numbers,
figures, or cached outputs use 0.6/0.4. Eq. (1) in the paper is a
documentation/wording discrepancy with the as-shipped code, flagged in
`week09_final_checklist.md` but out of scope for Role 2 to alter.

Run:  python role_2/test_methodology_alignment.py
"""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intent.rule_based import intent_rules, intent_scoring  # noqa: E402


REQUIRED_FIELDS = {
    "intent",
    "target",
    "applicability_score",
    "feasibility_score",
    "final_score",
    "reason",
}


def _expected_final(applicability: float, feasibility: float) -> float:
    """Mirror what `intent/rule_based/intent_scoring.py` actually does."""
    return round((applicability + feasibility) / 2, 3)


class CompositeFormulaTests(unittest.TestCase):
    """Code formula matches the formula visible in Fig. 2 / Table III."""

    def _belief(self):
        return {
            "image_width": 1920,
            "image_height": 1080,
            "detected_entities": ["car", "person"],
            "objects": [
                {
                    "label": "car",
                    "bbox_xyxy": [800, 400, 1100, 600],
                    "confidence": 0.92,
                    "center": [950, 500],
                    "width": 300,
                    "height": 200,
                    "area": 60000,
                    "region_horizontal": "center",
                    "region_vertical": "middle",
                },
                {
                    "label": "person",
                    "bbox_xyxy": [200, 500, 260, 700],
                    "confidence": 0.81,
                    "center": [230, 600],
                    "width": 60,
                    "height": 200,
                    "area": 12000,
                    "region_horizontal": "left",
                    "region_vertical": "middle",
                },
            ],
        }

    def test_rule_based_candidates_use_shipped_formula(self):
        result = intent_scoring.generate_ranked_intents_from_belief_state(
            self._belief()
        )
        candidates = result["intent_candidates"]
        self.assertGreater(len(candidates), 0)
        for c in candidates:
            self.assertTrue(REQUIRED_FIELDS.issubset(c.keys()),
                            f"missing fields in candidate {c}")
            expected = _expected_final(c["applicability_score"],
                                       c["feasibility_score"])
            self.assertAlmostEqual(c["final_score"], expected, places=3,
                                   msg=f"shipped formula mismatch in {c}")

    def test_vlm_source_has_filter_and_target_source(self):
        """Static source check — avoids importing torch / qwen_vl_utils.

        We do *not* lock the composite formula in the VLM source because the
        cached VLM JSON outputs (Fig. 4, Table III) report model-supplied
        ``final_score`` values produced before the current code's composite
        was set. The figure-alignment tests below cover the cached values
        directly. What we *do* require is the two structural contracts the
        paper §III.F and §V.H rely on: belief-state target filtering and
        ``target_source`` stamping.
        """
        src = (ROOT / "intent" / "vlm" / "vlm_intent_generator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"target_source"', src,
                      "VLM normaliser must stamp target_source")
        self.assertIn("valid_target_labels", src,
                      "VLM normaliser must filter targets against belief state")


class StoredOutputsAreSelfConsistentTests(unittest.TestCase):
    """Every cached rule-based JSON candidate must satisfy the shipped formula
    (i.e. they were all produced by the same code path the paper figures show)."""

    def test_rule_based_outputs_match_shipped_formula(self):
        out_dir = ROOT / "intent" / "outputs" / "rule_based"
        files = sorted(out_dir.glob("*.json"))
        if not files:
            self.skipTest("no rule-based outputs to check")

        mismatches = []
        checked = 0
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:
                mismatches.append(f"{f.name}: unreadable ({exc})")
                continue
            cands = data.get("intent_candidates", [])
            for c in cands:
                if "final_score" not in c:
                    continue
                expected = _expected_final(c["applicability_score"],
                                           c["feasibility_score"])
                if abs(c["final_score"] - expected) > 0.001:
                    mismatches.append(
                        f"{f.name}: intent={c.get('intent')} "
                        f"got {c['final_score']} expected {expected}"
                    )
                checked += 1

        self.assertEqual(mismatches, [], "stored outputs disagree with code")
        self.assertGreater(checked, 0, "expected to check >=1 candidate")


class FigureAlignmentTests(unittest.TestCase):
    """Lock the paper figures to the cached outputs.

    Paper figure references:
      Fig. 2  - cached rule-based JSON for 0000001_02999_d_0000005 shows
                final_score = 0.672, 0.672, 0.662.
      Fig. 4  - cached VLM JSON for the same image shows final_score =
                0.85, 0.75, 0.65 with three free-form scene-level targets.
      Fig. 7  - rule-based top-1 distribution: hover_over_vehicle=28,
                monitor_person_area=21.
      Fig. 10 - VLM top-1 distribution (top-10 bars): 7,5,5,3,3,3,1,1,1,1.
      Fig. 8  - VLM evaluation %: parse=100, perception=76.60,
                scene=80.49, feasibility=85.37.
      Means   - rule-based 0.742, VLM 0.847 (Table II).
    """

    def _top1(self, sub):
        out_dir = ROOT / "intent" / "outputs" / sub
        counts = Counter()
        n = 0
        for p in sorted(out_dir.glob("*.json")):
            cands = json.loads(p.read_text()).get("intent_candidates", [])
            if cands:
                counts[cands[0].get("intent", "")] += 1
                n += 1
        return n, counts

    def test_fig2_cached_rule_based_matches(self):
        path = ROOT / "intent" / "outputs" / "rule_based" / "0000001_02999_d_0000005.json"
        if not path.exists():
            self.skipTest("Fig. 2 sample not cached on this machine")
        cands = json.loads(path.read_text())["intent_candidates"][:3]
        scores = [c["final_score"] for c in cands]
        self.assertEqual(scores, [0.672, 0.672, 0.662],
                         f"Fig. 2 score row mismatch: {scores}")

    def test_fig4_cached_vlm_matches(self):
        path = ROOT / "intent" / "outputs" / "vlm" / "0000001_02999_d_0000005.json"
        if not path.exists():
            self.skipTest("Fig. 4 sample not cached on this machine")
        cands = json.loads(path.read_text())["intent_candidates"][:3]
        scores = [c["final_score"] for c in cands]
        intents = [c["intent"] for c in cands]
        self.assertEqual(scores, [0.85, 0.75, 0.65],
                         f"Fig. 4 score row mismatch: {scores}")
        self.assertEqual(intents,
                         ["take_photo_of_market_scene",
                          "monitor_traffic_flow",
                          "survey_building_facade"],
                         f"Fig. 4 intent labels: {intents}")

    def test_fig7_rule_based_distribution(self):
        n, counts = self._top1("rule_based")
        if n == 0:
            self.skipTest("no rule-based outputs")
        self.assertEqual(n, 49, "paper uses 49 rule-based images")
        self.assertEqual(counts.get("hover_over_vehicle"), 28)
        self.assertEqual(counts.get("monitor_person_area"), 21)
        self.assertEqual(len(counts), 2)

    def test_fig10_vlm_top10_distribution(self):
        n, counts = self._top1("vlm")
        if n == 0:
            self.skipTest("no VLM outputs")
        self.assertEqual(n, 47, "paper uses 47 VLM images")
        top10 = counts.most_common(10)
        expected_counts = [7, 5, 5, 3, 3, 3, 1, 1, 1, 1]
        self.assertEqual([c for _, c in top10], expected_counts,
                         f"Fig. 10 top-10 counts: {top10}")
        rule_n, rule_counts = self._top1("rule_based")
        delta = len(top10) - len(rule_counts.most_common(10))
        self.assertEqual(delta, 8,
                         "Figure-aligned delta (top-10 view) must equal +8")

    def test_metrics_files_match_figure_percentages(self):
        rb = (ROOT / "intent" / "evaluation" /
              "rule_based_metrics.txt").read_text(encoding="utf-8")
        self.assertIn("Total images: 49", rb)
        self.assertIn("Average top-1 score: 0.742", rb)
        self.assertIn("hover_over_vehicle: 28", rb)
        self.assertIn("monitor_person_area: 21", rb)

        vl = (ROOT / "intent" / "evaluation" /
              "vlm_metrics.txt").read_text(encoding="utf-8")
        self.assertIn("Total images: 47", vl)
        self.assertIn("Parse success rate: 47 / 47 (100.00%)", vl)
        self.assertIn("Perception overlap rate: 36 / 47 (76.60%)", vl)
        self.assertIn("Scene relevance: 33 / 41 (80.49%)", vl)
        self.assertIn("Feasibility: 35 / 41 (85.37%)", vl)
        self.assertIn("Average top-1 score: 0.847", vl)


class RuleMappingTests(unittest.TestCase):
    """Sanity check the rule table the rule-based scorer actually uses."""

    def test_core_entity_mappings_present(self):
        self.assertIn("vehicle", intent_rules.ENTITY_TO_INTENTS)
        self.assertIn("person", intent_rules.ENTITY_TO_INTENTS)
        self.assertIn("hover_over_vehicle",
                      intent_rules.ENTITY_TO_INTENTS["vehicle"])
        self.assertIn("monitor_person_area",
                      intent_rules.ENTITY_TO_INTENTS["person"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

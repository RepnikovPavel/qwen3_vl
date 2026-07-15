"""Tests for the nuScenes auto-labelling skills and their tolerant parsers.

These cover the structured-output parsers added for weak annotation
(2D detection, lane polylines, scene graph, drivable-area polygon). They run
without a GPU: the inputs are real-shaped model outputs (strict JSON and the
inline-prose form the 2B Thinking model actually emits), and the assertions
check that the parser recovers clean labels regardless of formatting.
"""

from __future__ import annotations

import unittest

from qwen3_vl.skills import SKILLS, SkillSpec, get_skill, resolve_skill
from qwen3_vl.skill_parsers import coord_scale, parse_skill


AUTOLABEL_KEYS = (
    "nuscenes_2d_detection",
    "nuscenes_lane",
    "nuscenes_scene_graph",
    "nuscenes_drivable_area",
)


class AutoLabelCatalogTest(unittest.TestCase):
    def test_four_autolabel_skills_are_registered(self):
        for key in AUTOLABEL_KEYS:
            self.assertIn(key, SKILLS)

    def test_autolabel_specs_are_valid(self):
        for key in AUTOLABEL_KEYS:
            spec = get_skill(key)
            self.assertIsInstance(spec, SkillSpec)
            self.assertTrue(spec.label)
            self.assertGreater(spec.default_max_new_tokens, 0)

    def test_spatial_flag_matches_coord_scale(self):
        # Skills carrying pixel coordinates must report is_spatial=True and a
        # non-zero coord_scale; the scene graph has no coords and must be False.
        spatial = {
            "nuscenes_2d_detection": True,
            "nuscenes_lane": True,
            "nuscenes_scene_graph": False,
            "nuscenes_drivable_area": True,
        }
        for key, expected in spatial.items():
            spec = get_skill(key)
            self.assertEqual(spec.is_spatial, expected, key)
            if expected:
                self.assertEqual(coord_scale(key), 1000, key)
            else:
                self.assertEqual(coord_scale(key), 0, key)

    def test_resolve_returns_full_plan(self):
        resolved = resolve_skill("nuscenes_2d_detection")
        self.assertEqual(resolved["skill"], "nuscenes_2d_detection")
        self.assertEqual(resolved["coord_scale"], 1000)
        self.assertIn("bbox_2d", resolved["prompt"])
        self.assertGreater(resolved["max_new_tokens"], 0)


class LaneParserTest(unittest.TestCase):
    def test_strict_json_multiple_lanes(self):
        text = (
            '[{"lane_id": 0, "points": [[100, 900], [120, 700], [140, 500]]}, '
            '{"lane_id": 1, "points": [[800, 900], [820, 700]]}]'
        )
        lanes = parse_skill("nuscenes_lane", text)
        self.assertEqual(len(lanes), 2)
        self.assertEqual(lanes[0]["lane_id"], 0)
        self.assertEqual(lanes[0]["points"][0], [100, 900])
        self.assertEqual(lanes[1]["lane_id"], 1)

    def test_prose_recovery_lane_keyword(self):
        # The form the 2B Thinking model often emits instead of clean JSON.
        text = (
            "Let me trace the lanes.\n"
            "lane 0: [[100, 900], [140, 500]]\n"
            "lane 1: [[800, 900], [820, 700]]\n"
        )
        lanes = parse_skill("nuscenes_lane", text)
        self.assertEqual(len(lanes), 2)
        self.assertEqual(lanes[0]["points"][0], [100, 900])

    def test_empty_and_garbage_return_empty_list(self):
        self.assertEqual(parse_skill("nuscenes_lane", ""), [])
        self.assertEqual(parse_skill("nuscenes_lane", "no coordinates here"), [])

    def test_bare_coordinate_pairs_collapse_into_single_lane(self):
        # If the model just lists points without a lane_id, we still recover
        # them as one lane so downstream tooling gets *something* to draw.
        lanes = parse_skill("nuscenes_lane", "saw points at [10, 20] and [30, 40]")
        self.assertEqual(len(lanes), 1)
        self.assertEqual(len(lanes[0]["points"]), 2)


class SceneGraphParserTest(unittest.TestCase):
    def test_strict_json_triples(self):
        text = (
            '[{"subject": "truck", "relation": "left_of", "object": "van"}, '
            '{"subject": "car", "relation": "ahead_of", "object": "truck"}]'
        )
        triples = parse_skill("nuscenes_scene_graph", text)
        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0], {
            "subject": "truck", "relation": "left_of", "object": "van",
        })

    def test_prose_paren_triples_and_quote_cleaning(self):
        # Inline '(subj, rel, obj)' with JSON-style quotes must be cleaned.
        text = 'The graph: ("truck", "left of", "van") and <car> ahead_of <truck>.'
        triples = parse_skill("nuscenes_scene_graph", text)
        # Both the paren and the angle-bracket patterns should fire.
        subjects = {t["subject"] for t in triples}
        self.assertIn("truck", subjects)
        self.assertIn("car", subjects)
        # No leaked quote characters.
        for triple in triples:
            for value in triple.values():
                self.assertNotIn('"', value)
        # Multi-word relations are normalized to snake_case.
        relations = {t["relation"] for t in triples}
        self.assertIn("left_of", relations)

    def test_empty_input_returns_empty(self):
        self.assertEqual(parse_skill("nuscenes_scene_graph", ""), [])


class DrivableAreaParserTest(unittest.TestCase):
    def test_strict_json_polygon(self):
        text = '{"polygon": [[200, 900], [800, 900], [600, 500], [400, 500]]}'
        result = parse_skill("nuscenes_drivable_area", text)
        self.assertEqual(len(result["polygon"]), 4)
        self.assertEqual(result["polygon"][0], [200, 900])

    def test_bare_list_is_treated_as_polygon(self):
        result = parse_skill(
            "nuscenes_drivable_area", "[[100, 800], [900, 800], [500, 300]]"
        )
        self.assertEqual(len(result["polygon"]), 3)

    def test_prose_coordinate_recovery(self):
        result = parse_skill(
            "nuscenes_drivable_area",
            "The drivable region corners are roughly [10, 20] and [30, 40].",
        )
        self.assertEqual(len(result["polygon"]), 2)

    def test_empty_returns_empty_polygon(self):
        self.assertEqual(
            parse_skill("nuscenes_drivable_area", ""),
            {"polygon": []},
        )


class DetectionParserTest(unittest.TestCase):
    """2D detection reuses parse_grounding_1000; sanity-check the wiring."""

    def test_strict_json_bboxes_parse(self):
        text = (
            '[{"class": "vehicle", "bbox_2d": [65, 245, 345, 675]}, '
            '{"class": "pedestrian", "bbox_2d": [365, 485, 405, 635]}]'
        )
        parsed = parse_skill("nuscenes_2d_detection", text)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["bbox_2d"], [65, 245, 345, 675])

    def test_inline_bbox_prose_recovery(self):
        # The 2B model frequently writes "[x1, y1, x2, y2] - class" in prose.
        text = "I see a truck at [65, 245, 345, 675] and a car at [625, 525, 665, 580]."
        parsed = parse_skill("nuscenes_2d_detection", text)
        self.assertGreaterEqual(len(parsed), 2)
        self.assertIn("bbox_2d", parsed[0])


if __name__ == "__main__":
    unittest.main()

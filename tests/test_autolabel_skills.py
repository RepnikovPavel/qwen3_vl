"""Tests for the driving-scene auto-labelling skills and their tolerant parsers.

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
    "detection_2d",
    "lane_polyline",
    "scene_graph",
    "drivable_area",
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
            "detection_2d": True,
            "lane_polyline": True,
            "scene_graph": False,
            "drivable_area": True,
        }
        for key, expected in spatial.items():
            spec = get_skill(key)
            self.assertEqual(spec.is_spatial, expected, key)
            if expected:
                self.assertEqual(coord_scale(key), 1000, key)
            else:
                self.assertEqual(coord_scale(key), 0, key)

    def test_resolve_returns_full_plan(self):
        resolved = resolve_skill("detection_2d")
        self.assertEqual(resolved["skill"], "detection_2d")
        self.assertEqual(resolved["coord_scale"], 1000)
        self.assertIn("bbox_2d", resolved["prompt"])
        self.assertGreater(resolved["max_new_tokens"], 0)


class LaneParserTest(unittest.TestCase):
    def test_strict_json_multiple_lanes(self):
        text = (
            '[{"lane_id": 0, "points": [[100, 900], [120, 700], [140, 500]]}, '
            '{"lane_id": 1, "points": [[800, 900], [820, 700]]}]'
        )
        lanes = parse_skill("lane_polyline", text)
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
        lanes = parse_skill("lane_polyline", text)
        self.assertEqual(len(lanes), 2)
        self.assertEqual(lanes[0]["points"][0], [100, 900])

    def test_empty_and_garbage_return_empty_list(self):
        self.assertEqual(parse_skill("lane_polyline", ""), [])
        self.assertEqual(parse_skill("lane_polyline", "no coordinates here"), [])

    def test_bare_coordinate_pairs_collapse_into_single_lane(self):
        # If the model just lists points without a lane_id, we still recover
        # them as one lane so downstream tooling gets *something* to draw.
        lanes = parse_skill("lane_polyline", "saw points at [10, 20] and [30, 40]")
        self.assertEqual(len(lanes), 1)
        self.assertEqual(len(lanes[0]["points"]), 2)

    def test_point_grounding_output_grouped_into_polylines(self):
        # The composed skill asks for ranked point_2d items per lane; the parser
        # must group by lane_id and order each lane's points by rank, even if
        # the model emits them interleaved.
        text = (
            '['
            '{"point_2d": [800, 900], "lane_id": 1, "rank": 0},'
            '{"point_2d": [100, 900], "lane_id": 0, "rank": 0},'
            '{"point_2d": [120, 700], "lane_id": 0, "rank": 1},'
            '{"point_2d": [820, 700], "lane_id": 1, "rank": 1},'
            '{"point_2d": [140, 500], "lane_id": 0, "rank": 2}'
            ']'
        )
        lanes = parse_skill("lane_polyline", text)
        self.assertEqual(len(lanes), 2)
        ego = next(l for l in lanes if l["lane_id"] == 0)
        self.assertEqual(ego["points"], [[100, 900], [120, 700], [140, 500]])
        right = next(l for l in lanes if l["lane_id"] == 1)
        self.assertEqual(right["points"], [[800, 900], [820, 700]])


class SceneGraphParserTest(unittest.TestCase):
    def test_strict_json_triples(self):
        text = (
            '[{"subject": "truck", "relation": "left_of", "object": "van"}, '
            '{"subject": "car", "relation": "ahead_of", "object": "truck"}]'
        )
        triples = parse_skill("scene_graph", text)
        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0], {
            "subject": "truck", "relation": "left_of", "object": "van",
        })

    def test_prose_paren_triples_and_quote_cleaning(self):
        # Inline '(subj, rel, obj)' with JSON-style quotes must be cleaned.
        text = 'The graph: ("truck", "left of", "van") and <car> ahead_of <truck>.'
        triples = parse_skill("scene_graph", text)
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
        self.assertEqual(parse_skill("scene_graph", ""), [])


class DrivableAreaParserTest(unittest.TestCase):
    def test_strict_json_polygon(self):
        text = '{"polygon": [[200, 900], [800, 900], [600, 500], [400, 500]]}'
        result = parse_skill("drivable_area", text)
        self.assertEqual(len(result["polygon"]), 4)
        self.assertEqual(result["polygon"][0], [200, 900])

    def test_bare_list_is_treated_as_polygon(self):
        result = parse_skill(
            "drivable_area", "[[100, 800], [900, 800], [500, 300]]"
        )
        self.assertEqual(len(result["polygon"]), 3)

    def test_prose_coordinate_recovery(self):
        result = parse_skill(
            "drivable_area",
            "The drivable region corners are roughly [10, 20] and [30, 40].",
        )
        self.assertEqual(len(result["polygon"]), 2)

    def test_empty_returns_empty_polygon(self):
        self.assertEqual(
            parse_skill("drivable_area", ""),
            {"polygon": []},
        )

    def test_point_grounding_output_assembled_into_polygon(self):
        # The composed skill asks for labelled point_2d boundary points; the
        # parser must assemble them in role order (left near->far, vanishing,
        # right far->near) into a single closed polygon.
        text = (
            '['
            '{"point_2d": [200, 900], "label": "left_edge_near"},'
            '{"point_2d": [350, 600], "label": "left_edge_mid"},'
            '{"point_2d": [450, 400], "label": "left_edge_far"},'
            '{"point_2d": [500, 320], "label": "vanishing_point"},'
            '{"point_2d": [550, 400], "label": "right_edge_far"},'
            '{"point_2d": [650, 600], "label": "right_edge_mid"},'
            '{"point_2d": [800, 900], "label": "right_edge_near"}'
            ']'
        )
        result = parse_skill("drivable_area", text)
        poly = result["polygon"]
        self.assertEqual(len(poly), 7)
        # Role order is enforced, not encounter order.
        self.assertEqual(poly[0], [200, 900])  # left_edge_near
        self.assertEqual(poly[3], [500, 320])  # vanishing_point
        self.assertEqual(poly[6], [800, 900])  # right_edge_near


class DetectionParserTest(unittest.TestCase):
    """2D detection parser: strict JSON, then prose recovery with class."""

    def test_strict_json_bboxes_parse_with_class(self):
        text = (
            '[{"class": "vehicle", "bbox_2d": [65, 245, 345, 675]}, '
            '{"class": "pedestrian", "bbox_2d": [365, 485, 405, 635]}]'
        )
        parsed = parse_skill("detection_2d", text)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["bbox_2d"], [65, 245, 345, 675])
        self.assertEqual(parsed[0]["class"], "vehicle")
        self.assertEqual(parsed[1]["class"], "pedestrian")

    def test_inline_bbox_prose_recovers_canonical_class(self):
        # The 2B Thinking model writes "<subject>: [x1,y1,x2,y2] - <class>".
        # Aliases (truck/car -> vehicle) must resolve to the canonical class.
        text = (
            "The truck: [65, 245, 345, 675] - vehicle.\n"
            "White van: [425, 500, 475, 580] - vehicle.\n"
            "Pedestrian: [365, 485, 405, 635] - pedestrian.\n"
            "Barrier: [675, 555, 725, 660] - barrier."
        )
        parsed = parse_skill("detection_2d", text)
        self.assertEqual(len(parsed), 4)
        classes = [item["class"] for item in parsed]
        self.assertEqual(classes, ["vehicle", "vehicle", "pedestrian", "barrier"])
        # The label keeps the human-readable mention; class is canonical.
        self.assertEqual(parsed[0]["label"], "vehicle")

    def test_alias_resolution_truck_car_become_vehicle(self):
        # No trailing "- class": the parser must still pull the class from the
        # window around the bbox ("truck at [...]").
        text = "I see a truck at [65, 245, 345, 675] and a car at [625, 525, 665, 580]."
        parsed = parse_skill("detection_2d", text)
        self.assertGreaterEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["class"], "vehicle")
        self.assertEqual(parsed[1]["class"], "vehicle")

    def test_em_dash_separator_is_recognised(self):
        # The model often uses an em dash (U+2014), not a hyphen.
        text = "Truck: [65, 245, 345, 675] — vehicle."
        parsed = parse_skill("detection_2d", text)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["class"], "vehicle")


if __name__ == "__main__":
    unittest.main()

"""Tests for the skill catalog and output parsers (no GPU / no model required)."""

from __future__ import annotations

import unittest

from skills import (
    SKILLS,
    SkillError,
    SkillSpec,
    get_skill,
    public_skills,
    resolve_skill,
)
from skill_parsers import coord_scale, parse_skill


class SkillCatalogTest(unittest.TestCase):
    def test_all_skills_resolve(self):
        for key, spec in SKILLS.items():
            self.assertIsInstance(spec, SkillSpec, f"{key} is not a SkillSpec")
            self.assertEqual(spec.key, key)

    def test_known_skills_present(self):
        expected = {
            "describe", "ocr", "ocr_spotting", "formula", "chart",
            "document_parsing_html", "document_parsing_md",
            "spatial_understanding", "think_detailed", "omni_recognition",
            "2d_grounding", "3d_grounding", "video_understanding",
            "long_document", "mmcode", "computer_use", "mobile_agent",
            # Auto-labelling skills (driving / nuScenes-style):
            "nuscenes_2d_detection", "nuscenes_lane",
            "nuscenes_scene_graph", "nuscenes_drivable_area",
        }
        self.assertEqual(set(SKILLS), expected)

    def test_get_skill_rejects_unknown(self):
        with self.assertRaises(SkillError):
            get_skill("does_not_exist")

    def test_coord_scale_matches_cookbook_conventions(self):
        # grounding / spatial / omni -> 0..1000
        self.assertEqual(coord_scale("2d_grounding"), 1000)
        self.assertEqual(coord_scale("spatial_understanding"), 1000)
        self.assertEqual(coord_scale("omni_recognition"), 1000)
        # OCR spotting (Qwen2.5-VL notebook) -> 0..999
        self.assertEqual(coord_scale("ocr_spotting"), 999)
        # mobile agent -> 0..999
        self.assertEqual(coord_scale("mobile_agent"), 999)
        # computer use -> 0..1000
        self.assertEqual(coord_scale("computer_use"), 1000)
        # free-text skills -> no coords
        self.assertEqual(coord_scale("describe"), 0)
        self.assertEqual(coord_scale("video_understanding"), 0)

    def test_grounding_skills_flagged(self):
        self.assertTrue(get_skill("2d_grounding").is_grounding)
        self.assertTrue(get_skill("3d_grounding").is_grounding)
        self.assertFalse(get_skill("describe").is_grounding)

    def test_public_skills_schema(self):
        payload = public_skills()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["skills"]), len(SKILLS))
        for item in payload["skills"]:
            self.assertIn("key", item)
            self.assertIn("prompt", item)
            self.assertIn("output_kind", item)

    def test_resolve_skill_custom_prompt_for_accepts_custom(self):
        resolved = resolve_skill("2d_grounding", custom_prompt="find all bicycles")
        self.assertEqual(resolved["prompt"], "find all bicycles")

    def test_resolve_skill_override_for_fixed_prompt(self):
        resolved = resolve_skill("describe", custom_prompt="count vehicles only")
        self.assertEqual(resolved["prompt"], "count vehicles only")

    def test_resolve_skill_uses_defaults_when_no_override(self):
        resolved = resolve_skill("describe")
        spec = get_skill("describe")
        self.assertEqual(resolved["prompt"], spec.prompt)
        self.assertEqual(resolved["max_new_tokens"], spec.default_max_new_tokens)

    def test_resolve_skill_rejects_nonpositive_tokens(self):
        with self.assertRaises(SkillError):
            resolve_skill("describe", max_new_tokens=0)

    def test_skill_key_with_leading_digit_allowed(self):
        # 2d_grounding / 3d_grounding legitimately start with a digit
        self.assertIn("2d_grounding", SKILLS)
        self.assertIn("3d_grounding", SKILLS)

    def test_parser_coord_scale_matches_skill_spec(self):
        # Single source of truth: skill_parsers.coord_scale must agree with
        # SkillSpec.coord_scale for every skill.
        for key, spec in SKILLS.items():
            self.assertEqual(
                coord_scale(key), spec.coord_scale,
                f"coord scale mismatch for {key}: parser={coord_scale(key)} spec={spec.coord_scale}",
            )


class GroundingParserTest(unittest.TestCase):
    def test_parse_strict_bbox_json(self):
        text = '[{"bbox_2d": [100, 200, 300, 400], "label": "car"}]'
        parsed = parse_skill("2d_grounding", text)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["bbox_2d"], [100, 200, 300, 400])
        self.assertEqual(parsed[0]["label"], "car")

    def test_parse_fenced_json(self):
        text = '```json\n[{"bbox_2d": [10, 20, 30, 40], "label": "person"}]\n```'
        parsed = parse_skill("2d_grounding", text)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["bbox_2d"], [10, 20, 30, 40])

    def test_parse_point_json(self):
        text = '[{"point_2d": [500, 250], "label": "center"}]'
        parsed = parse_skill("spatial_understanding", text)
        self.assertTrue(len(parsed) >= 1)
        self.assertEqual(parsed[0]["point_2d"], [500, 250])


class OcrSpottingParserTest(unittest.TestCase):
    def test_parse_ocr_spotting_999_scale(self):
        text = '[{"bbox_2d": [0, 0, 999, 50], "text_content": "HELLO"}]'
        parsed = parse_skill("ocr_spotting", text)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["text_content"], "HELLO")
        # coords in 0..999 convention
        self.assertEqual(parsed[0]["bbox_2d"], [0, 0, 999, 50])

    def test_parse_ocr_spotting_with_fences(self):
        text = '```json\n[{"bbox_2d": [10, 10, 200, 60], "text_content": "A"}]\n```'
        parsed = parse_skill("ocr_spotting", text)
        self.assertEqual(len(parsed), 1)


class FormulaChartParserTest(unittest.TestCase):
    def test_parse_formula_json(self):
        text = '{"formulas": ["E=mc^2", "a^2+b^2=c^2"]}'
        result = parse_skill("formula", text)
        self.assertEqual(result["formulas"], ["E=mc^2", "a^2+b^2=c^2"])

    def test_parse_chart_json(self):
        text = '{"title": "Sales", "panels": []}'
        result = parse_skill("chart", text)
        self.assertEqual(result["title"], "Sales")


class PlainParserTest(unittest.TestCase):
    def test_describe_returns_text(self):
        text = "  A street scene with cars.  "
        result = parse_skill("describe", text)
        self.assertEqual(result, "A street scene with cars.")


if __name__ == "__main__":
    unittest.main()

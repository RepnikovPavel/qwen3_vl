import json
import unittest
from dataclasses import FrozenInstanceError

from demo.tasks import (
    DEMO_PRESETS,
    DemoPreset,
    DemoTaskError,
    build_structured_result,
    evaluate_validated_fixture_answer,
    get_preset,
    public_presets,
    resolve_task,
)


def _chart(values_key="values"):
    series = {"name": "Series A", values_key: [1, 3, 2]}
    return {
        "title": "Demo chart",
        "panels": [
            {
                "chart_type": "line",
                "title": "Trend",
                "x_label": "Quarter",
                "y_label": "Value",
                "categories": ["Q1", "Q2", "Q3"],
                "series": [series],
            }
        ],
        "facts": [{"subject": "Series A", "relation": "maximum_at", "object": "Q2"}],
    }


class DemoPresetTest(unittest.TestCase):
    def test_registry_is_ordered_immutable_and_has_task_specific_defaults(self):
        self.assertEqual(
            list(DEMO_PRESETS),
            ["describe", "ocr", "formula", "chart", "custom"],
        )
        self.assertEqual(get_preset("describe").default_max_image_side, 640)
        self.assertEqual(get_preset("ocr").default_max_image_side, 1280)
        self.assertEqual(get_preset("formula").default_max_new_tokens, 4096)
        self.assertEqual(get_preset("chart").default_max_new_tokens, 16_384)
        self.assertTrue(get_preset("formula").structured_output)
        self.assertFalse(get_preset("ocr").structured_output)
        with self.assertRaises(FrozenInstanceError):
            get_preset("ocr").label = "changed"
        with self.assertRaises(TypeError):
            DEMO_PRESETS["new"] = get_preset("describe")

    def test_public_serialization_is_json_safe_and_does_not_expose_ground_truth(self):
        first = public_presets()
        rendered = json.dumps(first, ensure_ascii=False)

        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(len(first["tasks"]), 5)
        self.assertNotIn("ground_truth", rendered)
        self.assertNotIn("image_sha256", rendered)
        self.assertIsNone(first["tasks"][-1]["prompt"])
        first["tasks"][0]["label"] = "mutated"
        self.assertEqual(public_presets()["tasks"][0]["label"], "Describe")

    def test_invalid_preset_construction_is_rejected(self):
        cases = [
            {"key": "Bad", "label": "Bad", "prompt": "p", "output_kind": "text"},
            {"key": None, "label": "Bad", "prompt": "p", "output_kind": "text"},
            {"key": "bad", "label": "", "prompt": "p", "output_kind": "text"},
            {"key": "bad", "label": 7, "prompt": "p", "output_kind": "text"},
            {"key": "bad", "label": "Bad", "prompt": None, "output_kind": "text"},
            {"key": "bad", "label": "Bad", "prompt": "p", "output_kind": "audio"},
            {
                "key": "bad",
                "label": "Bad",
                "prompt": None,
                "output_kind": "text",
                "accepts_custom_prompt": 1,
            },
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(DemoTaskError):
                DemoPreset(
                    default_max_new_tokens=1,
                    default_max_image_side=64,
                    **values,
                )


class TaskResolutionTest(unittest.TestCase):
    def test_server_presets_and_custom_prompt_are_resolved_strictly(self):
        ocr = resolve_task("ocr")
        self.assertEqual(ocr["task"], "ocr")
        self.assertIn("Transcribe all visible text", ocr["prompt"])
        self.assertEqual(ocr["max_new_tokens"], 4096)
        self.assertEqual(ocr["max_image_side"], 1280)

        custom = resolve_task(
            "custom",
            custom_prompt="  Compare both panels.  ",
            max_new_tokens=777,
            max_image_side=900,
        )
        self.assertEqual(custom["prompt"], "Compare both panels.")
        self.assertEqual(custom["max_new_tokens"], 777)
        self.assertEqual(custom["max_image_side"], 900)

    def test_invalid_task_prompt_and_generation_limits_are_rejected(self):
        invalid_calls = [
            lambda: get_preset("OCR"),
            lambda: get_preset("missing"),
            lambda: resolve_task("custom"),
            lambda: resolve_task("ocr", custom_prompt="override"),
            lambda: resolve_task("describe", max_new_tokens=True),
            lambda: resolve_task("describe", max_new_tokens=0),
            lambda: resolve_task("describe", max_new_tokens=40_961),
            lambda: resolve_task("describe", max_image_side=63),
            lambda: resolve_task("describe", max_image_side=4097),
            lambda: resolve_task("custom", custom_prompt="x" * 200_001),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(DemoTaskError):
                call()


class StructuredResultTest(unittest.TestCase):
    def test_formula_strict_and_recovered_results_preserve_schema_status(self):
        strict = build_structured_result(
            "formula", json.dumps({"formulas": [r"E=mc^2"]})
        )
        recovered = build_structured_result(
            "formula", '{"formulas":["\\int_0^1 x\\,dx"]}'
        )

        self.assertTrue(strict["strict_schema_valid"])
        self.assertFalse(strict["recovery_applied"])
        self.assertEqual(strict["value"], {"formulas": [r"E=mc^2"]})
        self.assertFalse(recovered["strict_schema_valid"])
        self.assertTrue(recovered["recovery_applied"])
        self.assertEqual(recovered["recovery_steps"], ["escape_latex_backslashes"])
        self.assertEqual(recovered["value"]["formulas"], [r"\int_0^1 x\,dx"])

    def test_chart_alias_is_recovered_but_prose_is_not(self):
        aliased = build_structured_result("chart", json.dumps(_chart("numeric_values")))
        prose = build_structured_result("chart", "The line rises and then falls.")

        self.assertFalse(aliased["strict_schema_valid"])
        self.assertTrue(aliased["recovery_applied"])
        self.assertEqual(aliased["recovery_steps"], ["alias_numeric_values_to_values"])
        self.assertEqual(
            aliased["value"]["panels"][0]["series"][0]["values"], [1, 3, 2]
        )
        self.assertIsNone(prose["value"])
        self.assertFalse(prose["strict_schema_valid"])
        self.assertFalse(prose["recovery_applied"])
        self.assertIsNone(build_structured_result("ocr", "plain text"))

    def test_non_string_answer_is_rejected(self):
        with self.assertRaisesRegex(DemoTaskError, "answer must be a string"):
            build_structured_result("formula", {"formulas": []})


class FixtureEvaluationAdapterTest(unittest.TestCase):
    def test_adapter_returns_metrics_without_ground_truth_or_answer(self):
        fixture = {
            "id": "line_bar_chart",
            "task": "chart",
            "ground_truth": {**_chart(), "numeric_tolerance": 0.05},
        }
        result = evaluate_validated_fixture_answer(
            fixture, json.dumps(_chart("numeric_values"))
        )
        rendered = json.dumps(result)

        self.assertFalse(result["response_schema_valid"])
        self.assertTrue(result["response_recovery_applied"])
        self.assertEqual(result["metrics"]["label_exact_rate"], 1)
        self.assertEqual(result["metrics"]["numeric_within_tolerance_rate"], 1)
        self.assertNotIn("ground_truth", rendered)
        self.assertNotIn("Demo chart", rendered)

    def test_formula_adapter_uses_recovery_without_exposing_formula_text(self):
        result = evaluate_validated_fixture_answer(
            {
                "id": "formula",
                "task": "formula",
                "ground_truth": {"formulas": [r"\int_0^1 x\,dx"]},
            },
            '{"formulas":["\\int_0^1 x\\,dx"]}',
        )
        rendered = json.dumps(result)

        self.assertFalse(result["response_schema_valid"])
        self.assertTrue(result["response_recovery_applied"])
        self.assertEqual(result["metrics"]["normalized_exact_rate"], 1)
        self.assertNotIn("ground_truth", rendered)
        self.assertNotIn("\\int_0^1", rendered)

    def test_adapter_supports_text_and_rejects_invalid_fixture(self):
        result = evaluate_validated_fixture_answer(
            {"id": "text", "task": "text", "ground_truth": {"text": "Hello"}},
            "Hello",
        )

        self.assertTrue(result["response_schema_valid"])
        self.assertTrue(result["metrics"]["nfkc_exact"])
        with self.assertRaises(DemoTaskError):
            evaluate_validated_fixture_answer(
                {"id": "bad", "task": "text", "ground_truth": {"wrong": "x"}},
                "x",
            )


if __name__ == "__main__":
    unittest.main()

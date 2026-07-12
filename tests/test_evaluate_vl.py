import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from evaluate_vl import SchemaError, evaluate
from generate_eval_fixtures import generate_fixtures


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _chart_answer(ground_truth):
    return {
        "title": ground_truth["title"],
        "panels": ground_truth["panels"],
        "facts": ground_truth["facts"],
    }


def _perfect_responses(manifest):
    responses = []
    for fixture in manifest["fixtures"]:
        if fixture["task"] == "text":
            answer = fixture["ground_truth"]["text"]
        elif fixture["task"] == "formula":
            answer = json.dumps(fixture["ground_truth"], ensure_ascii=False)
        else:
            answer = json.dumps(
                _chart_answer(fixture["ground_truth"]), ensure_ascii=False
            )
        responses.append({"id": fixture["id"], "answer": answer})
    return {"schema_version": 1, "responses": responses}


class FixtureGenerationTest(unittest.TestCase):
    def test_generates_four_reproducible_images_and_manifest(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            first_manifest_path = generate_fixtures(first)
            second_manifest_path = generate_fixtures(second)
            first_manifest = _read_json(first_manifest_path)
            second_manifest = _read_json(second_manifest_path)

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(len(first_manifest["fixtures"]), 4)
            self.assertEqual(
                [fixture["task"] for fixture in first_manifest["fixtures"]],
                ["text", "formula", "chart", "chart"],
            )
            for fixture in first_manifest["fixtures"]:
                image = Path(first) / fixture["image"]
                self.assertTrue(image.is_file())
                self.assertEqual(
                    hashlib.sha256(image.read_bytes()).hexdigest(),
                    fixture["image_sha256"],
                )


class EvaluationTest(unittest.TestCase):
    def _workspace(self, temporary):
        root = Path(temporary)
        manifest_path = generate_fixtures(root)
        manifest = _read_json(manifest_path)
        responses_path = root / "responses.json"
        responses_path.write_text(
            json.dumps(_perfect_responses(manifest), ensure_ascii=False),
            encoding="utf-8",
        )
        return root, manifest_path, manifest, responses_path

    def test_perfect_responses_score_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest_path, _, responses_path = self._workspace(temporary)
            result = evaluate(manifest_path, responses_path)

        self.assertEqual(result["summary"]["response_schema_valid_rate"], 1.0)
        self.assertEqual(result["summary"]["response_recovery_applied_count"], 0)
        self.assertTrue(
            all(not entry["response_recovery_applied"] for entry in result["results"])
        )
        self.assertTrue(
            all(entry["response_recovery_steps"] == [] for entry in result["results"])
        )
        self.assertTrue(result["summary"]["text"]["nfkc_exact"])
        self.assertEqual(result["summary"]["text"]["cer"], 0.0)
        self.assertEqual(result["summary"]["text"]["wer"], 0.0)
        self.assertEqual(result["summary"]["formula"]["normalized_exact_rate"], 1.0)
        self.assertEqual(
            result["summary"]["formula"]["token_edit_similarity_mean"], 1.0
        )
        self.assertEqual(result["summary"]["formula"]["syntax_valid_rate"], 1.0)
        self.assertEqual(result["summary"]["charts"]["chart_type_exact_rate"], 1.0)
        self.assertEqual(result["summary"]["charts"]["label_exact_rate"], 1.0)
        self.assertEqual(
            result["summary"]["charts"]["numeric_within_tolerance_rate"], 1.0
        )
        self.assertEqual(result["summary"]["charts"]["fact_f1"], 1.0)

    def test_quality_errors_are_measured_and_invalid_inner_json_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest_path, manifest, responses_path = self._workspace(temporary)
            responses = _perfect_responses(manifest)
            by_id = {entry["id"]: entry for entry in responses["responses"]}
            by_id["multilingual_text_table"]["answer"] = by_id[
                "multilingual_text_table"
            ]["answer"].replace("English", "Engl1sh", 1)
            formula = deepcopy(manifest["fixtures"][1]["ground_truth"])
            formula["formulas"][0] = r"\frac{1}{2"
            by_id["formulas_mathtext"]["answer"] = json.dumps(formula)
            line_chart = _chart_answer(
                deepcopy(manifest["fixtures"][2]["ground_truth"])
            )
            line_chart["panels"][0]["series"][0]["values"][0] += 1
            line_chart["facts"].pop()
            by_id["line_bar_chart"]["answer"] = json.dumps(line_chart)
            by_id["scatter_heatmap_chart"]["answer"] = "not-json"
            responses_path.write_text(json.dumps(responses), encoding="utf-8")

            result = evaluate(manifest_path, responses_path)

        results = {entry["id"]: entry for entry in result["results"]}
        self.assertFalse(results["multilingual_text_table"]["metrics"]["nfkc_exact"])
        self.assertGreater(results["multilingual_text_table"]["metrics"]["cer"], 0)
        self.assertLess(results["formulas_mathtext"]["metrics"]["syntax_valid_rate"], 1)
        self.assertLess(
            results["line_bar_chart"]["metrics"]["numeric_within_tolerance_rate"], 1
        )
        self.assertLess(results["line_bar_chart"]["metrics"]["fact_recall"], 1)
        self.assertFalse(results["scatter_heatmap_chart"]["response_schema_valid"])
        self.assertFalse(results["scatter_heatmap_chart"]["response_recovery_applied"])
        self.assertEqual(
            results["scatter_heatmap_chart"]["metrics"]["label_exact_rate"], 0
        )
        self.assertEqual(
            results["scatter_heatmap_chart"]["metrics"][
                "numeric_within_tolerance_rate"
            ],
            0,
        )
        self.assertIn(
            "invalid JSON", results["scatter_heatmap_chart"]["response_schema_error"]
        )
        self.assertEqual(result["summary"]["response_schema_valid_count"], 3)

    def test_narrow_formula_and_numeric_values_recovery_preserves_strict_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest_path, manifest, responses_path = self._workspace(temporary)
            responses = _perfect_responses(manifest)
            by_id = {entry["id"]: entry for entry in responses["responses"]}
            formula_answer = json.dumps(
                manifest["fixtures"][1]["ground_truth"], ensure_ascii=False
            ).replace("\\\\", "\\")
            by_id["formulas_mathtext"]["answer"] = formula_answer
            line_chart = _chart_answer(
                deepcopy(manifest["fixtures"][2]["ground_truth"])
            )
            for panel in line_chart["panels"]:
                for series in panel["series"]:
                    series["numeric_values"] = series.pop("values")
            by_id["line_bar_chart"]["answer"] = json.dumps(line_chart)
            responses_path.write_text(json.dumps(responses), encoding="utf-8")

            result = evaluate(manifest_path, responses_path)

        results = {entry["id"]: entry for entry in result["results"]}
        formula_result = results["formulas_mathtext"]
        self.assertFalse(formula_result["response_schema_valid"])
        self.assertIsNotNone(formula_result["response_schema_error"])
        self.assertTrue(formula_result["response_recovery_applied"])
        self.assertEqual(
            formula_result["response_recovery_steps"],
            ["escape_latex_backslashes"],
        )
        self.assertEqual(formula_result["metrics"]["normalized_exact_rate"], 1)
        self.assertEqual(formula_result["metrics"]["syntax_valid_rate"], 1)
        chart_result = results["line_bar_chart"]
        self.assertFalse(chart_result["response_schema_valid"])
        self.assertIn("numeric_values", chart_result["response_schema_error"])
        self.assertTrue(chart_result["response_recovery_applied"])
        self.assertEqual(
            chart_result["response_recovery_steps"],
            ["alias_numeric_values_to_values"],
        )
        self.assertEqual(chart_result["metrics"]["label_exact_rate"], 1)
        self.assertEqual(chart_result["metrics"]["numeric_within_tolerance_rate"], 1)
        self.assertEqual(result["summary"]["response_schema_valid_count"], 2)
        self.assertEqual(result["summary"]["response_recovery_applied_count"], 2)

    def test_recovery_rejects_markdown_and_ambiguous_numeric_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest_path, manifest, responses_path = self._workspace(temporary)
            responses = _perfect_responses(manifest)
            by_id = {entry["id"]: entry for entry in responses["responses"]}
            by_id["formulas_mathtext"]["answer"] = (
                '```json\n{"formulas":["\\int_0^1 x\\,dx"]}\n```'
            )
            line_chart = _chart_answer(
                deepcopy(manifest["fixtures"][2]["ground_truth"])
            )
            line_chart["panels"][0]["series"][0]["numeric_values"] = line_chart[
                "panels"
            ][0]["series"][0]["values"]
            by_id["line_bar_chart"]["answer"] = json.dumps(line_chart)
            responses_path.write_text(json.dumps(responses), encoding="utf-8")

            result = evaluate(manifest_path, responses_path)

        results = {entry["id"]: entry for entry in result["results"]}
        for fixture_id in ("formulas_mathtext", "line_bar_chart"):
            self.assertFalse(results[fixture_id]["response_schema_valid"])
            self.assertFalse(results[fixture_id]["response_recovery_applied"])
            self.assertEqual(results[fixture_id]["response_recovery_steps"], [])
        self.assertEqual(
            results["formulas_mathtext"]["metrics"]["normalized_exact_rate"], 0
        )
        self.assertEqual(
            results["line_bar_chart"]["metrics"]["numeric_within_tolerance_rate"],
            0,
        )

    def test_wrapper_and_manifest_validation_are_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest_path, manifest, responses_path = self._workspace(temporary)
            responses = _perfect_responses(manifest)
            responses["unexpected"] = True
            responses_path.write_text(json.dumps(responses), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "keys mismatch"):
                evaluate(manifest_path, responses_path)

            valid_responses = _perfect_responses(manifest)
            responses_path.write_text(json.dumps(valid_responses), encoding="utf-8")
            image_path = root / manifest["fixtures"][0]["image"]
            image_path.write_bytes(image_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(SchemaError, "SHA-256 mismatch"):
                evaluate(manifest_path, responses_path)


if __name__ == "__main__":
    unittest.main()

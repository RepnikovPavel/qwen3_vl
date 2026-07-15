import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import qwen3_vl.qwen3_vl
from qwen3_vl.evaluate_vl import SchemaError, validate_responses
from qwen3_vl.run_vl_eval import run_manifest


def _chart_truth(title):
    return {
        "title": title,
        "panels": [],
        "facts": [],
        "numeric_tolerance": 0.05,
    }


def _manifest(root):
    definitions = [
        ("text", "text", {"text": "Hello"}),
        ("formula", "formula", {"formulas": [r"E=mc^2"]}),
        ("line_bar", "chart", _chart_truth("Line and bar")),
        ("scatter_heatmap", "chart", _chart_truth("Scatter and heatmap")),
    ]
    fixtures = []
    for fixture_id, task, ground_truth in definitions:
        image_name = f"{fixture_id}.png"
        image_path = root / image_name
        image_path.write_bytes(f"png:{fixture_id}".encode())
        fixtures.append(
            {
                "id": fixture_id,
                "task": task,
                "image": image_name,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "prompt": f"prompt:{fixture_id}",
                "ground_truth": ground_truth,
            }
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "fixtures": fixtures}),
        encoding="utf-8",
    )
    return manifest_path, fixtures


class _Runtime:
    def __init__(self, factory_kwargs, finish_reason="eos", fail_at=None):
        self.factory_kwargs = factory_kwargs
        self.finish_reason = finish_reason
        self.fail_at = fail_at
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_at == len(self.calls):
            raise RuntimeError("inference failed")
        answer = f"answer:{kwargs['prompt']}"
        return SimpleNamespace(answer=answer, finish_reason=self.finish_reason), []


class RunManifestTest(unittest.TestCase):
    def test_top_level_cli_dispatches_eval_run(self):
        with mock.patch("run_vl_eval.main", return_value=19) as runner:
            result = qwen3_vl.main(["eval-run", "--manifest", "m", "--output", "o"])

        self.assertEqual(result, 19)
        runner.assert_called_once_with(["--manifest", "m", "--output", "o"])

    def test_loads_one_runtime_runs_all_fixtures_and_writes_compatible_json(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, fixtures = _manifest(root)
            output_path = root / "responses.json"
            runtimes = []

            def factory(**kwargs):
                runtime = _Runtime(kwargs)
                runtimes.append(runtime)
                return runtime

            payload = run_manifest(
                manifest_path,
                output_path,
                model_size="4b",
                device="cpu",
                cpu_threads=7,
                seed=99,
                max_image_side=900,
                max_new_tokens=321,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=11,
                verbose=False,
                runtime_factory=factory,
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, written)
        self.assertEqual(len(runtimes), 1)
        runtime = runtimes[0]
        self.assertEqual(runtime.factory_kwargs["model_size"], "4b")
        self.assertEqual(runtime.factory_kwargs["device"], "cpu")
        self.assertEqual(runtime.factory_kwargs["cpu_threads"], 7)
        self.assertEqual(runtime.factory_kwargs["seed"], 99)
        self.assertEqual(len(runtime.calls), 4)
        for call, fixture in zip(runtime.calls, fixtures, strict=True):
            self.assertEqual(call["prompt"], fixture["prompt"])
            self.assertEqual(
                call["media_inputs"],
                [("image", str(manifest_path.parent / fixture["image"]))],
            )
            self.assertTrue(call["do_sample"])
            self.assertEqual(call["temperature"], 0.7)
            self.assertEqual(call["top_p"], 0.8)
            self.assertEqual(call["top_k"], 11)
            self.assertTrue(call["check_finite_logits"])
            self.assertEqual(call["max_image_side"], 900)
            self.assertEqual(call["max_new_tokens"], 321)
        validate_responses(written, {fixture["id"] for fixture in fixtures})

    def test_inference_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _manifest(root)
            output_path = root / "responses.json"
            output_path.write_text("previous\n", encoding="utf-8")

            def factory(**kwargs):
                return _Runtime(kwargs, fail_at=3)

            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                run_manifest(manifest_path, output_path, runtime_factory=factory)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "previous\n")
            self.assertEqual(list(root.glob(".responses.json.*.tmp")), [])

    def test_incomplete_generation_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _manifest(root)
            output_path = root / "responses.json"

            def factory(**kwargs):
                return _Runtime(kwargs, finish_reason="max_new_tokens")

            with self.assertRaisesRegex(RuntimeError, "max_new_tokens"):
                run_manifest(manifest_path, output_path, runtime_factory=factory)

            self.assertFalse(output_path.exists())

    def test_manifest_is_verified_before_runtime_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, fixtures = _manifest(root)
            (root / fixtures[0]["image"]).write_bytes(b"tampered")
            factory_calls = []

            def factory(**kwargs):
                factory_calls.append(kwargs)
                return _Runtime(kwargs)

            with self.assertRaisesRegex(SchemaError, "SHA-256 mismatch"):
                run_manifest(
                    manifest_path, root / "responses.json", runtime_factory=factory
                )

            self.assertEqual(factory_calls, [])


if __name__ == "__main__":
    unittest.main()

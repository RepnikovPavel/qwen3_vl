import io
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

import qwen3_vl
from web_ui import create_app


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LazyTopLevelCliTest(unittest.TestCase):
    def test_importing_top_level_cli_does_not_import_runtime_or_torch(self):
        script = (
            "import sys; import qwen3_vl; "
            "assert 'qwen3_vl_offline' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_download_and_verify_delegate_without_importing_runtime(self):
        calls = []
        fake_download_module = types.ModuleType("download_models")

        def fake_main(argv):
            calls.append(list(argv))
            return 17

        fake_download_module.main = fake_main
        with mock.patch.dict(
            sys.modules,
            {
                "download_models": fake_download_module,
                "qwen3_vl_offline": None,
            },
        ):
            self.assertEqual(qwen3_vl.main(["download", "2b", "--quick"]), 17)
            self.assertEqual(qwen3_vl.main(["verify", "4b", "--quick"]), 17)

        self.assertEqual(
            calls,
            [
                ["2b", "--quick"],
                ["4b", "--quick", "--verify-only"],
            ],
        )


class FakeRuntime:
    def __init__(self, device="cuda"):
        self.spec = SimpleNamespace(repo_id="Example/Qwen3-VL-Test")
        self.device = device
        self.load_seconds = 1.25
        self.fp8_names = ["layer.0", "layer.1"]


class FakeWebResult:
    answer = "A bounded test answer."

    def to_dict(self):
        return {
            "answer": self.answer,
            "finish_reason": "eos",
            "prompt_tokens": 12,
            "generated_tokens": 5,
        }


class FakeInferenceRuntime(FakeRuntime):
    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.infer_calls = []

    def infer(self, **kwargs):
        self.infer_calls.append(kwargs)
        return FakeWebResult(), []


class WebStatusTest(unittest.TestCase):
    def test_health_and_status_use_runtime_metadata_without_inference(self):
        app = create_app(FakeRuntime())
        with TestClient(app) as client:
            health = client.get("/healthz")
            status = client.get("/api/status")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(status.status_code, 200)
        self.assertEqual(
            status.json(),
            {
                "model": "Example/Qwen3-VL-Test",
                "device_mode": "gpu_fp8",
                "load_seconds": 1.25,
                "fp8_modules": 2,
            },
        )

    def test_cpu_status_reports_fp32_mode(self):
        app = create_app(FakeRuntime(device="cpu"))
        with TestClient(app) as client:
            status = client.get("/api/status")
        self.assertEqual(status.json()["device_mode"], "cpu_fp32")


class WebInferenceValidationTest(unittest.TestCase):
    def test_invalid_generation_parameters_are_rejected_before_runtime(self):
        runtime = FakeInferenceRuntime()
        app = create_app(runtime)
        invalid_cases = {
            "max_new_tokens below range": {"max_new_tokens": "0"},
            "max_new_tokens above range": {"max_new_tokens": "40961"},
            "max_image_side below range": {"max_image_side": "63"},
            "max_image_side above range": {"max_image_side": "4097"},
            "video frames below range": {"video_num_frames": "1"},
            "video frames above range": {"video_num_frames": "513"},
            "zero temperature": {"temperature": "0"},
            "non-finite temperature": {"temperature": "nan"},
            "top-p below range": {"top_p": "0"},
            "top-p above range": {"top_p": "1.01"},
            "top-k below range": {"top_k": "0"},
            "top-k above range": {"top_k": "1001"},
        }

        with TestClient(app) as client:
            for label, override in invalid_cases.items():
                with self.subTest(label=label):
                    response = client.post(
                        "/api/infer",
                        data={"prompt": "Validate settings.", **override},
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertIn("detail", response.json())

        self.assertEqual(runtime.infer_calls, [])

    def test_infer_preserves_first_pair_and_newest_pairs_within_history_bound(self):
        runtime = FakeInferenceRuntime()
        app = create_app(runtime)
        history = []
        for index in range(20):
            history.extend(
                [
                    {"role": "user", "content": f"old user {index}"},
                    {"role": "assistant", "content": f"old assistant {index}"},
                ]
            )
        image_buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "green").save(image_buffer, format="PNG")

        with TestClient(app) as client:
            response = client.post(
                "/api/infer",
                data={
                    "prompt": "Newest follow-up",
                    "history_json": json.dumps(history),
                },
                files={"files": ("fixture.png", image_buffer.getvalue(), "image/png")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        bounded = response.json()["history"]
        self.assertEqual(len(bounded), 40)
        self.assertEqual(bounded[:2], history[:2])
        self.assertEqual(
            bounded[-2:],
            [
                {"role": "user", "content": "Newest follow-up"},
                {"role": "assistant", "content": FakeWebResult.answer},
            ],
        )
        self.assertEqual(
            [item["role"] for item in bounded],
            ["user", "assistant"] * 20,
        )

        self.assertEqual(len(runtime.infer_calls), 1)
        call = runtime.infer_calls[0]
        self.assertEqual(call["history"], history)
        self.assertEqual(call["media_history_index"], 0)
        self.assertEqual(len(call["media_inputs"]), 1)
        self.assertEqual(call["media_inputs"][0][0], "image")
        self.assertIsInstance(call["media_inputs"][0][1], Image.Image)


if __name__ == "__main__":
    unittest.main()

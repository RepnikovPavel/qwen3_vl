import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from demo.model_manager import DemoBusyError, DemoModelManager


class FakeRuntime:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.spec = SimpleNamespace(repo_id="Qwen/Test-FP8")
        self.load_seconds = 1.5
        self.fp8_names = ["a", "b"]
        self.context_mode = "native_256k"
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                get_text_config=lambda: SimpleNamespace(max_position_embeddings=262144)
            )
        )
        self.hf_device_map = {"model": 0}


class DemoModelManagerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.calls = []

        def factory(**kwargs):
            self.calls.append(kwargs)
            return FakeRuntime(**kwargs)

        self.manager = DemoModelManager(
            self.temporary.name,
            runtime_factory=factory,
            idle_seconds=0,
        )

    def tearDown(self):
        self.manager.close()
        self.temporary.cleanup()

    @mock.patch.object(DemoModelManager, "_visible_gpu_count", return_value=2)
    def test_load_is_fp8_cuda_only_and_reuses_matching_runtime(self, _count):
        with self.manager.operation():
            first = self.manager.load("8b", "balanced")
        with self.manager.operation():
            second = self.manager.load("8b", "balanced")
        self.assertIs(first, second)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["device"], "cuda")
        self.assertEqual(self.calls[0]["gpu_placement"], "balanced")

    def test_operation_is_non_blocking(self):
        with self.manager.operation():
            with self.assertRaisesRegex(DemoBusyError, "busy"):
                with self.manager.operation():
                    pass

    @mock.patch.object(DemoModelManager, "_visible_gpu_count", return_value=2)
    def test_failed_checkpoint_preflight_does_not_unload_current_model(self, _count):
        with self.manager.operation():
            loaded = self.manager.load("8b", "single")
        self.manager._runtime_factory = None
        with mock.patch(
            "download_models.verify_checkpoint",
            side_effect=FileNotFoundError("checkpoint missing"),
        ):
            with self.manager.operation():
                with self.assertRaises(FileNotFoundError):
                    self.manager.load("4b", "single")
        self.assertIs(self.manager.runtime, loaded)

    @mock.patch.object(DemoModelManager, "_visible_gpu_count", return_value=1)
    def test_balanced_requires_two_visible_gpus(self, _count):
        with self.manager.operation():
            with self.assertRaisesRegex(ValueError, "two visible GPUs"):
                self.manager.load("2b", "balanced")

    @mock.patch.object(DemoModelManager, "_visible_gpu_count", return_value=2)
    def test_status_reports_context_and_device_map(self, _count):
        with self.manager.operation():
            self.manager.load("8b", "balanced")
        status = self.manager.status()
        self.assertTrue(status["loaded"])
        self.assertEqual(status["device"], "cuda_fp8")
        self.assertEqual(status["context_tokens"], 262144)
        self.assertEqual(status["device_map"], {"model": 0})

    @mock.patch("demo.model_manager._gpu_processes")
    def test_memory_reports_process_breakdown(self, mock_processes):
        mock_processes.return_value = [
            {"pid": 100, "gpu": 0, "used_bytes": 1024, "ours": True, "cmd": "python demo"},
            {"pid": 200, "gpu": 1, "used_bytes": 2048, "ours": False, "cmd": "other"},
        ]
        memory = self.manager.memory()
        self.assertEqual(memory["ours_vram_bytes"], 1024)
        self.assertEqual(memory["other_vram_bytes"], 2048)
        self.assertEqual(len(memory["processes"]), 2)
        self.assertFalse(memory["loaded"])

    @mock.patch.object(DemoModelManager, "_visible_gpu_count", return_value=2)
    def test_models_expose_only_catalogued_fp8_checkpoints(self, _count):
        root = Path(self.temporary.name)
        snapshot = (
            root / "models--Qwen--Qwen3-VL-8B-Thinking-FP8" / "snapshots" / "main"
        )
        snapshot.mkdir(parents=True)
        for filename in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
        ):
            (snapshot / filename).touch()
        models = self.manager.models()
        self.assertEqual([item["id"] for item in models], ["2b", "4b", "8b"])
        self.assertEqual([item["available"] for item in models], [False, False, True])
        self.assertTrue(all("FP8" in item["display_name"] for item in models))


if __name__ == "__main__":
    unittest.main()

"""L1: chat path must attach structured.overlays for coordinate skills (UI draw contract)."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from demo.generation import DemoGenerationResult
from demo.model_manager import DemoModelManager
from demo.server import create_app
from demo.sessions import SessionStore
from qwen3_vl.model_catalog import MODEL_SPECS, default_snapshot_path


def _png_bytes(w=80, h=60) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (40, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


def _sse_events(response) -> list[dict]:
    events = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class FakeRuntime:
    def __init__(self, **kwargs):
        self.model_size = kwargs["model_size"]
        self.gpu_placement = kwargs["gpu_placement"]
        self.spec = MODEL_SPECS[self.model_size]
        self.load_seconds = 0.01
        self.fp8_names = []
        self.context_mode = "native_256k"
        self.hf_device_map = {"visual": 0}
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                get_text_config=lambda: SimpleNamespace(max_position_embeddings=262_144)
            )
        )


class OverlayContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ckpt_dir = self.root / "checkpoints"
        self.state_dir = self.root / "state"
        snapshot = default_snapshot_path(self.ckpt_dir, "2b")
        snapshot.mkdir(parents=True)
        for filename in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
            (snapshot / filename).touch()

        self.manager = DemoModelManager(
            self.ckpt_dir,
            idle_seconds=0,
            runtime_factory=lambda **kw: FakeRuntime(**kw),
        )
        self.store = SessionStore(self.state_dir / "sessions.sqlite")
        self.gpu_patch = mock.patch.object(
            DemoModelManager, "_visible_gpu_count", return_value=1
        )
        self.gpu_patch.start()

        answer = (
            '[{"bbox_2d": [100, 100, 400, 400], "label": "car"}, '
            '{"point_2d": [500, 500], "label": "person"}]'
        )
        result = DemoGenerationResult(
            answer=answer,
            reasoning="think",
            finish_reason="eos",
            truncated=False,
            stopped=False,
            prompt_tokens=8,
            visual_tokens=0,
            generated_tokens=32,
            preprocess_seconds=0.01,
            generation_seconds=0.1,
            tokens_per_second=10.0,
            peak_vram_mib_per_device={},
        )

        def immediate_generation(*args, **kwargs):
            emit = args[12] if len(args) > 12 else kwargs.get("emit")
            if emit:
                emit({"type": "token", "phase": "answer", "text": answer})
            return result

        self.generation_patch = mock.patch(
            "demo.server.run_streaming_generation",
            side_effect=immediate_generation,
        )
        self.generation_patch.start()
        self.app = create_app(self.manager, self.store, self.state_dir)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.generation_patch.stop()
        self.gpu_patch.stop()
        self.temporary.cleanup()

    def test_2d_grounding_sse_includes_overlays(self):
        session = self.client.post("/api/sessions", json={"model_id": "2b"}).json()
        response = self.client.post(
            "/api/chat",
            data={
                "session_id": session["id"],
                "model_id": "2b",
                "task": "grounding_2d",
                "skill": "2d_grounding",
                "custom_prompt": "Locate cars. Report bbox coordinates in JSON format.",
                "do_sample": "true",
                "max_new_tokens": "128",
            },
            files={"files": ("scene.png", _png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        events = _sse_events(response)
        done = next((e for e in events if e.get("type") == "done"), None)
        self.assertIsNotNone(done, events)
        structured = done.get("structured")
        if structured is None and isinstance(done.get("result"), dict):
            structured = done["result"].get("structured")
        self.assertIsInstance(structured, dict, done)
        overlays = structured.get("overlays")
        self.assertIsInstance(overlays, list, structured)
        self.assertGreaterEqual(len(overlays), 1, structured)
        kinds = {o.get("kind") for o in overlays}
        self.assertTrue(kinds & {"box", "point"}, overlays)
        for o in overlays:
            for pt in o.get("pts") or []:
                self.assertGreaterEqual(pt[0], 0.0)
                self.assertLessEqual(pt[0], 1.0)


if __name__ == "__main__":
    unittest.main()

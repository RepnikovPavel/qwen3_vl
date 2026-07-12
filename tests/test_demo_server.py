import concurrent.futures
import io
import json
import threading
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
from model_catalog import MODEL_SPECS, default_snapshot_path


def _result(
    *,
    answer: str = '{"formulas":["E=mc^2"]}',
    reasoning: str | None = "I inspected the image.",
    stopped: bool = False,
) -> DemoGenerationResult:
    return DemoGenerationResult(
        answer=answer,
        reasoning=reasoning,
        finish_reason="stopped" if stopped else "eos",
        truncated=False,
        stopped=stopped,
        prompt_tokens=32,
        visual_tokens=12,
        generated_tokens=9,
        preprocess_seconds=0.125,
        generation_seconds=0.25,
        tokens_per_second=36.0,
        peak_vram_mib_per_device={"0": 128.0},
    )


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
        self.load_seconds = 0.125
        self.fp8_names = ["visual.proj", "language.layers.0.mlp"]
        self.context_mode = "native_256k"
        self.hf_device_map = {"visual": 0, "language.layers.0": 1}
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                get_text_config=lambda: SimpleNamespace(max_position_embeddings=262_144)
            )
        )


class DemoServerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ckpt_dir = self.root / "checkpoints"
        self.state_dir = self.root / "state"
        snapshot = default_snapshot_path(self.ckpt_dir, "2b")
        snapshot.mkdir(parents=True)
        for filename in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
        ):
            (snapshot / filename).touch()
        self.runtime_calls = []

        def runtime_factory(**kwargs):
            self.runtime_calls.append(kwargs)
            return FakeRuntime(**kwargs)

        self.manager = DemoModelManager(
            self.ckpt_dir,
            idle_seconds=0,
            runtime_factory=runtime_factory,
        )
        self.store = SessionStore(self.state_dir / "sessions.sqlite")
        self.gpu_patch = mock.patch.object(
            DemoModelManager,
            "_visible_gpu_count",
            return_value=2,
        )
        self.gpu_patch.start()
        self.immediate_result = _result()

        def immediate_generation(
            runtime,
            media_inputs,
            prompt,
            history,
            media_history_index,
            max_new_tokens,
            max_image_side,
            do_sample,
            temperature,
            top_p,
            top_k,
            stop_event,
            emit,
        ):
            emit(
                {
                    "type": "prompt",
                    "prompt_tokens": self.immediate_result.prompt_tokens,
                    "visual_tokens": self.immediate_result.visual_tokens,
                    "context_tokens": 262_144,
                    "preprocess_seconds": 0.125,
                }
            )
            if self.immediate_result.reasoning:
                emit(
                    {
                        "type": "token",
                        "phase": "reasoning",
                        "text": self.immediate_result.reasoning,
                    }
                )
            if self.immediate_result.answer:
                emit(
                    {
                        "type": "token",
                        "phase": "answer",
                        "text": self.immediate_result.answer,
                    }
                )
            emit(
                {
                    "type": "stats_live",
                    "generated_tokens": self.immediate_result.generated_tokens,
                    "tokens_per_second": self.immediate_result.tokens_per_second,
                    "elapsed_seconds": self.immediate_result.generation_seconds,
                }
            )
            return self.immediate_result

        self.generation_patch = mock.patch(
            "demo.server.run_streaming_generation",
            side_effect=immediate_generation,
        )
        self.generation_mock = self.generation_patch.start()
        self.app = create_app(self.manager, self.store, self.state_dir)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.generation_patch.stop()
        self.gpu_patch.stop()
        self.temporary.cleanup()

    def create_session(self, **values) -> dict:
        response = self.client.post("/api/sessions", json=values)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_health_tasks_models_and_readiness(self):
        index = self.client.get("/")
        health = self.client.get("/healthz")
        readiness = self.client.get("/readyz")
        models = self.client.get("/api/models")
        tasks = self.client.get("/api/tasks")

        self.assertEqual(index.status_code, 200)
        self.assertIn("Qwen3 VL FP8 Studio", index.text)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(readiness.status_code, 200)
        self.assertEqual(readiness.json()["status"], "ready")
        self.assertEqual(
            [item["id"] for item in models.json()["models"]], ["2b", "4b", "8b"]
        )
        self.assertEqual(
            [item["available"] for item in models.json()["models"]],
            [True, False, False],
        )
        self.assertTrue(
            all(
                item["placements"] == ["single", "balanced"]
                for item in models.json()["models"]
            )
        )
        self.assertEqual(tasks.status_code, 200)
        self.assertEqual(tasks.json()["schema_version"], 1)
        self.assertEqual(
            [item["key"] for item in tasks.json()["tasks"]],
            ["describe", "ocr", "formula", "chart", "custom"],
        )

        (default_snapshot_path(self.ckpt_dir, "2b") / "tokenizer.json").unlink()
        not_ready = self.client.get("/readyz")
        self.assertEqual(not_ready.status_code, 503)
        self.assertEqual(not_ready.json()["status"], "not_ready")

    def test_session_crud_and_reset_remove_persisted_conversation(self):
        session = self.create_session(model_id="2b", title="Initial")
        listed = self.client.get("/api/sessions")
        loaded = self.client.get(f"/api/sessions/{session['id']}")

        self.assertEqual(listed.json()["sessions"][0]["id"], session["id"])
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json()["messages"], [])
        self.assertIsNone(loaded.json()["generation"])

        renamed = self.client.patch(
            f"/api/sessions/{session['id']}",
            json={"title": "Renamed"},
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(
            self.client.get(f"/api/sessions/{session['id']}").json()["title"],
            "Renamed",
        )

        message = self.store.append_message(session["id"], "user", "Saved prompt")
        media_dir = self.state_dir / "media" / session["id"]
        media_dir.mkdir(parents=True)
        media_path = media_dir / "saved.png"
        media_path.write_bytes(b"saved")
        self.store.register_media(
            session["id"],
            message_id=message["id"],
            stored_path=media_path,
            media_type="image",
            original_name="saved.png",
            mime_type="image/png",
            size_bytes=5,
        )

        reset = self.client.post(f"/api/sessions/{session['id']}/reset")
        after_reset = self.client.get(f"/api/sessions/{session['id']}").json()

        self.assertEqual(reset.status_code, 200)
        self.assertFalse(media_path.exists())
        self.assertEqual(after_reset["title"], "New chat")
        self.assertEqual(after_reset["messages"], [])
        self.assertEqual(after_reset["media"], [])

        deleted = self.client.delete(f"/api/sessions/{session['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(
            self.client.get(f"/api/sessions/{session['id']}").status_code,
            404,
        )
        self.assertEqual(
            self.client.delete(f"/api/sessions/{session['id']}").status_code,
            404,
        )

    def test_image_upload_media_serving_and_path_hiding(self):
        session = self.create_session()
        image_buffer = io.BytesIO()
        Image.new("RGB", (13, 7), "navy").save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()

        chat = self.client.post(
            "/api/chat",
            data={
                "session_id": session["id"],
                "model_id": "2b",
                "placement": "single",
                "task": "describe",
            },
            files={"files": ("../chart.png", image_bytes, "image/png")},
        )

        self.assertEqual(chat.status_code, 200, chat.text)
        public_session = self.client.get(f"/api/sessions/{session['id']}").json()
        self.assertEqual(len(public_session["media"]), 1)
        public_media = public_session["media"][0]
        self.assertEqual(public_media["original_name"], "chart.png")
        self.assertEqual(public_media["metadata"], {"height": 7, "width": 13})
        rendered = json.dumps(public_session)
        self.assertNotIn("stored_path", rendered)
        self.assertNotIn(str(self.state_dir), rendered)

        media_response = self.client.get(f"/api/media/{public_media['id']}")
        self.assertEqual(media_response.status_code, 200)
        self.assertEqual(media_response.headers["content-type"], "image/png")
        self.assertEqual(media_response.content, image_bytes)
        media_inputs = self.generation_mock.call_args.args[1]
        self.assertEqual(media_inputs[0][0], "image")
        self.assertTrue(Path(media_inputs[0][1]).is_file())

        self.assertEqual(
            self.client.delete(f"/api/sessions/{session['id']}").status_code,
            200,
        )
        self.assertEqual(
            self.client.get(f"/api/media/{public_media['id']}").status_code,
            404,
        )

    def test_chat_sse_persists_answer_reasoning_metrics_and_structured_result(self):
        session = self.create_session(model_id="2b")

        response = self.client.post(
            "/api/chat",
            data={
                "session_id": session["id"],
                "model_id": "2b",
                "placement": "balanced",
                "task": "formula",
                "max_new_tokens": "512",
                "max_image_side": "900",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream")
        )
        self.assertEqual(response.headers["x-session-id"], session["id"])
        events = _sse_events(response)
        self.assertEqual(events[0]["type"], "snapshot")
        terminal = next(
            event
            for event in reversed(events)
            if event.get("type") in {"done", "snapshot"}
            and event.get("result") is not None
        )
        self.assertEqual(terminal["result"]["finish_reason"], "eos")
        self.assertEqual(terminal["result"]["generated_tokens"], 9)
        self.assertTrue(terminal["structured"]["strict_schema_valid"])
        self.assertEqual(terminal["structured"]["value"], {"formulas": ["E=mc^2"]})

        persisted = self.client.get(f"/api/sessions/{session['id']}").json()
        self.assertEqual(
            [message["role"] for message in persisted["messages"]],
            ["user", "assistant"],
        )
        assistant = persisted["messages"][1]
        self.assertEqual(assistant["content"], self.immediate_result.answer)
        self.assertEqual(assistant["reasoning"], self.immediate_result.reasoning)
        self.assertEqual(assistant["metrics"]["generation"]["finish_reason"], "eos")
        self.assertEqual(
            assistant["metrics"]["structured"]["value"],
            {"formulas": ["E=mc^2"]},
        )
        generation_call = self.generation_mock.call_args.args
        self.assertIn("Transcribe every displayed formula", generation_call[2])
        self.assertIn('{"formulas":["latex","..."]}', generation_call[2])
        self.assertEqual(generation_call[5], 512)
        self.assertEqual(generation_call[6], 900)
        status = self.client.get("/api/status").json()
        self.assertFalse(status["loaded"])
        self.assertFalse(status["keep_model_loaded"])
        self.assertEqual(status["unload_policy"], "after_generation")

    def test_busy_manager_returns_409_without_starting_generation(self):
        session = self.create_session()
        image_buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(image_buffer, format="PNG")
        self.manager.acquire()
        try:
            chat = self.client.post(
                "/api/chat",
                data={
                    "session_id": session["id"],
                    "model_id": "2b",
                    "task": "describe",
                },
                files={"files": ("busy.png", image_buffer.getvalue(), "image/png")},
            )
            load = self.client.post(
                "/api/load",
                json={"model_id": "2b", "placement": "single"},
            )
            unload = self.client.post("/api/unload")
        finally:
            self.manager.release()

        self.assertEqual(chat.status_code, 409)
        self.assertEqual(load.status_code, 409)
        self.assertEqual(unload.status_code, 409)
        self.assertEqual(self.generation_mock.call_count, 0)
        self.assertEqual(self.store.get_session(session["id"])["messages"], [])
        self.assertEqual(self.store.get_session(session["id"])["media"], [])

    def test_session_model_provenance_cannot_be_changed(self):
        session = self.create_session(model_id="2b")
        response = self.client.post(
            "/api/chat",
            data={
                "session_id": session["id"],
                "model_id": "4b",
                "task": "describe",
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("session model differs", response.json()["detail"])
        self.assertEqual(self.generation_mock.call_count, 0)

    def test_stop_and_reattach_return_snapshot_and_persist_stopped_result(self):
        session = self.create_session()
        started = threading.Event()

        def blocked_generation(
            runtime,
            media_inputs,
            prompt,
            history,
            media_history_index,
            max_new_tokens,
            max_image_side,
            do_sample,
            temperature,
            top_p,
            top_k,
            stop_event,
            emit,
        ):
            emit({"type": "token", "phase": "reasoning", "text": "Working"})
            started.set()
            if not stop_event.wait(timeout=5):
                raise RuntimeError("stop was not requested")
            return _result(answer="Partial", reasoning="Working", stopped=True)

        self.generation_mock.side_effect = blocked_generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            chat_future = executor.submit(
                self.client.post,
                "/api/chat",
                data={
                    "session_id": session["id"],
                    "model_id": "2b",
                    "task": "describe",
                },
            )
            self.assertTrue(started.wait(timeout=5))
            status = self.client.get("/api/status")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(
                status.json()["active_generations"][0]["session_id"],
                session["id"],
            )
            reattach_future = executor.submit(
                self.client.get,
                f"/api/stream/{session['id']}",
            )
            stopped = self.client.post(f"/api/stop/{session['id']}")
            chat = chat_future.result(timeout=5)
            reattached = reattach_future.result(timeout=5)

        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(stopped.json(), {"ok": True})
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(reattached.status_code, 200)
        events = _sse_events(reattached)
        self.assertEqual(events[0]["type"], "snapshot")
        terminal = next(
            event
            for event in reversed(events)
            if event.get("type") in {"done", "snapshot"}
            and event.get("result") is not None
        )
        self.assertTrue(terminal["result"]["stopped"])
        self.assertEqual(terminal["result"]["finish_reason"], "stopped")
        self.assertEqual(
            self.client.post(f"/api/stop/{session['id']}").json(),
            {"ok": False, "reason": "no active generation"},
        )
        persisted = self.store.get_session(session["id"])
        self.assertEqual(persisted["messages"][1]["content"], "Partial")
        self.assertTrue(persisted["messages"][1]["metrics"]["generation"]["stopped"])

    def test_load_and_retention_use_real_manager_with_fake_runtime(self):
        loaded = self.client.post(
            "/api/load",
            json={"model_id": "4b", "placement": "balanced"},
        )

        self.assertEqual(loaded.status_code, 200, loaded.text)
        self.assertEqual(
            loaded.json(),
            {
                "ok": True,
                "model_id": "4b",
                "repo_id": MODEL_SPECS["4b"].repo_id,
                "placement": "balanced",
                "load_seconds": 0.125,
                "keep_model_loaded": False,
                "unloaded": True,
            },
        )
        self.assertEqual(self.runtime_calls[0]["device"], "cuda")
        self.assertEqual(self.runtime_calls[0]["gpu_placement"], "balanced")
        status = self.client.get("/api/status").json()
        self.assertFalse(status["loaded"])
        self.assertTrue(status["auto_unload_after_generation"])

        retained = self.client.post(
            "/api/load",
            json={
                "model_id": "4b",
                "placement": "balanced",
                "keep_model_loaded": True,
            },
        )
        self.assertFalse(retained.json()["unloaded"])
        status = self.client.get("/api/status").json()
        self.assertTrue(status["loaded"])
        self.assertTrue(status["keep_model_loaded"])
        self.assertEqual(status["context_tokens"], 262_144)
        self.assertEqual(status["device_map"], {"visual": 0, "language.layers.0": 1})

        released = self.client.post(
            "/api/retention",
            json={"keep_model_loaded": False},
        )
        self.assertEqual(released.status_code, 200, released.text)
        self.assertFalse(released.json()["loaded"])
        self.assertFalse(released.json()["keep_model_loaded"])


if __name__ == "__main__":
    unittest.main()

"""Gateway tests for the multi-task worker-pool path (no real GPU/model).

We inject a fake pool to exercise the /api/jobs, /api/stop_job, and
/api/jobs/{id}/stream endpoints plus the /api/chat pool branch, without
spawning worker processes or loading the model. The real worker process +
loop detector are covered by the e2e GPU smoke (scripts/) and the unit tests.
"""

from __future__ import annotations

import threading
import time
import unittest
from collections import deque
from pathlib import Path

from fastapi.testclient import TestClient

from demo.server import create_app
from demo.model_manager import DemoModelManager
from demo.sessions import SessionStore


class _FakePool:
    """Minimal stand-in for WorkerPool: records submits, serves queued events."""

    def __init__(self) -> None:
        self.jobs_seen: list[dict] = []
        self._stop_calls: list[str] = []
        self._buffers: dict[str, deque] = {}
        self._conds: dict[str, threading.Condition] = {}
        self._done: dict[str, bool] = {}

    # API the gateway uses:
    def submit(self, *, session_id, task, skill, spec) -> str:
        import uuid

        job_id = uuid.uuid4().hex
        self.jobs_seen.append(
            {"job_id": job_id, "session_id": session_id, "task": task,
             "skill": skill, "spec": spec}
        )
        self._buffers[job_id] = deque()
        self._conds[job_id] = threading.Condition()
        self._done[job_id] = False
        return job_id

    def stop(self, job_id: str) -> bool:
        self._stop_calls.append(job_id)
        return True

    def jobs(self) -> list[dict]:
        return [
            {"job_id": j["job_id"], "session_id": j["session_id"],
             "task": j["task"], "skill": j["skill"], "state": "queued",
             "stop_requested": False, "created_at": time.time(),
             "age_seconds": 0.0, "generated_tokens": 0,
             "finish_reason": None, "error": None}
            for j in self.jobs_seen
        ]

    def replay_snapshot(self, job_id: str):
        if job_id not in self._buffers:
            return None
        return (list(self._buffers[job_id]), len(self._buffers[job_id]))

    def events(self, job_id: str, after_seq: int = 0, timeout=None):
        cond = self._conds[job_id]
        buf = self._buffers[job_id]
        cursor = after_seq
        while True:
            ready = [e for e in buf if e["seq"] > cursor]
            if ready:
                cursor = ready[-1]["seq"]
                for e in ready:
                    yield e["data"]
                if self._done.get(job_id) and cursor >= len(buf):
                    return
                continue
            if self._done.get(job_id):
                return
            with cond:
                cond.wait(timeout=0.2)

    # Test helper: push an event then a terminal done.
    def emit(self, job_id: str, event: dict) -> None:
        buf = self._buffers[job_id]
        buf.append({"seq": len(buf) + 1, "data": event})
        with self._conds[job_id]:
            self._conds[job_id].notify_all()

    def finish(self, job_id: str, answer: str = "ok") -> None:
        self.emit(job_id, {
            "type": "done", "answer": answer, "reasoning": None,
            "finish_reason": "eos", "generated_tokens": 5,
            "tokens_per_second": 10.0, "loop_detected": False,
        })
        self._done[job_id] = True
        with self._conds[job_id]:
            self._conds[job_id].notify_all()


def _make_app(fake_pool: _FakePool | None = None):
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="qwen3_pool_test_"))
    manager = DemoModelManager(tmp / "models", None, idle_seconds=0)
    store = SessionStore(tmp / "sessions.sqlite")
    return create_app(manager, store, tmp, pool=fake_pool), tmp


class PoolGatewayTest(unittest.TestCase):
    def test_jobs_endpoint_reports_disabled_without_pool(self):
        app, _tmp = _make_app(fake_pool=None)
        client = TestClient(app)
        resp = client.get("/api/jobs")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["enabled"])

    def test_jobs_endpoint_lists_pool_jobs(self):
        pool = _FakePool()
        app, _tmp = _make_app(fake_pool=pool)
        client = TestClient(app)
        pool.submit(session_id="s1", task="describe", skill=None,
                    spec={"prompt": "hi"})
        resp = client.get("/api/jobs")
        self.assertTrue(resp.json()["enabled"])
        self.assertEqual(len(resp.json()["jobs"]), 1)
        self.assertEqual(resp.json()["jobs"][0]["task"], "describe")

    def test_stop_job_calls_pool_stop(self):
        pool = _FakePool()
        app, _tmp = _make_app(fake_pool=pool)
        client = TestClient(app)
        job_id = pool.submit(session_id="s1", task="describe", skill=None,
                             spec={"prompt": "hi"})
        resp = client.post(f"/api/stop_job/{job_id}")
        self.assertTrue(resp.json()["ok"])
        self.assertEqual(pool._stop_calls, [job_id])

    def test_chat_uses_pool_and_streams_job(self):
        pool = _FakePool()
        app, tmp = _make_app(fake_pool=pool)
        client = TestClient(app)
        # Create a session first.
        sess = client.post("/api/sessions", json={"model_id": "2b"})
        session_id = sess.json()["id"]

        # Start chat in a thread (it streams until 'done').
        result_holder: dict = {}

        def run_chat() -> None:
            resp = client.post(
                "/api/chat",
                data={"session_id": session_id, "model_id": "2b",
                      "placement": "single", "task": "describe",
                      "custom_prompt": "hello"},
            )
            result_holder["status"] = resp.status_code
            result_holder["body"] = resp.text

        # Give the pool no workers that finish on their own; we finish manually.
        t = threading.Thread(target=run_chat)
        t.start()
        # Wait for submit to land.
        deadline = time.monotonic() + 5
        while not pool.jobs_seen and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(pool.jobs_seen, "pool.submit was not called")
        job_id = pool.jobs_seen[-1]["job_id"]
        pool.finish(job_id, answer="A short answer.")
        t.join(timeout=10)
        self.assertEqual(result_holder.get("status"), 200)
        self.assertIn("job", result_holder["body"])
        self.assertIn("A short answer.", result_holder["body"])

        # The completed turn must be persisted to the session.
        msgs = client.get(f"/api/sessions/{session_id}").json()["messages"]
        self.assertTrue(any("A short answer." in (m.get("content") or "")
                            for m in msgs), msgs)

    def test_pool_chat_does_not_delete_uploaded_media(self):
        # Regression: the pool path returns before `handed_to_worker` was set in
        # the old in-process flow, so the request `finally` reset the session
        # and deleted the uploaded image right after submit. The worker then
        # failed with "local image does not exist". The media must survive.
        pool = _FakePool()
        app, tmp = _make_app(fake_pool=pool)
        client = TestClient(app)
        sess = client.post("/api/sessions", json={"model_id": "2b"})
        session_id = sess.json()["id"]

        # A tiny valid PNG (1x1) as the upload.
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="PNG")
        buf.seek(0)

        result_holder: dict = {}

        def run_chat() -> None:
            resp = client.post(
                "/api/chat",
                data={"session_id": session_id, "model_id": "2b",
                      "placement": "single", "task": "describe",
                      "custom_prompt": "what is this"},
                files={"files": ("x.png", buf, "image/png")},
            )
            result_holder["status"] = resp.status_code

        t = threading.Thread(target=run_chat)
        t.start()
        deadline = time.monotonic() + 5
        while not pool.jobs_seen and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(pool.jobs_seen, "pool.submit was not called")
        # The submitted spec must carry a media path that still exists on disk.
        spec = pool.jobs_seen[-1]["spec"]
        media = spec.get("media_inputs") or []
        self.assertTrue(media, "media not forwarded to the pool")
        kind, path = media[0]
        self.assertEqual(kind, "image")
        self.assertTrue(Path(path).is_file(), f"uploaded media deleted: {path}")
        pool.finish(pool.jobs_seen[-1]["job_id"])
        t.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

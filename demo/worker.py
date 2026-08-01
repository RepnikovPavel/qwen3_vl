"""Resident FP8 inference worker process (one model, pinned to one GPU).

The demo gateway spawns one ``python3 -m demo.worker`` per visible GPU (each
with ``CUDA_VISIBLE_DEVICES`` set to a single card). The worker loads a single
``Qwen3VLRuntime`` once and keeps it resident — model load is the expensive
step (~2-3 s), so we never repeat it between jobs.

IPC is deliberately minimal and robust: JSON-lines over stdio.

* Commands arrive on **stdin**, one JSON object per line:
    {"cmd": "RUN", "job_id": "...", "media_inputs": [["image","/path"]],
     "prompt": "...", "history": [{"role","content"}],
     "media_history_index": 0, "params": {"max_new_tokens":..., ...}}
    {"cmd": "STOP", "job_id": "..."}
  STOP is best-effort: it sets the running generation's stop_event so HF's
  StoppingCriteria halts within ~1 token. (A wedged worker is reclaimed by the
  gateway killing the process — see worker_pool.py.)

* Events leave on **stdout**, one JSON object per line:
    {"type": "READY", "gpu": <index>, "pid": <pid>}
    {"job_id": "...", "type": "loading"|"prompt"|"token"|"stats_live"|
                              "loop_detected"|"done"|"error", ...}
  A ``done``/``error`` event closes a job. Each event is flushed immediately so
  the gateway can stream it to the client with low latency.

* All non-protocol logging goes to **stderr** (never stdout), so the JSON-lines
  channel stays clean.

The thinking trace (full raw model output, including ``<think>``) is written to
``<logs_dir>/<job_id>.jsonl`` on job completion, for incident analysis and skill
verification. Logs never go to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# The offline runtime sets strict-offline env + a network audit hook at import
# time. Import it once at module load so the env is locked before torch.
from qwen3_vl.qwen3_vl_offline import Qwen3VLRuntime  # noqa: F401  (side effects)
from qwen3_vl.loop_detector import LoopDetector, LoopVerdict

from demo.generation import run_streaming_generation, DemoGenerationResult


def _emit_event(event: dict[str, Any]) -> None:
    """Write one JSON event line to stdout and flush."""
    sys.stdout.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _log(message: str) -> None:
    """Diagnostic logging to stderr only (keep stdout = JSON protocol)."""
    sys.stderr.write(message.rstrip() + "\n")
    sys.stderr.flush()


class _Worker:
    def __init__(self, *, ckpt_dir: str, kernel_dir: str | None, logs_dir: Path,
                 model_size: str = "2b") -> None:
        self.ckpt_dir = ckpt_dir
        self.kernel_dir = kernel_dir
        self.logs_dir = logs_dir
        self.model_size = model_size
        self.runtime: Qwen3VLRuntime | None = None
        # The active job: only one generation runs at a time per worker.
        self._active_job: str | None = None
        self._stop_event: threading.Event | None = None
        self._job_thread: threading.Thread | None = None
        self._cmd_lock = threading.Lock()

    # --- lifecycle ----------------------------------------------------------

    def load_runtime(self) -> None:
        _log(f"[worker] loading Qwen3-VL {self.model_size} FP8 from {self.ckpt_dir}")
        load_started = time.monotonic()
        self.runtime = Qwen3VLRuntime(
            model_size=self.model_size,
            device="cuda",
            ckpt_dir=self.ckpt_dir,
            kernel_dir=self.kernel_dir,
            gpu_placement="single",
            yarn_1m=True,
        )
        elapsed = round(time.monotonic() - load_started, 2)
        _log(f"[worker] runtime ready in {elapsed}s")

    def announce_ready(self) -> None:
        gpu = _visible_gpu_index()
        _emit_event({"type": "READY", "gpu": gpu, "pid": os.getpid()})

    # --- command loop -------------------------------------------------------

    def run_command_loop(self) -> None:
        """Read JSON-lines commands from stdin until EOF."""
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as exc:
                _log(f"[worker] ignoring malformed command: {exc}")
                continue
            kind = cmd.get("cmd")
            if kind == "RUN":
                self._handle_run(cmd)
            elif kind == "STOP":
                self._handle_stop(cmd)
            elif kind == "PING":
                _emit_event({"type": "PONG", "ts": time.time()})
            else:
                _log(f"[worker] unknown command {kind!r}")
        _log("[worker] stdin EOF, exiting command loop")

    def _handle_stop(self, cmd: dict[str, Any]) -> None:
        job_id = cmd.get("job_id")
        with self._cmd_lock:
            if job_id == self._active_job and self._stop_event is not None:
                self._stop_event.set()
                _log(f"[worker] stop requested for job {job_id}")

    def _handle_run(self, cmd: dict[str, Any]) -> None:
        job_id = cmd.get("job_id") or _new_id()
        with self._cmd_lock:
            if self._active_job is not None:
                # A worker runs one job at a time; the gateway must not send a
                # second RUN while one is active. Reject so the gateway can
                # re-queue.
                _emit_event({
                    "job_id": job_id, "type": "error",
                    "message": "worker already busy with another job",
                })
                return
            self._active_job = job_id
            stop_event = threading.Event()
            self._stop_event = stop_event

        # Run the generation on a separate thread so the command loop keeps
        # reading stdin and can process a STOP mid-generation. (model.generate
        # blocks; if it ran on the command-loop thread, STOP would sit unread
        # in the pipe until the generation finished — defeating cancellation.)
        def _execute() -> None:
            try:
                self._run_job(job_id, cmd, stop_event)
            except BaseException as exc:  # noqa: BLE001 — surface any failure
                _log(f"[worker] job {job_id} crashed: {exc}\n{traceback.format_exc()}")
                _emit_event({"job_id": job_id, "type": "error", "message": str(exc)})
            finally:
                with self._cmd_lock:
                    if self._active_job == job_id:
                        self._active_job = None
                        self._stop_event = None

        job_thread = threading.Thread(target=_execute, daemon=True)
        with self._cmd_lock:
            self._job_thread = job_thread
        job_thread.start()

    # --- the actual generation ---------------------------------------------

    def _run_job(self, job_id: str, cmd: dict[str, Any],
                 stop_event: threading.Event) -> None:
        assert self.runtime is not None
        params = dict(cmd.get("params") or {})
        media_inputs = cmd.get("media_inputs") or []
        prompt = cmd.get("prompt") or ""
        history = cmd.get("history") or []
        media_history_index = cmd.get("media_history_index")
        skill = cmd.get("skill")  # for logging only
        loop = LoopDetector()

        def emit(event: dict[str, Any]) -> None:
            # Route token deltas through the loop detector; on a hit, stop.
            if event.get("type") == "token":
                verdict = loop.feed(event.get("text") or "")
                if verdict.detected and not stop_event.is_set():
                    stop_event.set()
                    _emit_event({
                        "job_id": job_id, "type": "loop_detected",
                        "reason": verdict.reason, "detail": verdict.detail,
                        "generated_chars": verdict.generated_chars,
                    })
            out = {"job_id": job_id, **event}
            _emit_event(out)

        _emit_event({"job_id": job_id, "type": "loading", "state": "generating"})
        started = time.monotonic()
        result: DemoGenerationResult = run_streaming_generation(
            self.runtime,
            media_inputs,
            prompt,
            history,
            media_history_index,
            int(params.get("max_new_tokens", 2048)),
            int(params.get("max_image_side", 0)),
            bool(params.get("do_sample", True)),
            float(params.get("temperature", 0.6)),
            float(params.get("top_p", 0.95)),
            int(params.get("top_k", 20)),
            stop_event,
            emit,
            video_num_frames=int(params.get("video_num_frames", 32)),
        )
        elapsed = round(time.monotonic() - started, 2)
        self._write_thinking_log(job_id, prompt, media_inputs, history, params,
                                 skill, result, loop.verdict)
        _emit_event({
            "job_id": job_id, "type": "done",
            "answer": result.answer, "reasoning": result.reasoning,
            "finish_reason": result.finish_reason, "stopped": result.stopped,
            "truncated": result.truncated,
            "prompt_tokens": result.prompt_tokens,
            "visual_tokens": result.visual_tokens,
            "generated_tokens": result.generated_tokens,
            "tokens_per_second": round(result.tokens_per_second, 2),
            "elapsed_seconds": elapsed,
            "peak_vram_mib_per_device": result.peak_vram_mib_per_device,
            "loop_detected": loop.verdict.detected if loop.verdict else False,
        })

    def _write_thinking_log(
        self, job_id: str, prompt: str, media_inputs: list, history: list,
        params: dict, skill: str | None, result: DemoGenerationResult,
        loop_verdict: LoopVerdict | None,
    ) -> None:
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "job_id": job_id, "skill": skill,
                "ts": time.time(),
                "prompt": prompt,
                "media_inputs": [
                    {"kind": k, "path": str(v)} for k, v in media_inputs
                ],
                "history": history, "params": params,
                "answer": result.answer, "reasoning": result.reasoning,
                "finish_reason": result.finish_reason,
                "generated_tokens": result.generated_tokens,
                "tokens_per_second": result.tokens_per_second,
                "loop_detected": bool(loop_verdict and loop_verdict.detected),
                "loop_reason": loop_verdict.reason if loop_verdict else None,
                "loop_detail": loop_verdict.detail if loop_verdict else None,
            }
            path = self.logs_dir / f"{job_id}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            _log(f"[worker] could not write thinking log for {job_id}: {exc}")


def _visible_gpu_index() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        pass
    return None


def _new_id() -> str:
    import uuid
    return uuid.uuid4().hex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resident FP8 inference worker.")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--kernel-dir", default=os.environ.get("QWEN3_FP8_KERNEL_DIR"))
    parser.add_argument("--logs-dir", default=os.environ.get("DEMO_THINKING_LOGS_DIR", ""))
    parser.add_argument("--model", default=os.environ.get("DEMO_WORKER_MODEL", "2b"))
    args = parser.parse_args(argv)

    logs_dir = Path(args.logs_dir).expanduser() if args.logs_dir else Path("/tmp/qwen3_vl_thinking_logs")
    worker = _Worker(
        ckpt_dir=args.ckpt_dir, kernel_dir=args.kernel_dir or None,
        logs_dir=logs_dir, model_size=args.model,
    )
    worker.load_runtime()
    worker.announce_ready()
    worker.run_command_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

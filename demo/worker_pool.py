"""Worker process pool + job queue for the demo gateway.

Replaces the old single global ``DemoModelManager._lease`` (which serialized
every request and wedged forever if a generation hung). The pool runs N worker
processes (one resident FP8 runtime per visible GPU) and dispatches jobs to
free workers, queueing the rest.

Key properties:
- **Concurrency**: up to ``len(gpus) * workers_per_gpu`` jobs run in parallel,
  one runtime each. The rest wait in a FIFO queue (state ``queued``).
- **Cancellation**: ``stop(job_id)`` sends a STOP command to the running
  worker (soft stop via the generation stop_event, ~1 token latency). Queued
  jobs are simply removed from the queue.
- **Reclaim on hang**: if a worker emits no event for ``stall_timeout`` seconds
  after a loop is detected (or otherwise stalls), the gateway kills and
  restarts the worker process — the only way to reclaim a wedged CUDA
  generation, since Python threads cannot be killed.
- **Per-job event buffers**: each job has a bounded deque + Condition (same
  pattern as ``Generation``), so the gateway can serve a reconnectable SSE
  stream from any thread.
- **Job registry**: ``jobs()`` returns running + queued jobs for the UI panel.

The pool is process-local (one gateway process). Workers communicate over
stdio JSON-lines (see demo/worker.py).
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# Job lifecycle states.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
STOPPED = "stopped"
LOOPED = "looped"

_ACTIVE = {QUEUED, RUNNING}


_EVENT_BUFFER_MAXLEN = 4096


@dataclass(slots=True)
class _Job:
    job_id: str
    session_id: str
    task: str
    skill: str | None
    created_at: float
    spec: dict[str, Any]
    state: str = QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    stop_requested: bool = False
    error: str | None = None
    finish_reason: str | None = None
    generated_tokens: int = 0
    progress_seq: int = 0  # last event sequence the job produced
    # Event buffer for the gateway SSE consumer (snapshot + deltas).
    _events: deque = field(
        default_factory=lambda: collections.deque(maxlen=_EVENT_BUFFER_MAXLEN)
    )
    _cond: threading.Condition = field(default_factory=threading.Condition)
    _done: bool = False


class _WorkerHandle:
    """One resident worker subprocess + its assignment."""

    def __init__(self, pool: "WorkerPool", gpu_index: int) -> None:
        self.pool = pool
        self.gpu_index = gpu_index
        self.proc: subprocess.Popen | None = None
        self.ready = False
        self.current_job: _Job | None = None
        self.last_event_at: float = 0.0
        self.lock = threading.Lock()  # guards current_job assignment + writes to proc.stdin
        self._reader: threading.Thread | None = None
        self._restarting = False

    def start(self) -> None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        cmd = [
            sys.executable, "-u", "-m", "demo.worker",
            "--ckpt-dir", self.pool.ckpt_dir,
            "--logs-dir", str(self.pool.logs_dir),
            "--model", self.pool.model_size,
        ]
        if self.pool.kernel_dir:
            cmd += ["--kernel-dir", self.pool.kernel_dir]
        self.proc = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1, text=True,
        )
        self.last_event_at = time.monotonic()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            # Worker stderr is diagnostics only; keep it out of the protocol.
            if self.pool.verbose:
                sys.stderr.write(f"[worker gpu{self.gpu_index}] {line}")

    def _read_loop(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.last_event_at = time.monotonic()
            self.pool._handle_worker_event(self, event)

    def send(self, command: dict[str, Any]) -> bool:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return False
        try:
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def terminate(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            self.proc = None
            self.ready = False


class WorkerPool:
    """A pool of resident worker processes with a job queue."""

    def __init__(
        self,
        *,
        ckpt_dir: str,
        kernel_dir: str | None,
        logs_dir: Path,
        gpu_ids: list[int],
        model_size: str = "2b",
        workers_per_gpu: int = 1,
        stall_timeout: float = 60.0,
        verbose: bool = False,
    ) -> None:
        if not gpu_ids:
            raise ValueError("WorkerPool needs at least one GPU id")
        if workers_per_gpu < 1:
            raise ValueError("workers_per_gpu must be >= 1")
        self.ckpt_dir = ckpt_dir
        self.kernel_dir = kernel_dir
        self.logs_dir = logs_dir
        self.model_size = model_size
        self.stall_timeout = stall_timeout
        self.verbose = verbose
        self._handles: list[_WorkerHandle] = [
            _WorkerHandle(self, gpu) for gpu in gpu_ids for _ in range(workers_per_gpu)
        ]
        self._jobs: dict[str, _Job] = {}
        self._queue: collections.deque[str] = collections.deque()
        self._lock = threading.RLock()
        self._closed = False
        self._watcher: threading.Thread | None = None

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for handle in self._handles:
            handle.start()
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher.start()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            handles = list(self._handles)
            jobs = list(self._jobs.values())
        for job in jobs:
            if not job._done:
                self._fail_job(job, "server shutting down")
        for handle in handles:
            handle.terminate()

    # --- public API: submit / stop / jobs / events --------------------------

    def submit(self, *, session_id: str, task: str, skill: str | None,
               spec: dict[str, Any]) -> str:
        """Enqueue a generation job. Returns the job_id (immediately)."""
        job_id = uuid.uuid4().hex
        job = _Job(
            job_id=job_id, session_id=session_id, task=task, skill=skill,
            created_at=time.time(), spec=spec,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("worker pool is shut down")
            self._jobs[job_id] = job
            self._queue.append(job_id)
        self._dispatch()
        return job_id

    def stop(self, job_id: str) -> bool:
        """Cancel a job: remove if queued, else soft-stop the running worker."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job._done:
                return False
            job.stop_requested = True
            if job.state == QUEUED:
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                job.state = STOPPED
                self._finish_job(job, finish_reason="stopped")
                return True
        # RUNNING: send STOP to the owning worker.
        handle = self._find_owner(job_id)
        if handle is not None:
            handle.send({"cmd": "STOP", "job_id": job_id})
            return True
        return False

    def get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._jobs.values())
        now = time.time()
        out = []
        for job in items:
            out.append({
                "job_id": job.job_id, "session_id": job.session_id,
                "task": job.task, "skill": job.skill, "state": job.state,
                "stop_requested": job.stop_requested,
                "created_at": job.created_at,
                "age_seconds": round(now - job.created_at, 1),
                "generated_tokens": job.generated_tokens,
                "finish_reason": job.finish_reason,
                "error": job.error,
            })
        return out

    def active_jobs(self) -> list[dict[str, Any]]:
        return [j for j in self.jobs() if j["state"] in _ACTIVE]

    def events(self, job_id: str, after_seq: int = 0,
               timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """Yield events for a job, optionally blocking for new ones.

        Used by the gateway SSE endpoint. Reconnect-safe: starts from
        ``after_seq`` (the snapshot sequence) and replays buffered events.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        cursor = after_seq
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                ready = [ev for ev in job._events if ev["seq"] > cursor]
                if ready:
                    cursor = ready[-1]["seq"]
                    for ev in ready:
                        yield ev["data"]
                    if job._done and cursor >= job.progress_seq:
                        return
                    continue
                if job._done:
                    return
                cond = job._cond
            # Wait outside the pool lock for new events.
            if deadline is not None and time.monotonic() >= deadline:
                return
            with cond:
                cond.wait(timeout=0.5)

    def replay_snapshot(self, job_id: str) -> tuple[list[dict[str, Any]], int] | None:
        """Return (snapshot events, last_seq) for a job, or None if unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return list(ev["data"] for ev in job._events), job.progress_seq

    # --- internal: dispatch / event handling / watcher ---------------------

    def _dispatch(self) -> None:
        with self._lock:
            if self._closed:
                return
            while self._queue:
                handle = self._next_free_handle()
                if handle is None:
                    return
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None or job._done:
                    continue
                with handle.lock:
                    if not handle.ready or handle.current_job is not None or handle.proc is None:
                        # Not actually free; requeue and stop.
                        self._queue.appendleft(job_id)
                        return
                    handle.current_job = job
                    job.state = RUNNING
                    job.started_at = time.time()
                    ok = handle.send({"cmd": "RUN", "job_id": job_id, **job.spec})
                if not ok:
                    with handle.lock:
                        handle.current_job = None
                    self._fail_job(job, "worker process not reachable")
                    continue
                self._emit(job, {"type": "loading", "state": "running"})
                break  # one dispatch per call; loop will call again

    def _next_free_handle(self) -> _WorkerHandle | None:
        for handle in self._handles:
            if handle.ready and handle.current_job is None and handle.proc is not None:
                return handle
        return None

    def _find_owner(self, job_id: str) -> _WorkerHandle | None:
        for handle in self._handles:
            if handle.current_job is not None and handle.current_job.job_id == job_id:
                return handle
        return None

    def _handle_worker_event(self, handle: _WorkerHandle, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "READY":
            handle.ready = True
            self._dispatch()
            return
        if etype == "PONG":
            return
        job_id = event.get("job_id")
        if not job_id:
            return
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return
        self._emit(job, {k: v for k, v in event.items() if k != "job_id"})
        # Track progress + completion.
        if etype == "stats_live":
            with self._lock:
                job.generated_tokens = int(event.get("generated_tokens", job.generated_tokens))
        elif etype == "loop_detected":
            # Soft stop was already issued by the worker. Note it; the watcher
            # will hard-reclaim if the worker then stalls.
            pass
        elif etype == "done":
            with handle.lock:
                if handle.current_job is job:
                    handle.current_job = None
            with self._lock:
                job.finish_reason = event.get("finish_reason")
                if event.get("loop_detected"):
                    job.state = LOOPED
                elif event.get("finish_reason") == "stopped" or job.stop_requested:
                    job.state = STOPPED
                else:
                    job.state = DONE
            self._finish_job(job, finish_reason=job.finish_reason)
            self._dispatch()
        elif etype == "error":
            with handle.lock:
                if handle.current_job is not None and handle.current_job.job_id == job_id:
                    handle.current_job = None
            self._fail_job(job, event.get("message", "worker error"))
            self._dispatch()

    def _emit(self, job: _Job, event: dict[str, Any]) -> None:
        with self._lock:
            job.progress_seq += 1
            job._events.append({"seq": job.progress_seq, "data": event})
            with job._cond:
                job._cond.notify_all()

    def _finish_job(self, job: _Job, *, finish_reason: str | None) -> None:
        with self._lock:
            if job._done:
                return
            job._done = True
            job.finished_at = time.time()
            if job.state not in {DONE, FAILED, STOPPED, LOOPED}:
                job.state = DONE
            job.finish_reason = finish_reason or job.finish_reason
            # Emit a terminal marker the SSE loop can rely on.
            job.progress_seq += 1
            job._events.append({
                "seq": job.progress_seq,
                "data": {"type": "_closed", "state": job.state,
                         "finish_reason": job.finish_reason},
            })
            with job._cond:
                job._cond.notify_all()
        self._prune()

    def _fail_job(self, job: _Job, message: str) -> None:
        with self._lock:
            job.error = message
        self._emit(job, {"type": "error", "message": message})
        with self._lock:
            job.state = FAILED
        self._finish_job(job, finish_reason="error")

    def _watch_loop(self) -> None:
        """Reclaim wedged workers and dispatch when workers become free."""
        while not self._closed:
            time.sleep(5.0)
            now = time.monotonic()
            for handle in list(self._handles):
                proc = handle.proc
                if proc is None:
                    continue
                if proc.poll() is not None:
                    # Worker died; fail its job and restart.
                    self._handle_dead_worker(handle)
                    continue
                # Stall check: a running job that produced no event for a while.
                with handle.lock:
                    job = handle.current_job
                    idle = now - handle.last_event_at
                if job is not None and idle > self.stall_timeout and not handle._restarting:
                    sys.stderr.write(
                        f"[pool] worker gpu{handle.gpu_index} stalled "
                        f"({idle:.0f}s) for job {job.job_id}; reclaiming\n")
                    self._reclaim_worker(handle, job, "worker stalled (generation hung)")

    def _handle_dead_worker(self, handle: _WorkerHandle) -> None:
        with handle.lock:
            job = handle.current_job
            handle.current_job = None
            handle.ready = False
        if job is not None:
            self._fail_job(job, "worker process died")
        # Restart the worker so the slot comes back.
        handle._restarting = True
        try:
            handle.start()
        finally:
            handle._restarting = False
        self._dispatch()

    def _reclaim_worker(self, handle: _WorkerHandle, job: _Job, reason: str) -> None:
        handle._restarting = True
        try:
            handle.terminate()
            with handle.lock:
                handle.current_job = None
                handle.ready = False
            self._fail_job(job, reason)
            handle.start()
        finally:
            handle._restarting = False
        self._dispatch()

    def _prune(self) -> None:
        """Drop finished jobs older than the retention window."""
        cutoff = time.time() - 600
        with self._lock:
            stale = [jid for jid, job in self._jobs.items()
                     if job._done and (job.finished_at or 0) < cutoff]
            for jid in stale:
                self._jobs.pop(jid, None)

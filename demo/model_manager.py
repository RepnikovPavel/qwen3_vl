from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from model_catalog import (
    MODEL_SPECS,
    default_snapshot_path,
    get_model_spec,
    normalize_model_size,
)


PLACEMENTS = ("single", "balanced")


class DemoBusyError(RuntimeError):
    pass


def _current_rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return 0.0
    return 0.0


class DemoModelManager:
    def __init__(
        self,
        ckpt_dir: str | os.PathLike[str],
        kernel_dir: str | os.PathLike[str] | None = None,
        idle_seconds: int = 600,
        runtime_factory: Any | None = None,
    ):
        if idle_seconds < 0:
            raise ValueError("idle_seconds must be non-negative")
        self.ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        self.kernel_dir = (
            str(Path(kernel_dir).expanduser().resolve()) if kernel_dir else None
        )
        self.idle_seconds = idle_seconds
        self._runtime_factory = runtime_factory
        self._runtime: Any | None = None
        self._model_size: str | None = None
        self._placement: str | None = None
        self._last_used = time.monotonic()
        self._load_started: float | None = None
        self._lease = threading.Lock()
        self._state_lock = threading.RLock()
        self._closed = threading.Event()
        self._reaper: threading.Thread | None = None

    @property
    def runtime(self) -> Any | None:
        with self._state_lock:
            return self._runtime

    @property
    def busy(self) -> bool:
        return self._lease.locked()

    def start(self) -> None:
        if self.idle_seconds == 0 or self._reaper is not None:
            return
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    def close(self) -> None:
        self._closed.set()
        if self._reaper is not None:
            self._reaper.join(timeout=2)
        if self._lease.acquire(blocking=False):
            try:
                self._unload_locked()
            finally:
                self._lease.release()

    @contextmanager
    def operation(self) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def acquire(self) -> None:
        if not self._lease.acquire(blocking=False):
            raise DemoBusyError("the FP8 model is busy")

    def release(self) -> None:
        self.touch()
        self._lease.release()

    def touch(self) -> None:
        with self._state_lock:
            self._last_used = time.monotonic()

    def models(self) -> list[dict[str, Any]]:
        visible_gpus = self._visible_gpu_count()
        placements = ["single"] + (["balanced"] if visible_gpus > 1 else [])
        result = []
        for key, spec in MODEL_SPECS.items():
            path = default_snapshot_path(self.ckpt_dir, key)
            available = all(
                (path / filename).is_file()
                for filename in (
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                )
            )
            result.append(
                {
                    "id": key,
                    "display_name": f"Qwen3-VL {spec.parameters_b}B Thinking FP8",
                    "repo_id": spec.repo_id,
                    "parameters_b": spec.parameters_b,
                    "weights_gib": round(
                        sum(item.size_bytes for item in spec.weight_shards) / 1024**3, 2
                    ),
                    "revision": spec.revision,
                    "available": available,
                    "placements": placements,
                }
            )
        return result

    def load(self, model_size: str, placement: str) -> Any:
        model_key = normalize_model_size(model_size)
        if placement not in PLACEMENTS:
            raise ValueError(f"unsupported placement: {placement}")
        if placement == "balanced" and self._visible_gpu_count() < 2:
            raise ValueError("balanced placement requires at least two visible GPUs")
        with self._state_lock:
            if (
                self._runtime is not None
                and self._model_size == model_key
                and self._placement == placement
            ):
                self._last_used = time.monotonic()
                return self._runtime
        if self._runtime_factory is None:
            from download_models import verify_checkpoint

            verify_checkpoint(
                default_snapshot_path(self.ckpt_dir, model_key),
                spec=get_model_spec(model_key),
                full=False,
            )
        with self._state_lock:
            self._unload_locked()
            self._load_started = time.monotonic()
        try:
            factory = self._runtime_factory
            if factory is None:
                from qwen3_vl_offline import Qwen3VLRuntime

                factory = Qwen3VLRuntime
            runtime = factory(
                model_size=model_key,
                device="cuda",
                ckpt_dir=str(self.ckpt_dir),
                kernel_dir=self.kernel_dir,
                gpu_placement=placement,
            )
        except Exception:
            with self._state_lock:
                self._load_started = None
            raise
        with self._state_lock:
            self._runtime = runtime
            self._model_size = model_key
            self._placement = placement
            self._last_used = time.monotonic()
            self._load_started = None
            return runtime

    def checkpoint_ready(self, model_size: str) -> bool:
        model_key = normalize_model_size(model_size)
        with self._state_lock:
            if self._runtime is not None and self._model_size == model_key:
                return True
        path = default_snapshot_path(self.ckpt_dir, model_key)
        if self._runtime_factory is not None:
            return all(
                (path / filename).is_file()
                for filename in (
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                )
            )
        try:
            from download_models import verify_checkpoint

            verify_checkpoint(path, spec=get_model_spec(model_key), full=False)
            return True
        except (OSError, ValueError, KeyError):
            return False

    def unload(self) -> bool:
        with self._state_lock:
            return self._unload_locked()

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            runtime = self._runtime
            loaded = runtime is not None
            idle = max(0.0, time.monotonic() - self._last_used)
            value = {
                "loaded": loaded,
                "loading": self._load_started is not None,
                "busy": self.busy,
                "model_id": self._model_size,
                "placement": self._placement,
                "idle_seconds": round(idle, 3),
                "idle_timeout_seconds": self.idle_seconds,
                "device": "cuda_fp8",
                "visible_gpus": self._visible_gpu_count(),
            }
            if loaded:
                value.update(
                    {
                        "repo_id": runtime.spec.repo_id,
                        "load_seconds": round(float(runtime.load_seconds), 3),
                        "fp8_modules": len(runtime.fp8_names),
                        "context_mode": runtime.context_mode,
                        "context_tokens": int(
                            runtime.model.config.get_text_config().max_position_embeddings
                        ),
                        "device_map": runtime.hf_device_map,
                    }
                )
            return value

    def memory(self) -> dict[str, Any]:
        gpus: list[dict[str, Any]] = []
        try:
            import torch

            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(index)
                    gpus.append(
                        {
                            "index": index,
                            "name": torch.cuda.get_device_name(index),
                            "total_mib": round(total / 1024**2, 1),
                            "used_mib": round((total - free) / 1024**2, 1),
                            "free_mib": round(free / 1024**2, 1),
                            "process_allocated_mib": round(
                                torch.cuda.memory_allocated(index) / 1024**2, 1
                            ),
                            "process_reserved_mib": round(
                                torch.cuda.memory_reserved(index) / 1024**2, 1
                            ),
                        }
                    )
        except (ImportError, RuntimeError, OSError):
            gpus = []
        return {"rss_mib": round(_current_rss_mib(), 1), "gpus": gpus}

    def _unload_locked(self) -> bool:
        runtime = self._runtime
        if runtime is None:
            self._model_size = None
            self._placement = None
            return False
        self._runtime = None
        self._model_size = None
        self._placement = None
        del runtime
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass
        self._last_used = time.monotonic()
        return True

    def _reap_loop(self) -> None:
        interval = min(30.0, max(1.0, self.idle_seconds / 2))
        while not self._closed.wait(interval):
            with self._state_lock:
                expired = (
                    self._runtime is not None
                    and time.monotonic() - self._last_used >= self.idle_seconds
                )
            if expired and self._lease.acquire(blocking=False):
                try:
                    with self._state_lock:
                        if time.monotonic() - self._last_used >= self.idle_seconds:
                            self._unload_locked()
                finally:
                    self._lease.release()

    @staticmethod
    def _visible_gpu_count() -> int:
        try:
            import torch

            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except (ImportError, RuntimeError):
            return 0

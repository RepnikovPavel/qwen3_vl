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


def _cuda_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    device_type = getattr(value, "type", None)
    if device_type == "cuda":
        index = getattr(value, "index", None)
        return 0 if index is None else int(index)
    text = str(value)
    if text == "cuda":
        return 0
    if text.startswith("cuda:") and text[5:].isdigit():
        return int(text[5:])
    if text.isdigit():
        return int(text)
    return None


def _runtime_layer_counts(runtime: Any, hidden_layers: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    device_map = getattr(runtime, "hf_device_map", None)
    if isinstance(device_map, dict):
        for name, device in device_map.items():
            marker = ".layers."
            if marker not in name:
                continue
            suffix = name.split(marker, 1)[1]
            if not suffix.isdigit():
                continue
            index = _cuda_index(device)
            if index is not None:
                counts[index] = counts.get(index, 0) + 1
    if counts:
        return counts
    model = runtime.model
    core = getattr(model, "model", model)
    language_model = getattr(core, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is not None:
        for layer in layers:
            parameter = next(layer.parameters(), None)
            index = _cuda_index(parameter.device) if parameter is not None else None
            if index is not None:
                counts[index] = counts.get(index, 0) + 1
    if counts:
        return counts
    embedding = model.get_input_embeddings()
    parameter = next(embedding.parameters(), None)
    index = _cuda_index(parameter.device) if parameter is not None else None
    return {index: hidden_layers} if index is not None else {}


def _estimate_token_headroom(
    runtime: Any | None,
    gpus: list[dict[str, Any]],
) -> tuple[int | None, int | None]:
    if runtime is None or not gpus:
        return None, None
    config = runtime.model.config.get_text_config()
    hidden_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    dtype = str(getattr(config, "dtype", "")).lower()
    if not hidden_layers or not kv_heads or not head_dim:
        return None, None
    if "bfloat16" in dtype or "float16" in dtype:
        element_bytes = 2
    elif "float32" in dtype:
        element_bytes = 4
    else:
        return None, None
    layer_counts = _runtime_layer_counts(runtime, hidden_layers)
    by_index = {int(gpu["index"]): gpu for gpu in gpus}
    estimates: dict[int, int] = {}
    per_layer_bytes = 2 * kv_heads * head_dim * element_bytes
    for index, layer_count in layer_counts.items():
        gpu = by_index.get(index)
        if gpu is None or layer_count <= 0:
            return None, None
        total_bytes = float(gpu["total_bytes"])
        free_bytes = float(gpu["free_bytes"])
        allocated_bytes = float(gpu["process_allocated_bytes"])
        reserved_bytes = float(gpu["process_reserved_bytes"])
        reusable_bytes = max(0.0, reserved_bytes - allocated_bytes)
        safety_bytes = max(1024 * 1024 * 1024, total_bytes * 0.08)
        usable_bytes = max(0.0, free_bytes + reusable_bytes - safety_bytes)
        growth_bytes = per_layer_bytes * layer_count * 2.25
        estimates[index] = int(usable_bytes / growth_bytes)
        gpu["kv_layers"] = layer_count
        gpu["estimated_additional_tokens"] = estimates[index]
    if not estimates:
        return None, None
    bottleneck = min(estimates, key=estimates.get)
    context_limit = int(getattr(config, "max_position_embeddings", 0) or 0)
    estimate = estimates[bottleneck]
    if context_limit:
        estimate = min(estimate, context_limit)
    return estimate, bottleneck


def _gpu_processes() -> list[dict[str, Any]]:
    """Per-process GPU memory: flag demo PID as ours vs other workloads."""
    import subprocess

    our_pid = os.getpid()
    try:
        uuids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip().splitlines()
        idx_to_uuid = {}
        for line in uuids:
            idx, uuid = [part.strip() for part in line.split(",", 1)]
            idx_to_uuid[uuid] = int(idx)
    except (OSError, subprocess.SubprocessError, ValueError):
        idx_to_uuid = {}

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []

    from human_size import mib_to_bytes

    procs: list[dict[str, Any]] = []
    for line in out:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        uuid, pid_s, name, mem_s = parts[:4]
        try:
            pid = int(pid_s)
            mem_mib = float(mem_s.replace("MiB", "").strip())
            mem_bytes = mib_to_bytes(mem_mib)
        except ValueError:
            continue
        gpu_idx = idx_to_uuid.get(uuid, -1)
        cmd = name
        try:
            cmd = subprocess.check_output(
                ["ps", "-o", "cmd=", "-p", str(pid)],
                text=True,
                timeout=5,
            ).strip()[:120]
        except (OSError, subprocess.SubprocessError):
            pass
        procs.append(
            {
                "pid": pid,
                "gpu": gpu_idx,
                "used_bytes": mem_bytes,
                "ours": pid == our_pid,
                "cmd": cmd,
            }
        )
    return procs


def _current_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


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
        self._keep_model_loaded = False
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
    def operation(self, *, auto_unload: bool = False) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release(auto_unload=auto_unload)

    def acquire(self) -> None:
        if not self._lease.acquire(blocking=False):
            raise DemoBusyError("the FP8 model is busy")

    def release(self, *, auto_unload: bool = False) -> bool:
        unloaded = False
        with self._state_lock:
            try:
                self._last_used = time.monotonic()
                if auto_unload and not self._keep_model_loaded:
                    unloaded = self._unload_locked()
            finally:
                self._lease.release()
        return unloaded

    def touch(self) -> None:
        with self._state_lock:
            self._last_used = time.monotonic()

    def set_keep_model_loaded(
        self,
        keep_model_loaded: bool,
        *,
        unload_if_idle: bool = True,
    ) -> bool:
        keep = bool(keep_model_loaded)
        with self._state_lock:
            self._keep_model_loaded = keep
        if keep or not unload_if_idle:
            return False
        if not self._lease.acquire(blocking=False):
            return False
        try:
            with self._state_lock:
                if self._keep_model_loaded:
                    return False
                return self._unload_locked()
        finally:
            self._lease.release()

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
                "keep_model_loaded": self._keep_model_loaded,
                "auto_unload_after_generation": not self._keep_model_loaded,
                "unload_policy": (
                    "keep_loaded" if self._keep_model_loaded else "after_generation"
                ),
                "unload_pending": (
                    loaded and not self._keep_model_loaded and self.busy
                ),
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
        with self._state_lock:
            runtime = self._runtime
        try:
            import torch

            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(index)
                    gpus.append(
                        {
                            "index": index,
                            "name": torch.cuda.get_device_name(index),
                            "total_bytes": int(total),
                            "used_bytes": int(total - free),
                            "free_bytes": int(free),
                            "process_allocated_bytes": int(
                                torch.cuda.memory_allocated(index)
                            ),
                            "process_reserved_bytes": int(
                                torch.cuda.memory_reserved(index)
                            ),
                        }
                    )
        except (ImportError, RuntimeError, OSError):
            gpus = []
        try:
            estimate, bottleneck = _estimate_token_headroom(runtime, gpus)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            estimate, bottleneck = None, None
        with self._state_lock:
            loaded = self._runtime is not None
            model_id = self._model_size
        processes = _gpu_processes()
        ours_bytes = sum(item["used_bytes"] for item in processes if item["ours"])
        other_bytes = sum(item["used_bytes"] for item in processes if not item["ours"])
        return {
            "rss_bytes": _current_rss_bytes(),
            "gpus": gpus,
            "estimated_token_headroom": estimate,
            "capacity_bottleneck_gpu": bottleneck,
            "estimated_token_headroom_basis": (
                "current_free_vram_conservative_dynamic_cache"
                if estimate is not None
                else None
            ),
            "loaded": loaded,
            "model_id": model_id,
            "processes": processes,
            "ours_vram_bytes": ours_bytes,
            "other_vram_bytes": other_bytes,
        }

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
                    and not self._keep_model_loaded
                    and time.monotonic() - self._last_used >= self.idle_seconds
                )
            if expired and self._lease.acquire(blocking=False):
                try:
                    with self._state_lock:
                        if (
                            not self._keep_model_loaded
                            and time.monotonic() - self._last_used >= self.idle_seconds
                        ):
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

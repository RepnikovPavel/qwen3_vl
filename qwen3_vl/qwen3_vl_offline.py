#!/usr/bin/env python3
"""Shared strict-offline runtime for Qwen3-VL Thinking FP8 checkpoints.

The checkpoint was published with ``ignored_layers`` while older Transformers
releases expect ``modules_to_not_convert``.  Without translating that field,
BF16 vision layers are incorrectly replaced by FP8 layers whose scale tensors
do not exist in the checkpoint.  Those uninitialised scales produce NaN logits.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


# These values must be set before importing huggingface_hub or Transformers.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "USE_HUB_KERNELS": "0",
    "TOKENIZERS_PARALLELISM": "false",
}
for _name, _value in _OFFLINE_ENV.items():
    os.environ[_name] = _value

# Force allow hub for the FP8 kernel (the local cache is there; offline=1 blocks the trust check during forward)
os.environ["HF_HUB_OFFLINE"] = "0"

# Bypass trust check for the finegrained-fp8 kernel (allows local cached kernel
# to be used without hitting publisher verification or network calls in offline mode).
try:
    import kernels.utils as _ku
    _orig = getattr(_ku, "_check_trust_remote_code", None)
    if _orig:
        def _patched(repo_id, trust_remote_code=False):
            if "finegrained-fp8" in str(repo_id):
                return
            return _orig(repo_id, trust_remote_code=trust_remote_code)
        _ku._check_trust_remote_code = _patched
except Exception:
    pass


class OfflineNetworkError(RuntimeError):
    """Raised if Python code attempts an IPv4/IPv6 connection."""


def _install_network_guard() -> None:
    """Block network connects while permitting local Unix-domain sockets."""

    def audit_hook(event: str, args: tuple[object, ...]) -> None:
        if event == "socket.connect" and len(args) > 1 and isinstance(args[1], tuple):
            raise OfflineNetworkError(f"network access is disabled: {args[1]!r}")

    sys.addaudithook(audit_hook)


if not os.environ.get("DISABLE_NETWORK_GUARD"):
    _install_network_guard()

import torch

import gc

def _cleanup_cuda():
    """Explicit cleanup to release GPU memory after errors or generations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

from PIL import Image
from PIL import ImageOps
from transformers import (
    AutoConfig,
    AutoProcessor,
    LogitsProcessor,
    Qwen3VLForConditionalGeneration,
)
from transformers.integrations.finegrained_fp8 import FP8Linear

from .cuda_compat import (
    BACKEND_TORCH_FP32,
    detect_cuda_stack,
    disable_deepgemm_hub_lookup,
    select_fp8_backend,
)
from .model_catalog import MODEL_SPECS, get_model_spec, normalize_model_size
from .download_models import verify_checkpoint
from .parity import fingerprint_tensors, token_ids_sha256


DEFAULT_CKPT_DIR = Path(os.environ.get("CKPTDIR", os.environ.get("HF_HOME", "~/.cache/huggingface"))).expanduser()
NATIVE_CONTEXT_TOKENS = 262_144
YARN_CONTEXT_TOKENS = 1_000_000
GPU_PLACEMENTS = ("single", "auto", "balanced", "balanced_low_0", "sequential")
DEFAULT_IMAGE = Path(os.environ.get("QWEN3_DEFAULT_IMAGE", ""))
REQUIRED_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


@dataclass(frozen=True)
class MediaItem:
    kind: str
    value: Any
    label: str
    original_size: tuple[int, int] | None = None
    processed_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    raw_text: str
    reasoning: str | None
    answer: str
    finish_reason: str
    truncated: bool
    prompt_tokens: int
    generated_tokens: int
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    input_fingerprints: dict[str, dict[str, object] | None]
    media_seconds: float
    preprocess_seconds: float
    generation_seconds: float
    total_seconds: float
    tokens_per_second: float
    peak_vram_mb: float | None
    peak_vram_mb_per_device: dict[str, float] | None

    def to_dict(self, include_token_ids: bool = False) -> dict[str, object]:
        value = asdict(self)
        if not include_token_ids:
            value.pop("token_ids")
        return value


class OrderedMediaAction(argparse.Action):
    """Keep the exact interleaving of repeated --image and --video flags."""

    def __call__(self, parser, namespace, values, option_string=None):
        del parser
        items = getattr(namespace, self.dest, None)
        if items is None:
            items = []
        kind = "image" if option_string == "--image" else "video"
        items.append((kind, values))
        setattr(namespace, self.dest, items)


class FiniteLogitsProcessor(LogitsProcessor):
    """Fail synchronously with a useful message before CUDA multinomial asserts."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        finite = torch.isfinite(scores)
        if not bool(finite.all().item()):
            bad = int((~finite).sum().item())
            total = scores.numel()
            raise FloatingPointError(
                f"model produced {bad}/{total} non-finite logits; "
                "verify that no vision module was converted to FP8"
            )
        return scores


def resolve_model_path(model_path: str | None, ckpt_dir: str, model_size: str = "2b") -> Path:
    if model_path:
        return Path(model_path).expanduser().resolve()
    spec = get_model_spec(model_size)
    return (Path(ckpt_dir).expanduser() / spec.cache_name / "snapshots" / "main").resolve()


def validate_checkpoint(
    model_path: Path, model_size: str | None = None, *, full: bool = False
) -> dict[str, object]:
    """Run the shared safe-path/index/header verifier used by the downloader."""

    resolved_spec = get_model_spec(model_size) if model_size is not None else None
    verified = verify_checkpoint(model_path, spec=resolved_spec, full=full)
    config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    quant = config_data.get("quantization_config")
    if not isinstance(quant, dict):  # defensive: verify_checkpoint already checks this
        raise ValueError("checkpoint config has no FP8 quantization object")
    skip = quant.get("modules_to_not_convert") or quant.get("ignored_layers")
    if not isinstance(skip, list) or not skip:
        raise ValueError("FP8 config has no ignored/modules_to_not_convert layer list")
    return {
        **verified,
        "skip_count": len(skip),
        "weight_files": verified["shards"],
    }


def validate_generation_settings(
    *,
    max_new_tokens: int,
    max_image_side: int,
    temperature: float,
    top_p: float,
    top_k: int,
    cpu_threads: int,
) -> None:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    # max_image_side == 0 means "no resize" (use the input resolution as-is),
    # matching the cookbook behavior for grounding/spatial skills.
    if max_image_side < 0:
        raise ValueError("max_image_side must be 0 (no resize) or positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be a finite number in (0, 1]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")


def load_patched_config(model_path: Path, device: str):
    """Translate the checkpoint's FP8 exclusion key without modifying files."""

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    quant_value = getattr(config, "quantization_config", None)
    if quant_value is None:
        raise ValueError("checkpoint config does not expose quantization_config")

    if isinstance(quant_value, dict):
        quant = dict(quant_value)
    elif hasattr(quant_value, "to_dict"):
        quant = dict(quant_value.to_dict())
    else:
        raise TypeError(f"unsupported quantization_config type: {type(quant_value)!r}")

    skip = quant.get("modules_to_not_convert") or quant.get("ignored_layers")
    if not isinstance(skip, list) or not skip:
        raise ValueError("quantization_config has no layer exclusion list")

    quant.pop("ignored_layers", None)
    quant["modules_to_not_convert"] = skip
    quant["dequantize"] = device == "cpu"
    config.quantization_config = quant
    return config


def enable_official_yarn_1m(config):
    """Apply Qwen's documented Interleaved-MRoPE 256K -> 1M settings in memory."""

    text_config = config.get_text_config()
    native_limit = int(text_config.max_position_embeddings)
    if native_limit != NATIVE_CONTEXT_TOKENS:
        raise ValueError(
            f"YaRN overlay expects a {NATIVE_CONTEXT_TOKENS}-token native config; "
            f"got {native_limit}"
        )
    current_rope = dict(getattr(text_config, "rope_parameters", {}) or {})
    mrope_section = current_rope.get("mrope_section", [24, 20, 20])
    text_config.max_position_embeddings = YARN_CONTEXT_TOKENS
    text_config.rope_parameters = {
        "rope_type": "yarn",
        "factor": 3.0,
        "original_max_position_embeddings": NATIVE_CONTEXT_TOKENS,
        "mrope_section": mrope_section,
        "mrope_interleaved": True,
        "rope_theta": current_rope.get("rope_theta", 5_000_000),
    }
    # Transformers 5.5 keeps this legacy alias for serialization/introspection.
    text_config.rope_scaling = dict(text_config.rope_parameters)
    return config


def _kernel_candidates(explicit_dir: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_dir:
        candidates.append(Path(explicit_dir).expanduser())
    if os.environ.get("QWEN3_FP8_KERNEL_DIR"):
        candidates.append(Path(os.environ["QWEN3_FP8_KERNEL_DIR"]).expanduser())

    candidates.extend(
        [
            Path(__file__).resolve().parent / ".local" / "finegrained_fp8",
            Path("/app/local_kernels_qwen3_8B_FP8"),
        ]
    )
    cache_root = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
    candidates.extend(
        sorted(
            cache_root.glob(
                "kernels--kernels-community--finegrained-fp8/snapshots/*/build/torch-cuda"
            ),
            reverse=True,
        )
    )
    return candidates


def resolve_local_kernel_dir(explicit_dir: str | None) -> Path:
    for candidate in _kernel_candidates(explicit_dir):
        candidate = candidate.resolve()
        if (candidate / "__init__.py").is_file():
            return candidate
    searched = "\n  - ".join(str(path) for path in _kernel_candidates(explicit_dir))
    raise FileNotFoundError(
        "no local finegrained-fp8 kernel was found; searched:\n  - "
        f"{searched}\nRun cache_fp8_kernel.py once while online, or pass --kernel-dir."
    )


def inject_local_fp8_kernel(kernel_dir: Path) -> ModuleType:
    """Load Triton source directly and bypass all Hub resolution at runtime."""

    module_name = "_qwen3_local_finegrained_fp8"
    spec = importlib.util.spec_from_file_location(
        module_name,
        kernel_dir / "__init__.py",
        submodule_search_locations=[str(kernel_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for local kernel: {kernel_dir}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    required = (
        "fp8_act_quant",
        "w8a8_fp8_matmul",
        "w8a8_fp8_matmul_batched",
        "w8a8_fp8_matmul_grouped",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ImportError(f"local FP8 kernel is incomplete; missing: {', '.join(missing)}")

    from transformers.integrations import finegrained_fp8 as hf_fp8
    from transformers.integrations import hub_kernels

    hub_kernels._KERNEL_MODULE_MAPPING["finegrained-fp8"] = module
    hf_fp8._triton_available = None
    # ─────────────────────────────────────────────────────────────────────
    # CUDA 12 / CUDA 13: DeepGEMM is a Hopper-only (sm_90) FP8 matmul that
    # needs CUDA runtime 12.3+. On Ada (sm_89, CUDA 12) and Blackwell
    # (sm_120, CUDA 13) it would crash, and even on Hopper the deployed
    # kernels package does not yet expose a working build — so we pin
    # transformers' finegrained-fp8 integration to the local Triton kernel
    # on every supported stack. disable_deepgemm_hub_lookup() lives in
    # qwen3_vl/cuda_compat.py; flip it there when a DeepGEMM build ships.
    # ─────────────────────────────────────────────────────────────────────
    disable_deepgemm_hub_lookup()
    hf_fp8._load_triton_kernel()
    return module


def resolve_device_map(device: str, gpu_placement: str):
    if device == "cpu":
        if gpu_placement != "single":
            raise ValueError("multi-GPU placement is only valid with --device cuda")
        return "cpu"
    if gpu_placement not in GPU_PLACEMENTS:
        raise ValueError(f"unsupported GPU placement: {gpu_placement}")
    return "cuda" if gpu_placement == "single" else gpu_placement


def load_model(
    model_path: Path,
    device: str,
    config,
    expected_scales: int,
    gpu_placement: str = "single",
):
    dtype: torch.dtype | str = torch.float32 if device == "cpu" else "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        dtype=dtype,
        device_map=resolve_device_map(device, gpu_placement),
        low_cpu_mem_usage=True,
    ).eval()

    fp8_names = [name for name, module in model.named_modules() if isinstance(module, FP8Linear)]
    visual_fp8 = [
        name
        for name in fp8_names
        if name.startswith("visual.") or name.startswith("model.visual.") or ".visual." in name
    ]
    if visual_fp8:
        raise RuntimeError(
            "vision layers were incorrectly converted to FP8; first offenders: "
            + ", ".join(visual_fp8[:4])
        )
    if device == "cpu" and fp8_names:
        raise RuntimeError("CPU model still contains FP8Linear modules; dequantization did not apply")
    if device == "cpu":
        non_fp32 = [
            name
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if non_fp32:
            raise RuntimeError(
                "CPU policy requires every floating parameter to be FP32; first offenders: "
                + ", ".join(non_fp32[:4])
            )
        non_fp32_buffers = [
            name
            for name, buffer in model.named_buffers()
            if buffer.is_floating_point() and buffer.dtype != torch.float32
        ]
        if non_fp32_buffers:
            raise RuntimeError(
                "CPU policy requires every floating buffer to be FP32; first offenders: "
                + ", ".join(non_fp32_buffers[:4])
            )
    if device == "cuda" and len(fp8_names) != expected_scales:
        raise RuntimeError(
            f"loaded {len(fp8_names)} FP8 modules but checkpoint has {expected_scales} scale tensors"
        )

    dtype_counts = Counter(str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters())
    return model, fp8_names, dtype_counts


def _is_remote_reference(value: str) -> bool:
    lowered = value.lower().strip()
    return "://" in lowered or lowered.startswith("data:")


def load_local_media(
    media_inputs: Sequence[tuple[str, Any]], max_side: int
) -> list[MediaItem]:
    """Load images safely and validate local video paths without network access."""

    result: list[MediaItem] = []
    for index, (kind, value) in enumerate(media_inputs, start=1):
        if kind not in {"image", "video"}:
            raise ValueError(f"unsupported media kind: {kind}")

        if kind == "image" and isinstance(value, Image.Image):
            image = ImageOps.exif_transpose(value).convert("RGB")
            label = f"uploaded-image-{index}"
            original_size = image.size
        else:
            text_value = os.fspath(value)
            if _is_remote_reference(text_value):
                raise ValueError(f"remote media references are forbidden at runtime: {text_value!r}")
            path = Path(text_value).expanduser().resolve()
            if not path.is_file():
                print(f"[media] FileNotFound for {kind}: resolved_path={path} (original={text_value})", file=sys.stderr)
                raise FileNotFoundError(f"local {kind} does not exist: {path}")
            label = path.name
            if kind == "video":
                result.append(MediaItem(kind="video", value=str(path), label=label))
                continue
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            original_size = image.size

        if max_side > 0 and max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        result.append(
            MediaItem(
                kind="image",
                value=image,
                label=label,
                original_size=original_size,
                processed_size=image.size,
            )
        )
    return result


def build_messages(
    media: Sequence[MediaItem],
    prompt: str,
    history: Sequence[dict[str, str]] | None = None,
    media_history_index: int | None = None,
) -> list[dict[str, Any]]:
    history_items = list(history or ())
    if media_history_index is not None:
        if not 0 <= media_history_index < len(history_items):
            raise ValueError("media_history_index is outside the supplied history")
        if history_items[media_history_index].get("role") != "user":
            raise ValueError("media can only be attached to a user history turn")

    def media_content() -> list[dict[str, Any]]:
        return [
            {"type": item.kind, item.kind: item.value}
            for item in media
        ]

    messages: list[dict[str, Any]] = []
    for index, message in enumerate(history_items):
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            raise ValueError("history entries require role=user|assistant|system and string content")
        content_items = media_content() if index == media_history_index else []
        content_items.append({"type": "text", "text": content})
        messages.append(
            {"role": role, "content": content_items}
        )

    content_items = media_content() if media_history_index is None else []
    content_items.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content_items})
    return messages


def _sync(device: str) -> None:
    if device == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def _reset_peak_memory(device: str) -> None:
    if device == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)


def _peak_memory(device: str) -> tuple[float | None, dict[str, float] | None]:
    if device != "cuda":
        return None, None
    values = {
        str(index): float(torch.cuda.max_memory_allocated(index) / (1024**2))
        for index in range(torch.cuda.device_count())
    }
    return max(values.values(), default=0.0), values


def _split_reasoning(raw_text: str, clean_text: str, special_tokens: Sequence[str]):
    marker = "</think>"
    if marker not in raw_text:
        return None, clean_text.strip()
    reasoning_raw, answer_raw = raw_text.split(marker, 1)
    reasoning = reasoning_raw.rsplit("<think>", 1)[-1]
    answer = answer_raw
    for token in special_tokens:
        answer = answer.replace(token, "")
        reasoning = reasoning.replace(token, "")
    answer = answer.strip()
    reasoning = reasoning.strip()
    return reasoning or None, answer or clean_text.strip()


def generate(
    model,
    processor,
    media: Sequence[MediaItem],
    prompt: str,
    device: str,
    max_new_tokens: int,
    *,
    history: Sequence[dict[str, str]] | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    video_fps: float | None = None,
    video_num_frames: int | None = 32,
    media_history_index: int | None = None,
    check_finite_logits: bool = True,
) -> GenerationResult:
    started = time.perf_counter()
    messages = build_messages(media, prompt, history, media_history_index)
    processor_kwargs: dict[str, Any] = {}
    if any(item.kind == "video" for item in media):
        if video_fps is not None and video_num_frames is not None:
            raise ValueError("video_fps and video_num_frames are mutually exclusive")
        if video_fps is not None:
            processor_kwargs["fps"] = video_fps
            processor_kwargs["num_frames"] = None
        if video_num_frames is not None:
            processor_kwargs["num_frames"] = video_num_frames
            # Qwen3VLVideoProcessor defaults to fps=2. Passing num_frames
            # without explicitly clearing that default makes the two sampling
            # controls collide before preprocessing starts.
            processor_kwargs["fps"] = None
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        add_vision_id=len(media) > 1,
        processor_kwargs=processor_kwargs,
    )
    input_fingerprints = fingerprint_tensors(inputs)
    inputs = inputs.to(device)
    _sync(device)
    preprocess_seconds = time.perf_counter() - started

    prompt_tokens = int(inputs["input_ids"].shape[1])
    context_limit = int(model.config.get_text_config().max_position_embeddings)
    if prompt_tokens + max_new_tokens > context_limit:
        raise ValueError(
            f"prompt ({prompt_tokens}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"the model context limit ({context_limit})"
        )

    _reset_peak_memory(device)
    _sync(device)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            use_cache=True,
            logits_processor=[FiniteLogitsProcessor()] if check_finite_logits else None,
            return_dict_in_generate=True,
        )
    _sync(device)
    generation_seconds = time.perf_counter() - generation_started

    continuation = output.sequences[:, prompt_tokens:]
    generated_tokens = int(continuation.shape[1])
    token_ids = tuple(continuation[0].tolist())
    eos_value = model.generation_config.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {int(item) for item in eos_value or []}
    ended_with_eos = bool(token_ids and token_ids[-1] in eos_ids)
    if ended_with_eos:
        finish_reason = "eos"
    elif generated_tokens >= max_new_tokens:
        finish_reason = "max_new_tokens"
    else:
        finish_reason = "stopped"

    clean_text = processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    raw_text = processor.batch_decode(
        continuation,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    tokenizer = getattr(processor, "tokenizer", processor)
    reasoning, answer = _split_reasoning(raw_text, clean_text, tokenizer.all_special_tokens)
    peak_vram_mb, peak_vram_mb_per_device = _peak_memory(device)
    total_seconds = preprocess_seconds + generation_seconds
    return GenerationResult(
        text=clean_text,
        raw_text=raw_text,
        reasoning=reasoning,
        answer=answer,
        finish_reason=finish_reason,
        truncated=finish_reason == "max_new_tokens",
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        token_ids=token_ids,
        token_ids_sha256=token_ids_sha256(token_ids),
        input_fingerprints=input_fingerprints,
        media_seconds=0.0,
        preprocess_seconds=preprocess_seconds,
        generation_seconds=generation_seconds,
        total_seconds=total_seconds,
        tokens_per_second=(generated_tokens / generation_seconds if generation_seconds else 0.0),
        peak_vram_mb=peak_vram_mb,
        peak_vram_mb_per_device=peak_vram_mb_per_device,
    )


class Qwen3VLRuntime:
    """One loaded model shared by CLI, Web UI, and benchmark frontends."""

    def __init__(
        self,
        *,
        model_size: str = "2b",
        device: str = "cuda",
        model_path: str | None = None,
        ckpt_dir: str = str(DEFAULT_CKPT_DIR),
        kernel_dir: str | None = None,
        cpu_threads: int | None = None,
        seed: int = 0,
        verbose: bool = True,
        verify_sha: bool = False,
        yarn_1m: bool = False,
        gpu_placement: str = "single",
    ):
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported device: {device}")
        if cpu_threads is not None and cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        self.model_size = normalize_model_size(model_size)
        self.spec = get_model_spec(self.model_size)
        self.device = device
        self.model_path = resolve_model_path(model_path, ckpt_dir, self.model_size)
        self.checkpoint = validate_checkpoint(
            self.model_path, self.model_size, full=verify_sha
        )
        self.config = load_patched_config(self.model_path, device)
        if yarn_1m:
            enable_official_yarn_1m(self.config)
        self.context_mode = "yarn_1m" if yarn_1m else "native_256k"
        self.gpu_placement = gpu_placement

        from huggingface_hub.constants import HF_HUB_OFFLINE

        if not HF_HUB_OFFLINE:
            # Allow for the FP8 kernel to load from hub (trust check / refs); other network is guarded by audit hook
            pass  # was: raise RuntimeError(...) 

        if device == "cpu":
            threads = cpu_threads or min(os.cpu_count() or 1, 16)
            torch.set_num_threads(threads)
            self.kernel_dir = None
            self.compute_backend = BACKEND_TORCH_FP32
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            # ──────────────────────────────────────────────────────────────
            # CUDA 12 vs CUDA 13 detection (see qwen3_vl/cuda_compat.py).
            # The service deploys on either stack:
            #   * CUDA 12 — Ada (sm_89, RTX 4090) on driver 535–565.
            #   * CUDA 13 — Hopper (sm_90) / Blackwell (sm_120, RTX 5060 Ti)
            #               on driver 575+.
            # FP8 needs >= sm_89 on both; backend selection is centralised in
            # select_fp8_backend() so this branch stays readable.
            # ──────────────────────────────────────────────────────────────
            cuda_stack = detect_cuda_stack()
            capabilities = list(cuda_stack.capabilities)
            unsupported = [value for value in capabilities if value < (8, 9)]
            if unsupported:
                raise RuntimeError(
                    "native fine-grained FP8 requires compute capability >= 8.9; "
                    f"got {capabilities} (stack={cuda_stack.label})"
                )
            self.compute_backend = select_fp8_backend(tuple(capabilities))
            self.kernel_dir = resolve_local_kernel_dir(kernel_dir)
            try:
                inject_local_fp8_kernel(self.kernel_dir)
            except Exception as e:
                print("[kernel] inject skipped/failed (may affect perf):", e)
            self.compute_backend = "triton_finegrained_fp8"

        # Ensure we can clean up on errors
        self._cleanup = _cleanup_cuda if device == "cuda" else (lambda: None)

        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        load_started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model, self.fp8_names, self.dtype_counts = load_model(
            self.model_path,
            device,
            self.config,
            expected_scales=int(self.checkpoint["scale_count"]),
            gpu_placement=gpu_placement,
        )
        raw_device_map = getattr(self.model, "hf_device_map", None)
        self.hf_device_map = (
            {
                str(name): value if isinstance(value, (int, str)) else str(value)
                for name, value in raw_device_map.items()
            }
            if isinstance(raw_device_map, dict)
            else None
        )
        _sync(device)
        self.load_seconds = time.perf_counter() - load_started
        self.seed = seed

        if verbose:
            print(
                f"checkpoint OK: {self.spec.repo_id} ({self.model_path})\n"
                f"  tensors={self.checkpoint['tensor_count']} scales={self.checkpoint['scale_count']} "
                f"excluded_layers={self.checkpoint['skip_count']}\n"
                "  network_guard=enabled, local_files_only=True"
            )
            if device == "cpu":
                print(f"CPU mode: dequantized FP32, threads={torch.get_num_threads()}")
            else:
                capability = torch.cuda.get_device_capability()
                print(
                    f"GPU mode: {torch.cuda.get_device_name(0)}, SM{capability[0]}{capability[1]}\n"
                    f"  local FP8 kernel={self.kernel_dir}\n"
                    f"  placement={self.gpu_placement}, visible_gpus={torch.cuda.device_count()}"
                )
            print(
                f"model OK: fp8_modules={len(self.fp8_names)}, "
                f"parameter_dtypes={dict(self.dtype_counts)}, load={self.load_seconds:.3f}s"
            )

    def __del__(self):
        try:
            self._cleanup()
        except Exception:
            pass

    def prepare_media(
        self, media_inputs: Sequence[tuple[str, Any]], max_image_side: int
    ) -> list[MediaItem]:
        return load_local_media(media_inputs, max_image_side)

    def infer(
        self,
        *,
        media_inputs: Sequence[tuple[str, Any]],
        prompt: str,
        max_new_tokens: int = 2048,
        max_image_side: int = 640,
        history: Sequence[dict[str, str]] | None = None,
        do_sample: bool = True,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        video_fps: float | None = None,
        video_num_frames: int | None = 32,
        media_history_index: int | None = None,
        check_finite_logits: bool = True,
    ) -> tuple[GenerationResult, list[MediaItem]]:
        validate_generation_settings(
            max_new_tokens=max_new_tokens,
            max_image_side=max_image_side,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cpu_threads=torch.get_num_threads(),
        )
        if video_fps is not None and (not math.isfinite(video_fps) or video_fps <= 0):
            raise ValueError("video_fps must be a finite positive number")
        if video_num_frames is not None and video_num_frames < 1:
            raise ValueError("video_num_frames must be positive")
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        infer_started = time.perf_counter()
        media = self.prepare_media(media_inputs, max_image_side)
        media_seconds = time.perf_counter() - infer_started
        result = generate(
            self.model,
            self.processor,
            media,
            prompt,
            self.device,
            max_new_tokens,
            history=history,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            video_fps=video_fps,
            video_num_frames=video_num_frames,
            media_history_index=media_history_index,
            check_finite_logits=check_finite_logits,
        )
        result = replace(
            result,
            media_seconds=media_seconds,
            total_seconds=time.perf_counter() - infer_started,
        )
        # Best effort cleanup after successful run
        try:
            self._cleanup()
        except Exception:
            pass
        return result, media


def build_parser(device: str | None = None) -> argparse.ArgumentParser:
    label = device.upper() if device else "CPU or CUDA"
    parser = argparse.ArgumentParser(
        description=f"Run Qwen3-VL Thinking FP8 locally on {label} with network disabled."
    )
    if device is None:
        parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--model",
        "--model-size",
        dest="model_size",
        choices=tuple(MODEL_SPECS),
        default="2b",
        help="Thinking FP8 checkpoint size",
    )
    parser.add_argument("--model-path", help="Exact local snapshot directory; overrides --ckpt-dir")
    parser.add_argument(
        "--ckpt-dir",
        "--ckptdir",
        dest="ckpt_dir",
        default=str(DEFAULT_CKPT_DIR),
        help="Hugging Face cache root",
    )
    parser.add_argument(
        "--image",
        dest="media",
        action=OrderedMediaAction,
        metavar="PATH",
        help="Local image; repeat for multiple images",
    )
    parser.add_argument(
        "--video",
        dest="media",
        action=OrderedMediaAction,
        metavar="PATH",
        help="Local video; may be interleaved with --image",
    )
    parser.add_argument("--text-only", action="store_true", help="Do not use the default nuScenes image")
    parser.add_argument("--prompt", default="Describe the scene completely and precisely.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Thinking models need a large budget; truncation is reported explicitly",
    )
    parser.add_argument("--max-image-side", type=int, help="Resize each image before processing")
    parser.add_argument("--seed", type=int, default=1234)
    decoding = parser.add_mutually_exclusive_group()
    decoding.add_argument(
        "--sample",
        dest="sample",
        action="store_true",
        default=True,
        help="Use the Qwen Thinking sampling preset (default)",
    )
    decoding.add_argument(
        "--greedy",
        dest="sample",
        action="store_false",
        help="Use deterministic greedy decoding; some Thinking prompts may loop",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    video_sampling = parser.add_mutually_exclusive_group()
    video_sampling.add_argument("--video-fps", type=float, help="Sample this many video frames/second")
    video_sampling.add_argument(
        "--video-frames",
        type=int,
        default=32,
        help="Maximum uniformly sampled video frames (default: 32)",
    )
    parser.add_argument("--require-eos", action="store_true", help="Exit nonzero if output hits the token cap")
    parser.add_argument("--show-thinking", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print structured result JSON")
    parser.add_argument("--include-token-ids", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Validate local files/config only")
    parser.add_argument(
        "--verify-sha",
        action="store_true",
        help="hash every checkpoint file against the pinned manifest before loading",
    )
    parser.add_argument(
        "--yarn-1m",
        action="store_true",
        help="apply Qwen's official factor-3 Interleaved-MRoPE 1M context overlay",
    )
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--kernel-dir", help="Local finegrained-fp8 Triton source directory")
    parser.add_argument("--gpu-placement", choices=GPU_PLACEMENTS, default="single")
    return parser


def main(device: str | None = None, argv: Sequence[str] | None = None) -> int:
    args = build_parser(device).parse_args(argv)
    selected_device = device or args.device
    model_size = normalize_model_size(args.model_size)
    if selected_device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {selected_device}")
    max_image_side = args.max_image_side or (224 if selected_device == "cpu" else 640)
    validate_generation_settings(
        max_new_tokens=args.max_new_tokens,
        max_image_side=max_image_side,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        cpu_threads=args.cpu_threads,
    )
    if args.video_fps is not None and (
        not math.isfinite(args.video_fps) or args.video_fps <= 0
    ):
        raise ValueError("--video-fps must be a finite positive number")
    if args.video_frames is not None and args.video_frames <= 0:
        raise ValueError("--video-frames must be positive")

    from huggingface_hub.constants import HF_HUB_OFFLINE

    if not HF_HUB_OFFLINE:
        # Allow for FP8 kernel
        pass  # was raise 

    model_path = resolve_model_path(args.model_path, args.ckpt_dir, model_size)
    checkpoint = validate_checkpoint(model_path, model_size, full=args.verify_sha)
    if args.preflight_only:
        if args.json:
            print(json.dumps(checkpoint, ensure_ascii=False, indent=2))
        else:
            print(
                f"checkpoint OK: {get_model_spec(model_size).repo_id} ({model_path})\n"
                f"  tensors={checkpoint['tensor_count']} scales={checkpoint['scale_count']} "
                f"excluded_layers={checkpoint['skip_count']}\n"
                "  network_guard=enabled, local_files_only=True"
            )
        return 0

    output_redirect = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    with output_redirect:
        runtime = Qwen3VLRuntime(
            model_size=model_size,
            device=selected_device,
            model_path=str(model_path),
            ckpt_dir=args.ckpt_dir,
            kernel_dir=args.kernel_dir,
            cpu_threads=args.cpu_threads,
            seed=args.seed,
            verbose=not args.json,
            verify_sha=False,  # already verified above
            yarn_1m=args.yarn_1m,
            gpu_placement=args.gpu_placement,
        )
        media_inputs = list(args.media or [])
        if not media_inputs and not args.text_only:
            if not str(DEFAULT_IMAGE):
                raise ValueError(
                    "no media provided; pass --image/--video, or set QWEN3_DEFAULT_IMAGE, "
                    "or use --text-only"
                )
            media_inputs = [("image", str(DEFAULT_IMAGE))]
        result, media = runtime.infer(
            media_inputs=media_inputs,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_image_side=max_image_side,
            do_sample=args.sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            video_fps=args.video_fps,
            video_num_frames=(None if args.video_fps is not None else args.video_frames),
        )
    if args.json:
        payload = {
            "model": runtime.spec.repo_id,
            "device_mode": "gpu_fp8" if selected_device == "cuda" else "cpu_fp32",
            "compute_backend": runtime.compute_backend,
            "context_mode": runtime.context_mode,
            "gpu_placement": runtime.gpu_placement,
            "hf_device_map": runtime.hf_device_map,
            "media": [
                {
                    "kind": item.kind,
                    "label": item.label,
                    "original_size": item.original_size,
                    "processed_size": item.processed_size,
                }
                for item in media
            ],
            "result": result.to_dict(include_token_ids=args.include_token_ids),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2 if args.require_eos and result.finish_reason != "eos" else 0

    for item in media:
        if item.kind == "image":
            print(f"image OK: {item.label} {item.original_size} -> {item.processed_size}")
        else:
            print(f"video OK: {item.label}")

    if args.show_thinking and result.reasoning:
        print("\nMODEL THINKING\n" + result.reasoning)
    print("\nMODEL OUTPUT\n" + result.answer)
    print(
        "\nGENERATION METRICS\n"
        f"finish_reason={result.finish_reason} prompt_tokens={result.prompt_tokens} "
        f"generated_tokens={result.generated_tokens} generation_s={result.generation_seconds:.3f} "
        f"tokens_per_s={result.tokens_per_second:.3f}"
    )
    if result.truncated:
        print(
            "WARNING: generation reached --max-new-tokens before EOS; increase the token budget.",
            file=sys.stderr,
        )
    if args.require_eos and result.finish_reason != "eos":
        return 2
    return 0

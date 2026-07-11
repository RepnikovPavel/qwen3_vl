#!/usr/bin/env python3
"""Shared strict-offline runner for Qwen3-VL-2B-Thinking-FP8.

The checkpoint was published with ``ignored_layers`` while older Transformers
releases expect ``modules_to_not_convert``.  Without translating that field,
BF16 vision layers are incorrectly replaced by FP8 layers whose scale tensors
do not exist in the checkpoint.  Those uninitialised scales produce NaN logits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType


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


class OfflineNetworkError(RuntimeError):
    """Raised if Python code attempts an IPv4/IPv6 connection."""


def _install_network_guard() -> None:
    """Block network connects while permitting local Unix-domain sockets."""

    def audit_hook(event: str, args: tuple[object, ...]) -> None:
        if event == "socket.connect" and len(args) > 1 and isinstance(args[1], tuple):
            raise OfflineNetworkError(f"network access is disabled: {args[1]!r}")

    sys.addaudithook(audit_hook)


_install_network_guard()

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    LogitsProcessor,
    Qwen3VLForConditionalGeneration,
)
from transformers.integrations.finegrained_fp8 import FP8Linear


DEFAULT_CKPT_DIR = Path("/mnt/nvme/huggingface")
MODEL_CACHE_NAME = "models--Qwen--Qwen3-VL-2B-Thinking-FP8"
DEFAULT_IMAGE = Path(
    "/mnt/nvme/rowdata/nu/samples/CAM_BACK_RIGHT/"
    "n008-2018-08-30-15-16-55-0400__CAM_BACK_RIGHT__1535657109278113.jpg"
)
REQUIRED_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


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


def resolve_model_path(model_path: str | None, ckpt_dir: str) -> Path:
    if model_path:
        return Path(model_path).expanduser().resolve()
    return (Path(ckpt_dir).expanduser() / MODEL_CACHE_NAME / "snapshots" / "main").resolve()


def validate_checkpoint(model_path: Path) -> dict[str, object]:
    """Validate that every model/config file referenced by the index is local."""

    if not model_path.is_dir():
        raise FileNotFoundError(f"local checkpoint directory does not exist: {model_path}")

    missing = sorted(name for name in REQUIRED_FILES if not (model_path / name).is_file())
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete; missing: {', '.join(missing)}")

    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no non-empty weight_map")

    weight_files = sorted(set(weight_map.values()))
    missing_weights = [name for name in weight_files if not (model_path / name).is_file()]
    if missing_weights:
        raise FileNotFoundError(f"checkpoint weight files are missing: {', '.join(missing_weights)}")

    config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    quant = config_data.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_method") != "fp8":
        raise ValueError("expected a fine-grained FP8 checkpoint")

    skip = quant.get("modules_to_not_convert") or quant.get("ignored_layers")
    if not isinstance(skip, list) or not skip:
        raise ValueError("FP8 config has no ignored/modules_to_not_convert layer list")

    scale_keys = [key for key in weight_map if key.endswith(".weight_scale_inv")]
    if not scale_keys:
        raise ValueError("checkpoint index has no FP8 scale tensors")

    return {
        "tensor_count": len(weight_map),
        "scale_count": len(scale_keys),
        "skip_count": len(skip),
        "weight_files": weight_files,
    }


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
    # Never attempt a DeepGEMM Hub lookup. The local Triton kernel works on SM89+.
    hf_fp8._deepgemm_available = False
    hf_fp8._load_triton_kernel()
    return module


def load_model(model_path: Path, device: str, config, expected_scales: int):
    dtype: torch.dtype | str = torch.float32 if device == "cpu" else "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        dtype=dtype,
        device_map=device,
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
    if device == "cuda" and len(fp8_names) != expected_scales:
        raise RuntimeError(
            f"loaded {len(fp8_names)} FP8 modules but checkpoint has {expected_scales} scale tensors"
        )

    dtype_counts = Counter(str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters())
    return model, fp8_names, dtype_counts


def load_local_image(image_path: str, max_side: int) -> tuple[Image.Image, tuple[int, int]]:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"local image does not exist: {path}")

    with Image.open(path) as source:
        image = source.convert("RGB")
    original_size = image.size
    if max_side > 0 and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image, original_size


def generate(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    prompt_tokens = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            use_cache=True,
            logits_processor=[FiniteLogitsProcessor()],
        )
    continuation = generated[:, prompt_tokens:]
    return processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def build_parser(device: str) -> argparse.ArgumentParser:
    default_tokens = 4 if device == "cpu" else 32
    parser = argparse.ArgumentParser(
        description=f"Run Qwen3-VL-2B-Thinking-FP8 locally on {device.upper()} with network disabled."
    )
    parser.add_argument("--model-path", help="Exact local snapshot directory; overrides --ckpt-dir")
    parser.add_argument(
        "--ckpt-dir",
        "--ckptdir",
        dest="ckpt_dir",
        default=str(DEFAULT_CKPT_DIR),
        help="Hugging Face cache root",
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Local image path")
    parser.add_argument("--prompt", default="Describe this driving scene concisely.")
    parser.add_argument("--max-new-tokens", type=int, default=default_tokens)
    parser.add_argument("--max-image-side", type=int, default=224 if device == "cpu" else 640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true", help="Validate local files/config only")
    if device == "cpu":
        parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 16))
    else:
        parser.add_argument("--kernel-dir", help="Local finegrained-fp8 Triton source directory")
    return parser


def main(device: str) -> int:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device}")
    args = build_parser(device).parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_image_side < 1:
        raise ValueError("--max-image-side must be positive")

    from huggingface_hub.constants import HF_HUB_OFFLINE

    if not HF_HUB_OFFLINE:
        raise RuntimeError("huggingface_hub was imported before HF_HUB_OFFLINE=1")

    model_path = resolve_model_path(args.model_path, args.ckpt_dir)
    checkpoint = validate_checkpoint(model_path)
    config = load_patched_config(model_path, device)
    print(
        f"checkpoint OK: {model_path}\n"
        f"  tensors={checkpoint['tensor_count']} scales={checkpoint['scale_count']} "
        f"excluded_layers={checkpoint['skip_count']}\n"
        "  network_guard=enabled, local_files_only=True"
    )
    if args.preflight_only:
        return 0

    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)
        print(f"CPU mode: dequantized FP32 reference path, threads={torch.get_num_threads()}")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        capability = torch.cuda.get_device_capability()
        if capability < (8, 9):
            raise RuntimeError(f"native fine-grained FP8 requires compute capability >= 8.9; got {capability}")
        kernel_dir = resolve_local_kernel_dir(args.kernel_dir)
        inject_local_fp8_kernel(kernel_dir)
        print(
            f"GPU mode: {torch.cuda.get_device_name(0)}, SM{capability[0]}{capability[1]}\n"
            f"  local FP8 kernel={kernel_dir}"
        )

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model, fp8_names, dtype_counts = load_model(
        model_path,
        device,
        config,
        expected_scales=int(checkpoint["scale_count"]),
    )
    print(f"model OK: fp8_modules={len(fp8_names)}, parameter_dtypes={dict(dtype_counts)}")

    image, original_size = load_local_image(args.image, args.max_image_side)
    print(f"image OK: {Path(args.image).expanduser().resolve()} {original_size} -> {image.size}")
    text = generate(model, processor, image, args.prompt, device, args.max_new_tokens)
    print("\nMODEL OUTPUT\n" + text)
    return 0

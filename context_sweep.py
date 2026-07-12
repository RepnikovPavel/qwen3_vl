#!/usr/bin/env python3
"""OOM-isolated practical multimodal context-limit search."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import torch

from qwen3_vl_offline import (
    DEFAULT_CKPT_DIR,
    DEFAULT_IMAGE,
    Qwen3VLRuntime,
    _sync,
    build_messages,
)
from model_catalog import get_model_spec


RESULT_PREFIX = "CONTEXT_RESULT_JSON="
FILLER_UNIT = " context"


def _prepare_for_target(runtime: Qwen3VLRuntime, image: str, target: int, max_side: int):
    media = runtime.prepare_media([("image", image)], max_side)

    def render(repetitions: int):
        prompt = "Measure multimodal long-context inference." + FILLER_UNIT * repetitions
        messages = build_messages(media, prompt)
        inputs = runtime.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return prompt, inputs, int(inputs["input_ids"].shape[1])

    _, base_inputs, base_tokens = render(0)
    if target <= base_tokens:
        return base_inputs.to(runtime.device), base_tokens

    low, high = 0, max(1, target - base_tokens)
    while render(high)[2] < target:
        high *= 2
    best_inputs, best_tokens = base_inputs, base_tokens
    while low <= high:
        middle = (low + high) // 2
        _, candidate_inputs, tokens = render(middle)
        if tokens <= target:
            best_inputs, best_tokens = candidate_inputs, tokens
            low = middle + 1
        else:
            high = middle - 1
    return best_inputs.to(runtime.device), best_tokens


def child_attempt(args) -> dict[str, object]:
    started = time.perf_counter()
    try:
        runtime = Qwen3VLRuntime(
            model_size=args.model,
            device=args.device,
            model_path=args.model_path,
            ckpt_dir=args.ckpt_dir,
            kernel_dir=args.kernel_dir,
            cpu_threads=args.cpu_threads,
            seed=args.seed,
            verbose=False,
            yarn_1m=args.yarn_1m,
            gpu_placement=args.gpu_placement,
        )
        inputs, prompt_tokens = _prepare_for_target(
            runtime, args.image, args.child_target, args.max_image_side
        )
        context_limit = int(runtime.model.config.get_text_config().max_position_embeddings)
        if prompt_tokens + args.reserve > context_limit:
            return {
                "status": "model_limit",
                "target_tokens": args.child_target,
                "prompt_tokens": prompt_tokens,
                "reserve_tokens": args.reserve,
                "context_limit": context_limit,
            }

        if args.device == "cuda":
            torch.cuda.empty_cache()
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
        _sync(args.device)
        infer_started = time.perf_counter()
        image_token_id = getattr(runtime.model.config, "image_token_id", None)
        visual_placeholder_tokens = (
            int((inputs["input_ids"] == image_token_id).sum().item())
            if image_token_id is not None
            else None
        )
        image_grid = inputs.get("image_grid_thw")
        with torch.inference_mode():
            output = runtime.model.generate(
                **inputs,
                min_new_tokens=args.reserve,
                max_new_tokens=args.reserve,
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                use_cache=True,
                logits_processor=None,
            )
        _sync(args.device)
        generated = int(output.shape[1] - inputs["input_ids"].shape[1])
        return {
            "status": "success",
            "target_tokens": args.child_target,
            "prompt_tokens": prompt_tokens,
            "reserve_tokens": args.reserve,
            "generated_tokens": generated,
            "visual_placeholder_tokens": visual_placeholder_tokens,
            "image_grid_thw": image_grid.detach().cpu().tolist() if image_grid is not None else None,
            "context_mode": runtime.context_mode,
            "context_limit": context_limit,
            "inference_seconds": time.perf_counter() - infer_started,
            "process_seconds": time.perf_counter() - started,
            "peak_allocated_mb": (
                round(torch.cuda.max_memory_allocated() / (1024**2), 2)
                if args.device == "cuda"
                else None
            ),
            "peak_reserved_mb": (
                round(torch.cuda.max_memory_reserved() / (1024**2), 2)
                if args.device == "cuda"
                else None
            ),
            "peak_allocated_mb_per_device": (
                {
                    str(index): round(torch.cuda.max_memory_allocated(index) / (1024**2), 2)
                    for index in range(torch.cuda.device_count())
                }
                if args.device == "cuda"
                else None
            ),
            "peak_reserved_mb_per_device": (
                {
                    str(index): round(torch.cuda.max_memory_reserved(index) / (1024**2), 2)
                    for index in range(torch.cuda.device_count())
                }
                if args.device == "cuda"
                else None
            ),
        }
    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        return {
            "status": "oom",
            "target_tokens": args.child_target,
            "reserve_tokens": args.reserve,
            "error_type": type(exc).__name__,
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return {
                "status": "oom",
                "target_tokens": args.child_target,
                "reserve_tokens": args.reserve,
                "error_type": type(exc).__name__,
            }
        raise


def _child_command(args, target: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--device",
        args.device,
        "--ckpt-dir",
        args.ckpt_dir,
        "--image",
        args.image,
        "--max-image-side",
        str(args.max_image_side),
        "--reserve",
        str(args.reserve),
        "--seed",
        str(args.seed),
        "--cpu-threads",
        str(args.cpu_threads),
        "--child-target",
        str(target),
        "--gpu-placement",
        args.gpu_placement,
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    if args.kernel_dir:
        command.extend(["--kernel-dir", args.kernel_dir])
    if args.yarn_1m:
        command.append("--yarn-1m")
    return command


def run_candidate(args, target: int) -> dict[str, object]:
    print(f"context candidate: {target} tokens", flush=True)
    try:
        completed = subprocess.run(
            _child_command(args, target),
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "target_tokens": target,
            "timeout_seconds": args.timeout_seconds,
            "exit_code": None,
        }
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
            result["exit_code"] = completed.returncode
            return result
    return {
        "status": "crash",
        "target_tokens": target,
        "exit_code": completed.returncode,
        "signal": -completed.returncode if completed.returncode < 0 else None,
    }


def search(args) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    target = args.start
    largest_success: int | None = None
    first_failure: int | None = None

    while target <= args.max_tokens:
        result = run_candidate(args, target)
        attempts.append(result)
        if result["status"] == "success":
            largest_success = target
            if target == args.max_tokens:
                break
            target = min(args.max_tokens, target * 2)
            if target == largest_success:
                break
        else:
            first_failure = target
            break

    if largest_success is None and first_failure is not None:
        candidate = (first_failure // 2 // args.resolution) * args.resolution
        while candidate >= args.resolution:
            result = run_candidate(args, candidate)
            attempts.append(result)
            if result["status"] == "success":
                largest_success = candidate
                break
            first_failure = candidate
            candidate = (candidate // 2 // args.resolution) * args.resolution

    if largest_success is not None and first_failure is not None:
        low = largest_success + args.resolution
        high = first_failure - args.resolution
        while low <= high:
            units = (low + high) // (2 * args.resolution)
            candidate = max(low, units * args.resolution)
            result = run_candidate(args, candidate)
            attempts.append(result)
            if result["status"] == "success":
                largest_success = candidate
                low = candidate + args.resolution
            else:
                first_failure = candidate
                high = candidate - args.resolution

    spec = get_model_spec(args.model)
    image_path = Path(args.image).expanduser().resolve()
    image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    successful_attempts = [item for item in attempts if item["status"] == "success"]
    largest_attempt = max(
        successful_attempts,
        key=lambda item: int(item["target_tokens"]),
        default=None,
    )
    environment: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [],
    }
    if args.device == "cuda" and torch.cuda.is_available():
        environment["gpus"] = [
            {
                "index": index,
                "name": (props := torch.cuda.get_device_properties(index)).name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_mb": round(props.total_memory / (1024**2), 2),
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": {
            "key": args.model,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
        },
        "device_mode": "gpu_fp8" if args.device == "cuda" else "cpu_fp32",
        "gpu_placement": args.gpu_placement,
        "context_mode": "yarn_1m" if args.yarn_1m else "native_256k",
        "method": "one-image text-dominant KV-cache pressure test",
        "image_sha256": image_digest,
        "max_image_side_pre_resize": args.max_image_side,
        "reserve_tokens": args.reserve,
        "start_tokens": args.start,
        "configured_max_tokens": args.max_tokens,
        "resolution_tokens": args.resolution,
        "largest_success_target": largest_success,
        "largest_success_prompt_tokens": (
            largest_attempt.get("prompt_tokens") if largest_attempt is not None else None
        ),
        "first_failure_target": first_failure,
        "practical_limit_interval": {
            "proven_at_least_tokens": largest_success,
            "failed_at_or_below_tokens": first_failure,
        },
        "environment": environment,
        "attempts": attempts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find the practical Qwen3-VL context limit")
    parser.add_argument("--model", choices=("2b", "4b", "8b"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-path")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--kernel-dir")
    parser.add_argument(
        "--gpu-placement",
        choices=("single", "auto", "balanced", "balanced_low_0", "sequential"),
        default="single",
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--max-image-side", type=int, default=224)
    parser.add_argument("--reserve", type=int, default=32)
    parser.add_argument("--start", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--yarn-1m",
        action="store_true",
        help="test Qwen's official factor-3 Interleaved-MRoPE 1M overlay",
    )
    parser.add_argument("--child-target", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tokens is None:
        args.max_tokens = 1_000_000 if args.yarn_1m else 262_144
    if (
        args.reserve < 1
        or args.start < 1
        or args.resolution < 1
        or args.timeout_seconds < 1
        or args.max_image_side < 1
        or args.cpu_threads < 1
        or args.max_tokens < 1
    ):
        raise ValueError("reserve/start/resolution/timeout/image-side/cpu-threads must be positive")
    if args.start > args.max_tokens:
        raise ValueError("--start cannot exceed --max-tokens")
    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(f"context image does not exist: {args.image}")
    if args.child_target:
        result = child_attempt(args)
        print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")))
        return 0 if result["status"] == "success" else 3

    result = search(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"context sweep JSON: {args.output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

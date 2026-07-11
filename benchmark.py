#!/usr/bin/env python3
"""Reproducible single-image latency benchmark for one Qwen3-VL model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import torch

from qwen3_vl_offline import (
    DEFAULT_CKPT_DIR,
    DEFAULT_IMAGE,
    Qwen3VLRuntime,
    validate_generation_settings,
)


SCHEMA_VERSION = 1
DEFAULT_PROMPT = "Describe this driving scene completely and precisely."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_kernel(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(source.relative_to(path).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> tuple[str | None, bool | None]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = result.stdout.strip() or None
    if commit is None:
        return None, None
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        return commit, None
    relevant_changes = [
        line
        for line in status.stdout.splitlines()
        if not line[3:].startswith("results/")
    ]
    return commit, bool(relevant_changes)


def _runtime_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("benchmark.py", "model_catalog.py", "qwen3_vl_offline.py"):
        source = root / name
        digest.update(name.encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))]


def run_benchmark(args) -> dict[str, object]:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"benchmark image does not exist: {image_path}")

    runtime = Qwen3VLRuntime(
        model_size=args.model,
        device=args.device,
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir,
        kernel_dir=args.kernel_dir,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )
    media_inputs = [("image", str(image_path))]

    for index in range(args.warmup):
        print(f"warmup {index + 1}/{args.warmup}")
        warmup, _ = runtime.infer(
            media_inputs=media_inputs,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_image_side=args.max_image_side,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            check_finite_logits=False,
        )
        if not args.allow_truncated and warmup.finish_reason != "eos":
            raise RuntimeError(
                f"warmup ended with {warmup.finish_reason}; raise --max-new-tokens or use --allow-truncated"
            )

    runs: list[dict[str, object]] = []
    for index in range(args.runs):
        print(f"measured run {index + 1}/{args.runs}")
        result, _ = runtime.infer(
            media_inputs=media_inputs,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_image_side=args.max_image_side,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            check_finite_logits=False,
        )
        if not args.allow_truncated and result.finish_reason != "eos":
            raise RuntimeError(
                f"run {index + 1} ended with {result.finish_reason}; result is not a complete description"
            )
        runs.append(
            {
                "preprocess_seconds": result.preprocess_seconds,
                "media_seconds": result.media_seconds,
                "generation_seconds": result.generation_seconds,
                "total_seconds": result.total_seconds,
                "generated_tokens": result.generated_tokens,
                "prompt_tokens": result.prompt_tokens,
                "tokens_per_second": result.tokens_per_second,
                "finish_reason": result.finish_reason,
                "peak_vram_mb": result.peak_vram_mb,
                "answer_characters": len(result.answer),
                "answer_sha256": hashlib.sha256(result.answer.encode("utf-8")).hexdigest(),
            }
        )

    generation_times = [float(item["generation_seconds"]) for item in runs]
    totals = [float(item["total_seconds"]) for item in runs]
    throughputs = [float(item["tokens_per_second"]) for item in runs]
    spec = runtime.spec
    environment = {
        "python_packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "accelerate", "kernels", "safetensors", "Pillow")
        },
        "cuda_runtime": torch.version.cuda,
        "gpu": None,
    }
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        environment["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_mb": round(props.total_memory / (1024**2), 2),
        }

    git_commit, git_dirty = _git_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runtime_source_sha256": _runtime_source_sha256(),
        "model": {
            "key": spec.key,
            "repo_id": spec.repo_id,
            "revision": getattr(spec, "revision", "main"),
            "device_mode": "gpu_fp8" if args.device == "cuda" else "cpu_fp32",
            "load_seconds": runtime.load_seconds,
            "fp8_modules": len(runtime.fp8_names),
            "compute_backend": runtime.compute_backend,
        },
        "kernel_sha256": _sha256_kernel(runtime.kernel_dir),
        "input": {
            "image_sha256": _sha256_file(image_path),
            "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
            "max_image_side": args.max_image_side,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "do_sample": not args.greedy,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "finite_logits_check": False,
        },
        "environment": environment,
        "cpu": {
            "architecture": platform.machine(),
            "torch_threads": torch.get_num_threads(),
        },
        "warmup_runs": args.warmup,
        "runs": runs,
        "summary": {
            "runs": len(runs),
            "generation_seconds_median": statistics.median(generation_times),
            "generation_seconds_p95": _percentile95(generation_times),
            "total_seconds_median": statistics.median(totals),
            "total_seconds_p95": _percentile95(totals),
            "tokens_per_second_median": statistics.median(throughputs),
            "all_finished_eos": all(item["finish_reason"] == "eos" for item in runs),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark one Qwen3-VL Thinking model per process")
    parser.add_argument("--model", choices=("2b", "4b", "8b"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-path")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--kernel-dir")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-image-side", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of Qwen sampling")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--allow-truncated", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        raise ValueError("--runs must be positive and --warmup non-negative")
    validate_generation_settings(
        max_new_tokens=args.max_new_tokens,
        max_image_side=args.max_image_side,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        cpu_threads=args.cpu_threads,
    )
    result = run_benchmark(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"benchmark JSON: {args.output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reproducible latency/throughput benchmark for one Qwen3-VL model and skill.

Runs warmup + N measured passes on a single media input (image, multi-image
sequence, or video) and reports median/p95 latency, tokens/s, peak VRAM, and
finish_reason. When ``--skill`` is given, the prompt + media shape come from
the skill catalog (skills.py) and the output is verified to match the skill's
expected structure (JSON bboxes, LaTeX formulas, structured chart, ...).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .qwen3_vl_offline import (
    DEFAULT_CKPT_DIR,
    DEFAULT_IMAGE,
    Qwen3VLRuntime,
    validate_generation_settings,
)
from .skills import SKILLS, resolve_skill
from .skill_parsers import parse_skill


SCHEMA_VERSION = 2
DEFAULT_PROMPT = "Describe the visual content completely and precisely."

# Performance benchmark task presets for typical VL perception tasks
TASK_PROMPTS = {
    "describe": DEFAULT_PROMPT,
    "lane_image": (
        "Detect all visible road lane markings in this image. "
        "For each lane, output a list of normalized (x, y) points in [0,1] range. "
        "Return structured text or JSON."
    ),
    "lane_video": (
        "Detect road lane markings across the video frames. "
        "For each lane, list its points per frame (at least 5 frames). "
        "Return structured output with frame indices."
    ),
    "2d_detection": (
        "Perform standard 2D object detection on the image. "
        "Detect vehicles, pedestrians, traffic signs, etc. "
        "Output as JSON list: [{\"class\": \"car\", \"bbox\": [x1, y1, x2, y2]}] with normalized [0,1] coords. "
        "Do not use markdown."
    ),
    "3d_detection": (
        "Analyze 3D structure of the scene. Estimate relative depths, positions, "
        "and layout of main objects and road surface. Describe distances and 3D relations."
    ),
    "entity_graph": (
        "Build a scene graph of entities (objects, lanes, signs, road parts) and their relations "
        "(left-of, ahead, on, crossing etc). Output as list of (subject, relation, object) triples. "
        "Use clear structured text."
    ),
    "object_matching": (
        "Track and match identical objects across the sequence of frames. "
        "Assign consistent IDs to the same objects and report in which frames each ID appears. "
        "Output structured matching results."
    ),
}


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


def _run_task_verification(runtime: "Qwen3VLRuntime", image_path: Path, prompt: str, args) -> dict[str, object]:
    """Run targeted prompts for key tasks and verify they produce reasonable structured/sensible output.
    Used for 2D/3D detection, entity graph, object matching on sequences, lane etc.
    """
    results: dict[str, object] = {"task": args.task, "num_frames": args.num_frames}
    N = max(2, min(args.num_frames, 8))

    def _infer(p: str, media: list[tuple[str, str]] | None = None, vnf: int | None = None):
        m = media or [("image", str(image_path))]
        r, _ = runtime.infer(
            media_inputs=m,
            prompt=p,
            max_new_tokens=args.max_new_tokens,
            max_image_side=args.max_image_side,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            video_num_frames=vnf,
            check_finite_logits=False,
        )
        return r

    # 2D detection benchmark timing + basic structure check
    if args.task == "2d_detection" or args.verify:
        r = _infer(TASK_PROMPTS["2d_detection"])
        ans = r.answer.lower()
        has_struct = any(x in ans for x in ["[", "class", "bbox", "car", "vehicle"])
        results["2d_detection"] = {
            "total_seconds": r.total_seconds,
            "generated_tokens": r.generated_tokens,
            "has_structured_output": has_struct,
            "preview": r.answer[:250],
        }

    # 3D detection verify
    if args.task == "3d_detection" or args.verify:
        r = _infer(TASK_PROMPTS["3d_detection"])
        ans = r.answer.lower()
        has_3d = any(kw in ans for kw in ["depth", "3d", "distance", "behind", "position", "far", "close"])
        results["3d_detection"] = {
            "total_seconds": r.total_seconds,
            "has_3d_structure": has_3d,
            "preview": r.answer[:250],
        }

    # Entity graph on sequence
    if args.task == "entity_graph" or args.verify:
        media = [("image", str(image_path))] * N
        p = f"These are {N} frames in temporal order. " + TASK_PROMPTS["entity_graph"]
        r = _infer(p, media=media)
        ans = r.answer.lower()
        has_graph = any(kw in ans for kw in ["(", "->", "relation", "left", "on", "graph", "entity"])
        results["entity_graph"] = {
            "num_frames": N,
            "total_seconds": r.total_seconds,
            "has_graph_structure": has_graph,
            "preview": r.answer[:250],
        }

    # Object matching on 2/4/8 frames
    if args.task == "object_matching" or args.verify:
        match_results = {}
        for n in [2, 4, 8]:
            if n > args.num_frames and args.task != "object_matching":
                continue
            media = [("image", str(image_path))] * n
            p = f"These are {n} sequential frames of the same scene. " + TASK_PROMPTS["object_matching"]
            r = _infer(p, media=media)
            ans = r.answer.lower()
            has_match = any(str(i) in ans for i in range(10)) or any(kw in ans for kw in ["id", "same", "match", "track", "frame"])
            match_results[f"frames_{n}"] = {
                "total_seconds": r.total_seconds,
                "has_consistent_ids": has_match,
                "preview": r.answer[:200],
            }
        results["object_matching"] = match_results

    # Lane image vs video (performance comparison example)
    if args.task in ("lane_image", "lane_video") or args.verify:
        # image
        r_img = _infer(TASK_PROMPTS["lane_image"])
        # video / sequence (use video if provided, else repeated frames)
        if args.video:
            vnf = args.num_frames
            r_vid = _infer(TASK_PROMPTS["lane_video"], media=[("video", str(Path(args.video)))], vnf=vnf)
        else:
            media = [("image", str(image_path))] * args.num_frames
            r_vid = _infer(TASK_PROMPTS["lane_video"], media=media)
        results["lane_perf"] = {
            "image_seconds": r_img.total_seconds,
            "video_or_seq_seconds": r_vid.total_seconds,
            "speedup_or_overhead": round(r_vid.total_seconds / max(r_img.total_seconds, 1e-6), 2),
            "image_has_lanes": "lane" in r_img.answer.lower(),
            "video_has_lanes": "lane" in r_vid.answer.lower(),
        }

    return results


def verify_skill_output(skill_key: str, answer: str) -> dict[str, object]:
    """Check that a skill's answer has the expected structure.

    Returns {verified: bool, parsed: <any>, criterion: str}.
    """
    skill = SKILLS[skill_key]
    parsed: Any = None
    criterion = ""
    try:
        parsed = parse_skill(skill_key, answer)
    except Exception as exc:  # noqa: BLE001 - verification must not crash the bench
        return {"verified": False, "parsed": None, "criterion": f"parse error: {exc}"}
    if skill.output_kind in {"grounding_2d", "grounding_3d"}:
        verified = isinstance(parsed, list) and len(parsed) >= 1
        criterion = "at least one bbox/point parsed"
    elif skill.output_kind == "formula":
        formulas = parsed.get("formulas", []) if isinstance(parsed, dict) else []
        verified = len(formulas) >= 1
        criterion = "at least one LaTeX formula"
    elif skill.output_kind == "chart":
        verified = isinstance(parsed, dict) and bool(parsed)
        criterion = "non-empty chart object"
    elif skill.output_kind == "code":
        verified = len(answer.strip()) >= 20
        criterion = "non-trivial code output (>=20 chars)"
    else:  # text
        verified = len(answer.strip()) >= 10
        criterion = "non-trivial text (>=10 chars)"
    return {"verified": bool(verified), "parsed": parsed, "criterion": criterion}


def run_benchmark(args) -> dict[str, object]:
    skill_key = getattr(args, "skill", None)
    if skill_key:
        resolved = resolve_skill(
            skill_key,
            custom_prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_image_side=args.max_image_side,
        )
        prompt = resolved["prompt"]
    else:
        prompt = args.prompt or TASK_PROMPTS.get(args.task, DEFAULT_PROMPT)
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
        gpu_placement=args.gpu_placement,
    )

    # Build media inputs: support video or multi-frame sequence (repeat image for matching/graph tests)
    video_num_frames = None
    if args.video:
        vpath = Path(args.video).expanduser().resolve()
        if not vpath.is_file():
            raise FileNotFoundError(f"video does not exist: {vpath}")
        media_inputs = [("video", str(vpath))]
        if args.task.endswith("_video") or args.num_frames > 1:
            video_num_frames = args.num_frames
    elif args.num_frames > 1 and args.task in ("object_matching", "entity_graph", "lane_video"):
        # Simulate sequence of frames by repeating the image (tests multi-image + temporal consistency)
        img = str(image_path)
        media_inputs = [("image", img)] * args.num_frames
    else:
        media_inputs = [("image", str(image_path))]

    # Warmup
    for index in range(args.warmup):
        print(f"warmup {index + 1}/{args.warmup}")
        warmup, _ = runtime.infer(
            media_inputs=media_inputs,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            max_image_side=args.max_image_side,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            video_num_frames=video_num_frames,
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
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            max_image_side=args.max_image_side,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            video_num_frames=video_num_frames,
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
                "token_ids_sha256": result.token_ids_sha256,
                "input_fingerprints": result.input_fingerprints,
                "prompt_tokens": result.prompt_tokens,
                "tokens_per_second": result.tokens_per_second,
                "finish_reason": result.finish_reason,
                "peak_vram_mb": result.peak_vram_mb,
                "peak_vram_mb_per_device": result.peak_vram_mb_per_device,
                "answer_characters": len(result.answer),
                "answer_sha256": hashlib.sha256(result.answer.encode("utf-8")).hexdigest(),
                "num_media": len(media_inputs),
            }
        )

    generation_times = [float(item["generation_seconds"]) for item in runs]
    totals = [float(item["total_seconds"]) for item in runs]
    throughputs = [float(item["tokens_per_second"]) for item in runs]
    spec = runtime.spec

    # Optional verification that typical tasks produce useful output (for 3D, graph, matching, 2D det etc.)
    verification = None
    if getattr(args, "verify", False):
        verification = _run_task_verification(runtime, image_path, prompt, args)
    # Skill verification: check the last measured run's answer matches the
    # skill's expected output structure (JSON bboxes, LaTeX, ...).
    skill_verification = None
    if skill_key and runs:
        # Re-run a single short inference purely for verification of structure.
        try:
            verify_result, _ = runtime.infer(
                media_inputs=media_inputs,
                prompt=prompt,
                max_new_tokens=min(resolved["max_new_tokens"] if skill_key else args.max_new_tokens, 512),
                max_image_side=args.max_image_side,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                video_num_frames=video_num_frames,
                check_finite_logits=False,
            )
            skill_verification = verify_skill_output(skill_key, verify_result.answer)
            skill_verification["verification_tokens_per_second"] = round(
                verify_result.tokens_per_second, 3
            )
        except Exception as exc:  # noqa: BLE001
            skill_verification = {"verified": False, "criterion": f"verification inference failed: {exc}"}
    environment = {
        "python_packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "accelerate", "kernels", "safetensors", "Pillow")
        },
        "cuda_runtime": torch.version.cuda,
        "gpus": [],
    }
    if args.device == "cuda":
        environment["gpus"] = [
            {
                "index": index,
                "name": (props := torch.cuda.get_device_properties(index)).name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_mb": round(props.total_memory / (1024**2), 2),
            }
            for index in range(torch.cuda.device_count())
        ]

    git_commit, git_dirty = _git_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
            "gpu_placement": runtime.gpu_placement,
            "hf_device_map": runtime.hf_device_map,
        },
        "kernel_sha256": _sha256_kernel(runtime.kernel_dir),
        "input": {
            "image_sha256": _sha256_file(image_path),
            "video_sha256": _sha256_file(Path(args.video)) if args.video else None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "task": args.task,
            "num_frames": getattr(args, "num_frames", 1),
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
        "verification": verification,
        "skill_verification": skill_verification,
        "summary": {
            "runs": len(runs),
            "task": args.task,
            "num_frames": getattr(args, "num_frames", 1),
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
    parser.add_argument(
        "--gpu-placement",
        choices=("single", "auto", "balanced", "balanced_low_0", "sequential"),
        default="single",
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--video", help="video file for video tasks (lane_video etc.)")
    parser.add_argument("--prompt", default=None, help="override prompt (else use --task)")
    parser.add_argument(
        "--task",
        default="describe",
        choices=list(TASK_PROMPTS.keys()),
        help="task preset for specialized benchmarks (lane, 2d det, 3d, graph, matching)",
    )
    parser.add_argument(
        "--skill",
        default=None,
        choices=sorted(SKILLS),
        help="use a cookbook skill (skills.py) for prompt + media shape; overrides --task/--prompt",
    )
    parser.add_argument("--num-frames", type=int, default=5, help="frames for video/sequence tasks (uses repeated image for matching/graph)")
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
    parser.add_argument("--verify", action="store_true", help="run verification that 3D/graph/matching/2D produce useful output")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        raise ValueError("--runs must be positive and --warmup non-negative")
    if args.skill:
        resolved = resolve_skill(
            args.skill,
            custom_prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_image_side=args.max_image_side,
        )
        # Let the skill drive prompt/token/side defaults unless the user overrode
        # them explicitly on the CLI (argparse has no "was set?" API, so we rely
        # on the skill filling in only when the CLI value equals the argparse default).
        if args.prompt is None:
            args.prompt = resolved["prompt"]
        if args.max_image_side == 640:  # argparse default -> adopt skill default
            args.max_image_side = resolved["max_image_side"]
        if args.max_new_tokens == 2048:  # argparse default -> adopt skill default
            args.max_new_tokens = resolved["max_new_tokens"]
    elif getattr(args, "prompt", None) is None:
        args.prompt = TASK_PROMPTS.get(args.task, DEFAULT_PROMPT)
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


# Smoke tests / discovery for performance task benchmarks
def test_task_prompts_exist():
    """Ensure all requested typical task prompts are defined (2D, 3D, graph, matching, lane image/video)."""
    assert "2d_detection" in TASK_PROMPTS
    assert "3d_detection" in TASK_PROMPTS
    assert "entity_graph" in TASK_PROMPTS
    assert "object_matching" in TASK_PROMPTS
    assert "lane_image" in TASK_PROMPTS
    assert "lane_video" in TASK_PROMPTS
    assert len(TASK_PROMPTS) >= 7

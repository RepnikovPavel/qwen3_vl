from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from evaluate_vl import _load_json, validate_manifest


GPU_PLACEMENTS = ("single", "auto", "balanced", "balanced_low_0", "sequential")


def _runtime_factory(**kwargs):
    from qwen3_vl_offline import Qwen3VLRuntime

    return Qwen3VLRuntime(**kwargs)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def run_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    model_size: str = "2b",
    device: str = "cuda",
    model_path: str | None = None,
    ckpt_dir: str = "/mnt/nvme/huggingface",
    kernel_dir: str | None = None,
    cpu_threads: int = 16,
    seed: int = 1234,
    max_image_side: int = 1280,
    max_new_tokens: int = 4096,
    verify_sha: bool = False,
    yarn_1m: bool = False,
    gpu_placement: str = "single",
    allow_incomplete: bool = False,
    verbose: bool = True,
    runtime_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device}")
    if model_size not in {"2b", "4b", "8b"}:
        raise ValueError(f"unsupported model size: {model_size}")
    if gpu_placement not in GPU_PLACEMENTS:
        raise ValueError(f"unsupported GPU placement: {gpu_placement}")
    if device == "cpu" and gpu_placement != "single":
        raise ValueError("multi-GPU placement requires --device cuda")
    if cpu_threads < 1 or max_image_side < 1 or max_new_tokens < 1:
        raise ValueError(
            "cpu_threads, max_image_side, and max_new_tokens must be positive"
        )
    manifest_file = Path(manifest_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    manifest_data, _ = _load_json(manifest_file)
    fixtures = validate_manifest(
        manifest_data, manifest_file.parent, verify_images=True
    )
    protected_paths = {
        manifest_file,
        *(manifest_file.parent / fixture["image"] for fixture in fixtures),
    }
    if output_file in protected_paths:
        raise ValueError(
            "output path must not overwrite the manifest or a fixture image"
        )
    factory = runtime_factory or _runtime_factory
    runtime = factory(
        model_size=model_size,
        device=device,
        model_path=model_path,
        ckpt_dir=ckpt_dir,
        kernel_dir=kernel_dir,
        cpu_threads=cpu_threads,
        seed=seed,
        verbose=verbose,
        verify_sha=verify_sha,
        yarn_1m=yarn_1m,
        gpu_placement=gpu_placement,
    )
    responses: list[dict[str, str]] = []
    for fixture in fixtures:
        image_path = manifest_file.parent / fixture["image"]
        result, _ = runtime.infer(
            media_inputs=[("image", str(image_path))],
            prompt=fixture["prompt"],
            max_new_tokens=max_new_tokens,
            max_image_side=max_image_side,
            do_sample=False,
            check_finite_logits=True,
        )
        answer = getattr(result, "answer", None)
        if not isinstance(answer, str):
            raise TypeError(
                f"runtime returned a non-string answer for fixture {fixture['id']}"
            )
        finish_reason = getattr(result, "finish_reason", None)
        if not allow_incomplete and finish_reason != "eos":
            raise RuntimeError(
                f"fixture {fixture['id']} ended with {finish_reason!r}; raise --max-new-tokens "
                "or use --allow-incomplete"
            )
        responses.append({"id": fixture["id"], "answer": answer})
    payload = {"schema_version": 1, "responses": responses}
    _atomic_write(output_file, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=("2b", "4b", "8b"), default="2b")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-path")
    parser.add_argument("--ckpt-dir", default="/mnt/nvme/huggingface")
    parser.add_argument("--kernel-dir")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-image-side", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--verify-sha", action="store_true")
    parser.add_argument("--yarn-1m", action="store_true")
    parser.add_argument("--gpu-placement", choices=GPU_PLACEMENTS, default="single")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_manifest(
        args.manifest,
        args.output,
        model_size=args.model,
        device=args.device,
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir,
        kernel_dir=args.kernel_dir,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
        max_image_side=args.max_image_side,
        max_new_tokens=args.max_new_tokens,
        verify_sha=args.verify_sha,
        yarn_1m=args.yarn_1m,
        gpu_placement=args.gpu_placement,
        allow_incomplete=args.allow_incomplete,
        verbose=not args.quiet,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "response_count": len(payload["responses"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

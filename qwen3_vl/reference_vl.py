from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from .parity import (
    SCHEMA_VERSION,
    TOKEN_ENCODING,
    build_parity_artifact,
    compare_artifacts,
)
from .qwen3_vl_offline import (
    DEFAULT_CKPT_DIR,
    DEFAULT_IMAGE,
    GPU_PLACEMENTS,
    Qwen3VLRuntime,
    _sync,
    validate_generation_settings,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eos_ids(model) -> set[int]:
    value = model.generation_config.eos_token_id
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value or []}


def _finish_reason(model, token_ids: Sequence[int], maximum: int) -> str:
    if token_ids and token_ids[-1] in _eos_ids(model):
        return "eos"
    if len(token_ids) >= maximum:
        return "max_new_tokens"
    return "stopped"


def _candidate_artifact(result, metadata: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_fingerprints": result.input_fingerprints,
        "continuation": {
            "encoding": TOKEN_ENCODING,
            "length": len(result.token_ids),
            "sha256": result.token_ids_sha256,
            "token_ids": list(result.token_ids),
        },
        "metadata": metadata,
    }


def _reference_generate(
    runtime,
    image_path: str,
    prompt: str,
    maximum: int,
    side: int,
    *,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
):
    media = runtime.prepare_media([("image", image_path)], side)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": media[0].value},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = runtime.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        add_vision_id=False,
        processor_kwargs={},
    )
    torch.manual_seed(runtime.seed)
    if runtime.device == "cuda":
        torch.cuda.manual_seed_all(runtime.seed)
    device_inputs = inputs.to(runtime.device)
    _sync(runtime.device)
    with torch.inference_mode():
        output = runtime.model.generate(
            **device_inputs,
            max_new_tokens=maximum,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            use_cache=True,
        )
    _sync(runtime.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    token_ids = output[0, prompt_tokens:].tolist()
    return build_parity_artifact(
        inputs,
        token_ids,
        metadata={
            "implementation": "direct_transformers_generate",
            "finish_reason": _finish_reason(runtime.model, token_ids, maximum),
            "prompt_tokens": prompt_tokens,
        },
    )


def _validate_args(args) -> None:
    validate_generation_settings(
        max_new_tokens=args.max_new_tokens,
        max_image_side=args.max_image_side,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        cpu_threads=args.cpu_threads,
    )


def run(args) -> dict[str, object]:
    _validate_args(args)
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    runtime = Qwen3VLRuntime(
        model_size=args.model,
        device=args.device,
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir,
        kernel_dir=args.kernel_dir,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
        verify_sha=args.verify_sha,
        gpu_placement=args.gpu_placement,
    )
    shared_metadata = {
        "model": runtime.spec.repo_id,
        "revision": runtime.spec.revision,
        "device": args.device,
        "gpu_placement": runtime.gpu_placement,
        "image_sha256": _sha256_file(image_path),
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "max_image_side": args.max_image_side,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "decode": "sampling" if args.sample else "greedy",
        "temperature": args.temperature if args.sample else None,
        "top_p": args.top_p if args.sample else None,
        "top_k": args.top_k if args.sample else None,
    }
    reference = _reference_generate(
        runtime,
        str(image_path),
        args.prompt,
        args.max_new_tokens,
        args.max_image_side,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    reference["metadata"].update(shared_metadata)
    candidate_result, _ = runtime.infer(
        media_inputs=[("image", str(image_path))],
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_image_side=args.max_image_side,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        check_finite_logits=False,
    )
    candidate = _candidate_artifact(
        candidate_result,
        {
            **shared_metadata,
            "implementation": "qwen3_vl_runtime",
            "finish_reason": candidate_result.finish_reason,
            "prompt_tokens": candidate_result.prompt_tokens,
        },
    )
    comparison = compare_artifacts(reference, candidate, require_token_ids=True)
    complete = (
        reference["metadata"]["finish_reason"] == "eos"
        and candidate["metadata"]["finish_reason"] == "eos"
    )
    return {
        "schema_version": 1,
        "proof_basis": "direct_generate_vs_runtime_wrapper_same_loaded_model",
        "proof_scope": "complete_answer" if complete else "generated_prefix",
        "reference": reference,
        "candidate": candidate,
        "comparison": comparison,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL direct-reference parity run")
    parser.add_argument("--model", choices=("2b", "4b", "8b"), default="2b")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-path")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--kernel-dir")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument(
        "--prompt", default="Describe the scene completely and precisely."
    )
    parser.add_argument("--max-image-side", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--gpu-placement", choices=GPU_PLACEMENTS, default="single")
    parser.add_argument("--verify-sha", action="store_true")
    parser.add_argument("--require-eos", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        separators=(",", ":") if args.compact else None,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["comparison"]["match"]:
        return 1
    if args.require_eos and result["proof_scope"] != "complete_answer":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

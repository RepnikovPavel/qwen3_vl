#!/usr/bin/env python3
"""Run a single Qwen3-VL skill locally (CLI entry point for `qwen3-vl skill`).

Loads the FP8 runtime once, runs the resolved skill prompt on the provided
media, post-processes the output with the skill parser, and (for grounding
skills) renders an annotated image. Mirrors the per-skill cookbook flow but
through the shared offline runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from qwen3_vl_offline import DEFAULT_CKPT_DIR, Qwen3VLRuntime
from skills import SkillError, get_skill, resolve_skill
from skill_parsers import coord_scale, parse_skill


def _draw_grounding(image_path: str, parsed: list[dict[str, Any]], scale: int, out_path: Path) -> str:
    """Render parsed grounding/points onto the image, honoring coord scale."""
    from PIL import Image
    from demo.grounding_viz import COLORS, _FONT
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    if not parsed:
        image.save(out_path)
        return str(out_path)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for index, item in enumerate(parsed):
        color = COLORS[index % len(COLORS)]
        label = item.get("label") or item.get("name") or ""
        if "bbox_2d" in item and len(item["bbox_2d"]) >= 4:
            x1 = int(item["bbox_2d"][0] / scale * width)
            y1 = int(item["bbox_2d"][1] / scale * height)
            x2 = int(item["bbox_2d"][2] / scale * width)
            y2 = int(item["bbox_2d"][3] / scale * height)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
            if label:
                draw.text((x1 + 6, y1 + 4), str(label), fill=color, font=_FONT)
        elif "point_2d" in item and len(item["point_2d"]) >= 2:
            x = int(item["point_2d"][0] / scale * width)
            y = int(item["point_2d"][1] / scale * height)
            radius = 5
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)
            if label:
                draw.text((x + radius + 2, y - radius), str(label), fill=color, font=_FONT)
    canvas.save(out_path)
    return str(out_path)


def run_skill(args) -> dict[str, Any]:
    skill = get_skill(args.skill)
    resolved = resolve_skill(
        args.skill,
        custom_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_image_side=args.max_image_side,
    )
    if not args.media:
        raise SkillError(f"skill {args.skill!r} requires --image/--video media input")
    runtime = Qwen3VLRuntime(
        model_size=args.model,
        device=args.device,
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir,
        kernel_dir=args.kernel_dir,
        seed=args.seed,
        gpu_placement=args.gpu_placement,
    )
    video_num_frames = None
    if resolved["video_num_frames"] is not None:
        video_num_frames = resolved["video_num_frames"]
    if args.video_frames is not None:
        video_num_frames = args.video_frames
    result, media = runtime.infer(
        media_inputs=args.media,
        prompt=resolved["prompt"],
        max_new_tokens=resolved["max_new_tokens"],
        max_image_side=resolved["max_image_side"],
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        video_num_frames=video_num_frames,
        check_finite_logits=False,
    )
    scale = coord_scale(args.skill)
    parsed = parse_skill(args.skill, result.answer) if scale or skill.output_kind in {"formula", "chart"} else None
    annotated_path: str | None = None
    if skill.is_grounding and isinstance(parsed, list) and parsed and args.image:
        out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(args.image).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(args.image).stem}_{args.skill}_annotated.png"
        annotated_path = _draw_grounding(args.image, parsed, scale or 1000, out_path)
    return {
        "skill": resolved,
        "model": runtime.spec.repo_id,
        "result": result.to_dict(),
        "parsed": parsed,
        "annotated_image": annotated_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Qwen3-VL skill locally on media (single GPU, FP8)."
    )
    parser.add_argument("--skill", required=True, help="skill key (see `qwen3-vl skills`)")
    parser.add_argument("--model", choices=("2b", "4b", "8b"), default="2b")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-path")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--kernel-dir")
    parser.add_argument("--gpu-placement", default="single",
                        choices=("single", "auto", "balanced", "balanced_low_0", "sequential"))
    parser.add_argument("--image", help="local image path (may repeat for multi-image skills)")
    parser.add_argument("--video", help="local video path")
    parser.add_argument("--prompt", help="override the skill prompt (for accepts_custom skills)")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-image-side", type=int)
    parser.add_argument("--video-frames", type=int, help="override sampled frame count for video skills")
    parser.add_argument("--greedy", action="store_true", help="greedy decoding (default: Qwen sampling)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", help="where to write annotated images")
    parser.add_argument("--json", action="store_true", help="print structured JSON result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    media: list[tuple[str, str]] = []
    if args.video:
        media.append(("video", args.video))
    if args.image:
        media.append(("image", args.image))
    args.media = media
    payload = run_skill(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    resolved = payload["skill"]
    result = payload["result"]
    print(f"skill: {resolved['label']} ({resolved['skill']})  cookbook: {resolved['cookbook']}")
    print(f"model: {payload['model']}  placement={args.gpu_placement}")
    if payload.get("annotated_image"):
        print(f"annotated image: {payload['annotated_image']}")
    print("\nMODEL OUTPUT\n" + result["answer"])
    if payload.get("parsed") is not None:
        print("\nPARSED\n" + json.dumps(payload["parsed"], ensure_ascii=False, indent=2, default=str))
    print(
        "\nMETRICS\n"
        f"finish_reason={result['finish_reason']} prompt_tokens={result['prompt_tokens']} "
        f"generated_tokens={result['generated_tokens']} tokens/s={result['tokens_per_second']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

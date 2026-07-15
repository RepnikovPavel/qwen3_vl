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
from pathlib import Path
from typing import Any, Sequence

from .qwen3_vl_offline import DEFAULT_CKPT_DIR, Qwen3VLRuntime
from .skills import SkillError, get_skill, resolve_skill
from .skill_parsers import coord_scale, parse_skill


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


def _draw_spatial(
    image_path: str,
    parsed: Any,
    skill: Any,
    scale: int,
    out_path: Path,
) -> str:
    """Render any spatial auto-labelling output (boxes/points/lanes/polygon)."""
    from PIL import Image
    from demo.grounding_viz import COLORS, _FONT
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    kind = skill.output_kind

    if kind in {"grounding_2d", "grounding_3d"} and isinstance(parsed, list):
        # Delegate bbox/point drawing to the shared helper.
        canvas.save(out_path)
        return _draw_grounding(image_path, parsed, scale, out_path)

    if kind == "lane" and isinstance(parsed, list):
        for index, lane in enumerate(parsed):
            if not isinstance(lane, dict):
                continue
            points = lane.get("points") or []
            color = COLORS[index % len(COLORS)]
            abs_pts = [
                (int(p[0] / scale * width), int(p[1] / scale * height))
                for p in points
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            if len(abs_pts) >= 2:
                draw.line(abs_pts, fill=color, width=4, joint="curve")
            for x, y in abs_pts:
                draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=color)
            label = f"lane {lane.get('lane_id', index)}"
            draw.text((abs_pts[0][0] + 6, abs_pts[0][1]), label, fill=color, font=_FONT)

    elif kind == "drivable_area" and isinstance(parsed, dict):
        polygon = parsed.get("polygon") or []
        abs_pts = [
            (int(p[0] / scale * width), int(p[1] / scale * height))
            for p in polygon
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        if len(abs_pts) >= 3:
            fill = (76, 175, 80, 96)  # translucent green
            # Overlay needs an RGBA canvas; paste back onto the RGB image.
            overlay = image.copy().convert("RGBA")
            ImageDraw.Draw(overlay).polygon(abs_pts, fill=fill, outline=(76, 175, 80))
            canvas = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            ImageDraw.Draw(canvas).line(abs_pts + [abs_pts[0]], fill=(76, 175, 80), width=3)

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
    structured_kinds = {"formula", "chart", "lane", "scene_graph", "drivable_area"}
    parsed = (
        parse_skill(args.skill, result.answer)
        if scale or skill.output_kind in structured_kinds
        else None
    )
    annotated_path: str | None = None
    # args.image is a list (action="append"); drawing uses the first frame.
    first_image = args.image[0] if args.image else None
    if skill.is_spatial and first_image and parsed:
        out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(first_image).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(first_image).stem}_{args.skill}_annotated.png"
        annotated_path = _draw_spatial(first_image, parsed, skill, scale or 1000, out_path)
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
    parser.add_argument(
        "--image", action="append", default=None,
        help="local image path; repeat for multi-image sequences (frame order preserved)",
    )
    parser.add_argument(
        "--image-dir", default=None,
        help="directory of images to load as an ordered sequence (taken in filename order)",
    )
    parser.add_argument("--num-frames", type=int, default=8,
                        help="max frames to load from --image-dir (uniformly sampled)")
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
    for path in args.image or []:
        media.append(("image", path))
    if args.image_dir:
        directory = Path(args.image_dir).expanduser()
        if not directory.is_dir():
            raise SkillError(f"--image-dir is not a directory: {directory}")
        frames = sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        if not frames:
            raise SkillError(f"--image-dir contains no images: {directory}")
        if len(frames) > args.num_frames:
            step = len(frames) / args.num_frames
            frames = [frames[min(len(frames) - 1, int(i * step))] for i in range(args.num_frames)]
        for frame in frames:
            media.append(("image", str(frame)))
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

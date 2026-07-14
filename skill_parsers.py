"""Output parsers for Qwen3-VL skills.

Each cookbook skill emits a slightly different structured format with its own
coordinate convention. This module gives a single place that maps
``skill_key -> (parse_fn, coord_scale)`` so the CLI / benchmark / web UI can
post-process any skill output uniformly.

Coordinate conventions (from the official Qwen3-VL cookbooks):
  * grounding / spatial / omni / document   -> relative 0..1000
  * OCR text spotting (qwen2.5-vl notebook) -> relative 0..999
  * 3D grounding                             -> metric bbox_3d + radians

Existing ``demo.grounding_viz.parse_grounding`` already handles 0..1000 bboxes
and points robustly; we re-export it and add the missing parsers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# Re-export the well-tested 0..1000 grounding parser for convenience.
from demo.grounding_viz import parse_grounding as _parse_grounding_1000


def parse_grounding_1000(text: str) -> list[dict[str, Any]]:
    """2D grounding / spatial points / omni recognition (coord scale 0..1000)."""
    return _parse_grounding_1000(text)


def _strip_fences(text: str) -> str:
    text = text.strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in ("```json", "```"):
            text = "\n".join(lines[i + 1 :])
            if "```" in text:
                text = text.split("```", 1)[0]
            break
    if "```" in text:
        parts = re.split(r"```(?:json)?", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            text = parts[1].split("```", 1)[0]
    return text.strip()


def parse_ocr_spotting(text: str) -> list[dict[str, Any]]:
    """OCR text spotting (qwen ocr cookbook): JSON list of
    ``{"bbox_2d": [x1,y1,x2,y2], "text_content": "..."}`` with coord scale 0..999.
    """
    if not text:
        return []
    cleaned = _strip_fences(text)
    start = cleaned.find("[")
    if start != -1:
        end = cleaned.rfind("]")
        if end > start:
            cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and "bbox_2d" in item:
            out.append(item)
    return out


def parse_point_2d(text: str) -> list[dict[str, Any]]:
    """Spatial-understanding point output (coord scale 0..1000)."""
    return parse_grounding_1000(text)


def parse_bbox_3d(text: str) -> list[dict[str, Any]]:
    """3D grounding output: JSON list of
    ``{"bbox_3d": [xc,yc,zc,xs,ys,zs,roll,pitch,yaw], "label": "..."}``.
    Delegates to demo.grounding_3d_viz.parse_bbox_3d_from_text.
    """
    from demo.grounding_3d_viz import parse_bbox_3d_from_text

    return parse_bbox_3d_from_text(text)


def parse_formula(text: str) -> dict[str, Any]:
    """Formula -> LaTeX. Cookbook schema: ``{"formulas": ["latex", ...]}``."""
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "formulas" in data:
            return data
    except Exception:
        pass
    # Fallback: treat non-empty lines as formulas.
    formulas = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return {"formulas": formulas}


def parse_chart(text: str) -> dict[str, Any]:
    """Chart -> structured JSON. Best-effort: return parsed JSON or raw text."""
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"raw": text.strip()}


def parse_plain(text: str) -> str:
    """Skills with free-form text output (describe, OCR full text, document, video)."""
    return text.strip()


# skill_key -> (parser, coordinate_scale)
# coordinate_scale: 1000 | 999 | 0 (0 = no spatial coords / N/A)
PARSERS: dict[str, tuple[Callable[[str], Any], int]] = {
    "describe": (parse_plain, 0),
    "ocr": (parse_plain, 0),
    "ocr_spotting": (parse_ocr_spotting, 999),
    "formula": (parse_formula, 0),
    "chart": (parse_chart, 0),
    "document_parsing_html": (parse_plain, 1000),
    "document_parsing_md": (parse_plain, 1000),
    "spatial_understanding": (parse_point_2d, 1000),
    "think_detailed": (parse_plain, 0),
    "omni_recognition": (parse_grounding_1000, 1000),
    "2d_grounding": (parse_grounding_1000, 1000),
    "3d_grounding": (parse_bbox_3d, 0),
    "video_understanding": (parse_plain, 0),
    "long_document": (parse_plain, 0),
    "mmcode": (parse_plain, 0),
    "computer_use": (parse_plain, 1000),
    "mobile_agent": (parse_plain, 999),
}


def parse_skill(skill_key: str, text: str) -> Any:
    """Dispatch to the parser registered for ``skill_key``."""
    if skill_key not in PARSERS:
        raise KeyError(f"no parser registered for skill {skill_key!r}")
    parser, _ = PARSERS[skill_key]
    return parser(text)


def coord_scale(skill_key: str) -> int:
    """Return the coordinate scale (0/999/1000) for a skill's structured output."""
    if skill_key not in PARSERS:
        raise KeyError(f"unknown skill {skill_key!r}")
    return PARSERS[skill_key][1]

"""2D Grounding visualization utilities adapted from the official Qwen3-VL 2d_grounding cookbook.

Supports:
- Bounding box grounding: list of {"bbox_2d": [x1,y1,x2,y2], "label": "...", ...extra}
- Point grounding: list of {"point_2d": [x,y], "label": "...", ...extra}

Coordinates are normalized 0-1000.
"""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# Try to load a nice font, fall back to default
_FONT = None
for font_path in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]:
    p = Path(font_path)
    if p.exists():
        try:
            _FONT = ImageFont.truetype(str(p), size=14)
            break
        except Exception:
            pass
if _FONT is None:
    _FONT = ImageFont.load_default()

COLORS = [
    "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "gray",
    "beige", "turquoise", "cyan", "magenta", "lime", "navy", "maroon", "teal",
    "olive", "coral", "lavender", "violet", "gold", "silver",
]


def _clean_json_text(text: str) -> str:
    """Strip markdown code fences and surrounding text."""
    text = text.strip()
    # ```json ... ``` or ``` ... ```
    if "```" in text:
        # take the first fenced block
        parts = re.split(r"```(?:json)?", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            text = parts[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    # sometimes the model adds extra prose; try to find the JSON array/object
    # look for first [ or { ... last ] or }
    start = min((text.find("["), text.find("{")))
    if start != -1:
        end = max(text.rfind("]"), text.rfind("}"))
        if end > start:
            candidate = text[start : end + 1]
            # basic sanity
            if candidate.count("[") <= candidate.count("]") + 1 and candidate.count("{") <= candidate.count("}") + 1:
                text = candidate
    return text.strip()


def parse_grounding(text: str) -> list[dict[str, Any]]:
    """Best-effort parse of grounding JSON from model text.

    Returns list of dicts. Each has either "bbox_2d" or "point_2d".
    """
    if not text:
        return []
    cleaned = _clean_json_text(text)
    try:
        data = json.loads(cleaned)
    except Exception:
        # fallback to ast.literal_eval on truncated
        try:
            import ast
            data = ast.literal_eval(cleaned)
        except Exception:
            # last resort: find the last [...] or {...} that looks like list of dicts
            m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
            if m:
                try:
                    data = json.loads(m.group(1))
                except Exception:
                    return []
            else:
                return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "bbox_2d" in item or "point_2d" in item:
            out.append(item)
    return out


def _scale_bbox(bbox: list[int | float], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _scale_point(pt: list[int | float], w: int, h: int) -> tuple[int, int]:
    x = int(pt[0] / 1000 * w)
    y = int(pt[1] / 1000 * h)
    return x, y


def draw_grounding(
    image: Image.Image,
    parsed: list[dict[str, Any]],
    *,
    draw_labels: bool = True,
    point_radius: int = 4,
    box_width: int = 3,
) -> Image.Image:
    """Draw boxes and/or points on a copy of the image.

    Supports mixed bbox and point items in one call.
    """
    if not parsed:
        return image.copy()

    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, item in enumerate(parsed):
        color = COLORS[i % len(COLORS)]
        label = item.get("label") or item.get("name") or ""

        if "bbox_2d" in item:
            try:
                x1, y1, x2, y2 = _scale_bbox(item["bbox_2d"], w, h)
                draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=box_width)
                if draw_labels and label:
                    draw.text((x1 + 6, y1 + 4), str(label), fill=color, font=_FONT)
                # optional extra info
                extra = []
                for k in ("type", "color", "role", "shirt_color"):
                    if k in item:
                        extra.append(f"{k}:{item[k]}")
                if extra and draw_labels:
                    draw.text((x1 + 6, y1 + 18), " ".join(extra), fill=color, font=_FONT)
            except Exception:
                continue

        elif "point_2d" in item:
            try:
                x, y = _scale_point(item["point_2d"], w, h)
                r = point_radius
                draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)
                if draw_labels and label:
                    draw.text((x + r + 2, y - r), str(label), fill=color, font=_FONT)
            except Exception:
                continue

    return img


def save_annotated(
    original_path: str | Path,
    parsed: list[dict[str, Any]],
    out_path: str | Path,
) -> str:
    """Convenience: load image, draw, save annotated version. Returns str path."""
    im = Image.open(original_path).convert("RGB")
    annotated = draw_grounding(im, parsed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path)
    return str(out_path)

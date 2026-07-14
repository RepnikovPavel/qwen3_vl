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
    """Strip markdown code fences and surrounding text. Matches logic from cookbooks/2d_grounding.ipynb parse_json."""
    if not text:
        return ""
    text = text.strip()
    # Notebook style: split on ```json line, take until next ```
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json" or line.strip() == "```":
            text = "\n".join(lines[i+1:])
            if "```" in text:
                text = text.split("```", 1)[0]
            break
    # Fallback fence strip
    if "```" in text:
        parts = re.split(r"```(?:json)?", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            text = parts[1].split("```", 1)[0]
    # Extract the JSON array/object if there is prose around it (common in responses)
    # Prefer the largest plausible [...] or {...} block
    start = min((text.find("["), text.find("{")))
    if start != -1:
        end = max(text.rfind("]"), text.rfind("}"))
        if end > start:
            candidate = text[start : end + 1]
            # sanity: balanced enough
            if candidate.count("[") <= candidate.count("]") + 2 and candidate.count("{") <= candidate.count("}") + 2:
                text = candidate
    return text.strip()


def parse_grounding(text: str) -> list[dict[str, Any]]:
    """Best-effort parse of grounding JSON from model text.
    Handles strict JSON objects and also loose [x1,y1,x2,y2] or [x,y] mentions in text.
    Returns list of dicts with "bbox_2d" or "point_2d" + generated label.
    """
    if not text:
        return []
    out = []
    cleaned = _clean_json_text(text)

    # 1. Try strict JSON first
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and ("bbox_2d" in item or "point_2d" in item):
                    out.append(item)
    except Exception:
        pass

    if out:
        return out

    # 2. Fallback: extract loose coordinate lists from the raw text
    # bbox like [282, 522, 359, 589] or (282,522,359,589)
    bbox_pattern = re.compile(r'[\(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]')
    for i, m in enumerate(bbox_pattern.finditer(text)):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append({"bbox_2d": [x1, y1, x2, y2], "label": f"obj{i+1}"})

    if out:
        return out

    # 3. Points [x, y]
    point_pattern = re.compile(r'[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]')
    seen = set()
    for i, m in enumerate(point_pattern.finditer(text)):
        x, y = map(int, m.groups())
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        out.append({"point_2d": [x, y], "label": f"pt{i+1}"})

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

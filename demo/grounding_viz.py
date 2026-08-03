"""2D Grounding visualization utilities adapted from the official Qwen3-VL 2d_grounding cookbook.

Supports:
- Bounding box grounding: list of {"bbox_2d": [x1,y1,x2,y2], "label": "...", ...extra}
- Point grounding: list of {"point_2d": [x,y], "label": "...", ...extra}

Coordinates are normalized 0-1000.
"""

from __future__ import annotations

import json
import re
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

    Always merges JSON hits with loose regex hits (Thinking models often put
    only a subset in a clean JSON block and more boxes inline in prose).
    """
    if not text:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def _add_box(x1: int, y1: int, x2: int, y2: int, label: str) -> None:
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 - x1 < 1 and y2 - y1 < 1:
            return
        key = ("b", x1, y1, x2, y2)
        if key in seen:
            return
        seen.add(key)
        out.append({"bbox_2d": [x1, y1, x2, y2], "label": label})

    def _add_pt(x: int, y: int, label: str) -> None:
        key = ("p", x, y)
        if key in seen:
            return
        seen.add(key)
        out.append({"point_2d": [x, y], "label": label})

    cleaned = _clean_json_text(text)

    # 1. Strict JSON (array / object / nested)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                        try:
                            _add_box(int(item[0]), int(item[1]), int(item[2]), int(item[3]), "obj")
                        except (TypeError, ValueError):
                            pass
                    continue
                label = str(item.get("label") or item.get("name") or item.get("class") or "")
                if "bbox_2d" in item and isinstance(item["bbox_2d"], (list, tuple)) and len(item["bbox_2d"]) >= 4:
                    b = item["bbox_2d"]
                    try:
                        _add_box(int(b[0]), int(b[1]), int(b[2]), int(b[3]), label)
                    except (TypeError, ValueError):
                        pass
                if "point_2d" in item and isinstance(item["point_2d"], (list, tuple)) and len(item["point_2d"]) >= 2:
                    p = item["point_2d"]
                    try:
                        _add_pt(int(p[0]), int(p[1]), label)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass

    # 1b. Multiple JSON arrays in text (thinking dumps several blocks)
    for m in re.finditer(r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]", text, flags=re.DOTALL):
        chunk = m.group(0)
        try:
            data = json.loads(chunk)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or item.get("class") or "")
            if "bbox_2d" in item and isinstance(item["bbox_2d"], (list, tuple)) and len(item["bbox_2d"]) >= 4:
                b = item["bbox_2d"]
                try:
                    _add_box(int(b[0]), int(b[1]), int(b[2]), int(b[3]), label)
                except (TypeError, ValueError):
                    pass
            if "point_2d" in item and isinstance(item["point_2d"], (list, tuple)) and len(item["point_2d"]) >= 2:
                p = item["point_2d"]
                try:
                    _add_pt(int(p[0]), int(p[1]), label)
                except (TypeError, ValueError):
                    pass

    # 2. Loose bbox tuples — always merge
    bbox_pattern = re.compile(
        r"[\(\[]\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*[\)\]]"
    )
    for i, m in enumerate(bbox_pattern.finditer(text)):
        x1, y1, x2, y2 = map(int, m.groups())
        if x2 > x1 and y2 > y1:
            _add_box(x1, y1, x2, y2, f"obj{i + 1}")

    # 3. Explicit point_2d fragments
    for m in re.finditer(
        r'point_2d["\']?\s*:\s*\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]', text
    ):
        _add_pt(int(m.group(1)), int(m.group(2)), "pt")

    # 4. Bare [x, y] only if we still have nothing spatial (avoid eating box pairs)
    if not out:
        point_pattern = re.compile(r"[\(\[]\s*(\d{1,4})\s*,\s*(\d{1,4})\s*[\)\]]")
        for i, m in enumerate(point_pattern.finditer(text)):
            _add_pt(int(m.group(1)), int(m.group(2)), f"pt{i + 1}")

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

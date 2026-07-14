"""3D Grounding visualization utilities adapted from the official Qwen3-VL 3d_grounding cookbook.

Supports:
- 3D bounding box grounding: list of {"bbox_3d": [x, y, z, x_size, y_size, z_size, pitch, yaw, roll], "label": "...", ...}
- Projection to 2D image using camera params (fx,fy,cx,cy)
- Drawing wireframe 3D boxes on image using PIL

Default camera params can be generated from image size (fov ~60deg).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# Try font
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

def parse_bbox_3d_from_text(text: str) -> list[dict[str, Any]]:
    """Parse 3D bbox JSON from model response text."""
    if not text:
        return []
    try:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
            else:
                json_str = text[start:].strip()
        else:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                json_str = text[start:end + 1]
            else:
                return []
        data = json.loads(json_str)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict) and "bbox_3d" in item]
    except Exception:
        return []

def generate_camera_params(image: Image.Image, fov: float = 60.0) -> dict[str, float]:
    """Generate default camera intrinsics based on image size and FoV."""
    w, h = image.size
    fx = round(w / (2 * math.tan(math.radians(fov) / 2)), 2)
    fy = round(h / (2 * math.tan(math.radians(fov) / 2)), 2)
    cx = round(w / 2, 2)
    cy = round(h / 2, 2)
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

def _rotate_xyz(point, pitch_deg, yaw_deg, roll_deg):
    x0, y0, z0 = point
    pitch, yaw, roll = map(math.radians, (pitch_deg, yaw_deg, roll_deg))

    # pitch
    x1 = x0
    y1 = y0 * math.cos(pitch) - z0 * math.sin(pitch)
    z1 = y0 * math.sin(pitch) + z0 * math.cos(pitch)

    # yaw
    x2 = x1 * math.cos(yaw) + z1 * math.sin(yaw)
    y2 = y1
    z2 = -x1 * math.sin(yaw) + z1 * math.cos(yaw)

    # roll
    x3 = x2 * math.cos(roll) - y2 * math.sin(roll)
    y3 = x2 * math.sin(roll) + y2 * math.cos(roll)
    z3 = z2

    return [x3, y3, z3]

def convert_3dbbox(bbox_3d: list[float], cam_params: dict[str, float]) -> list[list[float]]:
    """Project 3D bbox (center + size + euler) to 2D image corners."""
    if len(bbox_3d) < 9:
        return []
    x, y, z, sx, sy, sz, pitch, yaw, roll = bbox_3d[:9]
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    local = [
        [hx, hy, hz], [hx, hy, -hz], [hx, -hy, hz], [hx, -hy, -hz],
        [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, hz], [-hx, -hy, -hz],
    ]
    img_corners = []
    for corner in local:
        rx, ry, rz = _rotate_xyz(corner, pitch, yaw, roll)
        X, Y, Z = rx + x, ry + y, rz + z
        if Z > 0:
            ix = cam_params["fx"] * (X / Z) + cam_params["cx"]
            iy = cam_params["fy"] * (Y / Z) + cam_params["cy"]
            img_corners.append([ix, iy])
    return img_corners

EDGES = [
    (0,1), (2,3), (4,5), (6,7),
    (0,2), (1,3), (4,6), (5,7),
    (0,4), (1,5), (2,6), (3,7)
]

def draw_3d_bboxes(image: Image.Image, cam_params: dict[str, float], bbox_3d_list: list[dict], draw_labels: bool = True) -> Image.Image:
    """Draw wireframe 3D boxes projected on the image. Returns annotated copy."""
    if not bbox_3d_list:
        return image.copy()
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, item in enumerate(bbox_3d_list):
        bbox_3d = item.get("bbox_3d")
        if not bbox_3d or len(bbox_3d) < 9:
            continue
        label = item.get("label", "")
        color = ["red", "green", "blue", "yellow", "magenta", "cyan"][i % 6]

        corners = convert_3dbbox(bbox_3d, cam_params)
        if len(corners) < 8:
            continue

        for a, b in EDGES:
            try:
                p1 = (int(corners[a][0]), int(corners[a][1]))
                p2 = (int(corners[b][0]), int(corners[b][1]))
                draw.line([p1, p2], fill=color, width=2)
            except Exception:
                continue

        if draw_labels and label:
            # label near first corner
            cx, cy = int(corners[0][0]), int(corners[0][1])
            draw.text((cx + 4, cy + 4), str(label), fill=color, font=_FONT)

    return img

def save_annotated_3d(original_path: str | Path, cam_params: dict[str, float], bbox_3d_list: list[dict], out_path: str | Path) -> str:
    im = Image.open(original_path).convert("RGB")
    annotated = draw_3d_bboxes(im, cam_params, bbox_3d_list)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path)
    return str(out_path)

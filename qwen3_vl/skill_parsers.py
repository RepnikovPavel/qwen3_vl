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


# Recognised nuScenes-style detection classes (kept in sync with the
# nuscenes_2d_detection prompt so the auto-label parser can recover the class
# from prose, not just from strict JSON).
NUSCENES_CLASSES = (
    "vehicle", "pedestrian", "cyclist", "traffic_sign", "traffic_light",
    "barrier", "cone",
)
# Common concrete mentions the model writes instead of the canonical class
# (e.g. "truck", "car", "van" -> "vehicle"); resolved during prose recovery.
_CLASS_ALIASES = {
    "car": "vehicle", "truck": "vehicle", "van": "vehicle", "bus": "vehicle",
    "lorry": "vehicle", "vehicle": "vehicle", "motorcycle": "vehicle",
    "motorbike": "vehicle", "bicycle": "cyclist", "cyclist": "cyclist",
    "rider": "cyclist", "person": "pedestrian", "pedestrian": "pedestrian",
    "people": "pedestrian", "sign": "traffic_sign", "traffic_sign": "traffic_sign",
    "light": "traffic_light", "traffic_light": "traffic_light",
    "barrier": "barrier", "guardrail": "barrier", "railing": "barrier",
    "cone": "cone", "trafficcone": "cone",
}


def _resolve_class(token: str) -> str | None:
    """Map a free-form token to a canonical nuScenes class, or None."""
    key = re.sub(r"[^a-z_]", "", token.lower())
    if key in NUSCENES_CLASSES:
        return key
    return _CLASS_ALIASES.get(key)


def parse_nuscenes_detection(text: str) -> list[dict[str, Any]]:
    """2D detection auto-label parser: bboxes + recovered class (0..1000).

    Strict JSON first ({"class": "...", "bbox_2d": [...]}); then falls back to
    the cookbook grounding parser. If the JSON path produced label-less boxes
    (or there were none), recover the class for each bbox by scanning a small
    window of prose around the coordinates for a class mention or alias.
    """
    if not text:
        return []
    cleaned = _strip_fences(text)
    out: list[dict[str, Any]] = []
    # 1. Strict JSON carrying both class and bbox_2d.
    try:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end > start:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "bbox_2d" in item:
                        cls = item.get("class") or item.get("label")
                        out.append({
                            "bbox_2d": list(item["bbox_2d"]),
                            "label": cls or "object",
                            "class": _resolve_class(str(cls)) if cls else None,
                        })
    except Exception:
        out = []
    if out:
        return out
    # 2. Fallback to the cookbook grounding parser, then enrich label/class
    #    from the surrounding prose so callers get useful categories.
    parsed = _parse_grounding_1000(text)
    if not parsed:
        return []
    # Match each recovered bbox back to a class mentioned near it in the text.
    bbox_re = re.compile(
        r"[\(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]"
    )
    spans = [
        ((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))), m.span())
        for m in bbox_re.finditer(text)
    ]
    for item in parsed:
        bbox = item.get("bbox_2d")
        if not (isinstance(bbox, list) and len(bbox) >= 4):
            continue
        key = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        cls = None
        raw_token = None
        for coords, (start, end) in spans:
            if coords != key:
                continue
            # Look both before and after the bbox for a class mention.
            window = text[max(0, start - 60) : min(len(text), end + 40)].lower()
            # Prefer " - <class>" / "— <class>" trailing the coordinates.
            trailing = text[end : min(len(text), end + 40)]
            m = re.search(r"[\-—:]\s*([a-zA-Z_]+)", trailing)
            if m:
                resolved = _resolve_class(m.group(1))
                if resolved:
                    cls, raw_token = resolved, m.group(1)
                    break
            # Otherwise scan the window for any known class or alias.
            for word in re.findall(r"[a-zA-Z_]+", window):
                resolved = _resolve_class(word)
                if resolved:
                    cls, raw_token = resolved, word
                    break
            if cls:
                break
        item["label"] = raw_token or cls or item.get("label") or "object"
        item["class"] = cls
        out.append(item)
    return out


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


# --- Auto-labelling parsers (tolerant JSON + inline-prose recovery) ----------
#
# The 2B Thinking model frequently narrates its reasoning before answering, so
# these parsers first try a strict JSON block and then fall back to extracting
# coordinate lists / triples mentioned inline in prose. Anything returned is
# already in the skill's [0,coord_scale] frame; the CLI rescales to pixels.

_COORD_LIST = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*,\s*(-?\d+)\s*)?\]"
)


def _extract_json_block(text: str) -> Any:
    """Return the first JSON array/object decoded from text, or None.

    Strips markdown fences and narrows the text to the largest balanced
    [..] / {..} span before decoding, so prose around the JSON is ignored.
    """
    cleaned = _strip_fences(text)
    if not cleaned:
        return None
    # Try the whole stripped text first (common when the model obeys "JSON only").
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Fall back to the largest balanced bracket span.
    start = min((cleaned.find("["), cleaned.find("{")))
    if start != -1:
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return None


def parse_lane(text: str) -> list[dict[str, Any]]:
    """Lane polylines (coord scale 0..1000). Tolerant of prose.

    Returns a list of ``{"lane_id": int, "points": [[x, y], ...]}`` dicts.
    Recovers from strict JSON, from 'lane N: ...' prose, and finally from any
    bare coordinate pairs found in the text (collapsed into one lane).
    """
    if not text:
        return []
    data = _extract_json_block(text)
    lanes: list[dict[str, Any]] = []
    if isinstance(data, list):
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            raw_points = item.get("points")
            if raw_points is None:
                # Some models emit {"lane_id": 0, "x": [...], "y": [...]}.
                xs = item.get("x") or item.get("xs") or []
                ys = item.get("y") or item.get("ys") or []
                if isinstance(xs, list) and isinstance(ys, list):
                    raw_points = list(zip(xs, ys))
            if not isinstance(raw_points, list):
                continue
            points = [_coord_pair(p) for p in raw_points]
            points = [p for p in points if p is not None]
            if points:
                lanes.append({
                    "lane_id": item.get("lane_id", index),
                    "points": points,
                })
    if lanes:
        return lanes
    # Prose fallback: 'lane 0: (x,y) (x,y)' or 'lane 0:' followed by coords.
    for match in re.finditer(
        r"lane\s*(\d+)\s*[:\-]([^\n]*(?:\n(?!\s*lane\s*\d)[^\n]*)*)",
        text, flags=re.IGNORECASE,
    ):
        lane_id = int(match.group(1))
        body = match.group(2)
        points = [_coord_pair_from_match(m) for m in _COORD_LIST.finditer(body)]
        points = [p for p in points if p is not None]
        if points:
            lanes.append({"lane_id": lane_id, "points": points})
    if lanes:
        return lanes
    # Last resort: collect every 2-tuple in the text into a single lane.
    points = [_coord_pair_from_match(m) for m in _COORD_LIST.finditer(text)]
    points = [p for p in points if p is not None]
    if points:
        return [{"lane_id": 0, "points": points}]
    return []


def parse_scene_graph(text: str) -> list[dict[str, str]]:
    """Scene-graph triples. Tolerant of prose.

    Returns a list of ``{"subject": str, "relation": str, "object": str}``.
    Recovers from strict JSON or from inline '(subj, rel, obj)' / 'subj rel obj'
    mentions.
    """
    if not text:
        return []
    data = _extract_json_block(text)
    triples: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            triple = _triple_from_item(item)
            if triple:
                triples.append(triple)
    elif isinstance(data, dict) and isinstance(data.get("triples"), list):
        for item in data["triples"]:
            triple = _triple_from_item(item)
            if triple:
                triples.append(triple)
    if triples:
        return triples
    # Prose fallback: '(subject, relation, object)' or '<subj> relation <obj>'.
    for pattern in (
        r"\(\s*([^()<>|,\n]{1,60}?)\s*,\s*([^()<>|,\n]{1,40}?)\s*,\s*([^()<>|,\n]{1,60}?)\s*\)",
        r"<([^<>|,\n]{1,60}?)>\s*([a-z_]+)\s*<([^<>|,\n]{1,60}?)>",
    ):
        for match in re.finditer(pattern, text):
            triples.append({
                "subject": _clean_token(match.group(1)),
                "relation": _clean_token(match.group(2)).lower().replace(" ", "_"),
                "object": _clean_token(match.group(3)),
            })
    return triples


def parse_drivable_area(text: str) -> dict[str, Any]:
    """Drivable-area polygon (coord scale 0..1000). Tolerant of prose.

    Returns ``{"polygon": [[x, y], ...]}`` (possibly empty).
    """
    if not text:
        return {"polygon": []}
    data = _extract_json_block(text)
    if isinstance(data, dict):
        raw = data.get("polygon")
    elif isinstance(data, list):
        raw = data  # some models emit a bare coordinate list
    else:
        raw = None
    points: list[list[int]] = []
    if isinstance(raw, list):
        for p in raw:
            pair = _coord_pair(p)
            if pair is not None:
                points.append(pair)
    if points:
        return {"polygon": points}
    # Prose fallback: collect every coordinate pair in the text.
    for match in _COORD_LIST.finditer(text):
        pair = _coord_pair_from_match(match)
        if pair is not None:
            points.append(pair)
    return {"polygon": points}


def _coord_pair(value: Any) -> list[int] | None:
    """Coerce one JSON value into an [x, y] integer pair, or None."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _coord_pair_from_match(match: re.Match[str]) -> list[int] | None:
    """Coerce one _COORD_LIST regex match into an [x, y] (or bbox) pair."""
    groups = match.groups()
    if groups[2] is not None and groups[3] is not None:
        # A 4-tuple bbox was matched; take the first corner as a point.
        return [int(groups[0]), int(groups[1])]
    return [int(groups[0]), int(groups[1])]


def _clean_token(value: Any) -> str:
    """Strip surrounding quotes/whitespace from a recovered triple token."""
    text = str(value).strip().strip("\"'`")
    return text


def _triple_from_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    subject = item.get("subject") or item.get("src") or item.get("head")
    relation = item.get("relation") or item.get("rel") or item.get("predicate")
    obj = item.get("object") or item.get("dst") or item.get("tail")
    if subject and relation and obj:
        return {
            "subject": _clean_token(subject),
            "relation": _clean_token(relation).lower().replace(" ", "_"),
            "object": _clean_token(obj),
        }
    return None


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
    # Auto-labelling skills (driving / nuScenes-style):
    "nuscenes_2d_detection": (parse_nuscenes_detection, 1000),
    "nuscenes_lane": (parse_lane, 1000),
    "nuscenes_scene_graph": (parse_scene_graph, 0),
    "nuscenes_drivable_area": (parse_drivable_area, 1000),
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

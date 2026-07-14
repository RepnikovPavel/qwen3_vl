"""Declarative catalog of Qwen3-VL skills, derived from the official cookbooks.

Each skill maps a cookbook capability to a concrete local inference recipe:
the prompt, the expected output kind, the input modality (single image /
multi-image sequence / video / document), and how to verify the output.

The prompts follow the official cookbook phrasing where the cookbook gave a
verbatim string (grounding, OCR spotting, document parsing, 3D grounding), so
the model hits its trained output distribution. Skills that were API-only in
the cookbooks are reproduced as local ``model.generate`` calls here.

Reference: /home/pavel-repnikov/Qwen3-VL/cookbooks (2d_grounding, 3d_grounding,
computer_use, document_parsing, german_document_ocr, long_document_understanding,
mmcode, mobile_agent, ocr, omni_recognition, spatial_understanding,
think_with_images, video_understanding).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


class SkillError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """One reproducible Qwen3-VL capability."""

    key: str
    label: str
    cookbook: str
    prompt: str
    output_kind: str
    frames_kind: str
    coord_scale: int
    accepts_custom: bool = False
    default_max_new_tokens: int = 2048
    default_max_image_side: int = 0  # 0 = no resize, use input resolution
    video_num_frames: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", self.key):
            raise SkillError("skill key must match [a-z0-9][a-z0-9_]*")
        if self.output_kind not in {"text", "formula", "chart", "grounding_2d", "grounding_3d", "code"}:
            raise SkillError(f"unsupported output_kind: {self.output_kind!r}")
        if self.frames_kind not in {"single_image", "multi_image", "video", "document"}:
            raise SkillError(f"unsupported frames_kind: {self.frames_kind!r}")
        if self.coord_scale not in (0, 999, 1000):
            raise SkillError(f"coord_scale must be 0/999/1000, got {self.coord_scale}")
        if self.default_max_new_tokens < 1:
            raise SkillError("default_max_new_tokens must be positive")

    @property
    def is_grounding(self) -> bool:
        return self.output_kind in {"grounding_2d", "grounding_3d"}

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "cookbook": self.cookbook,
            "prompt": self.prompt,
            "output_kind": self.output_kind,
            "frames_kind": self.frames_kind,
            "coord_scale": self.coord_scale,
            "accepts_custom": self.accepts_custom,
            "default_max_new_tokens": self.default_max_new_tokens,
            "default_max_image_side": self.default_max_image_side,
            "video_num_frames": self.video_num_frames,
        }


# --- Single-image perception -------------------------------------------------

DESCRIBE = SkillSpec(
    key="describe",
    label="Describe the scene",
    cookbook="omni_recognition / spatial_understanding (general)",
    prompt="Describe the visual content completely and precisely.",
    output_kind="text",
    frames_kind="single_image",
    coord_scale=0,
    default_max_new_tokens=1024,
)

OCR = SkillSpec(
    key="ocr",
    label="OCR (full text)",
    cookbook="ocr.ipynb",
    prompt=(
        "Read all the text in the image. Preserve line breaks. "
        "Return only the text content."
    ),
    output_kind="text",
    frames_kind="single_image",
    coord_scale=0,
    default_max_new_tokens=2048,
    notes="cookbook used Qwen2.5-VL; reproduced on Qwen3-VL.",
)

OCR_SPOTTING = SkillSpec(
    key="ocr_spotting",
    label="OCR text spotting (bbox + text)",
    cookbook="ocr.ipynb",
    prompt=(
        "Spotting all the text in the image with line-level, and output in JSON "
        "format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...]."
    ),
    output_kind="text",
    frames_kind="single_image",
    coord_scale=999,
    default_max_new_tokens=2048,
    notes="coords are relative 0..999 (Qwen2.5-VL convention in the notebook).",
)

FORMULA = SkillSpec(
    key="formula",
    label="Formulas to LaTeX",
    cookbook="document_parsing / evaluation",
    prompt=(
        "Transcribe every displayed formula as LaTeX in top-to-bottom order. "
        'Return only a JSON object with schema {"formulas":["latex","..."]}. '
        "Do not use Markdown fences."
    ),
    output_kind="formula",
    frames_kind="single_image",
    coord_scale=0,
    default_max_new_tokens=2048,
)

CHART = SkillSpec(
    key="chart",
    label="Chart to structured JSON",
    cookbook="evaluation (chart fixtures)",
    prompt=(
        "Read the chart and return only JSON with keys title, panels, and facts. "
        "Each panel must contain chart_type, title, x_label, y_label, categories, "
        "and series. Each series must contain name and numeric values in values. "
        "Do not use Markdown fences."
    ),
    output_kind="chart",
    frames_kind="single_image",
    coord_scale=0,
    default_max_new_tokens=2048,
)

DOCUMENT_PARSING_HTML = SkillSpec(
    key="document_parsing_html",
    label="Document parsing (HTML)",
    cookbook="document_parsing.ipynb",
    prompt="qwenvl html",
    output_kind="text",
    frames_kind="document",
    coord_scale=1000,
    default_max_new_tokens=4096,
    notes="HTML with data-bbox attributes, coords 0..1000.",
)

DOCUMENT_PARSING_MD = SkillSpec(
    key="document_parsing_md",
    label="Document parsing (Markdown)",
    cookbook="document_parsing.ipynb",
    prompt="qwenvl markdown",
    output_kind="text",
    frames_kind="document",
    coord_scale=1000,
    default_max_new_tokens=4096,
    notes="Markdown with <!-- Table (x1,y1,x2,y2) --> comments, coords 0..1000.",
)

SPATIAL_UNDERSTANDING = SkillSpec(
    key="spatial_understanding",
    label="Spatial understanding",
    cookbook="spatial_understanding.ipynb",
    prompt=(
        "Locate objects by their spatial position. When asked for a location, "
        'output point coordinates in JSON format like '
        '{"point_2d": [x, y], "label": "object"}.'
    ),
    output_kind="grounding_2d",
    frames_kind="single_image",
    coord_scale=1000,
    default_max_new_tokens=1024,
)

THINK_DETAILED = SkillSpec(
    key="think_detailed",
    label="Think step-by-step",
    cookbook="think_with_images.ipynb",
    prompt=(
        "Examine the visual input carefully. Think step by step (internal "
        "reasoning first), then provide a clear, well-structured final answer."
    ),
    output_kind="text",
    frames_kind="single_image",
    coord_scale=0,
    default_max_new_tokens=4096,
)

OMNI_RECOGNITION = SkillSpec(
    key="omni_recognition",
    label="Omni recognition",
    cookbook="omni_recognition.ipynb",
    prompt=(
        "Identify the main subjects in the image and return their bounding box "
        "and name in JSON format."
    ),
    output_kind="grounding_2d",
    frames_kind="single_image",
    coord_scale=1000,
    default_max_new_tokens=1024,
)

# --- Grounding (verbatim cookbook prompts) -----------------------------------

GROUNDING_2D = SkillSpec(
    key="2d_grounding",
    label="2D Grounding (bbox / points)",
    cookbook="2d_grounding.ipynb",
    prompt=(
        'Locate every instance that belongs to the following categories: '
        '"car, person, vehicle". Report bbox coordinates in JSON format.'
    ),
    output_kind="grounding_2d",
    frames_kind="single_image",
    coord_scale=1000,
    accepts_custom=True,
    default_max_new_tokens=4096,
    notes='cookbook triggers JSON with "Report bbox coordinates in JSON format".',
)

GROUNDING_3D = SkillSpec(
    key="3d_grounding",
    label="3D Grounding (3D bboxes)",
    cookbook="3d_grounding.ipynb",
    prompt=(
        "Find all cars in this image. For each car, provide its 3D bounding box. "
        'The output format required is JSON: '
        '[{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, '
        'roll, pitch, yaw], "label":"category"}].'
    ),
    output_kind="grounding_3d",
    frames_kind="single_image",
    coord_scale=0,
    default_max_new_tokens=4096,
    notes="metric bbox_3d + radians; camera intrinsics via fov=60 fallback.",
)

# --- Multi-frame / video / document ------------------------------------------

VIDEO_UNDERSTANDING = SkillSpec(
    key="video_understanding",
    label="Video understanding",
    cookbook="video_understanding.ipynb",
    prompt=(
        "Analyze the video in detail. Describe the sequence of events, key "
        "actions, objects, any visible text or signs, and how the scene evolves "
        "over time. Be precise and chronological."
    ),
    output_kind="text",
    frames_kind="video",
    coord_scale=0,
    default_max_new_tokens=2048,
    video_num_frames=32,
    notes="cookbook default: total_pixels=20480*32*32, sample_fps=2, num_frames 64/128.",
)

LONG_DOCUMENT = SkillSpec(
    key="long_document",
    label="Long document (multi-page)",
    cookbook="long_document_understanding.ipynb",
    prompt=(
        "These are consecutive pages of a document. Read them in order and "
        "answer the question. Be precise and reference page content."
    ),
    output_kind="text",
    frames_kind="document",
    coord_scale=0,
    accepts_custom=True,
    default_max_new_tokens=4096,
    notes="cookbook: each page resized max_side<=1500, max_pixels=730*32*32 per page.",
)

# --- Code --------------------------------------------------------------------

MMCODE = SkillSpec(
    key="mmcode",
    label="Multimodal coding",
    cookbook="mmcode.ipynb",
    prompt=(
        "Analyze this screenshot and convert it to clean, functional and modern "
        "HTML code."
    ),
    output_kind="code",
    frames_kind="single_image",
    coord_scale=0,
    accepts_custom=True,
    default_max_new_tokens=4096,
    notes="cookbook also supports chart->matplotlib and MMCode problem solving.",
)

# --- Agent / function-call (best-effort local) -------------------------------

COMPUTER_USE = SkillSpec(
    key="computer_use",
    label="Computer use (actions)",
    cookbook="computer_use.ipynb",
    prompt=(
        "You are a GUI agent. Analyze the screenshot and decide the next action "
        "to accomplish the task. Output a JSON action with a 'coordinate' in "
        "[0,1000] logical display space when a click/move is needed."
    ),
    output_kind="code",
    frames_kind="single_image",
    coord_scale=1000,
    accepts_custom=True,
    default_max_new_tokens=2048,
    notes="cookbook uses ComputerUse tool-call + qwen-agent; local best-effort JSON.",
)

MOBILE_AGENT = SkillSpec(
    key="mobile_agent",
    label="Mobile agent (actions)",
    cookbook="mobile_agent.ipynb",
    prompt=(
        "You are a mobile GUI agent. Analyze the screenshot and decide the next "
        "action. Output a JSON action with a 'coordinate' in [0,999] logical "
        "screen space when a tap is needed."
    ),
    output_kind="code",
    frames_kind="single_image",
    coord_scale=999,
    accepts_custom=True,
    default_max_new_tokens=2048,
    notes="cookbook uses MobileUse tool-call; coords 0..999.",
)


_SKILL_ITEMS = (
    DESCRIBE,
    OCR,
    OCR_SPOTTING,
    FORMULA,
    CHART,
    DOCUMENT_PARSING_HTML,
    DOCUMENT_PARSING_MD,
    SPATIAL_UNDERSTANDING,
    THINK_DETAILED,
    OMNI_RECOGNITION,
    GROUNDING_2D,
    GROUNDING_3D,
    VIDEO_UNDERSTANDING,
    LONG_DOCUMENT,
    MMCODE,
    COMPUTER_USE,
    MOBILE_AGENT,
)

SKILLS: Mapping[str, SkillSpec] = MappingProxyType({item.key: item for item in _SKILL_ITEMS})


def get_skill(key: str) -> SkillSpec:
    if not isinstance(key, str) or key not in SKILLS:
        raise SkillError(f"unsupported skill: {key!r}; choose from {sorted(SKILLS)}")
    return SKILLS[key]


def public_skills() -> dict[str, Any]:
    return {"schema_version": 1, "skills": [item.to_public_dict() for item in _SKILL_ITEMS]}


def resolve_skill(
    key: str,
    *,
    custom_prompt: str | None = None,
    max_new_tokens: int | None = None,
    max_image_side: int | None = None,
) -> dict[str, Any]:
    """Resolve a skill key + optional overrides into a concrete inference plan."""
    skill = get_skill(key)
    prompt = skill.prompt
    if skill.accepts_custom:
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
    elif custom_prompt and custom_prompt.strip():
        # Allow free-form override even for fixed-prompt skills.
        prompt = custom_prompt.strip()
    tokens = max_new_tokens if max_new_tokens is not None else skill.default_max_new_tokens
    if tokens < 1:
        raise SkillError("max_new_tokens must be positive")
    side = max_image_side if max_image_side is not None else skill.default_max_image_side
    if side != 0 and side < 1:
        raise SkillError("max_image_side must be 0 (no resize) or positive")
    return {
        "skill": skill.key,
        "label": skill.label,
        "cookbook": skill.cookbook,
        "prompt": prompt,
        "output_kind": skill.output_kind,
        "frames_kind": skill.frames_kind,
        "coord_scale": skill.coord_scale,
        "max_new_tokens": tokens,
        "max_image_side": side,
        "video_num_frames": skill.video_num_frames,
    }

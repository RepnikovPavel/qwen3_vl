from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from evaluate_vl import (
    SchemaError,
    _evaluate_chart,
    _evaluate_formula,
    _evaluate_text,
    _structured_answer,
    _validate_chart_object,
    _validate_formula_object,
)


MAX_NEW_TOKENS = 131_072  # increased for longer thinking / large context
MIN_IMAGE_SIDE = 64
MAX_IMAGE_SIDE = 4096
MAX_CUSTOM_PROMPT_CHARACTERS = 200_000


class DemoTaskError(ValueError):
    pass


def _validate_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DemoTaskError(f"{name} must be an integer")
    return value


def _validate_limits(max_new_tokens: Any, max_image_side: Any) -> tuple[int, int]:
    tokens = _validate_integer(max_new_tokens, "max_new_tokens")
    side = _validate_integer(max_image_side, "max_image_side")
    if not 1 <= tokens <= MAX_NEW_TOKENS:
        raise DemoTaskError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    if not MIN_IMAGE_SIDE <= side <= MAX_IMAGE_SIDE:
        raise DemoTaskError(
            f"max_image_side must be between {MIN_IMAGE_SIDE} and {MAX_IMAGE_SIDE}"
        )
    return tokens, side


@dataclass(frozen=True, slots=True)
class DemoPreset:
    key: str
    label: str
    prompt: str | None
    output_kind: str
    default_max_new_tokens: int
    default_max_image_side: int
    accepts_custom_prompt: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", self.key
        ):
            raise DemoTaskError("preset key must match [a-z][a-z0-9_]*")
        if (
            not isinstance(self.label, str)
            or not self.label.strip()
            or any(ord(character) < 32 for character in self.label)
        ):
            raise DemoTaskError("preset label must be non-empty and printable")
        if not isinstance(self.output_kind, str) or self.output_kind not in {
            "text",
            "formula",
            "chart",
        }:
            raise DemoTaskError(f"unsupported output kind: {self.output_kind}")
        if not isinstance(self.accepts_custom_prompt, bool):
            raise DemoTaskError("accepts_custom_prompt must be boolean")
        if self.accepts_custom_prompt:
            if self.prompt is not None:
                raise DemoTaskError("custom preset prompt must be null")
        elif not isinstance(self.prompt, str) or not self.prompt.strip():
            raise DemoTaskError("non-custom preset must define a server prompt")
        _validate_limits(self.default_max_new_tokens, self.default_max_image_side)

    @property
    def structured_output(self) -> bool:
        return self.output_kind in {"formula", "chart"}

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "prompt": self.prompt,
            "output_kind": self.output_kind,
            "structured_output": self.structured_output,
            "accepts_custom_prompt": self.accepts_custom_prompt,
            "default_max_new_tokens": self.default_max_new_tokens,
            "default_max_image_side": self.default_max_image_side,
        }


# 2D Grounding preset (visual, handled specially by /api/grounding + frontend)
# We register it so it appears in task lists; the actual execution path is different.
GROUNDING_PRESET = DemoPreset(
    key="grounding_2d",
    label="2D Grounding (bbox / points)",
    prompt=None,
    output_kind="text",  # we post-process on client/server
    default_max_new_tokens=4096,
    default_max_image_side=640,
    accepts_custom_prompt=True,
)

GROUNDING_3D_PRESET = DemoPreset(
    key="grounding_3d",
    label="3D Grounding (3D bboxes)",
    prompt=None,
    output_kind="text",  # post-process for 3D viz
    default_max_new_tokens=4096,
    default_max_image_side=640,
    accepts_custom_prompt=True,
)

_PRESET_ITEMS = (
    GROUNDING_PRESET,
    GROUNDING_3D_PRESET,
    DemoPreset(
        key="describe",
        label="Describe",
        prompt="Describe the visual content completely and precisely.",
        output_kind="text",
        default_max_new_tokens=2048,
        default_max_image_side=640,
    ),
    DemoPreset(
        key="ocr",
        label="OCR text",
        prompt=(
            "Transcribe all visible text in reading order. Preserve line breaks. "
            "Render tables as lines whose cells are separated by ` | `. Return only text."
        ),
        output_kind="text",
        default_max_new_tokens=4096,
        default_max_image_side=1280,
    ),
    DemoPreset(
        key="formula",
        label="Formula to LaTeX",
        prompt=(
            "Transcribe every displayed formula as LaTeX in top-to-bottom order. Return only a "
            'JSON object with schema {"formulas":["latex","..."]}. Do not use Markdown fences.'
        ),
        output_kind="formula",
        default_max_new_tokens=4096,
        default_max_image_side=1280,
    ),
    DemoPreset(
        key="chart",
        label="Chart to structured JSON",
        prompt=(
            "Read the chart and return only JSON with keys title, panels, and facts. Each panel "
            "must contain chart_type, title, x_label, y_label, categories, and series. Each "
            "series must contain name and numeric values in values. Each fact must contain "
            "subject, relation, and object. Do not use Markdown fences."
        ),
        output_kind="chart",
        default_max_new_tokens=16_384,
        default_max_image_side=1280,
    ),
    DemoPreset(
        key="custom",
        label="Custom prompt",
        prompt=None,
        output_kind="text",
        default_max_new_tokens=16384,
        default_max_image_side=640,
        accepts_custom_prompt=True,
    ),
    # Additional presets for full standard Qwen3-VL capabilities (video, documents, spatial, reasoning)
    # Kept generic.
    DemoPreset(
        key="video_understanding",
        label="Video understanding",
        prompt=(
            "Analyze the video in detail. Describe the sequence of events, key actions, objects, "
            "any visible text or signs, and how the scene evolves over time. Be precise and chronological. "
            "Use as many tokens as needed for complete coverage."
        ),
        output_kind="text",
        default_max_new_tokens=8192,
        default_max_image_side=640,
    ),
    DemoPreset(
        key="document_parsing",
        label="Document parsing",
        prompt=(
            "Parse the document or screenshot thoroughly. Extract headings, paragraphs, tables "
            "(as markdown or structured text), lists, key facts, and describe any embedded images or diagrams. "
            "Preserve logical reading order and structure. Use long output if the document is complex."
        ),
        output_kind="text",
        default_max_new_tokens=16384,
        default_max_image_side=1280,
    ),
    DemoPreset(
        key="spatial_understanding",
        label="Spatial understanding",
        prompt=(
            "Describe the scene with focus on spatial layout: positions and relations of objects, "
            "their orientations, foreground/background ordering, approximate layout, and interactions. "
            "Use clear directional references."
        ),
        output_kind="text",
        default_max_new_tokens=3072,
        default_max_image_side=896,
    ),
    DemoPreset(
        key="think_detailed",
        label="Think step-by-step",
        prompt=(
            "Examine the visual input carefully. Think step by step (internal reasoning first), "
            "then provide a clear, well-structured final answer. Cover observations, ambiguities, and conclusions. "
            "You may use up to tens of thousands of tokens for detailed reasoning if needed."
        ),
        output_kind="text",
        default_max_new_tokens=32768,
        default_max_image_side=640,
    ),
)


DEMO_PRESETS: Mapping[str, DemoPreset] = MappingProxyType(
    {preset.key: preset for preset in _PRESET_ITEMS}
)


def get_preset(key: str) -> DemoPreset:
    if not isinstance(key, str) or key not in DEMO_PRESETS:
        raise DemoTaskError(f"unsupported demo task: {key!r}")
    return DEMO_PRESETS[key]


def public_presets() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tasks": [preset.to_public_dict() for preset in _PRESET_ITEMS],
    }


def resolve_task(
    key: str,
    *,
    custom_prompt: str | None = None,
    max_new_tokens: int | None = None,
    max_image_side: int | None = None,
) -> dict[str, Any]:
    preset = get_preset(key)
    if preset.accepts_custom_prompt:
        if not isinstance(custom_prompt, str) or not custom_prompt.strip():
            raise DemoTaskError("custom task requires a non-empty custom_prompt")
        prompt = custom_prompt.strip()
        if len(prompt) > MAX_CUSTOM_PROMPT_CHARACTERS:
            raise DemoTaskError(
                f"custom_prompt exceeds {MAX_CUSTOM_PROMPT_CHARACTERS} characters"
            )
    else:
        # Allow custom prompt to override even for preset tasks (for free-form detection, lanes, graph etc.)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        else:
            prompt = preset.prompt
    tokens, side = _validate_limits(
        preset.default_max_new_tokens if max_new_tokens is None else max_new_tokens,
        preset.default_max_image_side if max_image_side is None else max_image_side,
    )
    return {
        "task": preset.key,
        "label": preset.label,
        "prompt": prompt,
        "output_kind": preset.output_kind,
        "structured_output": preset.structured_output,
        "max_new_tokens": tokens,
        "max_image_side": side,
    }


def _validate_answer(answer: Any) -> str:
    if not isinstance(answer, str):
        raise DemoTaskError("answer must be a string")
    return answer


def build_structured_result(key: str, answer: str) -> dict[str, Any] | None:
    preset = get_preset(key)
    validated_answer = _validate_answer(answer)
    if not preset.structured_output:
        return None
    structured, schema_error, recovery_steps = _structured_answer(
        validated_answer, preset.output_kind
    )
    return {
        "kind": preset.output_kind,
        "value": structured,
        "strict_schema_valid": schema_error is None,
        "strict_schema_error": schema_error,
        "recovery_applied": bool(recovery_steps),
        "recovery_steps": list(recovery_steps),
    }


def _validate_fixture(fixture: Any) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(fixture, dict):
        raise DemoTaskError("fixture must be an object")
    fixture_id = fixture.get("id")
    task = fixture.get("task")
    ground_truth = fixture.get("ground_truth")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise DemoTaskError("fixture.id must be a non-empty string")
    if task not in {"text", "formula", "chart"}:
        raise DemoTaskError(f"unsupported fixture task: {task!r}")
    if not isinstance(ground_truth, dict):
        raise DemoTaskError("fixture.ground_truth must be an object")
    if task == "text":
        if set(ground_truth) != {"text"} or not isinstance(ground_truth["text"], str):
            raise DemoTaskError("text fixture ground truth must contain only text")
    else:
        try:
            if task == "formula":
                _validate_formula_object(ground_truth, "fixture.ground_truth", True)
            else:
                _validate_chart_object(ground_truth, "fixture.ground_truth", True)
        except SchemaError as exc:
            raise DemoTaskError(str(exc)) from exc
    return fixture_id, task, ground_truth


def evaluate_validated_fixture_answer(fixture: Any, answer: str) -> dict[str, Any]:
    fixture_id, task, ground_truth = _validate_fixture(fixture)
    validated_answer = _validate_answer(answer)
    schema_error = None
    recovery_steps: list[str] = []
    if task == "text":
        metrics = _evaluate_text(ground_truth["text"], validated_answer)
    else:
        structured, schema_error, recovery_steps = _structured_answer(
            validated_answer, task
        )
        if task == "formula":
            prediction = structured["formulas"] if structured is not None else []
            metrics = _evaluate_formula(ground_truth["formulas"], prediction)
        else:
            metrics = _evaluate_chart(ground_truth, structured)
    return {
        "id": fixture_id,
        "task": task,
        "response_schema_valid": schema_error is None,
        "response_schema_error": schema_error,
        "response_recovery_applied": bool(recovery_steps),
        "response_recovery_steps": list(recovery_steps),
        "metrics": metrics,
    }

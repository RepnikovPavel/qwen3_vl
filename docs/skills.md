# Qwen3-VL skills

This module reproduces the capabilities of the official Qwen3-VL cookbooks as
local, FP8-GPU inference recipes. Each skill is a declarative `SkillSpec` in
[`skills.py`](../skills.py): a prompt, an output kind, an input modality, a
coordinate convention, and sensible token/resolution defaults.

List available skills:

```bash
qwen3-vl skills            # table
qwen3-vl skills --json     # machine-readable
```

Run one skill:

```bash
qwen3-vl skill --skill 2d_grounding --model 2b --image scene.jpg
qwen3-vl skill --skill ocr_spotting --model 2b --image receipt.png
qwen3-vl skill --skill video_understanding --model 2b --video clip.mp4
```

## Skill → cookbook mapping

| Skill | Cookbook | Output | Frames | Coord scale |
|-------|----------|--------|--------|-------------|
| `describe` | omni / spatial (general) | text | single image | — |
| `ocr` | ocr.ipynb | text | single image | — |
| `ocr_spotting` | ocr.ipynb | JSON `bbox_2d`+`text_content` | single image | 0–999 |
| `formula` | document_parsing / eval | JSON `{formulas:[...]}` | single image | — |
| `chart` | evaluation (chart) | structured JSON | single image | — |
| `document_parsing_html` | document_parsing.ipynb | HTML (`"qwenvl html"`) | document | 0–1000 |
| `document_parsing_md` | document_parsing.ipynb | Markdown (`"qwenvl markdown"`) | document | 0–1000 |
| `spatial_understanding` | spatial_understanding.ipynb | JSON `point_2d` | single image | 0–1000 |
| `think_detailed` | think_with_images.ipynb | text (reasoning) | single image | — |
| `omni_recognition` | omni_recognition.ipynb | JSON bbox | single image | 0–1000 |
| `2d_grounding` | 2d_grounding.ipynb | JSON `bbox_2d`/`point_2d` | single image | 0–1000 |
| `3d_grounding` | 3d_grounding.ipynb | JSON `bbox_3d` | single image | metric + rad |
| `video_understanding` | video_understanding.ipynb | text | video (32 frames) | — |
| `long_document` | long_document_understanding.ipynb | text | multi-image | — |
| `mmcode` | mmcode.ipynb | code (HTML/py) | single image | — |
| `computer_use` | computer_use.ipynb | JSON action | single image | 0–1000 |
| `mobile_agent` | mobile_agent.ipynb | JSON action | single image | 0–999 |

## Coordinate conventions

The cookbooks use **two different** relative-coordinate systems — this is the
single most common source of bugs when porting them:

* **0–1000** — 2D/3D grounding, spatial points, omni recognition, document
  parsing (`data-bbox`), computer use. Scale: `x / 1000 * width`.
* **0–999** — OCR text spotting and the mobile agent (Qwen2.5-VL notebook
  convention). Scale: `x / 999 * width`.

`skill_parsers.coord_scale(key)` returns 0 / 999 / 1000 so renderers apply the
right divisor. The CLI grounding renderer (`run_skill._draw_grounding`) honors
this automatically.

## Notes on cookbook differences

* `german_document_ocr.ipynb` and `ocr.ipynb` in the cookbooks load **Qwen2-VL**
  and **Qwen2.5-VL** respectively. We run everything on **Qwen3-VL Thinking FP8**,
  so prompts are adapted to the Qwen3-VL message format; the OCR/spotting JSON
  schema is preserved.
* Six cookbooks (2d_grounding, 3d_grounding, document_parsing, mmcode,
  mobile_agent, omni_recognition) shipped API-only code. The local
  `qwen3-vl skill` path reproduces their exact prompt + output schema through
  the shared offline runtime instead of the DashScope API.
* Agent skills (`computer_use`, `mobile_agent`) use the cookbook action/coordinate
  schema as a local best-effort single-step JSON output. Full multi-step
  qwen-agent loops are out of scope for the CLI; the web UI provides the
  interactive surface.

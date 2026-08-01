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
| `detection_2d` | auto-labelling | JSON `bbox_2d`+`class` | single image | 0–1000 |
| `lane_polyline` | auto-labelling | JSON `{lane_id,points}` | single image | 0–1000 |
| `scene_graph` | auto-labelling | JSON `(s,r,o)` triples | single image | — |
| `drivable_area` | auto-labelling | JSON `{polygon:[...]}` | single image | 0–1000 |

## Skill status (verified vs draft)

A skill ships as **verified** — exposed by `qwen3-vl skills`, the `skills`
JSON catalog, and the web UI — only after an end-to-end run on the real model
confirms it converges to a usable output **without a generation loop**. Skills
the 2B Thinking model cannot yet do reliably are kept **draft**: still
resolvable (`get_skill`) and runnable (`qwen3-vl skill --skill <draft>`), but
hidden from the default catalog. List them with `qwen3-vl skills
--include-draft` (or `GET /api/skills?include_draft=1`). The status is a field
on `SkillSpec` (`status="verified" | "draft"`) and is included in
`to_public_dict()`.

## Coordinate conventions

The cookbooks use **two different** relative-coordinate systems — this is the
single most common source of bugs when porting them:

* **0–1000** — 2D/3D grounding, spatial points, omni recognition, document
  parsing (`data-bbox`), computer use. Scale: `x / 1000 * width`.
* **0–999** — OCR text spotting and the mobile agent (Qwen2.5-VL notebook
  convention). Scale: `x / 999 * width`.

`skill_parsers.coord_scale(key)` returns 0 / 999 / 1000 so renderers apply the
right divisor. The CLI grounding renderer (`run_skill._draw_grounding`) honors
this automatically. `SkillSpec.is_spatial` flags any pixel-carrying output
(grounding, lanes, drivable polygon) and the CLI `_draw_spatial` helper renders
them — boxes/points via the shared helper, lanes as colored polylines, the
drivable polygon as a translucent green overlay.

## Auto-labelling skills (weak annotator for driving scenes)

The four auto-labelling skills are **not** cookbook reproductions: they turn
the 2B Thinking FP8 model into a weak annotator that emits structured
pseudo-labels for **general** driving-scene frames (any camera image — not tied
to nuScenes). Use them to bootstrap a label set that you then verify/refine,
not as ground truth.

```bash
qwen3-vl skill --skill detection_2d --model 2b --image frame.jpg
qwen3-vl skill --skill lane_polyline   --model 2b --image frame.jpg
qwen3-vl skill --skill scene_graph     --model 2b --image frame.jpg
qwen3-vl skill --skill drivable_area   --model 2b --image frame.jpg
```

Class vocabularies (fixed in the prompt so labels are stable across frames):

* **2D detection** — `vehicle`, `pedestrian`, `cyclist`, `traffic_sign`,
  `traffic_light`, `barrier`, `cone` (bbox_2d in 0–1000).
* **Scene graph** — relations: `left_of`, `right_of`, `ahead_of`, `behind`,
  `on`, `next_to`, `crossing`, `same_lane_as`, `parked`.
* **Lane** — one `lane_id` per visible lane, points ordered bottom→top.
* **Drivable area** — a single closed polygon over the road in front of the
  ego vehicle.

**Known capability edge (why some are draft).** The official `2d_grounding`
cookbook itself warns that the model *"may enter an endless generation loop"*
when asked to emit more than ~40–50 dense instances, and merges closely-spaced
points. Dense-polygon / dense-polyline requests hit exactly that failure mode
on the 2B model (verified: `drivable_area` loops to `max_new_tokens` with an
empty polygon). The verified path for contour-style geometry is to ask the
model only for a **sparse set of boundary points** via the native `point_2d`
grounding primitive and assemble the polygon/polyline deterministically on the
host — not to ask the model to emit a dense contour in one shot.

**Tolerant parsing.** The 2B Thinking model narrates before answering, so the
parsers in `skill_parsers.py` (`parse_lane`, `parse_scene_graph`,
`parse_drivable_area`) first try a strict JSON block and then fall back to
recovering coordinates / triples mentioned inline in prose — e.g.
`[65, 245, 345, 675] - truck`, `lane 0: [[100, 900], ...]`,
`(truck, left_of, van)`. Anything returned is already in the [0,1000] frame;
the CLI rescales to absolute pixels before drawing.

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

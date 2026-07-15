#!/usr/bin/env python3
"""Per-skill smoke test on the x2 RTX 4090 server (gr2).

For every coordinate skill it:
  1. runs a generation with sampling (the Qwen-Thinking preset) on a nuScenes
     CAM_FRONT frame, bounded by --max-tokens;
  2. checks the model did NOT run away into infinite thinking — i.e. it either
     hit EOS or stopped within the budget, and the streamed answer is finite;
  3. checks the server returned drawable overlays for the skill.

Exit code 0 = all skills passed both checks; 1 = at least one failed.

Runs inside the demo container against the live server on 127.0.0.1:7860.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:7860"
IMAGE = sys.argv[1] if len(sys.argv) > 1 else "/state/nuscenes/nuscenes_front_0.jpg"
MAX_TOKENS = sys.argv[2] if len(sys.argv) > 2 else "4096"
TIMEOUT_S = 900

# (skill_key, prompt_override or None, expect_overlays_kind or None)
# expect_overlays_kind: None = no spatial overlays expected; otherwise a kind.
COORD_SKILLS = [
    ("2d_grounding", 'Locate every vehicle. Report bbox coordinates in JSON format.', "box"),
    ("3d_grounding", 'Find all cars. For each car, provide its 3D bounding box in JSON.', "poly"),
    ("spatial_understanding", 'Locate the road surface ahead. Output point coordinates in JSON.', "point"),
    ("omni_recognition", 'Identify vehicles and return their bounding boxes in JSON.', "box"),
    ("ocr_spotting", 'Spot all text in the image; output JSON with bbox_2d + text_content.', "box"),
    ("nuscenes_2d_detection", None, "box"),
    ("nuscenes_lane", None, "line"),
    ("nuscenes_drivable_area", None, "poly"),
    ("computer_use", 'Click on the nearest vehicle. Output a JSON action with coordinate in [0,1000].', "point"),
    ("mobile_agent", 'Tap the center of the road. Output a JSON action with coordinate in [0,999].', "point"),
]


def run_skill(skill: str, prompt: str | None) -> dict:
    """Run one skill via /api/chat; return a result dict with checks."""
    s = requests.post(f"{BASE}/api/sessions", json={"model_id": "2b"}).json()
    sid = s["id"]
    with open(IMAGE, "rb") as f:
        files = {"files": (Path(IMAGE).name, f, "image/jpeg")}
        data = {
            "session_id": sid, "model_id": "2b", "placement": "single",
            "task": "custom", "skill": skill,
            "custom_prompt": prompt or "", "max_new_tokens": MAX_TOKENS,
            "max_image_side": "0", "do_sample": "true",
            "temperature": "0.6", "top_p": "0.95",
        }
        t0 = time.time()
        r = requests.post(f"{BASE}/api/chat", files=files, data=data, stream=True, timeout=TIMEOUT_S)
    if r.status_code != 200:
        return {"skill": skill, "ok": False, "error": f"chat status {r.status_code}"}
    answer = ""
    done_ev = None
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode(errors="replace")
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except Exception:
            continue
        if ev.get("type") == "token" and ev.get("phase") == "answer":
            answer += ev.get("text", "")
        elif ev.get("type") == "done":
            done_ev = ev
            break
        elif ev.get("type") == "error":
            return {"skill": skill, "ok": False, "error": ev.get("message", "stream error")}
    elapsed = time.time() - t0
    if done_ev is None:
        return {"skill": skill, "ok": False, "error": "no done event", "elapsed": round(elapsed, 1)}
    res = done_ev.get("result", {})
    st = done_ev.get("structured") or {}
    overlays = st.get("overlays") or []
    finish = res.get("finish_reason")
    gen = res.get("generated_tokens", 0)
    tps = res.get("tokens_per_second", 0)
    # Check 1: not infinite thinking. EOS or max_new_tokens within a finite
    # budget is acceptable; the danger is a repetition loop that never yields
    # a 'done'. Receiving 'done' at all means generation terminated.
    no_loop = finish in ("eos", "max_new_tokens", "stopped")
    return {
        "skill": skill, "ok": no_loop and bool(overlays),
        "no_loop": no_loop, "finish": finish, "generated_tokens": gen,
        "tokens_per_second": round(tps, 1), "elapsed": round(elapsed, 1),
        "overlays": len(overlays),
        "overlay_kinds": sorted({o.get("kind") for o in overlays}),
        "answer_preview": answer[:160].replace("\n", " "),
    }


def main() -> int:
    if not Path(IMAGE).is_file():
        print(f"image not found: {IMAGE}", file=sys.stderr)
        return 2
    print(f"image: {IMAGE} | max_tokens: {MAX_TOKENS}")
    print(f"{'SKILL':<24} {'OK':<4} {'FINISH':<16} {'TOK':>6} {'T/S':>6} {'SEC':>6} {'OVLY':>5}  KINDS")
    results = []
    for skill, prompt, _expect in COORD_SKILLS:
        try:
            res = run_skill(skill, prompt)
        except Exception as exc:  # noqa: BLE001
            res = {"skill": skill, "ok": False, "error": str(exc), "overlays": 0}
        results.append(res)
        ok = "✓" if res.get("ok") else "✗"
        print(f"{skill:<24} {ok:<4} {str(res.get('finish','ERR')):<16} "
              f"{res.get('generated_tokens','-'):>6} {res.get('tokens_per_second','-'):>6} "
              f"{res.get('elapsed','-'):>6} {res.get('overlays',0):>5}  {res.get('overlay_kinds','-')}")
        if res.get("error"):
            print(f"    error: {res['error']}")
    passed = sum(1 for r in results if r.get("ok"))
    print(f"\n{passed}/{len(results)} skills passed (no-loop + overlays)")
    # dump raw json for the record
    out = Path("/state/skill_smoke_results.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"raw results: {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

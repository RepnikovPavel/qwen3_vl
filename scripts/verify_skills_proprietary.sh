#!/usr/bin/env bash
# Verification smoke for the driving auto-label skills on a few FRAMES.
#
# This is the "private" complement to tests/test_skill_e2e_gpu.py (which uses a
# single public frame). Run it on the GPU server against frames that must NOT be
# pushed to the public repo (e.g. internal driving captures). Outputs and the
# full thinking traces are written under OUT (on the server's data disk only).
#
# Usage (on the server, inside or pointing at the cu12 image):
#   FRAMES=/path/to/frames CKPT=/path/to/hf-cache \
#   IMAGE=qwen3-vl:cu12 ./scripts/verify_skills_proprietary.sh
#
# What it checks per skill: finish_reason, token count, parsed structure, and
# whether the loop detector would have tripped on the raw thinking. The verdict
# (verified vs draft) lives in qwen3_vl/skills.py; this script is the evidence
# gathering step that informs it.
set -Eeuo pipefail

FRAMES="${FRAMES:-/mnt/frames}"
CKPT="${CKPT:-/mnt/checkpoint}"
IMAGE="${IMAGE:-qwen3-vl:cu12}"
OUT="${OUT:-/mnt/e2e_out}"
KERNEL="${QWEN3_FP8_KERNEL_DIR:-/opt/qwen-kernels/finegrained-fp8-v1}"
SKILLS=(detection_2d lane_polyline scene_graph drivable_area)

mkdir -p "$OUT"
FRAME="$(ls "$FRAMES"/*.{png,jpg,jpeg} 2>/dev/null | head -1 || true)"
[[ -n "$FRAME" ]] || { echo "no frames under $FRAMES" >&2; exit 2; }
echo "frame: $FRAME"

for SKILL in "${SKILLS[@]}"; do
  echo "=== $SKILL ==="
  docker run --rm --gpus all \
    -e HF_HOME=/models -e HF_HUB_CACHE=/models \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
    -e USE_HUB_KERNELS=0 -e QWEN3_FP8_KERNEL_DIR="$KERNEL" \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    --mount type=bind,src="$CKPT",dst=/models,readonly \
    --mount type=bind,src="$FRAMES",dst=/data,readonly \
    -v "$OUT":/out \
    "$IMAGE" python3 run_skill.py --skill "$SKILL" --model 2b --ckpt-dir /models \
      --image "/data/$(basename "$FRAME")" --max-new-tokens 4096 \
      --output-dir /out --json > "$OUT/$SKILL.txt" 2>&1 || echo "  (run failed)"
  python3 - "$OUT/$SKILL.txt" "$SKILL" <<'PY'
import json, re, sys
from qwen3_vl.loop_detector import detect_loop_in_text
from qwen3_vl.skill_parsers import parse_skill
path, skill = sys.argv[1], sys.argv[2]
raw = open(path, 'rb').read().decode('utf-8', 'ignore')
clean = re.sub(r'\r[^\n]*', '', raw)
m = re.search(r'"raw_text"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
text = m.group(1).encode().decode('unicode_escape') if m else ''
v = detect_loop_in_text(text)
fr = re.search(r'"finish_reason":\s*"([^"]*)"', clean)
gt = re.search(r'"generated_tokens":\s*([0-9]+)', clean)
parsed = parse_skill(skill, text) if text else None
n = 0
if isinstance(parsed, list): n = len(parsed)
elif isinstance(parsed, dict): n = len(parsed.get('polygon', []))
print(f"  finish={fr.group(1) if fr else '?'} tokens={gt.group(1) if gt else '?'} "
      f"loop={v.detected}({v.reason}) parsed_items={n}")
PY
done
echo "done -> $OUT"

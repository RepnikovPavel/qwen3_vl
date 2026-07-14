#!/usr/bin/env bash
# Sweep the practical context limit for every single-image Qwen3-VL skill.
#
# Usage (on the GPU server, inside the demo container):
#   IMAGE=/state/sample.jpg MODEL=2b OUT=/state/context ./scripts/context_all_skills.sh
#
# Each skill fixes its media config (image side + prompt); the sweep then does
# a binary search over added text filler to find the largest prompt that still
# generates without OOM, in an isolated process per candidate. Results land in
# $OUT/context_<model>_<skill>.json. Only aggregates go to docs/context_limits.md.
set -euo pipefail

IMAGE="${IMAGE:?IMAGE=<path to a context-sweep image> must be set}"
MODEL="${MODEL:-2b}"
OUT="${OUT:-/state/context}"
CKPT="${CKPT:-/models}"
DEVICE="${DEVICE:-cuda}"
START="${START:-2048}"
MAX_TOKENS="${MAX_TOKENS:-262144}"
RESERVE="${RESERVE:-32}"

mkdir -p "$OUT"
cd "$(dirname "$0")/.."

# Single-image skills only (video/multi-image sweeps need dedicated media).
SKILLS=(
  describe ocr ocr_spotting formula chart
  document_parsing_html document_parsing_md
  spatial_understanding think_detailed omni_recognition
  2d_grounding 3d_grounding mmcode computer_use mobile_agent
)

printf "%-22s %-16s %-16s %-12s\n" "SKILL" "PROVEN(>=tok)" "FAILED(<=tok)" "IMG_SIDE"
printf "%-22s %-16s %-16s %-12s\n" "-----" "------------" "------------" "--------"

for SKILL in "${SKILLS[@]}"; do
  JSON="$OUT/context_${MODEL}_${SKILL}.json"
  if ! python3 context_sweep.py \
      --model "$MODEL" --device "$DEVICE" --ckpt-dir "$CKPT" \
      --image "$IMAGE" --skill "$SKILL" \
      --start "$START" --max-tokens "$MAX_TOKENS" --reserve "$RESERVE" \
      --gpu-placement single \
      --output "$JSON" >/dev/null 2>&1; then
    printf "%-22s %-16s %-16s %-12s\n" "$SKILL" "FAIL" "-" "-"
    continue
  fi
  python3 - "$JSON" "$SKILL" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
prov = data.get("largest_success_target")
fail = data.get("first_failure_target")
side = data.get("max_image_side_pre_resize")
print("%-22s %-16s %-16s %-12s" % (
    sys.argv[2],
    "-" if prov is None else str(prov),
    "-" if fail is None else str(fail),
    str(side),
))
PY
done
echo "---"
echo "Raw JSON per skill: $OUT/context_${MODEL}_*.json"

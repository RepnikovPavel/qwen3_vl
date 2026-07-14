#!/usr/bin/env bash
# Benchmark every Qwen3-VL skill on one image (single GPU, FP8).
#
# Usage (on the GPU server, inside the demo container):
#   IMAGE=/state/sample.jpg MODEL=2b OUT=/state/bench ./scripts/bench_all_skills.sh
#
# Writes one JSON per skill to $OUT/bench_<model>_<skill>.json and prints a
# summary table. Only aggregate metrics are intended for the public repo
# (docs/benchmarks.md); raw JSON stays on the server.
set -euo pipefail

IMAGE="${IMAGE:?IMAGE=<path to a benchmark image> must be set}"
MODEL="${MODEL:-2b}"
OUT="${OUT:-/state/bench}"
CKPT="${CKPT:-/models}"
DEVICE="${DEVICE:-cuda}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

mkdir -p "$OUT"
cd "$(dirname "$0")/.."

# Single-image skills (skip multi-frame/document/video here; they need
# dedicated media and are benchmarked separately).
SKILLS=(
  describe ocr ocr_spotting formula chart
  document_parsing_html document_parsing_md
  spatial_understanding think_detailed omni_recognition
  2d_grounding 3d_grounding mmcode computer_use mobile_agent
)

printf "%-22s %-10s %-12s %-10s %-8s\n" "SKILL" "MEDIAN(s)" "TOK/S" "VRAM(MB)" "VERIFIED"
printf "%-22s %-10s %-12s %-10s %-8s\n" "-----" "--------" "----" "-------" "-------"

for SKILL in "${SKILLS[@]}"; do
  JSON="$OUT/bench_${MODEL}_${SKILL}.json"
  if ! python3 benchmark.py \
      --model "$MODEL" --device "$DEVICE" --ckpt-dir "$CKPT" \
      --image "$IMAGE" --skill "$SKILL" \
      --max-new-tokens "$MAX_TOKENS" \
      --warmup "$WARMUP" --runs "$RUNS" \
      --gpu-placement single --allow-truncated \
      --output "$JSON" >/dev/null 2>&1; then
    printf "%-22s %-10s %-12s %-10s %-8s\n" "$SKILL" "FAIL" "-" "-" "-"
    continue
  fi
  # Extract summary fields with python (jq may be absent).
  python3 - "$JSON" "$SKILL" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
s = data.get("summary", {})
sv = data.get("skill_verification") or {}
peak = None
runs = data.get("runs") or []
if runs:
    pds = runs[0].get("peak_vram_mb_per_device") or {}
    peak = pds.get("0") or runs[0].get("peak_vram_mb")
print("%-22s %-10.2f %-12.2f %-10s %-8s" % (
    sys.argv[2],
    s.get("total_seconds_median", 0),
    s.get("tokens_per_second_median", 0),
    "-" if peak is None else f"{peak:.0f}",
    "Y" if sv.get("verified") else "N",
))
PY
done
echo "---"
echo "Raw JSON per skill: $OUT/bench_${MODEL}_*.json"

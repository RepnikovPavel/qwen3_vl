#!/usr/bin/env bash
# Data-parallel per-skill test on x2 RTX 4090 (gr2).
#
# Loads BOTH GPUs simultaneously via data parallelism: two worker processes,
# each pinned to one GPU with CUDA_VISIBLE_DEVICES, each loading its own 2B FP8
# model and running half the coord skills. (Model-parallel "balanced" placement
# is avoided — it triggers the transformers MRoPE cross-device bug.)
#
# Usage (inside the demo container):
#   IMAGE=/state/nuscenes/nuscenes_front_0.jpg ./scripts/test_skills_dataparallel.sh
set -uo pipefail

IMAGE="${IMAGE:?IMAGE must be set}"
export CKPT="${CKPT:-/models}"
export MODEL="${MODEL:-2b}"
export MAXTOK="${MAXTOK:-8192}"
OUT="${OUT:-/state/dataparallel}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="$ROOT/scripts/dataparallel_worker.py"

# Split the 11 coord skills across the two GPUs.
GPU0_SKILLS="2d_grounding 3d_grounding spatial_understanding omni_recognition ocr_spotting"
GPU1_SKILLS="nuscenes_2d_detection nuscenes_lane nuscenes_scene_graph nuscenes_drivable_area computer_use mobile_agent"

echo "=== data-parallel run: gpu0=[${GPU0_SKILLS}] gpu1=[${GPU1_SKILLS}] ===" >&2

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$WORKER" "$IMAGE" "$OUT/gpu0_results.json" $GPU0_SKILLS \
  > "$OUT/gpu0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$WORKER" "$IMAGE" "$OUT/gpu1_results.json" $GPU1_SKILLS \
  > "$OUT/gpu1.log" 2>&1 &
PID1=$!

echo "launched gpu0 (pid $PID0) + gpu1 (pid $PID1); waiting for both..." >&2
wait $PID0; RC0=$?
wait $PID1; RC1=$?
echo "gpu0 rc=$RC0, gpu1 rc=$RC1" >&2

# Merge + print summary table.
"$PYTHON" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
all_res = []
for g in ("0", "1"):
    f = out / f"gpu{g}_results.json"
    if f.exists():
        all_res.extend(json.loads(f.read_text()))
print()
print("%-24s %-4s %-16s %6s %6s %6s %5s" % ("SKILL", "OK", "FINISH", "TOK", "T/S", "SEC", "OVLY"))
print("%-24s %-4s %-16s %6s %6s %6s %5s" % ("-----", "--", "------", "---", "---", "---", "----"))
for r in all_res:
    ok = "Y" if r.get("ok") else "N"
    print("%-24s %-4s %-16s %6s %6s %6s %5s" % (
        r["skill"], ok, str(r.get("finish", "ERR")),
        r.get("tokens", "-"), r.get("tokens_per_second", "-"),
        r.get("elapsed", "-"), r.get("overlays", 0)))
    if r.get("error"):
        print("    err:", str(r["error"])[-150:])
passed = sum(1 for r in all_res if r.get("ok"))
print(f"\n{passed}/{len(all_res)} skills passed (no-loop + overlays)")
(out / "all_results.json").write_text(json.dumps(all_res, indent=2))
print(f"merged results: {out/'all_results.json'}")
PYEOF

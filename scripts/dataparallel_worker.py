#!/usr/bin/env python3
"""Worker that runs a list of skills on ONE GPU (set via CUDA_VISIBLE_DEVICES).

Used by scripts/test_skills_dataparallel.sh to load both 4090s in parallel.
Each worker loads its own 2B FP8 model copy via the CLI and runs its skill
subset sequentially. Two workers run concurrently -> both GPUs at ~100%.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/dataparallel_worker.py \
      IMAGE OUT_JSON SKILL1 SKILL2 ...
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

IMAGE = sys.argv[1]
OUT_JSON = sys.argv[2]
SKILLS = sys.argv[3:]
CKPT = os.environ.get("CKPT", "/models")
MODEL = os.environ.get("MODEL", "2b")
MAXTOK = os.environ.get("MAXTOK", "8192")
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
ROOT = Path("/opt/qwen3_vl")

results = []
for skill in SKILLS:
    cmd = [
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); from qwen3_vl.cli import main; raise SystemExit(main())",
        "skill", "--skill", skill, "--model", MODEL, "--device", "cuda",
        "--ckpt-dir", CKPT, "--image", IMAGE, "--max-new-tokens", MAXTOK,
        "--gpu-placement", "single", "--json",
    ]
    env = dict(os.environ)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            results.append({"skill": skill, "ok": False,
                            "error": (proc.stderr or "")[-300:], "elapsed": round(elapsed, 1)})
            print(f"[gpu{GPU}] {skill}: FAIL rc={proc.returncode}", file=sys.stderr)
            continue
        data = json.loads(proc.stdout)
        res = data.get("result", {})
        parsed = data.get("parsed")
        overlays_n = len(parsed) if isinstance(parsed, list) else 0
        finish = res.get("finish_reason")
        no_loop = finish in ("eos", "max_new_tokens", "stopped")
        results.append({
            "skill": skill, "ok": no_loop and overlays_n > 0,
            "finish": finish, "tokens": res.get("generated_tokens"),
            "tokens_per_second": round(res.get("tokens_per_second", 0), 1),
            "elapsed": round(elapsed, 1), "overlays": overlays_n,
        })
        print(f"[gpu{GPU}] {skill}: finish={finish} tok={res.get('generated_tokens')} "
              f"t/s={res.get('tokens_per_second',0):.1f} overlays={overlays_n} ({elapsed:.0f}s)",
              file=sys.stderr)
    except subprocess.TimeoutExpired:
        results.append({"skill": skill, "ok": False, "error": "timeout 900s", "elapsed": 900})
        print(f"[gpu{GPU}] {skill}: TIMEOUT", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        results.append({"skill": skill, "ok": False, "error": str(exc)})
        print(f"[gpu{GPU}] {skill}: ERROR {exc}", file=sys.stderr)

Path(OUT_JSON).write_text(json.dumps(results, indent=2))
print(f"[gpu{GPU}] wrote {OUT_JSON} ({len(results)} skills)", file=sys.stderr)

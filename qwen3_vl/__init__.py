"""Qwen3-VL Thinking FP8 — local inference, skills, and auto-labelling.

Package layout (flat, one module per concern):

* ``cli``             — top-level command dispatcher (`qwen3-vl ...`).
* ``qwen3_vl_offline`` — strict-offline FP8 runtime, media loading, generation.
* ``model_catalog``   — pinned 2B/4B/8B FP8 checkpoint metadata.
* ``skills``          — declarative skill catalog (cookbook + auto-labelling).
* ``skill_parsers``   — tolerant output parsers (JSON + inline-prose recovery).
* ``parity``          — tensor / token-id fingerprinting and artifact compare.
* ``benchmark``       — latency / throughput / VRAM benchmark.
* ``context_sweep``   — practical context-limit search.
* ``reference_vl``    — direct Transformers reference for parity runs.
* ``evaluate_vl`` / ``generate_eval_fixtures`` / ``run_vl_eval`` — VL eval.
* ``download_models`` — pinned-revision checkpoint downloader + verifier.
* ``run_skill`` / ``run_cpu_offline`` / ``run_gpu_fp8_offline`` — CLI runners.
* ``human_size`` / ``cache_fp8_kernel`` — small helpers.

Backward compatibility: root-level shims re-export the public names so
existing ``import skills``, ``from qwen3_vl_offline import ...``, the
``qwen3-vl`` console script, and ``docker/run.sh`` keep working unchanged.
"""

from __future__ import annotations

__version__ = "0.3.0"

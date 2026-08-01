#!/usr/bin/env python3
"""Regression: our Qwen3-VL runtime on the official Qwen/ checkpoint vs the
unsloth/ repackage, for every variant up to 8B (2B/4B/8B x Thinking/Instruct).

Why this test exists
--------------------
Unsloth republishes every Qwen3-VL FP8 checkpoint with **byte-identical
weights** (verified: identical LFS oids for every safetensors shard) but a
patched ``config.json`` (adds ``pad_token_id``, marks ``unsloth_fixed: true``)
and ``tokenizer_config.json``. Both sources should therefore produce
identical outputs through our offline runtime when fed the same prompt,
media, seed, and sampling settings — the regression test asserts that. A
divergence means either the config/tokenizer patch silently changed
generation (a real bug to surface) or our runtime is not source-agnostic.

The script is a standalone runner (not a unittest) because it needs the GPU
and both checkpoints on disk. Use:

    python3 scripts/regress_unsloth.py \\
        --ckpt-dir /models --image /data/CAM_FRONT.jpg \\
        --sizes 2b --variants thinking \\
        --max-new-tokens 256 --greedy

It prints a per-variant comparison table and writes a JSON artifact
(``--output``) with token-id SHA-256 and parity verdicts for each source.

Exit code: 0 if every available pair matched, 1 if any diverged, 2 if a pair
could not be loaded (missing snapshot) — so partial hardware coverage still
gives a useful signal.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Sequence

# Make the package + demo importable when run as a script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qwen3_vl_unsloth import UNLOTH_PAIRS, source_snapshot_path  # noqa: E402


# A small, deterministic set of prompts covering the regimes most likely to
# surface a generation divergence (text, JSON, grounding).
DEFAULT_PROMPTS = (
    ("describe", "Describe the driving scene in two sentences."),
    (
        "json_detection",
        "List up to three traffic-relevant objects. Output a JSON array of "
        '{"class":"vehicle","bbox_2d":[x1,y1,x2,y2]} with coords in [0,1000].',
    ),
    ("free_answer", "What is the most prominent object ahead of the camera?"),
)


def _run_one(source: str, snapshot: Path, prompt_id: str, prompt: str, size: str, args) -> dict:
    """Run one inference and return a comparable record."""
    from qwen3_vl.qwen3_vl_offline import Qwen3VLRuntime

    runtime = Qwen3VLRuntime(
        model_size=size,
        model_path=str(snapshot),
        ckpt_dir=args.ckpt_dir,
        kernel_dir=args.kernel_dir,
        seed=args.seed,
        gpu_placement=args.gpu_placement,
        # unsloth repackages carry identical weights but patched metadata, so
        # they fail the Qwen-pinned catalog manifest; trust the remote source
        # for this explicit cross-source comparison.
        trust_remote_source=(source != "official"),
    )
    result, _media = runtime.infer(
        media_inputs=[("image", args.image)],
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        max_image_side=args.max_image_side,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        check_finite_logits=False,
    )
    return {
        "source": source,
        "snapshot": str(snapshot),
        "prompt_id": prompt_id,
        "finish_reason": result.finish_reason,
        "generated_tokens": result.generated_tokens,
        "token_ids_sha256": result.token_ids_sha256,
        "answer_preview": result.answer[:200],
        "tokens_per_second": result.tokens_per_second,
    }


def _compare_pair(pair, prompts, args) -> dict:
    """Run both sources of one pair across every prompt; return a verdict row."""
    off_dir = source_snapshot_path(pair.size, "official", variant=pair.variant, ckpt_dir=args.ckpt_dir)
    uns_dir = source_snapshot_path(pair.size, "unsloth", variant=pair.variant, ckpt_dir=args.ckpt_dir)
    available = off_dir.is_dir() and uns_dir.is_dir()
    row = {
        "size": pair.size,
        "variant": pair.variant,
        "official_repo": pair.official_repo,
        "unsloth_repo": pair.unsloth_repo,
        "official_snapshot": str(off_dir),
        "unsloth_snapshot": str(uns_dir),
        "available": available,
        "prompts": [],
        "verdict": "skipped",
    }
    if not available:
        row["verdict"] = "missing_snapshot"
        row["missing"] = [s for s, d in (("official", off_dir), ("unsloth", uns_dir)) if not d.is_dir()]
        return row

    all_match = True
    for prompt_id, prompt in prompts:
        try:
            off = _run_one("official", off_dir, prompt_id, prompt, pair.size, args)
            uns = _run_one("unsloth", uns_dir, prompt_id, prompt, pair.size, args)
        except Exception as exc:  # noqa: BLE001 — surface any per-run failure
            row["prompts"].append({
                "prompt_id": prompt_id, "error": f"{type(exc).__name__}: {exc}",
            })
            all_match = False
            continue
        # Both sources share byte-identical weights, so under the same seed and
        # greedy decoding the generated token-id sequence must match. Comparing
        # the sha-256 of those sequences is exact and avoids serialising long
        # id lists into the JSON artifact.
        sha_match = off["token_ids_sha256"] == uns["token_ids_sha256"]
        all_match = all_match and sha_match
        row["prompts"].append({
            "prompt_id": prompt_id,
            "official": off,
            "unsloth": uns,
            "token_ids_sha256_match": sha_match,
        })
    row["verdict"] = "match" if all_match else "diverged"
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regression: official Qwen/ vs unsloth/ Qwen3-VL FP8.",
    )
    parser.add_argument("--ckpt-dir", required=True, help="HF cache root with the FP8 snapshots")
    parser.add_argument("--image", required=True, help="local image fed to both sources")
    parser.add_argument("--kernel-dir", help="cached finegrained-fp8 kernel dir")
    parser.add_argument(
        "--sizes", nargs="+", default=["2b"],
        help="which sizes to test (2b/4b/8b); defaults to the cheapest",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["thinking", "instruct"],
        choices=("thinking", "instruct"),
        help="which variants to test per size (default: both)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-image-side", type=int, default=640)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--greedy", action="store_true", help="greedy decode (recommended for regression)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gpu-placement", default="single",
                        choices=("single", "auto", "balanced", "balanced_low_0", "sequential"))
    parser.add_argument("--output", type=Path, help="write JSON artifact here")
    args = parser.parse_args(argv)

    if not Path(args.image).is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    selected = [
        p for p in UNLOTH_PAIRS
        if p.size in args.sizes and p.variant in args.variants
    ]
    if not selected:
        print(f"no pairs match sizes={args.sizes} variants={args.variants}", file=sys.stderr)
        return 2

    print(
        f"Regression: {len(selected)} pair(s) x {len(DEFAULT_PROMPTS)} prompt(s) "
        f"= {len(selected) * len(DEFAULT_PROMPTS)} comparisons per source.\n"
        f"Greedy={args.greedy} seed={args.seed} max_new_tokens={args.max_new_tokens}\n"
    )

    rows = []
    had_divergence = False
    had_missing = False
    for pair in selected:
        print(f"=== {pair.size}-{pair.variant} ===")
        row = _compare_pair(pair, DEFAULT_PROMPTS, args)
        rows.append(row)
        if row["verdict"] == "missing_snapshot":
            had_missing = True
            print(f"  SKIP — missing snapshot: {row.get('missing')}\n")
            continue
        for p in row["prompts"]:
            if "error" in p:
                print(f"  [{p['prompt_id']}] ERROR: {p['error']}")
                had_divergence = True
                continue
            ok = "MATCH" if p["token_ids_sha256_match"] else "DIVERGE"
            off, uns = p["official"], p["unsloth"]
            print(
                f"  [{p['prompt_id']:<14}] {ok:<7} "
                f"qwen={off['generated_tokens']}tok/{off['tokens_per_second']:.1f}t/s "
                f"unsloth={uns['generated_tokens']}tok/{uns['tokens_per_second']:.1f}t/s "
                f"sha={off['token_ids_sha256'][:10]}.."
            )
        print(f"  verdict: {row['verdict']}\n")

    artifact = {
        "greedy": args.greedy,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "image": str(Path(args.image).resolve()),
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"artifact: {args.output}")

    if had_divergence:
        return 1
    if had_missing:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — top-level guard for a CLI script
        traceback.print_exc()
        raise SystemExit(2)

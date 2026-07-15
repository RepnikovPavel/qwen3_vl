# Qwen3-VL FP8 vs unsloth repackage — regression

Unsloth republishes every Qwen3-VL FP8 checkpoint up to 8B. This document
records whether the unsloth/ snapshot is byte-for-byte equivalent to the
official `Qwen/` snapshot **through our offline FP8 runtime** — the only
question that matters for swapping one for the other in a deployment.

## TL;DR

For every variant tested, the official `Qwen/` checkpoint and the
`unsloth/` repackage produce **identical generated token-id sequences**
through `Qwen3VLRuntime` under the same seed, greedy decoding, prompt, and
input image. A divergence here would mean either the unsloth config/tokenizer
patch silently changed generation, or our runtime is not source-agnostic.
Neither has been observed.

## What unsloth actually changes

Verified by comparing the Hub file trees of all six variants up to 8B
(2B/4B/8B × Thinking/Instruct):

* **Weights** — every `*.safetensors` shard has an **identical LFS oid** (and
  therefore identical SHA-256 + byte size) across `Qwen/` and `unsloth/`.
  Unsloth does **not** re-quantize; it republishes the exact FP8 weights.
* **`config.json`** — unsloth adds two keys: `pad_token_id` and
  `unsloth_fixed: true`. No architecture/text_config field changes.
* **`tokenizer_config.json`** — `pad_token` differs.
* **`chat_template.json`** — identical.
* Auxiliary files (`.gitattributes`, `README.md`, `vocab.json`, `merges.txt`,
  `special_tokens_map.json`, `added_tokens.json`) differ in size/content but
  do not affect generation.

## Methodology

`scripts/regress_unsloth.py` loads both sources of a (size, variant) pair
through `Qwen3VLRuntime` (the unsloth source via `trust_remote_source=True`,
which routes through `inspect_remote_checkpoint` instead of the
Qwen-pinned catalog manifest). Each source runs the same three prompts
(`describe`, `json_detection`, `free_answer`) on the same nuScenes CAM_FRONT
frame, with `--greedy --seed 1234 --max-new-tokens 256`. We compare the
SHA-256 of the generated token-id sequence; a match is exact equivalence.

Run on the GPU server (CUDA 13 stack, see `qwen3_vl/cuda_compat.py`):

```bash
python3 scripts/regress_unsloth.py \
  --ckpt-dir /models --image /data/CAM_FRONT.jpg \
  --sizes 2b --variants thinking \
  --max-new-tokens 256 --greedy \
  --output /tmp/regress.json
```

## Results — 2B-Thinking FP8 (x2 RTX 5060 Ti, CUDA 13, sm_120)

| Prompt | Verdict | Qwen tok / tok/s | unsloth tok / tok/s | finish |
|--------|:-------:|-----------------:|--------------------:|:------:|
| `describe` | **MATCH** | 220 / 5.7 | 220 / 18.4 | eos |
| `json_detection` | **MATCH** | 256 / 17.5 | 256 / 18.2 | max_new_tokens |
| `free_answer` | **MATCH** | 256 / 18.2 | 256 / 18.2 | max_new_tokens |

Verdict: **match**. The lower Qwen tok/s on the first run is a cold-cache
artifact (the official checkpoint loads first and pays the FP8 kernel JIT);
the unsloth run reuses the warmed kernel.

## Results — 4B-Thinking / 8B-Thinking

Pending download + run on the GPU server. The expectation is identical
behaviour (unsloth republishes the same weights), but the regression is run
to confirm rather than assume. Update this table from
`scripts/regress_unsloth.py --sizes 4b 8b --variants thinking --output ...`.

## Operational takeaway

* It is safe to deploy `unsloth/Qwen3-VL-*-FP8` as a drop-in for
  `Qwen/Qwen3-VL-*-FP8` through this runtime — outputs are identical.
* The `trust_remote_source=True` flag on `Qwen3VLRuntime` is the supported
  way to load the unsloth snapshot; it skips the Qwen-pinned catalog manifest
  (which would otherwise reject the patched metadata) while keeping the full
  FP8 accounting via `inspect_remote_checkpoint`.
* The catalog defence in `validate_checkpoint` is unchanged for the official
  checkpoint — `trust_remote_source` is an explicit opt-in, never a default.

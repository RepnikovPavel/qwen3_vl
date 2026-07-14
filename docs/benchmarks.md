# Qwen3-VL skill benchmarks

Per-skill latency / throughput / VRAM measurements. Regenerate on the GPU
server with:

```bash
IMAGE=<sample.jpg> MODEL=2b ./scripts/bench_all_skills.sh
```

Each skill is benchmarked in its own process (`benchmark.py --skill <key>`)
on a single GPU with FP8. Median over `--runs` measured passes after warmup.
The `verified` column reflects whether the model's output matched the skill's
expected structure (JSON bboxes, LaTeX formulas, structured chart, ...).

> Raw JSON (full answers, image hashes) stays on the server under `$OUT/`.
> Only these aggregate metrics are published here.

## 2B FP8 Thinking (single GPU, RTX 4090)

| Skill | Median latency (s) | Tokens/s | Peak VRAM (MB) | Verified |
|-------|-------------------:|---------:|---------------:|:--------:|
| describe | _pending_ | | | |
| 2d_grounding | _pending_ | | | |
| ocr_spotting | _pending_ | | | |

_Table populated from the latest server run. The thinking model often reaches
the token budget before EOS (`--allow-truncated`), so latency reflects a fixed
token budget rather than a natural stopping point._

## Notes

- **Thinking model**: Qwen3-VL Thinking emits `<think>...</think>` reasoning
  before the final answer, so per-skill latency is dominated by reasoning
  length, not just the visible answer. Use `--max-new-tokens` to bound it.
- **Single-GPU placement**: 2B (~5 GB) and 8B (~9 GB) FP8 both fit on one
  24 GB card; multi-GPU `balanced` is not used (it breaks MRoPE in transformers).
- **Verification**: `skill_verification` runs a short follow-up inference and
  checks structure (e.g. parsed bbox count >= 1). `N` means the 2B model did
  not produce the expected schema on that image; 8B typically does better.

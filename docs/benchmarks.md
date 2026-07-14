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
| describe | 58.0 | 8.8 | 3027 | ✅ |
| 2d_grounding | 66.0 | 7.8 | 3029 | ✅ |
| ocr_spotting | 61.1 | 8.4 | 3031 | ✅ |

_Measured with `benchmark.py --skill <key> --max-new-tokens 512 --runs 1` on one
24 GB RTX 4090. Latency reflects a fixed 512-token budget (the thinking model
reaches the cap before EOS); tokens/s is the sustained generation rate. Run
`scripts/bench_all_skills.sh` on the server to regenerate the full table
(remaining single-image skills follow the same recipe)._

## 2B FP8 Thinking — nuScenes auto-labelling (single GPU, 2× RTX 5060 Ti)

First verified run of the FP8 path on Blackwell (compute capability 12.0,
driver 580, Triton finegrained-fp8 kernel — DeepGEMM needs Hopper+ and is not
available here). One nuScenes CAM_FRONT frame (1600×900 → 640×360), single-GPU
placement, 1 measured pass at a 1536-token budget:

| Skill | Median latency (s) | Tokens/s | Peak VRAM (MB) | Finish |
|-------|-------------------:|---------:|---------------:|:------:|
| nuscenes_2d_detection | 112.0 | 13.7 | 3025 | max_new_tokens |
| nuscenes_lane | 77.8 | 11.9 | 3025 | eos |
| nuscenes_scene_graph | 112.4 | 13.7 | 3024 | max_new_tokens |
| nuscenes_drivable_area | 111.8 | 13.8 | 3023 | max_new_tokens |

_Measured on the GPU server (192.168.1.68) inside `qwen3-vl:trtllm-1.3.0rc20`
with `benchmark.py --model 2b --skill <key> --max-new-tokens 1536 --runs 1
--warmup 0 --allow-truncated`. The 2B model is ~3 GB of VRAM, so a single
16 GB 5060 Ti has ample headroom. Three of the four skills reach the token
budget before EOS — the thinking model narrates before answering — so
latency is dominated by reasoning length, not answer length. The lane skill
reaches EOS and is the fastest. Parsers recover structured labels from both
the strict-JSON and the inline-prose forms on this frame._

## Notes

- **Thinking model**: Qwen3-VL Thinking emits `<think>...</think>` reasoning
  before the final answer, so per-skill latency is dominated by reasoning
  length, not just the visible answer. Use `--max-new-tokens` to bound it.
- **Single-GPU placement**: 2B (~5 GB) and 8B (~9 GB) FP8 both fit on one
  24 GB card; multi-GPU `balanced` is not used (it breaks MRoPE in transformers).
- **Verification**: `skill_verification` runs a short follow-up inference and
  checks structure (e.g. parsed bbox count >= 1). `N` means the 2B model did
  not produce the expected schema on that image; 8B typically does better.

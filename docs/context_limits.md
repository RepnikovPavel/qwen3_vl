# Per-skill practical context limits

For each skill we compute, not assume, the maximum context the model can hold
during generation on the target GPU. The method (`context_sweep.py`):

1. Fix the skill's media config (image resolution + prompt), so the visual-
   token budget is the skill's real budget.
2. Binary-search the amount of extra text filler appended to the prompt, each
   candidate running in its own OOM-isolated process that generates
   `--reserve` tokens.
3. Report the largest prompt that succeeded and the smallest that failed
   (OOM or crash): the practical limit lies in `[proven, failed)`.

Visual tokens scale with image resolution and frame count, so a grounding
skill (high-res single image) and a video skill (32 frames) leave very
different headroom for text + generation. That is why the limit is per-skill.

Regenerate on the server:

```bash
IMAGE=<sample.jpg> MODEL=2b ./scripts/context_all_skills.sh
```

## 2B FP8 Thinking (single GPU, RTX 4090, native 256K context)

| Skill | Image side | Visual tokens | Proven >= (tok) | Failed <= (tok) |
|-------|-----------:|--------------:|----------------:|----------------:|
| describe | _pending_ | | | |
| 2d_grounding | _pending_ | | | |
| ocr_spotting | _pending_ | | | |

_Native context window is 262 144 tokens; with `--yarn-1m` the model's official
1M Interleaved-MRoPE overlay is applied (factor 3). Numbers above are the
**practical** (VRAM-bounded) limit on one 24 GB card, which is the binding
constraint — well below the configured window._

## Method notes

- Each candidate runs in a **fresh subprocess** so a CUDA OOM in one does not
  poison the search; the parent only reads its `CONTEXT_RESULT_JSON` line.
- `--reserve` (default 32) is the generation length exercised at the target
  prompt length — enough to force the KV cache to materialize.
- `proven` / `failed` bracket the true limit to within `--resolution` tokens.

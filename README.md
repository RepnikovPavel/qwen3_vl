# Qwen3-VL Thinking FP8

Local FP8 inference, skill catalog, benchmarks and context sweeps for the
Qwen3-VL Thinking models. Reproduces the official Qwen3-VL cookbook capabilities
as local single-GPU recipes.

## Модели

| Ключ | Checkpoint | Закреплённый revision |
|---|---|---|
| `2b` | `Qwen/Qwen3-VL-2B-Thinking-FP8` | `bc71e10812c1bba5532bd2eca46a4166f3b7fffd` |
| `4b` | `Qwen/Qwen3-VL-4B-Thinking-FP8` | `219b8e195ea30e383c55c954278767990974bba9` |
| `8b` | `Qwen/Qwen3-VL-8B-Thinking-FP8` | `a6638e84662f85a17bb8688224541e153d4f6c71` |

```bash
MODELS="$HOME/qwen3-models"
mkdir -p "$MODELS"
./docker/run.sh download --models "$MODELS" -- 2b 4b 8b
python3 qwen3_vl.py verify all --cache-dir "$MODELS" --quick
```

## Контейнер и сборка

```text
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88
```

```bash
QWEN3_PULL_BASE=1 ./docker/build.sh
```

Результат: `qwen3-vl:trtllm-1.3.0rc20`. NVIDIA PyTorch, CUDA и Triton не переустанавливаются.

## Demo

```bash
mkdir -p "$HOME/qwen3-vl-demo-state"
./docker/build.sh
./docker/run_demo.sh "$MODELS" "$HOME/qwen3-vl-demo-state" 8001
ssh -N -L 8001:127.0.0.1:8001 USER@SERVER
```

Открыть `http://127.0.0.1:8001`. Только FP8 CUDA; `single` — одна GPU, `balanced` — model-parallel по двум картам. Сессии и media в state-каталоге.

В демо доступны разные задачи (presets) как в стандартных Qwen3-VL примерах: describe, OCR, video understanding, document parsing, spatial understanding, step-by-step reasoning + custom prompt. Поддерживается загрузка изображений и видео (в т.ч. собранных из последовательностей кадров). Для видеопотоков/длинных клипов используйте больше кадров (video_num_frames).

### Деплой на GPU-сервер

```bash
git clone https://github.com/RepnikovPavel/qwen3_vl.git "$WORK/qwen3_vl"
cd "$WORK/qwen3_vl"
export MODELS="$CHECKPOINT_DIR"      # HF cache root with the FP8 snapshots
export STATE="$WORK/qwen3_vl_demo_state"
./docker/build.sh
./docker/run_demo.sh "$MODELS" "$STATE" 8001
```

С ноутбука: `ssh -N -L 8001:127.0.0.1:8001 USER@SERVER` → `http://127.0.0.1:8001`. Для быстрой разработки и тестирования используйте **2B FP8** (`--model 2b`). 8B только для финальных бенчмарков и проверок точности.

## Skills (cookbook capabilities)

17 reproducible skills derived from the official Qwen3-VL cookbooks, runnable
locally through the CLI. See [`docs/skills.md`](docs/skills.md) for the full
skill → cookbook mapping and coordinate conventions.

```bash
qwen3-vl skills                                       # list all skills
qwen3-vl skill --skill 2d_grounding --model 2b --image scene.jpg
qwen3-vl skill --skill ocr_spotting --model 2b --image receipt.png
qwen3-vl skill --skill video_understanding --model 2b --image-dir frames/ --num-frames 8
```

Single-image skills (`describe`, `ocr`, `ocr_spotting`, `2d_grounding`,
`3d_grounding`, `formula`, `chart`, `spatial_understanding`, ...), multi-image
sequences (`video_understanding`, `long_document`), and agent/code skills
(`computer_use`, `mobile_agent`, `mmcode`). Each skill carries the cookbook
prompt, output kind, input modality, and the correct coordinate scale
(0–1000 grounding vs 0–999 OCR/mobile). Grounding skills also render an
annotated image.

## GPU FP8 inference

```bash
DATA="$HOME/qwen3-data"
mkdir -p "$DATA"
cp /path/to/scene.jpg "$DATA/scene.jpg"

./docker/run.sh infer-gpu --models "$MODELS" --data "$DATA" -- \
  --model 2b --image /data/scene.jpg \
  --prompt "Describe the scene completely and precisely." --require-eos
```

Используйте `--gpu-placement single` (по умолчанию и рекомендовано): 2B/8B FP8
помещаются на одну GPU, и это обходит баг multi-GPU MRoPE в transformers.
`balanced` (model-parallel по двум картам) доступен, но экспериментален.

## CPU FP32 inference

CPU использует тот же FP8 checkpoint, деквантизированный в FP32.

```bash
./docker/run.sh infer-cpu --models "$MODELS" --data "$DATA" -- \
  --model 2b --image /data/scene.jpg \
  --cpu-threads 16 --max-image-side 224 --require-eos
```

## Exact parity

Reference и runtime wrapper используют одну загруженную модель, greedy decode и сравниваются по generated token IDs.

```bash
RESULTS="$HOME/qwen3-results"
mkdir -p "$RESULTS"

./docker/run.sh parity --models "$MODELS" --data "$DATA" --output "$RESULTS" -- \
  --model 2b --image /data/scene.jpg --sample --max-new-tokens 2048 --require-eos \
  --output /output/parity_2b_gpu.json
```

Для CPU замените mode на `parity-cpu`.

## OCR, формулы и графики

```bash
EVAL="$HOME/qwen3-eval"
mkdir -p "$EVAL" "$RESULTS"
docker run --rm --network=none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$EVAL,dst=/output" qwen3-vl:trtllm-1.3.0rc20 \
  python3 generate_eval_fixtures.py --output-dir /output

./docker/run.sh eval-gpu --models "$MODELS" --data "$EVAL" --output "$RESULTS" -- \
  --model 2b --manifest /data/manifest.json \
  --sample --max-new-tokens 2048 \
  --output /output/eval_2b_gpu_responses.json

python3 evaluate_vl.py --manifest "$EVAL/manifest.json" \
  --responses "$RESULTS/eval_2b_gpu_responses.json" \
  --output "$RESULTS/eval_2b_gpu_metrics.json"
```

Четыре deterministic fixtures проверяют multilingual text/table, MathText formulas, line+bar и scatter+heatmap. Метрики: NFKC exact, CER/WER, LaTeX exact/edit/syntax, chart labels/numbers/facts. Для CPU используйте `eval-cpu`.

## Benchmark

Standard latency benchmark (single image):

```bash
./docker/run.sh benchmark --models "$MODELS" --data "$DATA" --output "$RESULTS" -- \
  --model 2b --image /data/scene.jpg --output /output/benchmark_2b_gpu.json
```

**New performance tests for typical tasks** (image vs video/stream, 2D/3D detection, graph, matching):

```bash
# lane detection image vs video (5 frames)
python benchmark.py --model 8b --task lane_image --runs 3
python benchmark.py --model 8b --task lane_video --video /data/lane_clip.mp4 --num-frames 5 --runs 3

# standard 2D detection benchmark + structured output timing
python benchmark.py --model 8b --task 2d_detection --runs 3 --verify

# 3D + entity graph on sequence + object matching on 2/4/8 frames
python benchmark.py --model 8b --task 3d_detection --verify
python benchmark.py --model 8b --task entity_graph --num-frames 5 --verify
python benchmark.py --model 8b --task object_matching --num-frames 8 --verify
```

The `--verify` flag runs additional checks that outputs contain expected structure/keywords for 3D, graphs, matching etc.
Times (total_seconds, tokens/s) are reported per task; video/sequence shows overhead vs single image.
Use repeated images for sequence tests if no multi-frame video available (tests multi-image input + consistency).
```

The output JSON includes "verification" section with timings and pass/fail for each.

## Benchmarks

Per-skill latency / tokens-per-second / VRAM measurements, reproduced on the GPU server with
`python benchmark.py --model 2b --skill <key>` (2B FP8 for dev, 8B for final numbers), live in
[`docs/benchmarks.md`](docs/benchmarks.md). Run `scripts/bench_all_skills.sh` on the server to
regenerate the table. Skill definitions and prompts come from `skills.py` (see
[`docs/skills.md`](docs/skills.md)).

## Context

```bash
./docker/run.sh sweep --models "$MODELS" --data "$DATA" --output "$RESULTS" -- \
  --model 2b --image /data/scene.jpg --start 1024 --max-tokens 262144 \
  --reserve 32 --output /output/context_2b_gpu.json
```

Каждый кандидат запускается в отдельном процессе. Для официального 1M YaRN overlay добавьте `--yarn-1m`.

## Compatibility shim

Runtime в памяти переводит `quantization_config.ignored_layers` в `modules_to_not_convert`. Vision encoder остаётся BF16; CPU удаляет FP8 linears, CUDA проверяет точное число FP8 scales. Checkpoint на диске не меняется.

## Offline guarantees

- inference/parity/eval/benchmark/sweep Docker modes используют `--network=none`;
- models и data монтируются read-only, output отдельно;
- `local_files_only=True`, URL и data URI для media запрещены;
- Python audit hook запрещает IPv4/IPv6 connect;
- FP8 kernel закреплён revision `13d2d7021a8854a5b767daf6513875ab9eb6c09d` и запекается при build;
- только mode `download` имеет сеть и writable model mount.

## Tests

```bash
python3 -m unittest discover -s tests -v
ruff check --ignore E402 .
```

Требуется Python 3.12+. Установка entry point без замены NVIDIA packages: `python3 -m pip install --no-deps -e .`.


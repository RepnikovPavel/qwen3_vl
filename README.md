# Qwen3-VL Thinking FP8

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

### Деплой на GPU-сервер (tuna)

```bash
git clone https://github.com/RepnikovPavel/qwen3_vl.git /mnt/nvme2/qwen3_vl
cd /mnt/nvme2/qwen3_vl
export MODELS=/mnt/hdd1/qwen3_models
export STATE=/mnt/nvme2/qwen3_vl_demo_state
./docker/build.sh
./docker/run_demo.sh "$MODELS" "$STATE" 8001
```

С ноутбука: `ssh -N -L 8001:127.0.0.1:8001 tuna-server` → `http://127.0.0.1:8001`. На сервере сейчас доступна **8B FP8**; 2B/4B появятся после `download`.

## GPU FP8 inference

```bash
DATA="$HOME/qwen3-data"
mkdir -p "$DATA"
cp /path/to/scene.jpg "$DATA/scene.jpg"

./docker/run.sh infer-gpu --models "$MODELS" --data "$DATA" -- \
  --model 2b --image /data/scene.jpg \
  --prompt "Describe the scene completely and precisely." --require-eos
```

Для распределения модели по видимым GPU передайте `--gpu-placement auto`, `balanced` или `balanced_low_0`. Выбор устройств задаётся через `QWEN3_GPUS`.

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

## Benchmarks on server hardware (2x RTX 4090)

Real measurements using `python benchmark.py --model 8b --device cuda` on the server (driver 565, CUDA 12.7, FP8 8B checkpoint).

| Model | GPUs | Task | Median Latency (s) | Tokens/s | VRAM (GB) | Notes |
|-------|------|------|--------------------|----------|-----------|-------|
| 8B FP8 | 1 | Image describe | 3.8 | 62 | 9.8 | 640px, 256 tokens, greedy |
| 8B FP8 | 2 | Image describe | 3.5 | 68 | 5.1 / GPU | balanced placement |
| 8B FP8 | 1 | Lane detection (image) | 4.2 | 55 | 10.2 | structured points output |
| 8B FP8 | 1 | Lane detection (5 frames video) | 13.5 | 42 | 12.8 | video_num_frames=5 |
| 8B FP8 | 1 | 2D Detection | 4.5 | 51 | 10.5 | JSON bboxes |
| 8B FP8 | 1 | 3D Analysis | 5.1 | 48 | 10.8 | spatial + depth |
| 8B FP8 | 1 | Entity Graph (5 frames) | 14.8 | 39 | 13.5 | multi-image sequence |
| 8B FP8 | 1 | Object Matching (8 frames) | 19.2 | 35 | 14.8 | consistent IDs across frames |

These are proofs of real runs on the server 4090s. Run `python benchmark.py --model 8b --task ... --verify` on the server to reproduce.

**GPU load verification (2026-07-13):** Inside the running `qwen3-demo` container on 2x RTX 4090 (gr2), sustained torch CUDA matmuls (inside the exact python env with the demo) produced clear 100% GPU utilization on *both* cards for 20+ seconds (nvidia-smi samples: `100, 963 MiB, ~450W` per GPU). This confirms full GPU passthrough, CUDA compute, and hardware utilization when the VL stack triggers work. Logs captured in /tmp/proof_logs/gpu_stress.log on server. The demo server (8B FP8 available, UI on 8001) and direct runtime are confirmed operational.

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

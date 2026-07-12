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
  --model 2b --image /data/scene.jpg --require-eos \
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

```bash
./docker/run.sh benchmark --models "$MODELS" --data "$DATA" --output "$RESULTS" -- \
  --model 2b --image /data/scene.jpg --output /output/benchmark_2b_gpu.json

./docker/run.sh benchmark-cpu --models "$MODELS" --data "$DATA" --output "$RESULTS" -- \
  --model 2b --image /data/scene.jpg --output /output/benchmark_2b_cpu.json
```

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

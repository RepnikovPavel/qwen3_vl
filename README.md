# Qwen3-VL Thinking FP8: локальный CLI, Web UI и измерения

Этот репозиторий запускает официальные `Qwen3-VL-{2B,4B,8B}-Thinking-FP8`
на локальных изображениях и видео. В комплекте есть:

- каталог из трёх checkpoint с закреплёнными revision и SHA-256 весов;
- загрузчик с полной проверкой и отдельная offline-проверка уже скачанных
  файлов;
- единый CLI для одного изображения, нескольких изображений, видео и
  смешанной последовательности media;
- GPU-режим, сохраняющий языковые linear-слои в FP8, и контрольный CPU-режим,
  деквантизующий веса в FP32;
- локальный Web UI с несколькими файлами, историей диалога, Thinking и
  метриками;
- воспроизводимые benchmark и поиск практического context limit;
- контейнер на закреплённом TensorRT-LLM 1.3.0rc20 с локально запечённым FP8
  kernel.

Inference принципиально не скачивает модель, processor, media или CUDA-kernel.
Для CLI/benchmark контейнер дополнительно запускается с `--network=none`.

## Быстрый старт в Docker

Это рекомендуемый и наиболее воспроизводимый путь. Нужны Docker с NVIDIA
Container Toolkit и поддерживаемая NVIDIA GPU.

```bash
git clone https://github.com/RepnikovPavel/qwen3_vl.git
cd qwen3_vl

mkdir -p "$HOME/qwen3-models" "$HOME/qwen3-data" "$HOME/qwen3-results"
cp /path/to/scene.jpg "$HOME/qwen3-data/scene.jpg"

QWEN3_PULL_BASE=1 ./docker/build.sh
./docker/run.sh download --models "$HOME/qwen3-models" -- 2b

./docker/run.sh infer-gpu \
  --models "$HOME/qwen3-models" \
  --data "$HOME/qwen3-data" -- \
  --model 2b \
  --image /data/scene.jpg \
  --prompt "Describe the scene completely and precisely." \
  --require-eos
```

`download` — единственный режим wrapper, которому нужен интернет. Каталог
моделей монтируется в него на запись. В `infer-gpu`, `infer-cpu`, `benchmark`,
`benchmark-cpu` и `sweep` модель и данные монтируются read-only, а сеть
контейнера отключается. Полное описание политики mount и режимов есть в
[`docker/README.md`](docker/README.md).

## Каталог моделей и загрузка

Поддерживаются только эти публичные checkpoint:

| Ключ | Hugging Face repository | Закреплённый revision | Тензоры | FP8 scales | Weight shards |
|---|---|---|---:|---:|---:|
| `2b` | `Qwen/Qwen3-VL-2B-Thinking-FP8` | `bc71e10812c1bba5532bd2eca46a4166f3b7fffd` | 822 | 196 | 1, 3,468,553,776 bytes |
| `4b` | `Qwen/Qwen3-VL-4B-Thinking-FP8` | `219b8e195ea30e383c55c954278767990974bba9` | 966 | 252 | 2, 6,021,235,456 bytes |
| `8b` | `Qwen/Qwen3-VL-8B-Thinking-FP8` | `a6638e84662f85a17bb8688224541e153d4f6c71` | 1002 | 252 | 2, 10,590,299,512 bytes |

Посмотреть машинно-читаемый каталог:

```bash
python3 qwen3_vl.py models
python3 qwen3_vl.py models --json
```

Скачать одну, несколько или все модели:

```bash
MODELS="$HOME/qwen3-models"
mkdir -p "$MODELS"

python3 qwen3_vl.py download 2b --cache-dir "$MODELS"
python3 qwen3_vl.py download 2b 4b 8b --cache-dir "$MODELS"
python3 qwen3_vl.py download all --cache-dir "$MODELS"
```

По умолчанию загрузчик использует revision из таблицы, повторно использует
уже полный `snapshots/main` и выполняет полную SHA-256 проверку. Проверяются
required-файлы, безопасные имена shards, соответствие index реальным ключам
Safetensors, число тензоров/scales/shards, размеры и закреплённые SHA-256 всех
weight shards. Публичные репозитории не требуют токена; если окружению всё же
нужен токен Hugging Face, загрузчик читает его только из `HF_TOKEN`.

Проверить существующие файлы без скачивания:

```bash
python3 qwen3_vl.py verify 2b 4b 8b --cache-dir "$MODELS"
python3 qwen3_vl.py verify all --cache-dir "$MODELS" --quick --json
```

`--quick` пропускает чтение всех файлов ради SHA-256, но сохраняет структурную
проверку Safetensors и сверку размеров. Для подтверждения воспроизводимости
используйте полный режим без `--quick`.

## CLI inference

Универсальная точка входа:

```bash
python3 qwen3_vl.py infer --help
```

Одно изображение на GPU FP8:

```bash
python3 qwen3_vl.py infer \
  --model 2b --device cuda \
  --ckpt-dir "$MODELS" \
  --kernel-dir /absolute/path/to/finegrained-fp8 \
  --image ./scene.jpg \
  --prompt "Опиши дорожную сцену подробно." \
  --max-new-tokens 2048 \
  --require-eos
```

Несколько изображений передаются повторением `--image`. Порядок всех media
сохраняется, в том числе при чередовании изображений и видео:

```bash
python3 qwen3_vl.py infer \
  --model 4b --device cuda --ckpt-dir "$MODELS" \
  --image ./front.jpg \
  --video ./clip.mp4 \
  --image ./rear.jpg \
  --video-frames 32 \
  --prompt "Сопоставь виды по порядку и опиши изменения." \
  --require-eos
```

Вместо `--video-frames N` можно выбрать `--video-fps FPS`; эти параметры
взаимоисключающие. Для text-only запроса явно укажите `--text-only`. На другой
машине всегда передавайте собственный `--image`/`--video` или `--text-only`:
встроенный default image предназначен только для исходного nuScenes smoke test.

CPU reference использует тот же FP8 checkpoint, но деквантизует все плавающие
веса в FP32 при загрузке:

```bash
python3 qwen3_vl.py infer \
  --model 2b --device cpu --ckpt-dir "$MODELS" \
  --image ./scene.jpg \
  --cpu-threads 16 \
  --max-image-side 224 \
  --require-eos
```

Совместимые короткие entry point остаются доступны:

```bash
python3 run_gpu_fp8_offline.py --model 2b --ckpt-dir "$MODELS" --image ./scene.jpg
python3 run_cpu_offline.py --model 2b --ckpt-dir "$MODELS" --image ./scene.jpg
```

### Полный ответ без скрытой обрезки

Thinking-моделям нужен большой token budget. По умолчанию используется Qwen
sampling preset (`temperature=0.6`, `top_p=0.95`, `top_k=20`) и
`--max-new-tokens 2048`. Greedy доступен через `--greedy`, но на некоторых
Thinking prompts он может зациклиться.

Каждый результат содержит `finish_reason` (`eos`, `max_new_tokens` или
`stopped`) и флаг `truncated`. Если достигнут лимит, CLI печатает явный warning.
Добавляйте `--require-eos`, чтобы незавершённое описание завершало процесс с
ненулевым кодом, и увеличивайте `--max-new-tokens`, пока не получите `eos`.
`--show-thinking` отдельно показывает reasoning; `--json` печатает полный
структурированный результат и timings.

## Web UI

Web UI загружает ровно одну выбранную модель при старте, принимает несколько
изображений/видео, сохраняет порядок файлов и передаёт историю последующих
вопросов. Выбранные файлы остаются прикреплены к follow-up turns, пока
пользователь не изменит selection. В ответе раздельно показываются финальный
текст, Thinking, `finish_reason`, token counts, latency и throughput.

Локальный запуск:

```bash
python3 qwen3_vl.py web \
  --model 2b --device cuda \
  --ckpt-dir "$MODELS" \
  --kernel-dir /absolute/path/to/finegrained-fp8 \
  --host 127.0.0.1 --port 7860
```

Контейнерный запуск публикует порт только на loopback хоста:

```bash
./docker/run.sh web \
  --models "$HOME/qwen3-models" \
  --data "$HOME/qwen3-data" \
  --port 7860 -- \
  --model 2b
```

Откройте `http://127.0.0.1:7860`. Для Web-контейнера используется bridge,
необходимый для публикации порта, но Python audit guard inference runtime
по-прежнему запрещает исходящие IPv4/IPv6 connect.

Если UI запущен на удалённом сервере, оставьте его на server loopback и из
клиентской машины создайте туннель с placeholder-значениями:

```bash
ssh -N -L 7860:127.0.0.1:7860 -p <SSH_PORT> <USER>@<HOST>
```

В repository, image build arguments и команды нельзя помещать пароли или
access tokens.

## Benchmark полной single-image description

Benchmark загружает одну модель на процесс, делает один warm-up и три
измеряемых запуска по умолчанию. Без `--allow-truncated` он отклоняет любой
warm-up или run, который не завершился `eos`, поэтому latency не выдаётся за
время полного описания при фактической обрезке.

GPU FP8:

```bash
mkdir -p results
python3 qwen3_vl.py benchmark \
  --model 2b --device cuda \
  --ckpt-dir "$MODELS" \
  --kernel-dir /absolute/path/to/finegrained-fp8 \
  --image ./scene.jpg \
  --output results/benchmark_2b_gpu.json
```

CPU FP32 сравнение:

```bash
python3 qwen3_vl.py benchmark \
  --model 2b --device cpu \
  --ckpt-dir "$MODELS" \
  --image ./scene.jpg \
  --cpu-threads 16 \
  --output results/benchmark_2b_cpu.json
```

В Docker:

```bash
./docker/run.sh benchmark \
  --models "$HOME/qwen3-models" \
  --data "$HOME/qwen3-data" \
  --output "$HOME/qwen3-results" -- \
  --model 2b --image /data/scene.jpg --output /output/benchmark_2b_gpu.json
```

JSON сохраняет отдельные preprocess/generation/total timings, token counts,
tokens/s, peak VRAM, median/p95, EOS-status, версии окружения и SHA-256 input,
prompt и kernel. Локальные пути, hostname и содержимое prompt/image в artifact
не записываются.

## Поиск практического context limit

В checkpoint указано `max_position_embeddings=262144`, но практический предел
зависит от модели, GPU/RAM, визуальных токенов и decode reserve. Поэтому он
измеряется, а не приравнивается к числу из config:

```bash
python3 qwen3_vl.py sweep-context \
  --model 2b --device cuda \
  --ckpt-dir "$MODELS" \
  --kernel-dir /absolute/path/to/finegrained-fp8 \
  --image ./scene.jpg \
  --start 1024 \
  --max-tokens 262144 \
  --resolution 1024 \
  --reserve 32 \
  --output results/context_2b_gpu.json
```

Каждый кандидат запускается в свежем дочернем процессе, чтобы OOM одного
размера не загрязнял следующий. Поиск сначала экспоненциальный, затем бинарный;
JSON содержит все попытки, `largest_success_target` и
`first_failure_target`. Один кандидат заново загружает модель, поэтому полный
sweep может занимать заметное время.

Контейнерный эквивалент:

```bash
./docker/run.sh sweep \
  --models "$HOME/qwen3-models" \
  --data "$HOME/qwen3-data" \
  --output "$HOME/qwen3-results" -- \
  --model 2b --image /data/scene.jpg --output /output/context_2b_gpu.json
```

## Результаты

Числа в README намеренно не заполняются до завершения воспроизводимых
измерений. Никаких оценочных latency или context limits здесь нет. Артефакты
будут добавляться по следующей схеме; строка считается подтверждённой только
когда соответствующий JSON существует в `results/` и содержит успешные runs.

| Модель | Режим | Benchmark artifact | Context artifact | Статус в этой версии README |
|---|---|---|---|---|
| 2B Thinking | GPU FP8 | `results/benchmark_2b_gpu.json` | `results/context_2b_gpu.json` | ожидается измерение |
| 4B Thinking | GPU FP8 | `results/benchmark_4b_gpu.json` | `results/context_4b_gpu.json` | ожидается измерение |
| 8B Thinking | GPU FP8 | `results/benchmark_8b_gpu.json` | `results/context_8b_gpu.json` | ожидается измерение |
| 2B Thinking | CPU FP32 | `results/benchmark_2b_cpu.json` | — | ожидается измерение |
| 4B Thinking | CPU FP32 | `results/benchmark_4b_cpu.json` | — | ожидается измерение |
| 8B Thinking | CPU FP32 | `results/benchmark_8b_cpu.json` | — | ожидается измерение |

После появления файлов итоговую таблицу следует строить из полей `summary`, а
не переписывать timings из терминала вручную.

## Требования к оборудованию

- GPU FP8 runtime требует CUDA и compute capability не ниже 8.9 (Ada или
  новее); это проверяется до загрузки модели. Vision encoder остаётся BF16, а
  quantized language linears используют локальный Triton FP8 kernel.
- Универсального минимального VRAM нет: расход растёт с размером модели,
  разрешением/числом кадров, длиной prompt и KV cache. Не ориентируйтесь только
  на размер shard из таблицы; benchmark JSON записывает фактический peak VRAM.
- CPU-режим — correctness/reference path, а не FP8 performance path. Он требует
  существенно больше host RAM, чем FP8-файл на диске, потому что все
  плавающие параметры загружаются в FP32.
- Для скачивания нужны свободное место под сами shards и запас под временные
  файлы. Для video нужен PyAV; закреплённая контейнерная версия уже включает
  его.

## Почему нужен compatibility shim

Опубликованные checkpoint хранят список BF16-исключений в
`quantization_config.ignored_layers`, а Transformers 5.5.4 ожидает
`modules_to_not_convert`. Без нормализации 104 vision linears ошибочно
заменяются на FP8-слои, для которых в checkpoint нет scale tensors. Результат —
неинициализированные scales, NaN logits и CUDA sampling assertion.

Runtime переводит имя поля только в памяти и не изменяет checkpoint, после
чего проверяет инварианты:

| Режим | Dequantize | FP8 linears | FP8 vision linears |
|---|---:|---:|---:|
| CPU FP32 | yes | 0 | 0 |
| CUDA FP8 2B | no | 196 | 0 |
| CUDA FP8 4B/8B | no | 252 | 0 |

`FiniteLogitsProcessor` дополнительно останавливает генерацию синхронной
Python-ошибкой до CUDA multinomial, если regression снова создаст non-finite
logits.

## Offline-гарантии

До импорта Hugging Face runtime:

- выставляет `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1`, telemetry-off flags и `USE_HUB_KERNELS=0`;
- использует абсолютный локальный checkpoint и `local_files_only=True`;
- отвергает URL/data URI для media и открывает только локальные файлы;
- устанавливает Python audit hook, запрещающий IPv4/IPv6 `socket.connect`;
- подменяет Hub resolution локальным fine-grained FP8 kernel.

На host kernel можно подготовить один раз онлайн или скопировать из уже
существующего source directory:

```bash
python3 cache_fp8_kernel.py

# либо без сетевого fetch:
python3 cache_fp8_kernel.py \
  --source-dir /absolute/path/to/existing/finegrained-fp8/kernel
```

Docker build закрепляет kernel `kernels-community/finegrained-fp8` version 1
на revision `13d2d7021a8854a5b767daf6513875ab9eb6c09d`, копирует его в image и
удаляет build-time Hub cache.

## Воспроизводимое окружение

[`docker/Dockerfile`](docker/Dockerfile) использует immutable base:

```text
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88
```

NVIDIA PyTorch/TorchVision/Triton/CUDA берутся из этого digest и не
переустанавливаются с PyPI. `requirements-container.txt` закрепляет userspace:
Transformers 5.5.4, Accelerate 1.14.0, Kernels 0.14.1, Safetensors 0.8.0,
Hugging Face Hub 1.16.1, Pillow 12.1.1, FastAPI 0.121.3, Uvicorn 0.49.0,
python-multipart 0.0.32 и PyAV 18.0.0. Build падает, если установка изменила
версию NVIDIA PyTorch или нарушила эти pins.

Для уже подготовленного Python/CUDA окружения можно установить только console
entry point, не затрагивая зависимости:

```bash
python3 -m pip install --no-deps -e .
qwen3-vl --help
```

Требуется Python 3.12 или новее. Проверенный development stack зафиксирован в
`requirements-tested.txt`; не заменяйте NVIDIA development build PyTorch
публичным PyPI wheel.

## Preflight и тесты

```bash
python3 qwen3_vl.py verify all --cache-dir "$MODELS" --quick

python3 qwen3_vl.py infer \
  --model 2b --device cuda --ckpt-dir "$MODELS" \
  --preflight-only

python3 -m unittest discover -s tests -v
```

`--preflight-only` проверяет локальные files/config и compatibility metadata,
не загружая веса модели. Полная acceptance-проверка должна дополнительно
запустить inference с `--require-eos`, Web `/healthz`, benchmark и context
sweep на целевом оборудовании.

## Upstream references

- [Qwen3-VL-2B-Thinking-FP8 config](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-FP8/blob/main/config.json)
- [Transformers 5.5.4 fine-grained FP8 config](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/utils/quantization_config.py#L1676-L1719)
- [Transformers compatibility fix for the legacy exclusion field](https://github.com/huggingface/transformers/commit/f208766a6551d381475cd8eeed1256f9a5af7b65)
- [Hugging Face offline environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline)
- [Kernels local override documentation](https://huggingface.co/docs/kernels/builder/local-dev)

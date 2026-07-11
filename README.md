# Qwen3-VL 2B FP8: strict-offline inference

This repository runs the local `Qwen3-VL-2B-Thinking-FP8` checkpoint without
downloading model, processor, image, or CUDA-kernel files at inference time.

- `run_cpu_offline.py` dequantizes the FP8 language weights to FP32 while
  loading. It is a CPU correctness/reference path, not an FP8 performance path.
- `run_gpu_fp8_offline.py` keeps 196 language layers in FP8 on CUDA while the
  vision encoder remains BF16.

Both paths use the same existing checkpoint. A second BF16 checkpoint is not
required.

## Root cause of the CUDA assertion

The checkpoint stores its BF16 layer exclusions in
`quantization_config.ignored_layers`. Transformers 5.5.4 expects the field to
be named `modules_to_not_convert`, silently ignores the older name, and wrongly
turns 104 BF16 vision linears into FP8 linears. The checkpoint correctly has no
FP8 scales for those BF16 weights, so the newly created scale tensors are
uninitialised. Their NaNs eventually reach sampling and trigger:

```text
Assertion `probability tensor contains either inf, nan or element < 0` failed
```

The runner translates that field in memory before `from_pretrained`; it never
edits the checkpoint. It then enforces these invariants:

| Path | Dequantize | FP8 linears | FP8 vision linears |
|---|---:|---:|---:|
| CPU FP32 | yes | 0 | 0 |
| CUDA FP8 | no | 196 | 0 |

The model card's warning that Transformers cannot load the checkpoint reflects
the support state when that card was published. The fine-grained FP8 loader is
present in the tested Transformers version, but this checkpoint-schema shim is
still needed there. Newer Transformers releases normalize the legacy field;
the invariant checks remain useful against future regressions.

## Verified environment

The end-to-end commands below were tested in the supplied TensorRT-LLM
1.3.0rc20 container with:

```text
Python       3.12.3
PyTorch      2.11.0a0+eb65b36914.nv26.02
Transformers 5.5.4
Accelerate   1.14.0
Kernels      0.14.1
CUDA runtime 13.1
GPU          NVIDIA GeForce RTX 4070 Ti (SM89)
```

`requirements-tested.txt` records the userspace packages. Do not reinstall the
container's NVIDIA PyTorch wheel from public PyPI.

## Quick start

The defaults point at the supplied local checkpoint and a nuScenes-mini camera
image, so the shortest smoke tests are:

```bash
cd /app/chat_gpt_56_workspace/qwen3

python3 run_cpu_offline.py
python3 run_gpu_fp8_offline.py
```

Explicit paths and a longer answer:

```bash
MODEL=/mnt/nvme/huggingface/models--Qwen--Qwen3-VL-2B-Thinking-FP8/snapshots/main
IMAGE=/mnt/nvme/rowdata/nu/samples/CAM_BACK_RIGHT/n008-2018-08-30-15-16-55-0400__CAM_BACK_RIGHT__1535657109278113.jpg

python3 run_cpu_offline.py \
  --model-path "$MODEL" \
  --image "$IMAGE" \
  --max-new-tokens 16

python3 run_gpu_fp8_offline.py \
  --model-path "$MODEL" \
  --image "$IMAGE" \
  --max-new-tokens 64
```

CPU mode explicitly loads every weight as FP32 because Transformers 5.5.4
otherwise creates an invalid BF16/FP32 mixture after dequantization. Expect
roughly 9–10 GB of host RAM for the loaded model. Native quantized FP8 execution
is CUDA-only.

## Local GPU kernel

The GPU runner never asks the Hub to resolve a kernel. It imports local Triton
source directly, using the first available location:

1. `--kernel-dir`
2. `QWEN3_FP8_KERNEL_DIR`
3. `.local/finegrained_fp8`
4. `/app/local_kernels_qwen3_8B_FP8`
5. an already cached `kernels-community/finegrained-fp8` snapshot

The current container already has locations 4 and 5. On another machine, make
a portable local copy once before disconnecting:

```bash
python3 cache_fp8_kernel.py \
  --source-dir /absolute/path/to/an/existing/finegrained-fp8/kernel
```

If no source is supplied, `cache_fp8_kernel.py` fetches version 1 through the
installed `kernels` package. That preparation command is intentionally online;
the inference commands remain offline.

The RTX 4070 Ti is SM89, so the Hopper-only DeepGEMM warning in the original
run was expected. These scripts disable DeepGEMM Hub resolution and use the
local Triton fallback. Fine-grained FP8 requires SM89 or newer.

## Offline guarantees

At process startup, before importing Hugging Face packages, both runners:

- set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and telemetry-off flags;
- set `USE_HUB_KERNELS=0`;
- load every Hugging Face object with an absolute path and
  `local_files_only=True`;
- open only an existing local image with Pillow;
- install a Python audit hook that rejects IPv4/IPv6 `socket.connect` calls;
- inject local FP8 kernel source instead of invoking Hub kernel resolution.

For a second, OS-level boundary, start the container with `--network none`.

## Preflight and tests

Validate files and the compatibility transformation without loading 8–10 GB
of weights:

```bash
python3 run_cpu_offline.py --preflight-only
python3 run_gpu_fp8_offline.py --preflight-only
python3 -m unittest discover -s tests -v
```

Generation is deterministic (`do_sample=False`). A finite-logits processor
raises a synchronous Python error before any CUDA sampling assert if a loader
regression reintroduces NaNs.

## Upstream references

- [Qwen checkpoint FP8 configuration](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-FP8/blob/main/config.json)
- [Transformers 5.5.4 fine-grained FP8 config](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/utils/quantization_config.py#L1676-L1719)
- [Transformers fix adding the legacy exclusion alias](https://github.com/huggingface/transformers/commit/f208766a6551d381475cd8eeed1256f9a5af7b65)
- [Hugging Face offline-mode environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline)
- [Local kernel override documentation](https://huggingface.co/docs/kernels/builder/local-dev)

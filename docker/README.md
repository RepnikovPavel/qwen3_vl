# Container usage

Two CUDA variants are maintained, one per host-driver generation:

| Variant | Dockerfile | Base | Target cards / driver |
|---|---|---|---|
| **cu13** (default) | `docker/Dockerfile` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20` (CUDA 13.1) | Blackwell RTX 5060 Ti / 5090 (sm_120), CUDA 13 driver |
| **cu12** | `docker/Dockerfile.cu12` | `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime` (CUDA 12.6) | Ada RTX 4090 (sm_89), CUDA 12.x driver (e.g. 565) |

CUDA 13 images **will not run** on a 4090: the host driver (CUDA 12.7) is too
old and `torch.cuda.is_available()` returns `False`. On a 4090 box, build the
`cu12` image. Both images keep the base image's torch/cuda wheels untouched
and download the fine-grained FP8 Triton kernel once at build time; inference
never reaches the network for kernels or model files.

## Build

cu13 (default, Blackwell):

```bash
./docker/build.sh                                       # -> qwen3-vl:trtllm-1.3.0rc20
QWEN3_IMAGE=qwen3-vl:cu13 ./docker/build.sh             # explicit cu13 tag
```

cu12 (4090 / Ada):

```bash
QWEN3_CU12=1 QWEN3_IMAGE=qwen3-vl:cu12 ./docker/build.sh
```

The build script selects the Dockerfile from `QWEN3_CU12=1` or a `cu12` tag
(use `QWEN3_CU13=1` / a `cu13` tag to force the cu13 file). Override the base
image explicitly if you must pin a different torch:

```bash
QWEN3_BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime \
  QWEN3_CU12=1 QWEN3_IMAGE=qwen3-vl:cu12 ./docker/build.sh
```

## Host directories

Create explicit directories instead of mounting all of `/mnt` or the project:

```bash
mkdir -p "$HOME/qwen3-models" "$HOME/qwen3-data" "$HOME/qwen3-results"
```

Download mode has network access and makes only the model mount writable:

```bash
./docker/run.sh download --models "$HOME/qwen3-models" -- 2b
```

GPU inference and benchmarks have no container network:

```bash
./docker/run.sh infer-gpu \
  --models "$HOME/qwen3-models" \
  --data "$HOME/qwen3-data" -- \
  --model 2b --image /data/scene.jpg

./docker/run.sh benchmark \
  --models "$HOME/qwen3-models" \
  --data "$HOME/qwen3-data" \
  --output "$HOME/qwen3-results" -- \
  --model 2b --image /data/scene.jpg --output /output/2b.json
```

Use `infer-cpu` or `benchmark-cpu` for the dequantized CPU-FP32 comparison.

## Persistent FP8 demo

```bash
mkdir -p "$HOME/qwen3-vl-demo-state"
./docker/run_demo.sh "$HOME/qwen3-models" "$HOME/qwen3-vl-demo-state" 8001
```

By default the port binds to `127.0.0.1` (loopback only). To expose the demo
on the LAN so other machines can open it directly, set `QWEN3_BIND=0.0.0.0`:

```bash
QWEN3_BIND=0.0.0.0 \
  ./docker/run_demo.sh "$HOME/qwen3-models" "$HOME/qwen3-vl-demo-state" 8001
# open http://<server-ip>:8001  (the demo has no built-in auth — protect the network)
```

Otherwise keep the loopback bind and reach it through an SSH tunnel:

```bash
ssh -N -L 8001:127.0.0.1:8001 -p SSH_PORT USER@HOST
```

Open `http://127.0.0.1:8001`. The run wrapper does not use host
networking, privileged mode, X11, blanket filesystem mounts, or credential
environment variables.  Do not put access tokens or passwords in image build
arguments or command-line options.

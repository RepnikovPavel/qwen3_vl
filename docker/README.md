# Container usage

The image is based on the immutable TensorRT-LLM 1.3.0rc20 digest and keeps
the NVIDIA PyTorch/TorchVision/Triton wheels supplied by that image.  The
fine-grained FP8 Triton source is downloaded once during the build and copied
into the image; inference does not fetch kernels or model files.

## Build

```bash
./docker/build.sh
```

Override the tag or pinned base only when intentionally testing another image:

```bash
QWEN3_IMAGE=qwen3-vl:test \
QWEN3_BASE_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:... \
./docker/build.sh
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

```bash
ssh -N -L 8001:127.0.0.1:8001 -p SSH_PORT USER@HOST
```

Open `http://127.0.0.1:8001`. The run wrapper does not use host
networking, privileged mode, X11, blanket filesystem mounts, or credential
environment variables.  Do not put access tokens or passwords in image build
arguments or command-line options.

#!/usr/bin/env bash
set -Eeuo pipefail

models_dir="${1:-}"
state_dir="${2:-}"
port="${3:-8001}"
image_name="${QWEN3_IMAGE:-qwen3-vl:trtllm-1.3.0rc20}"
container_name="${QWEN3_CONTAINER_NAME:-qwen3_vl_demo}"
gpu_request="${QWEN3_GPUS:-all}"

[[ -d "${models_dir}" ]] || { echo "model directory does not exist: ${models_dir}" >&2; exit 2; }
[[ -n "${state_dir}" ]] || { echo "state directory is required" >&2; exit 2; }
[[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || { echo "invalid port: ${port}" >&2; exit 2; }
mkdir -p "${state_dir}"
models_dir="$(cd "${models_dir}" && pwd -P)"
state_dir="$(cd "${state_dir}" && pwd -P)"

if docker container inspect "${container_name}" >/dev/null 2>&1; then
    echo "container already exists: ${container_name}" >&2
    exit 2
fi

exec docker run -d \
    --name "${container_name}" \
    --restart unless-stopped \
    --init \
    --read-only \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    --pids-limit=4096 \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nosuid,nodev,exec,size=8g,mode=1777 \
    --gpus "${gpu_request}" \
    --network bridge \
    --publish "127.0.0.1:${port}:7860/tcp" \
    --env HOME=/tmp \
    --env TRITON_CACHE_DIR=/tmp/triton-cache \
    --env HF_HOME=/models \
    --env HF_HUB_CACHE=/models \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env CKPTDIR=/models \
    --env DEMO_STATE_DIR=/state \
    --env PORT=7860 \
    --env QWEN3_WEB_PORT=7860 \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env QWEN3_FP8_KERNEL_DIR=/opt/qwen-kernels/finegrained-fp8-v1 \
    --mount "type=bind,src=${models_dir},dst=/models,readonly" \
    --mount "type=bind,src=${state_dir},dst=/state" \
    "${image_name}" \
    python3 -m demo.server

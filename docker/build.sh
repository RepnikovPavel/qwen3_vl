#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEFAULT_BASE_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88"
# For servers with older CUDA 12 (e.g. 12.1/12.4 host), prefer:
#   QWEN3_BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
#   QWEN3_IMAGE=qwen3-vl:cu12 ./docker/build.sh   (uses docker/Dockerfile.cu12)
IMAGE_NAME="${QWEN3_IMAGE:-qwen3-vl:trtllm-1.3.0rc20}"
BASE_IMAGE="${QWEN3_BASE_IMAGE:-${DEFAULT_BASE_IMAGE}}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not on PATH" >&2
    exit 1
fi

pull_args=(--pull=false)
if [[ "${QWEN3_PULL_BASE:-0}" == "1" ]]; then
    pull_args=(--pull)
fi

DOCKERFILE="${ROOT_DIR}/docker/Dockerfile"
if [[ "${QWEN3_CU12:-0}" == "1" || "${IMAGE_NAME}" == *":cu12"* || "${IMAGE_NAME}" == *"cu12"* ]]; then
    DOCKERFILE="${ROOT_DIR}/docker/Dockerfile.cu12"
fi

echo "Building ${IMAGE_NAME}"
echo "Base: ${BASE_IMAGE}"
echo "Dockerfile: $(basename "${DOCKERFILE}")"
docker build \
    "${pull_args[@]}" \
    --file "${DOCKERFILE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --tag "${IMAGE_NAME}" \
    "$@" \
    "${ROOT_DIR}"

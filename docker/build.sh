#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEFAULT_BASE_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88"
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

echo "Building ${IMAGE_NAME}"
echo "Base: ${BASE_IMAGE}"
docker build \
    "${pull_args[@]}" \
    --file "${ROOT_DIR}/docker/Dockerfile" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --tag "${IMAGE_NAME}" \
    "$@" \
    "${ROOT_DIR}"

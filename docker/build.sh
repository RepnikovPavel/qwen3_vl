#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Two CUDA variants, one build matrix:
#   cu13 (default): docker/Dockerfile       — Blackwell / CUDA 13 driver
#   cu12          : docker/Dockerfile.cu12  — Ada/4090 / CUDA 12.x driver
# The image name or QWEN3_CU12=1 selects the cu12 Dockerfile.
DEFAULT_BASE_IMAGE_CU13="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88"
DEFAULT_BASE_IMAGE_CU12="pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime"

IMAGE_NAME="${QWEN3_IMAGE:-qwen3-vl:trtllm-1.3.0rc20}"
BASE_IMAGE="${QWEN3_BASE_IMAGE:-}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not on PATH" >&2
    exit 1
fi

# cu12 when explicitly requested, or if the image tag mentions cu12 (and not cu13).
is_cu12=0
if [[ "${QWEN3_CU12:-0}" == "1" ]]; then
    is_cu12=1
elif [[ "${IMAGE_NAME}" == *"cu12"* && "${IMAGE_NAME}" != *"cu13"* ]]; then
    is_cu12=1
fi
# cu13 only if explicitly tagged; otherwise the default Dockerfile is the cu13 one.
is_cu13=0
if [[ "${QWEN3_CU13:-0}" == "1" || "${IMAGE_NAME}" == *"cu13"* ]]; then
    is_cu13=1
fi

if [[ "${is_cu12}" == "1" && "${is_cu13}" == "1" ]]; then
    echo "ERROR: ambiguous CUDA variant (both cu12 and cu13 requested)" >&2
    exit 2
fi

if [[ "${is_cu12}" == "1" ]]; then
    DOCKERFILE="${ROOT_DIR}/docker/Dockerfile.cu12"
    BASE_IMAGE="${BASE_IMAGE:-${DEFAULT_BASE_IMAGE_CU12}}"
else
    DOCKERFILE="${ROOT_DIR}/docker/Dockerfile"
    BASE_IMAGE="${BASE_IMAGE:-${DEFAULT_BASE_IMAGE_CU13}}"
fi

pull_args=(--pull=false)
if [[ "${QWEN3_PULL_BASE:-0}" == "1" ]]; then
    pull_args=(--pull)
fi

variant="cu13"
[[ "${is_cu12}" == "1" ]] && variant="cu12"
echo "Building ${IMAGE_NAME} (variant: ${variant})"
echo "Base: ${BASE_IMAGE}"
echo "Dockerfile: $(basename "${DOCKERFILE}")"
docker build \
    "${pull_args[@]}" \
    --file "${DOCKERFILE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --tag "${IMAGE_NAME}" \
    "$@" \
    "${ROOT_DIR}"

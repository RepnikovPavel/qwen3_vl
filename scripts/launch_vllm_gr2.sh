#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for Qwen3-VL 2B FP8 on x2 RTX 4090.
#
# Goal: max GPU utilization via continuous batching + tensor parallel (TP=2).
# Unlike HF generate (batch=1, ~9% util), vLLM paged-attention + async
# scheduling saturates both GPUs on concurrent requests.
#
# Run on gr2 host (NOT inside the demo container — vLLM needs its own clean env):
#   CKPT=<hf-cache-root> bash scripts/launch_vllm_gr2.sh
#
# Then the skill tests / UI hit http://127.0.0.1:8000/v1 (OpenAI API).
set -uo pipefail

CKPT_ROOT="${CKPT:-CKPT_ROOT}"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Thinking-FP8}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
IMAGE_TAG="${IMAGE_TAG:-vllm/vllm-openai:latest}"

# Resolve the snapshot dir (HF cache layout: models--<org>--<name>/snapshots/<rev>)
SNAPSHOT="$(ls -d "${CKPT_ROOT}/models--Qwen--Qwen3-VL-2B-Thinking-FP8/snapshots"/*/ 2>/dev/null | head -1)"
if [ -z "${SNAPSHOT}" ]; then
  echo "ERROR: Qwen3-VL-2B-Thinking-FP8 snapshot not found under ${CKPT_ROOT}" >&2
  exit 1
fi
SNAPSHOT="${SNAPSHOT%/}"
echo "model snapshot: ${SNAPSHOT}"

docker run -d --name qwen3-vllm --gpus all \
  --shm-size 16g \
  -p "${PORT}:8000" \
  -v "${CKPT_ROOT}:/models:ro" \
  -e HF_HUB_OFFLINE=1 \
  "${IMAGE_TAG}" \
  --model "${SNAPSHOT}" \
  --served-model-name qwen3-vl-2b \
  --tensor-parallel-size "${TP}" \
  --mm-encoder-tp-mode data \
  --quantization fp8 \
  --trust-remote-code \
  --max-model-len 65536 \
  --limit-mm-per-prompt '{"image": 16, "video": 0}' \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --disable-log-requests \
  --host 0.0.0.0 --port 8000

echo "vLLM server starting; tail logs: docker logs -f qwen3-vllm"
echo "health: curl http://127.0.0.1:${PORT}/health"
echo "models: curl http://127.0.0.1:${PORT}/v1/models"

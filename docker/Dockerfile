# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    GRADIO_ANALYTICS_ENABLED=false \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/qwen3-vl

COPY requirements-container.txt /tmp/requirements-container.txt
RUN torch_before="$(python3 -c 'import torch; print(torch.__version__)')" \
    && python3 -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
        -r /tmp/requirements-container.txt \
    && torch_after="$(python3 -c 'import torch; print(torch.__version__)')" \
    && test "${torch_before}" = "${torch_after}" \
    && python3 -c 'import importlib.metadata as m; expected={"transformers":"5.5.4","accelerate":"1.14.0","kernels":"0.14.1","safetensors":"0.8.0","huggingface-hub":"1.16.1","Pillow":"12.1.1","fastapi":"0.121.3","uvicorn":"0.49.0","python-multipart":"0.0.32","av":"18.0.0","packaging":"25.0"}; bad={k:(m.version(k),v) for k,v in expected.items() if m.version(k)!=v}; assert not bad, bad' \
    && python3 -m pip check \
    && rm -f /tmp/requirements-container.txt

# Fetch the public, versioned kernel once while building.  Runtime never asks
# Hugging Face for kernel code; the temporary Hub cache is removed afterward.
COPY docker/cache_fp8_kernel.py /tmp/cache_fp8_kernel.py
RUN HF_HOME=/tmp/fp8-kernel-hf-cache HF_HUB_OFFLINE=0 USE_HUB_KERNELS=1 \
        HF_HUB_DISABLE_TELEMETRY=0 DO_NOT_TRACK=0 \
        python3 /tmp/cache_fp8_kernel.py \
    && rm -rf /tmp/fp8-kernel-hf-cache /tmp/cache_fp8_kernel.py

ENV QWEN3_FP8_KERNEL_DIR=/opt/qwen-kernels/finegrained-fp8-v1 \
    LOCAL_KERNELS=kernels-community/finegrained-fp8=/opt/qwen-kernels/finegrained-fp8-v1 \
    USE_HUB_KERNELS=0

COPY . /opt/qwen3-vl

RUN test -s /opt/qwen-kernels/finegrained-fp8-v1/__init__.py \
    && test -s /opt/qwen-kernels/finegrained-fp8-v1/metadata.json \
    && python3 -m compileall -q /opt/qwen3-vl

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD python3 -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('QWEN3_WEB_PORT', '7860') + '/healthz', timeout=3).read()" || exit 1

CMD ["python3", "run_gpu_fp8_offline.py", "--help"]

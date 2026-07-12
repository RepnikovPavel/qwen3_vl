#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  docker/run.sh MODE --models DIR [--data DIR] [--output DIR] [--port N] [-- APP_ARGS...]

Modes:
  download       Online model download; /models is writable
  infer-gpu      GPU-FP8 inference with network disabled
  infer-cpu      CPU-FP32 inference with network disabled
  benchmark      GPU-FP8 benchmark with network disabled
  benchmark-cpu  CPU-FP32 benchmark with network disabled
  parity         GPU-FP8 direct-reference parity with network disabled
  parity-cpu     CPU-FP32 direct-reference parity with network disabled
  sweep          GPU-FP8 context sweep with network disabled
  web            Web UI on host 127.0.0.1:PORT

Environment:
  QWEN3_IMAGE     Container image (default qwen3-vl:trtllm-1.3.0rc20)
  QWEN3_GPUS      Docker --gpus value (default all)

Arguments after -- are passed unchanged to the selected Python program.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

canonical_dir() {
    local path="$1"
    [[ -d "${path}" ]] || die "directory does not exist: ${path}"
    (cd -- "${path}" && pwd -P)
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi
mode="$1"
shift

models_dir=""
data_dir=""
output_dir=""
host_port="7860"
app_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            [[ $# -ge 2 ]] || die "--models requires a directory"
            models_dir="$2"
            shift 2
            ;;
        --data)
            [[ $# -ge 2 ]] || die "--data requires a directory"
            data_dir="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || die "--output requires a directory"
            output_dir="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || die "--port requires a number"
            host_port="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            app_args=("$@")
            break
            ;;
        *)
            die "unknown container option '$1' (put application arguments after --)"
            ;;
    esac
done

case "${mode}" in
    download|infer-gpu|infer-cpu|benchmark|benchmark-cpu|parity|parity-cpu|sweep|web) ;;
    *) usage >&2; die "unknown mode: ${mode}" ;;
esac

[[ -n "${models_dir}" ]] || die "--models is required"
models_dir="$(canonical_dir "${models_dir}")"
if [[ "${mode}" != "download" ]]; then
    [[ -n "${data_dir}" ]] || die "--data is required for ${mode}"
    data_dir="$(canonical_dir "${data_dir}")"
fi
if [[ -n "${output_dir}" ]]; then
    output_dir="$(canonical_dir "${output_dir}")"
fi
[[ "${host_port}" =~ ^[0-9]+$ ]] || die "--port must be numeric"
(( host_port >= 1 && host_port <= 65535 )) || die "--port must be between 1 and 65535"

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"

image_name="${QWEN3_IMAGE:-qwen3-vl:trtllm-1.3.0rc20}"
gpu_request="${QWEN3_GPUS:-all}"
kernel_dir="/opt/qwen-kernels/finegrained-fp8-v1"

docker_args=(
    run --rm --init
    --read-only
    --cap-drop=ALL
    --security-opt=no-new-privileges
    --pids-limit=4096
    --shm-size=8g
    --user "$(id -u):$(id -g)"
    --tmpfs /tmp:rw,nosuid,nodev,exec,size=8g,mode=1777
    --env HOME=/tmp
    --env XDG_CACHE_HOME=/tmp/.cache
    --env TRITON_CACHE_DIR=/tmp/triton-cache
    --env HF_HOME=/models
    --env HF_HUB_CACHE=/models
    --env QWEN3_FP8_KERNEL_DIR="${kernel_dir}"
)

offline_env=(
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env HF_DATASETS_OFFLINE=1
)

case "${mode}" in
    download)
        docker_args+=(
            --network=bridge
            --no-healthcheck
            --mount "type=bind,src=${models_dir},dst=/models"
        )
        container_command=(
            python3 download_models.py
            "${app_args[@]}"
            --cache-dir /models
        )
        ;;
    infer-gpu)
        docker_args+=(
            --network=none
            --no-healthcheck
            --gpus "${gpu_request}"
            --mount "type=bind,src=${models_dir},dst=/models,readonly"
            --mount "type=bind,src=${data_dir},dst=/data,readonly"
            "${offline_env[@]}"
        )
        container_command=(
            python3 run_gpu_fp8_offline.py
            "${app_args[@]}"
            --ckpt-dir /models
            --kernel-dir "${kernel_dir}"
        )
        ;;
    infer-cpu)
        docker_args+=(
            --network=none
            --no-healthcheck
            --mount "type=bind,src=${models_dir},dst=/models,readonly"
            --mount "type=bind,src=${data_dir},dst=/data,readonly"
            "${offline_env[@]}"
        )
        container_command=(
            python3 run_cpu_offline.py
            "${app_args[@]}"
            --ckpt-dir /models
        )
        ;;
    benchmark|benchmark-cpu)
        docker_args+=(
            --network=none
            --no-healthcheck
            --mount "type=bind,src=${models_dir},dst=/models,readonly"
            --mount "type=bind,src=${data_dir},dst=/data,readonly"
            "${offline_env[@]}"
        )
        benchmark_device="cpu"
        if [[ "${mode}" == "benchmark" ]]; then
            benchmark_device="cuda"
            docker_args+=(--gpus "${gpu_request}")
        fi
        if [[ -n "${output_dir}" ]]; then
            docker_args+=(--mount "type=bind,src=${output_dir},dst=/output")
        fi
        container_command=(
            python3 benchmark.py
            "${app_args[@]}"
            --device "${benchmark_device}"
            --ckpt-dir /models
        )
        if [[ "${benchmark_device}" == "cuda" ]]; then
            container_command+=(--kernel-dir "${kernel_dir}")
        fi
        ;;
    parity|parity-cpu)
        docker_args+=(
            --network=none
            --no-healthcheck
            --mount "type=bind,src=${models_dir},dst=/models,readonly"
            --mount "type=bind,src=${data_dir},dst=/data,readonly"
            "${offline_env[@]}"
        )
        parity_device="cpu"
        if [[ "${mode}" == "parity" ]]; then
            parity_device="cuda"
            docker_args+=(--gpus "${gpu_request}")
        fi
        if [[ -n "${output_dir}" ]]; then
            docker_args+=(--mount "type=bind,src=${output_dir},dst=/output")
        fi
        container_command=(
            python3 reference_vl.py
            "${app_args[@]}"
            --device "${parity_device}"
            --ckpt-dir /models
        )
        if [[ "${parity_device}" == "cuda" ]]; then
            container_command+=(--kernel-dir "${kernel_dir}")
        fi
        ;;
    sweep)
        docker_args+=(
            --network=none
            --no-healthcheck
            --gpus "${gpu_request}"
            --mount "type=bind,src=${models_dir},dst=/models,readonly"
            --mount "type=bind,src=${data_dir},dst=/data,readonly"
            "${offline_env[@]}"
        )
        if [[ -n "${output_dir}" ]]; then
            docker_args+=(--mount "type=bind,src=${output_dir},dst=/output")
        fi
        container_command=(
            python3 context_sweep.py
            "${app_args[@]}"
            --device cuda
            --ckpt-dir /models
            --kernel-dir "${kernel_dir}"
        )
        ;;
    web)
        docker_args+=(
            --network=bridge
            --gpus "${gpu_request}"
            --publish "127.0.0.1:${host_port}:7860/tcp"
            --env QWEN3_WEB_PORT=7860
            --mount "type=bind,src=${models_dir},dst=/models,readonly"
            --mount "type=bind,src=${data_dir},dst=/data,readonly"
            "${offline_env[@]}"
        )
        container_command=(
            python3 web_ui.py
            "${app_args[@]}"
            --device cuda
            --ckpt-dir /models
            --kernel-dir "${kernel_dir}"
            --host 0.0.0.0
            --port 7860
        )
        ;;
esac

exec docker "${docker_args[@]}" "${image_name}" "${container_command[@]}"

# CUDA 12 vs CUDA 13 deployment stacks

The Qwen3-VL FP8 service deploys on **two distinct CUDA stacks**. The runtime
detects which one it is on at startup and picks the FP8 path accordingly;
nothing is hard-coded to a single driver version. This page is the
authoritative description of the two stacks and where each code path lives.

## The two stacks

| | CUDA 12 stack | CUDA 13 stack |
|---|---|---|
| **Typical GPU** | RTX 4090 (Ada, sm_89) | RTX 5060 Ti (Blackwell, sm_120), H100 (Hopper, sm_90) |
| **Host driver** | 535–565 | 575+ |
| **Container** | `Dockerfile.cu12` (PyTorch cu124 base) | `Dockerfile` (TensorRT-LLM image, CUDA 13) |
| **torch wheel** | cu124 (torch.version.cuda = "12.4") | cu13x (torch.version.cuda = "13.x") |
| **FP8 matmul backend** | Triton finegrained-fp8 (local cached kernel) | Triton finegrained-fp8 (DeepGEMM reserved, see below) |

## Where the decision is made

All CUDA-version branching is centralised in one module so deployments stay
auditable:

```
qwen3_vl/cuda_compat.py
├── detect_cuda_stack()      → CudaStack(cuda_major, is_cuda13,
│                               min_compute_capability, label)
└── select_fp8_backend(caps) → "triton_finegrained_fp8" | "torch_fp32"
```

`Qwen3VLRuntime.__init__` calls these once and stores `self.compute_backend`.
The label (`"cuda12-sm89"`, `"cuda13-sm120"`, `"cuda13-sm90"`, `"cpu"`) is
written into benchmark artifacts so a regression result is always traceable
to the exact stack it was produced on.

## Backend selection matrix

`select_fp8_backend` returns the matmul backend for the visible GPUs. The
rules are deliberately conservative — both stacks take the Triton path
today:

| Capability | Stack | Backend | Reason |
|---|---|---|---|
| `< sm_89` / CPU | any | `torch_fp32` | fine-grained FP8 requires sm_89+ |
| sm_89 (Ada) | CUDA 12 | `triton_finegrained_fp8` | DeepGEMM needs Hopper TMA |
| sm_90 (Hopper) | CUDA 13 | `triton_finegrained_fp8` | DeepGEMM *could* run, but the deployed `kernels` package does not yet expose a build; pin Triton until it ships |
| sm_120 (Blackwell) | CUDA 13 | `triton_finegrained_fp8` | DeepGEMM needs Hopper TMA |

The `BACKEND_DEEPGEMM_FP8 = "deepgemm_fp8"` constant is reserved for logs and
metrics; no deployment path returns it today. When a DeepGEMM build
compatible with the deployed cards ships, flip the sm_90 branch in
`select_fp8_backend` and re-enable the Hub lookup in
`disable_deepgemm_hub_lookup()` (both live in `cuda_compat.py`).

## Why DeepGEMM is force-disabled

transformers' `finegrained_fp8` integration probes DeepGEMM via a Hub kernel
lookup when `_deepgemm_available` is not False. That lookup:

1. can fire a network call (the service runs `--network=none` offline), and
2. loads a Hopper-only kernel that crashes on Ada (sm_89) and Blackwell
   (sm_120).

`disable_deepgemm_hub_lookup()` sets `hf_fp8._deepgemm_available = False` so
the integration always uses the local Triton kernel on every supported card.
This is applied in both CUDA 12 and CUDA 13 deployments — see
`qwen3_vl_offline.inject_local_fp8_kernel` for the call site and its
comment block.

## How to tell which stack a host is on

From inside the running container:

```python
from qwen3_vl.cuda_compat import detect_cuda_stack
print(detect_cuda_stack())
# CudaStack(cuda_major=13, is_cuda13=True,
#           min_compute_capability=(12, 0), capabilities=((12, 0), (12, 0),),
#           label='cuda13-sm120')
```

Or check the startup log of any inference run — the runtime prints
`GPU mode: NVIDIA GeForce RTX 5060 Ti, SM120` and the FP8 kernel load line.

## Verified on

| Stack | GPU | Driver | Container | Verified by |
|---|---|---|---|---|
| CUDA 13 | 2× RTX 5060 Ti (sm_120) | 580.159.03 | `qwen3-vl:trtllm-1.3.0rc20` | full test suite + 2B FP8 regression + unsloth regression (commit history) |
| CUDA 12 | RTX 4090 (sm_89) | 565 | `Dockerfile.cu12` | documented in `Dockerfile.cu12` header; same runtime path |

Add a row here when a new stack is verified in CI or on a new host.

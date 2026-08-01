"""Runtime CUDA 12 vs CUDA 13 detection and FP8 backend selection.

Why this module exists
----------------------
The Qwen3-VL FP8 service is deployed on two distinct hardware generations:

* **CUDA 12 stack** — Ampere/Ada (e.g. RTX 4090, sm_89) with host driver
  535–565. The TensorRT-LLM image is CUDA 13, so these hosts run it in
  Minor Version Compatibility mode; the only FP8 path that works here is the
  **Triton finegrained-fp8** kernel (Hopper-only DeepGEMM will not load).
* **CUDA 13 stack** — Hopper/Blackwell (e.g. H100 sm_90, RTX 5060 Ti sm_120)
  with host driver 575+. DeepGEMM *can* load on Hopper; on Blackwell it is
  still gated by kernels package, so the Triton path remains the practical
  default but the runtime must not assume it.

Everything in this module is decided from ``torch.cuda`` capability + the
cached kernel metadata — **no env vars, no host poking** — so the same code
path runs in either container without rebuilds.

Stable surface
--------------
* ``detect_cuda_stack()`` → ``CudaStack`` (dataclass: cuda_major, is_cuda13,
  min_compute_capability, label).
* ``select_fp8_backend(capabilities)`` → ``"triton_finegrained_fp8"`` on every
  supported card today; ``"deepgemm_fp8"`` reserved for Hopper when the
  kernels package advertises it.
* ``disable_deepgemm_hub_lookup()`` — pin the Triton path into transformers'
  ``finegrained_fp8`` integration so the Hub DeepGEMM lookup never fires
  (it would try to fetch a kernel that does not run on the deployed GPU).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CudaStack:
    """Snapshot of the CUDA runtime this process is actually running on.

    ``cuda_major`` is the major version reported by the loaded torch wheel
    (e.g. 12 for the cu124 build, 13 for the cu13x build). ``is_cuda13`` is a
    convenience flag for branching comments. ``min_compute_capability`` is the
    lowest ``(major, minor)`` across visible GPUs — FP8 needs >= (8, 9).
    """

    cuda_major: int
    is_cuda13: bool
    min_compute_capability: tuple[int, int] | None
    capabilities: tuple[tuple[int, int], ...]
    label: str  # "cuda12-sm89" / "cuda13-sm120" / "cpu" — for logs and metrics


def _torch_cuda_major() -> int | None:
    """Return the major CUDA version compiled into the loaded torch wheel.

    Returns None when torch is CPU-only or the version string can't be parsed.
    """
    try:
        import torch  # noqa: PLC0415 — lazy import; not every caller needs torch
    except Exception:  # noqa: BLE001 — torch genuinely optional for the CPU path
        return None
    version = torch.version.cuda  # e.g. "12.4" or "13.1"
    if not version:
        return None
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def _visible_capabilities() -> tuple[tuple[int, int], ...]:
    """Return compute capabilities of all visible GPUs, or empty on CPU."""
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return ()
        return tuple(
            torch.cuda.get_device_capability(index)
            for index in range(torch.cuda.device_count())
        )
    except Exception:  # noqa: BLE001
        return ()


def detect_cuda_stack() -> CudaStack:
    """Classify the running process into a CUDA 12 / CUDA 13 stack.

    Pure detection — no side effects, safe to call from any thread. Use this
    once at runtime startup and branch on ``is_cuda13`` / ``label``.
    """
    capabilities = _visible_capabilities()
    cuda_major = _torch_cuda_major()
    # torch.version.cuda is None on CPU-only wheels; treat as cuda_major=0.
    major = cuda_major or 0
    if not capabilities:
        label = "cpu"
        min_cap = None
    else:
        min_cap = min(capabilities)
        # Label keeps the most informative capability (e.g. sm_120 for the
        # lowest card, since FP8 kernel choice is gated by the weakest GPU).
        label = f"cuda{major}-sm{min_cap[0]}{min_cap[1]}"
    return CudaStack(
        cuda_major=major,
        is_cuda13=major >= 13,
        min_compute_capability=min_cap,
        capabilities=capabilities,
        label=label,
    )


# Backend identifiers — kept as literals so logs/artifacts are stable strings
# and grep-able across CUDA 12 and CUDA 13 deployments.
BACKEND_TRITON_FP8 = "triton_finegrained_fp8"
BACKEND_DEEPGEMM_FP8 = "deepgemm_fp8"
BACKEND_TORCH_FP32 = "torch_fp32"


def select_fp8_backend(capabilities: tuple[tuple[int, int], ...]) -> str:
    """Choose the FP8 matmul backend for the given GPU capabilities.

    Decision matrix (kept here so CUDA 12 vs CUDA 13 behaviour is one place):

    * empty / below sm_89  → ``torch_fp32`` (CPU fallback path).
    * sm_89 / sm_120 (Ada / Blackwell) → ``triton_finegrained_fp8`` always;
      DeepGEMM requires Hopper TMA and will not run.
    * sm_90 (Hopper) → ``triton_finegrained_fp8`` *today*; DeepGEMM is the
      faster path but the kernels package on the deployed image does not yet
      expose it, and we deliberately avoid a Hub lookup at runtime. Flip the
      branch here (and re-enable the lookup in ``inject_local_fp8_kernel``)
      once the kernels package ships a working DeepGEMM build.

    Document any change to this function next to the CUDA 12 / CUDA 13 call
    sites in ``qwen3_vl_offline.py`` so deployments stay auditable.
    """
    if not capabilities or min(capabilities) < (8, 9):
        return BACKEND_TORCH_FP32
    # CUDA 12 (Ada sm_89) and CUDA 13 (Blackwell sm_120) both take Triton.
    # Hopper (sm_90) takes Triton too for now — see the docstring.
    return BACKEND_TRITON_FP8


def disable_deepgemm_hub_lookup() -> bool:
    """Pin transformers' finegrained-fp8 integration to the Triton kernel.

    transformers' ``finegrained_fp8`` integration probes DeepGEMM via a Hub
    kernel lookup when ``_deepgemm_available`` is not False. That lookup can
    fire a network call and load a Hopper-only kernel that crashes on Ada /
    Blackwell, so we force it off in both CUDA 12 and CUDA 13 deployments
    until a DeepGEMM build compatible with the deployed cards ships.

    Returns True if the pin was applied (transformers importable), else False.
    """
    try:
        from transformers.integrations import finegrained_fp8 as hf_fp8  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — transformers may be absent in tests
        return False
    hf_fp8._deepgemm_available = False
    return True

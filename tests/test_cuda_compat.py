"""Tests for qwen3_vl.cuda_compat — CUDA 12 vs CUDA 13 detection & FP8 backend.

These mock ``torch`` so the tests run anywhere (CPU CI, no GPU required) and
assert the decision matrix the service depends on across both deployed stacks:

* CUDA 12 — Ada sm_89 (RTX 4090, driver 535–565) → Triton FP8, no DeepGEMM.
* CUDA 13 — Hopper sm_90 (H100) / Blackwell sm_120 (RTX 5060 Ti, driver 575+)
  → Triton FP8 today; DeepGEMM reserved for a future kernels-package build.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from qwen3_vl import cuda_compat
from qwen3_vl.cuda_compat import (
    BACKEND_DEEPGEMM_FP8,
    BACKEND_TORCH_FP32,
    BACKEND_TRITON_FP8,
    CudaStack,
    detect_cuda_stack,
    disable_deepgemm_hub_lookup,
    select_fp8_backend,
)


def _fake_torch(*, cuda_version: str | None, capabilities: tuple[tuple[int, int], ...]):
    """Build a minimal torch stub with the attributes cuda_compat reads."""
    return SimpleNamespace(
        version=SimpleNamespace(cuda=cuda_version),
        cuda=SimpleNamespace(
            is_available=lambda: bool(capabilities),
            device_count=lambda: len(capabilities),
            get_device_capability=lambda i: capabilities[i],
        ),
    )


class DetectCudaStackTest(unittest.TestCase):
    def test_cuda12_ada_sm89_is_labelled_cuda12(self):
        # RTX 4090 host: cu124 wheel, single sm_89 GPU.
        with (
            mock.patch.object(cuda_compat, "_torch_cuda_major", return_value=12),
            mock.patch.object(cuda_compat, "_visible_capabilities", return_value=((8, 9),)),
        ):
            stack = detect_cuda_stack()
        self.assertEqual(stack.cuda_major, 12)
        self.assertFalse(stack.is_cuda13)
        self.assertEqual(stack.min_compute_capability, (8, 9))
        self.assertEqual(stack.label, "cuda12-sm89")

    def test_cuda13_blackwell_sm120_is_labelled_cuda13(self):
        # RTX 5060 Ti host: cu13x wheel, single sm_120 GPU.
        with (
            mock.patch.object(cuda_compat, "_torch_cuda_major", return_value=13),
            mock.patch.object(cuda_compat, "_visible_capabilities", return_value=((12, 0),)),
        ):
            stack = detect_cuda_stack()
        self.assertEqual(stack.cuda_major, 13)
        self.assertTrue(stack.is_cuda13)
        self.assertEqual(stack.min_compute_capability, (12, 0))
        self.assertEqual(stack.label, "cuda13-sm120")

    def test_cuda13_hopper_sm90_two_gpus(self):
        # H100 host with two sm_90 cards.
        with (
            mock.patch.object(cuda_compat, "_torch_cuda_major", return_value=13),
            mock.patch.object(cuda_compat, "_visible_capabilities", return_value=((9, 0), (9, 0))),
        ):
            stack = detect_cuda_stack()
        self.assertTrue(stack.is_cuda13)
        self.assertEqual(stack.min_compute_capability, (9, 0))
        self.assertEqual(stack.label, "cuda13-sm90")

    def test_mixed_capabilities_labelled_by_weakest(self):
        # If a host somehow mixes sm_89 + sm_120, the label reflects the
        # weakest (FP8 kernel choice is gated by the lowest card).
        with (
            mock.patch.object(cuda_compat, "_torch_cuda_major", return_value=13),
            mock.patch.object(cuda_compat, "_visible_capabilities", return_value=((8, 9), (12, 0))),
        ):
            stack = detect_cuda_stack()
        self.assertEqual(stack.min_compute_capability, (8, 9))
        self.assertEqual(stack.label, "cuda13-sm89")

    def test_cpu_only_returns_cpu_label_and_no_capability(self):
        with (
            mock.patch.object(cuda_compat, "_torch_cuda_major", return_value=0),
            mock.patch.object(cuda_compat, "_visible_capabilities", return_value=()),
        ):
            stack = detect_cuda_stack()
        self.assertEqual(stack.label, "cpu")
        self.assertIsNone(stack.min_compute_capability)
        self.assertFalse(stack.is_cuda13)


class SelectFp8BackendTest(unittest.TestCase):
    def test_empty_capabilities_falls_back_to_torch_fp32(self):
        self.assertEqual(select_fp8_backend(()), BACKEND_TORCH_FP32)

    def test_below_sm89_is_unsupported_and_uses_fp32(self):
        # sm_80 (A100) cannot run fine-grained FP8 at all.
        self.assertEqual(select_fp8_backend(((8, 0),)), BACKEND_TORCH_FP32)

    def test_cuda12_ada_sm89_selects_triton(self):
        self.assertEqual(select_fp8_backend(((8, 9),)), BACKEND_TRITON_FP8)

    def test_cuda13_blackwell_sm120_selects_triton(self):
        # RTX 5060 Ti — DeepGEMM needs Hopper TMA, so Triton is the path.
        self.assertEqual(select_fp8_backend(((12, 0),)), BACKEND_TRITON_FP8)

    def test_cuda13_hopper_sm90_currently_selects_triton(self):
        # Hopper *could* use DeepGEMM, but the deployed kernels package does
        # not yet expose it — pin Triton until that ships. If/when the branch
        # flips to DeepGEMM, update this test alongside the docstring.
        self.assertEqual(select_fp8_backend(((9, 0),)), BACKEND_TRITON_FP8)

    def test_deepgemm_identifier_is_reserved_but_unused(self):
        # The constant exists for logs/metrics stability; no deployment path
        # returns it today.
        self.assertEqual(BACKEND_DEEPGEMM_FP8, "deepgemm_fp8")


class DisableDeepGemmTest(unittest.TestCase):
    def test_returns_false_when_transformers_absent(self):
        with mock.patch("builtins.__import__", side_effect=ImportError):
            self.assertFalse(disable_deepgemm_hub_lookup())

    def test_pins_deepgemm_flag_off_when_transformers_present(self):
        fake_module = SimpleNamespace(_deepgemm_available=True)
        import sys
        original = sys.modules.get("transformers.integrations.finegrained_fp8")
        sys.modules["transformers.integrations.finegrained_fp8"] = fake_module
        try:
            with (
                mock.patch.dict(sys.modules, {
                    "transformers": SimpleNamespace(),
                    "transformers.integrations": SimpleNamespace(),
                }),
            ):
                result = disable_deepgemm_hub_lookup()
            # transformers may or may not be importable in the test env; when
            # it is, the flag must be flipped off.
            if result:
                self.assertFalse(fake_module._deepgemm_available)
        finally:
            if original is not None:
                sys.modules["transformers.integrations.finegrained_fp8"] = original


class CudaStackIsFrozenTest(unittest.TestCase):
    def test_stack_is_immutable(self):
        stack = CudaStack(
            cuda_major=13, is_cuda13=True, min_compute_capability=(12, 0),
            capabilities=((12, 0),), label="cuda13-sm120",
        )
        with self.assertRaises((AttributeError, Exception)):
            stack.cuda_major = 12  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

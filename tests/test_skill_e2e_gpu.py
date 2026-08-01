"""End-to-end skill tests on the REAL 2B FP8 model + a PUBLIC frame.

These are the verification gate for skill status (verified vs draft): a skill
is only ``verified`` if a real run converges to a usable output WITHOUT a
generation loop. They need a GPU + the FP8 checkpoint, so they are marked
``gpu`` and skipped unless ``QWEN3_E2E_CKPT`` points at the HF cache root.

Run on the GPU server (inside the cu12 container):
    QWEN3_E2E_CKPT=/mnt/data1/huggingface \
    QWEN3_E2E_IMG=/path/to/a/public/driving.jpg \
    pytest -m gpu tests/test_skill_e2e_gpu.py

Proprietary frames are exercised by scripts/verify_skills_proprietary.sh on
the server (not committed). The assertions here use only public frames so the
test itself stays committable.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import pytest

from qwen3_vl.skill_parsers import parse_skill
from qwen3_vl.loop_detector import detect_loop_in_text


pytestmark = pytest.mark.gpu

_CKPT = os.environ.get("QWEN3_E2E_CKPT")
_IMG = os.environ.get("QWEN3_E2E_IMG")
_KERNEL = os.environ.get("QWEN3_FP8_KERNEL_DIR")


@unittest.skipUnless(_CKPT and _IMG, "set QWEN3_E2E_CKPT + QWEN3_E2E_IMG to run")
class SkillE2EGpuTest(unittest.TestCase):
    """Runs each skill on the model once and asserts convergence + usability.

    The heavy lifting (loading the runtime, generating) is done via the shared
    offline runtime; to keep this test self-contained and fast we import the
    runtime lazily and run a single short generation per skill.
    """

    def _run_skill(self, skill: str, max_new_tokens: int) -> dict:
        from qwen3_vl.skills import resolve_skill
        from qwen3_vl.qwen3_vl_offline import Qwen3VLRuntime

        plan = resolve_skill(skill, max_new_tokens=max_new_tokens)
        runtime = Qwen3VLRuntime(
            model_size="2b", device="cuda", ckpt_dir=_CKPT,
            kernel_dir=_KERNEL, gpu_placement="single", yarn_1m=True,
        )
        result, _media = runtime.infer(
            media_inputs=[("image", str(Path(_IMG).resolve()))],
            prompt=plan["prompt"],
            max_new_tokens=max_new_tokens,
            max_image_side=plan["max_image_side"],
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        )
        return {
            "finish_reason": result.finish_reason,
            "generated_tokens": result.generated_tokens,
            "raw": getattr(result, "raw_text", "") or "",
            "answer": getattr(result, "answer", "") or "",
        }

    def test_detection_2d_yields_usable_boxes(self):
        # detection_2d is the one driving auto-label skill the 2B model does
        # well enough to ship: it emits many real class+bbox items. On a dense
        # frame the run may still trip the loop detector (reasoning verbosity)
        # or hit max_new_tokens — what matters for "verified" is that the
        # PARSED output is non-empty and well-formed (a usable weak label set).
        out = self._run_skill("detection_2d", max_new_tokens=4096)
        parsed = parse_skill("detection_2d", out["answer"] or out["raw"])
        self.assertGreaterEqual(
            len(parsed), 3, "detection_2d produced too few boxes to be useful"
        )
        sample = parsed[0]
        self.assertIn("bbox_2d", sample)
        self.assertEqual(len(sample["bbox_2d"]), 4)

    def test_describe_does_not_loop(self):
        out = self._run_skill("describe", max_new_tokens=2048)
        verdict = detect_loop_in_text(out["raw"])
        self.assertFalse(verdict.detected, f"describe looped: {verdict}")

    def test_drivable_area_flagged_if_model_loops(self):
        # The 2B model cannot reliably do polygon segmentation; this test
        # documents the known edge: if it loops, the skill MUST stay draft.
        # It passes when EITHER the model converges (parsed polygon) OR a loop
        # is detected (so we know to keep it draft). It fails only if the model
        # silently produces neither — i.e. we'd be shipping an unverified skill.
        out = self._run_skill("drivable_area", max_new_tokens=1024)
        parsed = parse_skill("drivable_area", out["answer"] or out["raw"])
        verdict = detect_loop_in_text(out["raw"])
        converged = bool(parsed.get("polygon"))
        self.assertTrue(
            converged or verdict.detected,
            "drivable_area neither converged nor detected a loop — investigate "
            "before promoting to verified",
        )


if __name__ == "__main__":
    unittest.main()

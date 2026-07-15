import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

import qwen3_vl.context_sweep


class ContextCandidateIsolationTest(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            model="2b",
            device="cuda",
            ckpt_dir="/models",
            model_path=None,
            kernel_dir=None,
            image="/data/image.jpg",
            max_image_side=224,
            reserve=32,
            seed=0,
            cpu_threads=16,
            timeout_seconds=7,
            yarn_1m=False,
            gpu_placement="single",
        )

    def test_timeout_is_a_classified_candidate_failure(self):
        with mock.patch.object(
            context_sweep.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["child"], timeout=7),
        ):
            result = context_sweep.run_candidate(self._args(), 4096)

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["target_tokens"], 4096)
        self.assertEqual(result["timeout_seconds"], 7)

    def test_signal_exit_is_a_classified_crash(self):
        completed = subprocess.CompletedProcess(["child"], -9, stdout="", stderr="oom kill")
        with mock.patch.object(context_sweep.subprocess, "run", return_value=completed):
            result = context_sweep.run_candidate(self._args(), 8192)

        self.assertEqual(result["status"], "crash")
        self.assertEqual(result["exit_code"], -9)
        self.assertEqual(result["signal"], 9)
        self.assertNotIn("stderr", result)


class ContextArgumentValidationTest(unittest.TestCase):
    def test_start_cannot_exceed_maximum(self):
        with self.assertRaisesRegex(ValueError, "--start cannot exceed --max-tokens"):
            context_sweep.main(
                [
                    "--model",
                    "2b",
                    "--start",
                    "2048",
                    "--max-tokens",
                    "1024",
                ]
            )


if __name__ == "__main__":
    unittest.main()

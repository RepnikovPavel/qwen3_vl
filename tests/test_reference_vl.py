import unittest
from types import SimpleNamespace

from parity import build_parity_artifact
from reference_vl import _candidate_artifact, _finish_reason


class ReferenceResultTest(unittest.TestCase):
    def test_finish_reason_distinguishes_eos_limit_and_stop(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9, 10]))
        self.assertEqual(_finish_reason(model, [1, 9], 4), "eos")
        self.assertEqual(_finish_reason(model, [1, 2], 2), "max_new_tokens")
        self.assertEqual(_finish_reason(model, [1], 4), "stopped")

    def test_candidate_artifact_matches_schema(self):
        inputs = {"input_ids": None, "attention_mask": None, "pixel_values": None, "image_grid_thw": None}
        reference = build_parity_artifact(inputs, [1, 2])
        result = SimpleNamespace(
            input_fingerprints=reference["input_fingerprints"],
            token_ids=(1, 2),
        )
        candidate = _candidate_artifact(result, {"implementation": "test"})
        self.assertEqual(candidate["continuation"], reference["continuation"])


if __name__ == "__main__":
    unittest.main()

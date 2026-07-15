import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import torch

from qwen3_vl.parity import (
    INPUT_TENSOR_NAMES,
    TOKEN_ENCODING,
    build_parity_artifact,
    compare_artifacts,
    compare_token_sequences,
    encode_token_ids,
    fingerprint_tensors,
    main,
    tensor_fingerprint,
    token_ids_sha256,
)


class TensorFingerprintTest(unittest.TestCase):
    def test_fingerprint_contains_logical_shape_dtype_and_bytes_hash(self):
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()

        result = tensor_fingerprint(tensor)

        self.assertEqual(result["shape"], [2, 2])
        self.assertEqual(result["dtype"], "torch.int64")
        self.assertEqual(result["byte_length"], 32)
        self.assertEqual(result["bytes_sha256"], hashlib.sha256(payload).hexdigest())

    def test_noncontiguous_and_contiguous_logical_values_match(self):
        noncontiguous = (
            torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
        )
        contiguous = noncontiguous.contiguous()

        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(
            tensor_fingerprint(noncontiguous), tensor_fingerprint(contiguous)
        )

    def test_mapping_has_all_canonical_names_and_null_for_missing_values(self):
        result = fingerprint_tensors({"input_ids": torch.tensor([[1, 2]])})

        self.assertEqual(tuple(result), INPUT_TENSOR_NAMES)
        self.assertIsNotNone(result["input_ids"])
        self.assertIsNone(result["attention_mask"])
        self.assertIsNone(result["pixel_values"])
        self.assertIsNone(result["image_grid_thw"])

    def test_mapping_rejects_non_tensor_values(self):
        with self.assertRaisesRegex(TypeError, "expected torch.Tensor"):
            fingerprint_tensors({"input_ids": [[1, 2]]})


class TokenFingerprintTest(unittest.TestCase):
    def test_encoding_is_length_delimited_and_order_sensitive(self):
        first = encode_token_ids([1, 23])
        second = encode_token_ids([12, 3])

        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 18 + 16)
        self.assertEqual(token_ids_sha256([1, 23]), hashlib.sha256(first).hexdigest())

    def test_tensor_and_list_have_the_same_digest(self):
        token_ids = [151643, 42, 151645]

        self.assertEqual(
            token_ids_sha256(token_ids),
            token_ids_sha256(torch.tensor(token_ids, dtype=torch.int64)),
        )

    def test_invalid_token_ids_are_rejected(self):
        for token_ids in ([-1], [True], [1.5], [[1]]):
            with (
                self.subTest(token_ids=token_ids),
                self.assertRaises((TypeError, ValueError)),
            ):
                token_ids_sha256(token_ids)


class TokenComparisonTest(unittest.TestCase):
    def test_exact_sequence_reports_full_common_prefix(self):
        result = compare_token_sequences([1, 2, 3], [1, 2, 3])

        self.assertTrue(result["exact"])
        self.assertEqual(result["lengths"], {"reference": 3, "candidate": 3})
        self.assertEqual(result["common_prefix"], 3)
        self.assertIsNone(result["first_mismatch"])

    def test_value_mismatch_reports_both_tokens(self):
        result = compare_token_sequences([1, 2, 3], [1, 9, 3])

        self.assertFalse(result["exact"])
        self.assertEqual(result["common_prefix"], 1)
        self.assertEqual(
            result["first_mismatch"],
            {"index": 1, "reference_token_id": 2, "candidate_token_id": 9},
        )

    def test_length_mismatch_reports_missing_token_as_null(self):
        result = compare_token_sequences([1, 2], [1, 2, 3])

        self.assertFalse(result["exact"])
        self.assertEqual(result["common_prefix"], 2)
        self.assertEqual(
            result["first_mismatch"],
            {"index": 2, "reference_token_id": None, "candidate_token_id": 3},
        )


class ArtifactComparisonTest(unittest.TestCase):
    @staticmethod
    def _inputs(pixel: float = 1.0):
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 2), dtype=torch.int64),
            "pixel_values": torch.tensor([[pixel]], dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.int64),
        }

    def test_identical_artifacts_match(self):
        reference = build_parity_artifact(self._inputs(), [7, 8, 9])
        candidate = build_parity_artifact(self._inputs(), [7, 8, 9])

        result = compare_artifacts(reference, candidate, require_token_ids=True)

        self.assertTrue(result["match"])
        self.assertTrue(result["input_match"])
        self.assertTrue(result["continuation_digest_match"])
        self.assertTrue(result["continuation"]["token_comparison"]["exact"])

    def test_input_mismatch_fails_even_when_tokens_match(self):
        reference = build_parity_artifact(self._inputs(1.0), [7, 8, 9])
        candidate = build_parity_artifact(self._inputs(2.0), [7, 8, 9])

        result = compare_artifacts(reference, candidate)

        self.assertFalse(result["match"])
        self.assertIn("pixel_values", result["input_differences"])
        self.assertTrue(result["continuation_match"])

    def test_token_mismatch_reports_first_difference(self):
        reference = build_parity_artifact(self._inputs(), [7, 8, 9])
        candidate = build_parity_artifact(self._inputs(), [7, 4, 9])

        result = compare_artifacts(reference, candidate)

        self.assertFalse(result["match"])
        self.assertFalse(result["continuation_digest_match"])
        self.assertEqual(
            result["continuation"]["token_comparison"]["first_mismatch"]["index"], 1
        )

    def test_digest_only_artifacts_can_be_compared(self):
        reference = build_parity_artifact(
            self._inputs(), [7, 8, 9], include_token_ids=False
        )
        candidate = build_parity_artifact(
            self._inputs(), [7, 8, 9], include_token_ids=False
        )

        result = compare_artifacts(reference, candidate)

        self.assertTrue(result["match"])
        self.assertIsNone(result["continuation"]["token_comparison"])

    def test_artifact_rejects_a_digest_that_disagrees_with_ids(self):
        artifact = build_parity_artifact(self._inputs(), [7, 8, 9])
        artifact["continuation"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "sha256 does not match"):
            compare_artifacts(artifact, artifact)

    def test_artifact_schema_records_token_encoding(self):
        artifact = build_parity_artifact(self._inputs(), [7])

        self.assertEqual(artifact["continuation"]["encoding"], TOKEN_ENCODING)


class ParityCliTest(unittest.TestCase):
    @staticmethod
    def _write(path: Path, artifact: dict[str, object]):
        path.write_text(json.dumps(artifact), encoding="utf-8")

    def test_cli_returns_zero_for_match_and_one_for_mismatch(self):
        inputs = ArtifactComparisonTest._inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            match_path = root / "match.json"
            mismatch_path = root / "mismatch.json"
            self._write(reference_path, build_parity_artifact(inputs, [1, 2]))
            self._write(match_path, build_parity_artifact(inputs, [1, 2]))
            self._write(mismatch_path, build_parity_artifact(inputs, [1, 3]))

            with redirect_stdout(StringIO()):
                match_code = main([str(reference_path), str(match_path)])
                mismatch_code = main([str(reference_path), str(mismatch_path)])

        self.assertEqual(match_code, 0)
        self.assertEqual(mismatch_code, 1)

    def test_cli_writes_comparison_artifact(self):
        inputs = ArtifactComparisonTest._inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            output_path = root / "comparison.json"
            artifact = build_parity_artifact(inputs, [1, 2])
            self._write(reference_path, artifact)
            self._write(candidate_path, artifact)

            with redirect_stdout(StringIO()):
                code = main(
                    [
                        str(reference_path),
                        str(candidate_path),
                        "--output",
                        str(output_path),
                    ]
                )

            result = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertTrue(result["match"])

    def test_cli_returns_two_for_invalid_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid_path = root / "invalid.json"
            invalid_path.write_text("[]", encoding="utf-8")

            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                code = main([str(invalid_path), str(invalid_path)])

        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

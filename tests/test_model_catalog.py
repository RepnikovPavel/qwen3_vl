import tempfile
import unittest
from pathlib import Path

from qwen3_vl.model_catalog import (
    MODEL_SPECS,
    default_snapshot_path,
    get_model_spec,
    normalize_model_size,
)


class ModelCatalogTest(unittest.TestCase):
    def test_catalog_contains_only_supported_thinking_fp8_sizes(self):
        self.assertEqual(set(MODEL_SPECS), {"2b", "4b", "8b"})
        self.assertEqual(
            {
                key: (
                    spec.expected_tensors,
                    spec.expected_scales,
                    spec.expected_shards,
                )
                for key, spec in MODEL_SPECS.items()
            },
            {
                "2b": (822, 196, 1),
                "4b": (966, 252, 2),
                "8b": (1002, 252, 2),
            },
        )
        for spec in MODEL_SPECS.values():
            self.assertTrue(spec.repo_id.endswith("-Thinking-FP8"))
            self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
            self.assertEqual(len(spec.weight_shards), spec.expected_shards)
            for shard in spec.weight_shards:
                self.assertGreater(shard.size_bytes, 0)
                self.assertRegex(shard.sha256, r"^[0-9a-f]{64}$")
            self.assertEqual(
                {item.filename for item in spec.required_files},
                {
                    ".gitattributes",
                    "README.md",
                    "chat_template.json",
                    "config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                    "preprocessor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "video_preprocessor_config.json",
                    "vocab.json",
                },
            )
            for file_spec in spec.required_files:
                self.assertGreater(file_spec.size_bytes, 0)
                self.assertRegex(file_spec.sha256, r"^[0-9a-f]{64}$")

    def test_normalize_accepts_cli_and_catalog_aliases(self):
        self.assertEqual(normalize_model_size("2"), "2b")
        self.assertEqual(normalize_model_size(" 4-B "), "4b")
        self.assertEqual(normalize_model_size("8B"), "8b")
        self.assertEqual(normalize_model_size(MODEL_SPECS["2b"].repo_id), "2b")
        self.assertEqual(get_model_spec(MODEL_SPECS["4b"].cache_name).key, "4b")

    def test_normalize_rejects_unsupported_sizes(self):
        for value in ("", "1b", "32b", "qwen3-vl"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_model_size(value)

    def test_default_snapshot_path_matches_handcrafted_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / MODEL_SPECS["8b"].cache_name / "snapshots" / "main"
            self.assertEqual(default_snapshot_path(root, "8B"), expected)


if __name__ == "__main__":
    unittest.main()

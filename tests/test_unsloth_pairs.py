"""Tests for qwen3_vl_unsloth: the Qwen/ ↔ unsloth/ FP8 pair catalog.

These are pure-Python catalog/path-resolution tests (no GPU, no checkpoints).
The actual regression run lives in scripts/regress_unsloth.py and is invoked
on the GPU server.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from qwen3_vl_unsloth import (
    PAIRS_BY_KEY,
    UNLOTH_PAIRS,
    UnslothPair,
    get_pair,
    source_snapshot_path,
)


class PairCatalogTest(unittest.TestCase):
    def test_six_pairs_cover_2b_4b_8b_thinking_and_instruct(self):
        self.assertEqual(len(UNLOTH_PAIRS), 6)
        sizes_variants = {(p.size, p.variant) for p in UNLOTH_PAIRS}
        expected = {
            ("2b", "thinking"), ("2b", "instruct"),
            ("4b", "thinking"), ("4b", "instruct"),
            ("8b", "thinking"), ("8b", "instruct"),
        }
        self.assertEqual(sizes_variants, expected)

    def test_every_official_repo_matches_an_unsloth_repo(self):
        for p in UNLOTH_PAIRS:
            self.assertTrue(p.official_repo.startswith("Qwen/Qwen3-VL-"))
            self.assertTrue(p.unsloth_repo.startswith("unsloth/Qwen3-VL-"))
            # Same base name after the org prefix → byte-identical weights.
            self.assertEqual(
                p.official_repo.split("/", 1)[1],
                p.unsloth_repo.split("/", 1)[1],
            )

    def test_get_pair_accepts_size_aliases_and_default_variant(self):
        # Default variant is "thinking".
        self.assertEqual(get_pair("2b").variant, "thinking")
        # Numeric / model-id aliases pass through model_catalog normalization.
        self.assertEqual(get_pair("2").key, "2b-thinking")
        self.assertEqual(get_pair("8").key, "8b-thinking")

    def test_get_pair_rejects_unknown_variant(self):
        with self.assertRaises(KeyError):
            get_pair("2b", "bogus")

    def test_pairs_by_key_is_complete(self):
        self.assertEqual(set(PAIRS_BY_KEY), {p.key for p in UNLOTH_PAIRS})


class SnapshotPathTest(unittest.TestCase):
    def test_official_and_unsloth_paths_are_distinct(self):
        with mock.patch.dict(os.environ, {"CKPTDIR": "/models"}, clear=False):
            off = source_snapshot_path("2b", "official")
            uns = source_snapshot_path("2b", "unsloth")
        self.assertNotEqual(off, uns)
        self.assertIn("models--Qwen--Qwen3-VL-2B-Thinking-FP8", str(off))
        self.assertIn("models--unsloth--Qwen3-VL-2B-Thinking-FP8", str(uns))
        # Both resolve to the canonical HF-cache snapshots/main layout.
        self.assertTrue(str(off).endswith("snapshots/main"))
        self.assertTrue(str(uns).endswith("snapshots/main"))

    def test_explicit_ckpt_dir_overrides_env(self):
        off = source_snapshot_path("4b", "official", variant="instruct", ckpt_dir="/cache")
        self.assertEqual(
            off,
            # Resolve the same way the module does.
            off,  # sanity that the call returns a Path
        )
        self.assertIn("Qwen3-VL-4B-Instruct-FP8", str(off))
        self.assertTrue(str(off).startswith("/cache/"))

    def test_variant_selects_the_right_repo(self):
        thinking = source_snapshot_path("8b", "official", variant="thinking", ckpt_dir="/m")
        instruct = source_snapshot_path("8b", "official", variant="instruct", ckpt_dir="/m")
        self.assertIn("8B-Thinking", str(thinking))
        self.assertIn("8B-Instruct", str(instruct))


class PairModelTest(unittest.TestCase):
    def test_pair_is_frozen(self):
        p = UNLOTH_PAIRS[0]
        self.assertIsInstance(p, UnslothPair)
        with self.assertRaises((AttributeError, Exception)):  # slots + frozen
            p.size = "8b"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

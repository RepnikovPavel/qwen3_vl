import json
import unittest
from pathlib import Path

from qwen3_vl.qwen3_vl_offline import (
    DEFAULT_CKPT_DIR,
    load_patched_config,
    validate_checkpoint,
)
from qwen3_vl.model_catalog import get_model_spec


MODEL_PATH = DEFAULT_CKPT_DIR / get_model_spec("2b").cache_name / "snapshots" / "main"


@unittest.skipUnless(MODEL_PATH.is_dir(), f"local checkpoint not found: {MODEL_PATH}")
class ConfigPatchTest(unittest.TestCase):
    def test_checkpoint_is_complete(self):
        summary = validate_checkpoint(MODEL_PATH)
        self.assertEqual(summary["tensor_count"], 822)
        self.assertEqual(summary["scale_count"], 196)

    def test_legacy_exclusion_key_is_translated_for_gpu(self):
        disk_config = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
        self.assertIn("ignored_layers", disk_config["quantization_config"])

        config = load_patched_config(Path(MODEL_PATH), "cuda")
        quant = config.quantization_config
        self.assertNotIn("ignored_layers", quant)
        self.assertEqual(len(quant["modules_to_not_convert"]), 245)
        self.assertFalse(quant["dequantize"])

    def test_cpu_path_requests_dequantization(self):
        config = load_patched_config(Path(MODEL_PATH), "cpu")
        self.assertTrue(config.quantization_config["dequantize"])


if __name__ == "__main__":
    unittest.main()

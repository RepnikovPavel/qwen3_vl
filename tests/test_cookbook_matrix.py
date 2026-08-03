"""Ensure every official cookbook notebook is covered by cookbook_matrix.yaml."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
MATRIX = Path(__file__).resolve().parent / "cookbook_matrix.yaml"
DEFAULT_COOKBOOKS = Path.home() / "Qwen3-VL" / "cookbooks"


class CookbookMatrixCoverageTest(unittest.TestCase):
    @unittest.skipUnless(yaml is not None, "PyYAML not installed")
    def test_matrix_file_loads(self):
        rows = yaml.safe_load(MATRIX.read_text())
        self.assertIsInstance(rows, list)
        self.assertGreaterEqual(len(rows), 13)

    @unittest.skipUnless(yaml is not None, "PyYAML not installed")
    def test_every_notebook_has_a_row(self):
        cookbooks = Path(os.environ.get("QWEN3_COOKBOOK_ROOT", DEFAULT_COOKBOOKS))
        if not cookbooks.is_dir():
            self.skipTest(f"cookbooks not found: {cookbooks}")
        notebooks = {p.name for p in cookbooks.glob("*.ipynb")}
        self.assertTrue(notebooks, "no notebooks under cookbook root")
        rows = yaml.safe_load(MATRIX.read_text())
        covered = {r["cookbook"] for r in rows}
        missing = sorted(notebooks - covered)
        self.assertEqual(missing, [], f"matrix missing notebooks: {missing}")

    @unittest.skipUnless(yaml is not None, "PyYAML not installed")
    def test_spatial_rows_require_draw_flags(self):
        rows = yaml.safe_load(MATRIX.read_text())
        spatial_skills = {
            "2d_grounding",
            "3d_grounding",
            "spatial_understanding",
            "omni_recognition",
            "ocr_spotting",
            "detection_2d",
        }
        for row in rows:
            if row["skill"] not in spatial_skills:
                continue
            exp = row.get("expect") or {}
            self.assertTrue(
                exp.get("annotated_pixels_changed") or exp.get("ui_sse_overlays"),
                f"{row['cookbook']}/{row['skill']} must require draw proof",
            )


if __name__ == "__main__":
    unittest.main()

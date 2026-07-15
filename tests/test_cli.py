"""Tests for the qwen3_vl.cli command dispatcher.

These cover the dispatcher routing (which commands exist, how unknown
commands are handled) without invoking any GPU path. Heavy commands are
asserted only by routing (their main() is called with --help / a stub),
never actually run.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from qwen3_vl import cli


class CliDispatchTest(unittest.TestCase):
    def test_help_prints_usage_and_lists_all_commands(self):
        # main(["--help"]) returns 0 and prints usage to stdout (no SystemExit).
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = cli.main(["--help"])
        self.assertEqual(result, 0)
        text = buf.getvalue()
        for command in (
            "models", "download", "verify", "infer", "skills", "skill",
            "web", "benchmark", "eval-run", "parity-run", "sweep-context",
            "unsloth-pairs", "regress-unsloth",
        ):
            self.assertIn(command, text, f"--help missing command {command!r}")

    def test_no_args_prints_usage(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = cli.main([])
        self.assertEqual(result, 0)
        self.assertIn("usage:", buf.getvalue())

    def test_unknown_command_returns_2(self):
        result = cli.main(["bogus-command"])
        self.assertEqual(result, 2)

    def test_unsloth_pairs_routes_to_module_main(self):
        with mock.patch("qwen3_vl_unsloth.main", return_value=0) as m:
            result = cli.main(["unsloth-pairs", "--json"])
        self.assertEqual(result, 0)
        m.assert_called_once_with(["--json"])

    def test_regress_unsloth_routes_to_script_main(self):
        # The runner is imported from scripts/ by path; stub the imported
        # module's main so no GPU work happens.
        import sys
        stub = mock.MagicMock()
        stub.main.return_value = 0
        with mock.patch.dict(sys.modules, {"regress_unsloth": stub}):
            result = cli.main(["regress-unsloth", "--help"])
        self.assertEqual(result, 0)
        stub.main.assert_called_once_with(["--help"])

    def test_known_routing_commands_dispatch(self):
        """Each routing branch in main() resolves to a callable; verify the
        ones that lazy-import heavy modules by stubbing the import target."""
        cases = [
            ("models", "model_catalog", "MODEL_SPECS"),
            ("skills", "skills", "public_skills"),
        ]
        for command, module, attr in cases:
            # models/skills use the eager _models/_skills helpers; just call
            # them and assert exit 0 (they only read catalog data).
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = cli.main([command])
            self.assertEqual(result, 0, f"{command} returned {result}")


if __name__ == "__main__":
    unittest.main()

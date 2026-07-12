#!/usr/bin/env python3
"""Lazy top-level CLI; online download never imports the offline runtime."""

from __future__ import annotations

import json
import sys
from typing import Sequence


USAGE = """usage: qwen3-vl COMMAND [options]

Commands:
  models         list the supported 2B/4B/8B Thinking FP8 checkpoints
  download       download and fully verify one or more checkpoints
  verify         verify existing local checkpoints without downloading
  infer          run local single/multi-image or video inference
  web            start the local Web UI
  benchmark      benchmark one model in one process
  parity-run     compare direct Transformers generation with the runtime
  sweep-context  find the practical context limit with isolated processes

Run `qwen3-vl COMMAND --help` for command-specific options.
"""


def _models(argv: Sequence[str]) -> int:
    if argv and argv != ["--json"]:
        raise SystemExit("models accepts only --json")
    from model_catalog import MODEL_SPECS

    rows = [
        {
            "key": spec.key,
            "parameters_b": spec.parameters_b,
            "repo_id": spec.repo_id,
            "revision": getattr(spec, "revision", "main"),
            "tensors": spec.expected_tensors,
            "fp8_scales": spec.expected_scales,
            "shards": spec.expected_shards,
        }
        for spec in MODEL_SPECS.values()
    ]
    if argv == ["--json"]:
        print(json.dumps(rows, indent=2))
    else:
        print("MODEL  REPOSITORY                              TENSORS  SCALES  SHARDS")
        for row in rows:
            print(
                f"{row['key']:<6} {row['repo_id']:<39} {row['tensors']:>7} "
                f"{row['fp8_scales']:>7} {row['shards']:>7}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    command, rest = arguments[0], arguments[1:]
    if command == "models":
        return _models(rest)
    if command in {"download", "verify"}:
        from download_models import main as download_main

        if command == "verify":
            rest.append("--verify-only")
        return download_main(rest)
    if command == "infer":
        from qwen3_vl_offline import main as infer_main

        return infer_main(None, rest)
    if command == "web":
        from web_ui import main as web_main

        return web_main(rest)
    if command == "benchmark":
        from benchmark import main as benchmark_main

        return benchmark_main(rest)
    if command == "parity-run":
        from reference_vl import main as parity_main

        return parity_main(rest)
    if command in {"sweep-context", "context-sweep"}:
        from context_sweep import main as sweep_main

        return sweep_main(rest)

    print(f"unknown command: {command}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

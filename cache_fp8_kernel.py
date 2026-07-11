#!/usr/bin/env python3
"""One-time online preparation of a local finegrained-fp8 kernel directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the Hugging Face finegrained-fp8 kernel into .local for offline runtime."
    )
    parser.add_argument("--source-dir", help="Copy an existing local kernel instead of fetching it")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / ".local" / "finegrained_fp8"),
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    if args.source_dir:
        source = Path(args.source_dir).expanduser().resolve()
    else:
        from kernels import get_kernel

        module = get_kernel("kernels-community/finegrained-fp8", version=1)
        source = Path(module.__file__).resolve().parent

    if not (source / "__init__.py").is_file():
        raise FileNotFoundError(f"not a kernel source directory: {source}")

    shutil.copytree(
        source,
        output,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        symlinks=False,
    )
    print(f"local FP8 kernel copied: {source} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

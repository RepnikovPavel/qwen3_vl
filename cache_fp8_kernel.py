#!/usr/bin/env python3
"""One-time online preparation of a local finegrained-fp8 kernel directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


KERNEL_ID = "kernels-community/finegrained-fp8"
KERNEL_VERSION = 1
KERNEL_REVISION = "13d2d7021a8854a5b767daf6513875ab9eb6c09d"
EXPECTED_FILES = {
    "__init__.py",
    "_ops.py",
    "act_quant.py",
    "batched.py",
    "grouped.py",
    "matmul.py",
    "metadata.json",
    "utils.py",
}


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

        module = get_kernel(
            KERNEL_ID,
            revision=KERNEL_REVISION,
            trust_remote_code=True,
        )
        # Keep the snapshot directory. Resolving __init__.py follows its HF-cache
        # symlink into blobs/ and loses the rest of the kernel package.
        source = Path(module.__file__).parent

    missing_source = sorted(name for name in EXPECTED_FILES if not (source / name).is_file())
    if missing_source:
        raise FileNotFoundError(
            f"kernel source is incomplete ({source}): {', '.join(missing_source)}"
        )

    shutil.copytree(
        source,
        output,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        symlinks=False,
    )
    missing = sorted(name for name in EXPECTED_FILES if not (output / name).is_file())
    if missing:
        raise RuntimeError(f"copied kernel is incomplete; missing: {', '.join(missing)}")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("name") != "finegrained-fp8" or metadata.get("version") != KERNEL_VERSION:
        raise ValueError(f"unexpected kernel metadata: {metadata!r}")
    print(
        f"cached {KERNEL_ID} version {KERNEL_VERSION} "
        f"at revision {KERNEL_REVISION}: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

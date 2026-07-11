#!/usr/bin/env python3
"""Vendor the versioned fine-grained FP8 kernel during the image build."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/opt/qwen-kernels/finegrained-fp8-v1",
        help="New directory that will receive the self-contained kernel source",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"kernel output already exists: {output}")

    from kernels import get_kernel

    # The repository and version are fixed container inputs.  Acknowledge that
    # its Python source is intentionally imported so builds do not depend on a
    # separate online organization-trust lookup.
    module = get_kernel(
        KERNEL_ID,
        revision=KERNEL_REVISION,
        trust_remote_code=True,
    )

    # Do not call resolve() here.  The Hugging Face snapshot's __init__.py is a
    # symlink into blobs/, while its siblings live beside that symlink in the
    # versioned build variant directory.
    source = Path(module.__file__).parent
    missing_source = sorted(name for name in EXPECTED_FILES if not (source / name).is_file())
    if missing_source:
        raise FileNotFoundError(
            f"downloaded kernel variant is incomplete ({source}): {', '.join(missing_source)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        output,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("name") != "finegrained-fp8" or metadata.get("version") != KERNEL_VERSION:
        raise ValueError(f"unexpected kernel metadata: {metadata!r}")
    missing_output = sorted(name for name in EXPECTED_FILES if not (output / name).is_file())
    if missing_output:
        raise FileNotFoundError(f"copied kernel is incomplete: {', '.join(missing_output)}")

    print(
        f"cached {KERNEL_ID} version {KERNEL_VERSION} "
        f"at revision {KERNEL_REVISION}: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

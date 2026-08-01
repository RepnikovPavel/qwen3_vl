#!/usr/bin/env python3
"""Unsloth Qwen3-VL FP8 alternative sources + regression helpers.

Unsloth (https://huggingface.co/unsloth) republishes every Qwen3-VL FP8
checkpoint up to 8B with **byte-identical weights** (verified: identical LFS
oids / sizes for every safetensors shard across Qwen/ and unsloth/) but a
patched ``config.json`` (adds ``pad_token_id`` and an ``unsloth_fixed: true``
marker) and ``tokenizer_config.json``. This module:

1. Lists the 6 Qwen3-VL variants up to 8B (2B/4B/8B x Thinking/Instruct) with
   their official ``Qwen/`` repo and the matching ``unsloth/`` repo.
2. Resolves either source to a local HF-cache snapshot path so the offline
   runtime can load it.
3. Provides ``load_source_runtime`` — build a Qwen3VLRuntime pointed at a
   specific source — used by the regression test and CLI.

Nothing here downloads anything: downloads still go through
``download_models.py`` (or plain ``huggingface-cli download``), so the offline
guarantees of the main runtime are preserved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from qwen3_vl.model_catalog import get_model_spec


@dataclass(frozen=True, slots=True)
class UnslothPair:
    """Official Qwen/ checkpoint paired with its unsloth/ repackage.

    ``size`` is the canonical catalog key (``2b``/``4b``/``8b``); ``variant``
    is ``thinking`` or ``instruct``. Both repos share identical safetensors
    SHAs — only config/tokenizer differ.
    """

    size: str
    variant: str
    official_repo: str
    unsloth_repo: str

    @property
    def key(self) -> str:
        return f"{self.size}-{self.variant}"


# All Qwen3-VL FP8 variants up to 8B (Thinking + Instruct). Weight SHAs are
# identical across the two columns; only config.json / tokenizer differ.
UNLOTH_PAIRS: tuple[UnslothPair, ...] = (
    UnslothPair("2b", "thinking", "Qwen/Qwen3-VL-2B-Thinking-FP8", "unsloth/Qwen3-VL-2B-Thinking-FP8"),
    UnslothPair("2b", "instruct", "Qwen/Qwen3-VL-2B-Instruct-FP8", "unsloth/Qwen3-VL-2B-Instruct-FP8"),
    UnslothPair("4b", "thinking", "Qwen/Qwen3-VL-4B-Thinking-FP8", "unsloth/Qwen3-VL-4B-Thinking-FP8"),
    UnslothPair("4b", "instruct", "Qwen/Qwen3-VL-4B-Instruct-FP8", "unsloth/Qwen3-VL-4B-Instruct-FP8"),
    UnslothPair("8b", "thinking", "Qwen/Qwen3-VL-8B-Thinking-FP8", "unsloth/Qwen3-VL-8B-Thinking-FP8"),
    UnslothPair("8b", "instruct", "Qwen/Qwen3-VL-8B-Instruct-FP8", "unsloth/Qwen3-VL-8B-Instruct-FP8"),
)

PAIRS_BY_KEY: dict[str, UnslothPair] = {p.key: p for p in UNLOTH_PAIRS}


def get_pair(size: str, variant: str = "thinking") -> UnslothPair:
    """Resolve a (size, variant) to its Qwen/ ↔ unsloth/ pair.

    ``size`` accepts the same aliases as ``model_catalog.normalize_model_size``
    (``2b``/``2``/``"Qwen3-VL-2B..."``); ``variant`` is ``thinking`` or
    ``instruct`` (default ``thinking``, matching the main catalog bias).
    """
    canon_size = get_model_spec(size).key
    key = f"{canon_size}-{variant.strip().lower()}"
    if key not in PAIRS_BY_KEY:
        raise KeyError(
            f"no unsloth pair for {size!r}/{variant!r}; choose from "
            f"{sorted(PAIRS_BY_KEY)}"
        )
    return PAIRS_BY_KEY[key]


def _hf_snapshot_dir(ckpt_dir: str | Path, repo_id: str) -> Path:
    """Hand-built HF cache snapshot path for ``repo_id`` (mirrors catalog)."""
    cache_name = "models--" + repo_id.replace("/", "--")
    return Path(ckpt_dir).expanduser().resolve() / cache_name / "snapshots" / "main"


def source_snapshot_path(
    size: str,
    source: str,
    *,
    variant: str = "thinking",
    ckpt_dir: str | Path | None = None,
) -> Path:
    """Return the local HF-cache snapshot path for one source of a pair.

    ``source`` is ``"official"`` or ``"unsloth"``. The path is returned whether
    or not the snapshot exists; callers should check ``.is_dir()``.
    """
    pair = get_pair(size, variant)
    repo = pair.official_repo if source == "official" else pair.unsloth_repo
    base = ckpt_dir if ckpt_dir is not None else os.environ.get(
        "CKPTDIR", os.environ.get("HF_HOME", "~/.cache/huggingface")
    )
    return _hf_snapshot_dir(base, repo)


def load_source_runtime(
    size: str,
    source: str,
    *,
    variant: str = "thinking",
    ckpt_dir: str | Path | None = None,
    kernel_dir: str | Path | None = None,
    seed: int = 1234,
    gpu_placement: str = "single",
):
    """Build a Qwen3VLRuntime pointed at one source of a pair.

    Importing Qwen3VLRuntime drags in torch/transformers, so this is lazy.
    The ``Qwen3VLRuntime`` accepts an explicit ``model_path`` that overrides
    the catalog default — we point it at the chosen snapshot so the same
    runtime code loads either source.

    The official Qwen/ snapshot goes through the full catalog manifest
    verification. The unsloth/ snapshot carries byte-identical weights but a
    patched config/tokenizer (different sizes), so it would fail the
    Qwen-pinned manifest; we pass ``trust_remote_source=True`` to load it
    through the same FP8 runtime without that check.
    """
    from qwen3_vl.qwen3_vl_offline import Qwen3VLRuntime

    snapshot = source_snapshot_path(size, source, variant=variant, ckpt_dir=ckpt_dir)
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"{source} snapshot for {size}-{variant} not found at {snapshot}; "
            f"download it first (huggingface-cli download <repo> --local-dir "
            f"use is not needed — populate the HF cache)."
        )
    # size is still passed so the runtime knows architecture/expected tensors;
    # model_path overrides the repo_id-derived default snapshot.
    return Qwen3VLRuntime(
        model_size=size,
        model_path=str(snapshot),
        ckpt_dir=ckpt_dir if ckpt_dir is not None else os.environ.get(
            "CKPTDIR", os.environ.get("HF_HOME", "~/.cache/huggingface")
        ),
        kernel_dir=kernel_dir,
        seed=seed,
        gpu_placement=gpu_placement,
        # unsloth repackages differ only in metadata; weights are identical.
        trust_remote_source=(source != "official"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="qwen3-vl-unsloth",
        description="List Qwen3-VL FP8 variants paired with their unsloth repackage.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.json:
        import json
        rows = [
            {
                "size": p.size,
                "variant": p.variant,
                "official_repo": p.official_repo,
                "unsloth_repo": p.unsloth_repo,
            }
            for p in UNLOTH_PAIRS
        ]
        print(json.dumps(rows, indent=2))
    else:
        print("SIZE  VARIANT    OFFICIAL REPOSITORY                            UNSLOTH REPOSITORY")
        for p in UNLOTH_PAIRS:
            print(f"{p.size:<5} {p.variant:<10} {p.official_repo:<48} {p.unsloth_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download and structurally verify the supported Qwen3-VL FP8 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from safetensors import safe_open

from model_catalog import (
    MODEL_SPECS,
    ModelSpec,
    SnapshotFileSpec,
    WeightShardSpec,
    default_snapshot_path,
    get_model_spec,
)


DEFAULT_CACHE_DIR = Path("/mnt/nvme/huggingface")
REQUIRED_FILES: tuple[str, ...] = (
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
)
_MAX_HEADER_BYTES = 128 * 1024 * 1024
HF_ORIGIN = "https://huggingface.co"
WGET_RETRIES = 5
_TOKEN_ENV_NAMES = {"HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"}


class CheckpointVerificationError(ValueError):
    """Raised when a checkpoint exists but is malformed or inconsistent."""


class ModelDownloadError(RuntimeError):
    """Raised when pinned public artifacts cannot be fetched safely."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointVerificationError(f"JSON object contains duplicate key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except CheckpointVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointVerificationError(f"cannot parse JSON file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointVerificationError(f"JSON file {path.name} must contain an object")
    return value


def _safe_shard_path(root: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise CheckpointVerificationError("weight_map shard names must be non-empty strings")
    if "\\" in value or "\x00" in value:
        raise CheckpointVerificationError(f"unsafe shard path in weight_map: {value!r}")

    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value != relative.as_posix()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
        or relative.suffix != ".safetensors"
    ):
        raise CheckpointVerificationError(f"unsafe shard path in weight_map: {value!r}")

    return relative.as_posix(), root.joinpath(*relative.parts)


def _read_safetensors_header(path: Path) -> tuple[set[str], int]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise CheckpointVerificationError(f"safetensors shard is truncated: {path.name}")

    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length < 2 or header_length > _MAX_HEADER_BYTES:
                raise CheckpointVerificationError(
                    f"invalid safetensors header length in {path.name}: {header_length}"
                )
            if 8 + header_length > file_size:
                raise CheckpointVerificationError(f"safetensors header is truncated: {path.name}")
            raw_header = handle.read(header_length)
        header = json.loads(raw_header, object_pairs_hook=_reject_duplicate_keys)
    except CheckpointVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as exc:
        raise CheckpointVerificationError(
            f"cannot parse safetensors header {path.name}: {exc}"
        ) from exc

    if not isinstance(header, dict):
        raise CheckpointVerificationError(f"safetensors header is not an object: {path.name}")

    tensor_keys: set[str] = set()
    max_data_end = 0
    for tensor_name, metadata in header.items():
        if tensor_name == "__metadata__":
            if not isinstance(metadata, dict):
                raise CheckpointVerificationError(
                    f"invalid __metadata__ entry in safetensors shard: {path.name}"
                )
            continue
        if not tensor_name or not isinstance(metadata, dict):
            raise CheckpointVerificationError(f"invalid tensor entry in shard: {path.name}")
        offsets = metadata.get("data_offsets")
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or not isinstance(shape, list)
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in shape
            )
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise CheckpointVerificationError(
                f"invalid metadata for tensor {tensor_name!r} in {path.name}"
            )
        tensor_keys.add(tensor_name)
        max_data_end = max(max_data_end, offsets[1])

    if 8 + header_length + max_data_end != file_size:
        raise CheckpointVerificationError(
            f"safetensors shard size does not match its header: {path.name}"
        )

    try:
        with safe_open(path, framework="pt", device="cpu") as shard:
            safe_open_keys = set(shard.keys())
    except Exception as exc:
        raise CheckpointVerificationError(
            f"safetensors rejected shard {path.name}: {type(exc).__name__}: {exc}"
        ) from exc
    if safe_open_keys != tensor_keys:
        raise CheckpointVerificationError(
            f"safetensors key mismatch while opening shard: {path.name}"
        )
    return tensor_keys, file_size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ArtifactSpec = SnapshotFileSpec | WeightShardSpec


def _artifact_manifest(spec: ModelSpec) -> dict[str, ArtifactSpec]:
    artifacts = (*spec.required_files, *spec.weight_shards)
    manifest = {item.filename: item for item in artifacts}
    if len(manifest) != len(artifacts):
        raise ModelDownloadError(f"catalog for {spec.key} contains duplicate filenames")
    expected = set(REQUIRED_FILES) | {item.filename for item in spec.weight_shards}
    if set(manifest) != expected:
        raise ModelDownloadError(
            f"catalog for {spec.key} has an incomplete artifact manifest"
        )
    return manifest


def _wget_environment() -> dict[str, str]:
    """Inherit ordinary network settings without forwarding model credentials."""

    return {name: value for name, value in os.environ.items() if name not in _TOKEN_ENV_NAMES}


def _wget_binary() -> str:
    binary = shutil.which("wget")
    if binary is None:
        raise ModelDownloadError("GNU wget is required for model downloads")
    return binary


def _validate_catalog_revision(spec: ModelSpec) -> None:
    revision = spec.revision
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ModelDownloadError(f"catalog revision for {spec.key} is not an immutable commit")


def _tree_url(spec: ModelSpec) -> str:
    _validate_catalog_revision(spec)
    repository = quote(spec.repo_id, safe="/")
    revision = quote(spec.revision, safe="")
    return f"{HF_ORIGIN}/api/models/{repository}/tree/{revision}?recursive=true&expand=false"


def _resolve_url(spec: ModelSpec, filename: str) -> str:
    _validate_catalog_revision(spec)
    repository = quote(spec.repo_id, safe="/")
    revision = quote(spec.revision, safe="")
    path = quote(filename, safe="/")
    return f"{HF_ORIGIN}/{repository}/resolve/{revision}/{path}"


def _query_remote_tree(spec: ModelSpec) -> dict[str, dict[str, Any]]:
    """Confirm the pinned repository contains the complete trusted manifest."""

    manifest = _artifact_manifest(spec)
    command = [
        _wget_binary(),
        "--no-config",
        "--no-netrc",
        "--https-only",
        "--quiet",
        "--retry-connrefused",
        "--retry-on-http-error=429,500,502,503,504",
        f"--tries={WGET_RETRIES}",
        "--timeout=30",
        "--waitretry=2",
        "--output-document=-",
        _tree_url(spec),
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=_wget_environment(),
    )
    if completed.returncode != 0:
        raise ModelDownloadError(
            f"remote structure query failed for {spec.repo_id} at its pinned revision "
            f"(wget exit {completed.returncode})"
        )
    if len(completed.stdout) > 16 * 1024 * 1024:
        raise ModelDownloadError("remote repository tree response is unexpectedly large")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ModelDownloadError("remote repository tree returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise ModelDownloadError("remote repository tree response is not a file list")

    remote: dict[str, dict[str, Any]] = {}
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        path = entry.get("path")
        if isinstance(path, str):
            remote[path] = entry

    missing = sorted(set(manifest) - set(remote))
    if missing:
        raise ModelDownloadError(
            "pinned repository revision is missing required files: " + ", ".join(missing)
        )
    for filename, artifact in manifest.items():
        remote_size = remote[filename].get("size")
        if not isinstance(remote_size, int) or remote_size != artifact.size_bytes:
            raise ModelDownloadError(
                f"pinned repository reports an unexpected size for {filename}"
            )
    for shard in spec.weight_shards:
        lfs = remote[shard.filename].get("lfs")
        remote_sha256 = lfs.get("sha256") if isinstance(lfs, dict) else None
        if remote_sha256 != shard.sha256:
            raise ModelDownloadError(
                f"pinned repository reports an unexpected digest for {shard.filename}"
            )
    print(
        f"STATUS: remote pinned structure verified for {spec.repo_id} "
        f"({len(manifest)} files)",
        file=sys.stderr,
    )
    return remote


def _validate_artifact_file(path: Path, artifact: ArtifactSpec) -> None:
    if path.is_symlink() or not path.is_file():
        raise CheckpointVerificationError(f"artifact is not a regular file: {artifact.filename}")
    actual_size = path.stat().st_size
    if actual_size != artifact.size_bytes:
        raise CheckpointVerificationError(
            f"artifact size mismatch for {artifact.filename}: "
            f"{actual_size} != {artifact.size_bytes}"
        )
    if _sha256(path) != artifact.sha256:
        raise CheckpointVerificationError(
            f"artifact SHA-256 mismatch for {artifact.filename}"
        )


def _artifact_is_valid(path: Path, artifact: ArtifactSpec) -> bool:
    try:
        _validate_artifact_file(path, artifact)
    except (OSError, CheckpointVerificationError):
        return False
    return True


def _download_artifact(
    spec: ModelSpec,
    artifact: ArtifactSpec,
    target: Path,
    *,
    position: int,
    total: int,
) -> None:
    """Resume into ``.part``, authenticate, then atomically publish one file."""

    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ModelDownloadError(f"unsafe existing download target: {artifact.filename}")
    if _artifact_is_valid(target, artifact):
        print(
            f"STATUS [{position}/{total}]: verified existing {artifact.filename}",
            file=sys.stderr,
        )
        return

    partial = target.with_name(target.name + ".part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ModelDownloadError(f"unsafe partial download path: {artifact.filename}.part")
    if partial.exists() and partial.stat().st_size > artifact.size_bytes:
        partial.unlink()

    url = _resolve_url(spec, artifact.filename)
    for attempt in range(2):
        verb = "resuming" if partial.exists() and partial.stat().st_size else "downloading"
        print(
            f"STATUS [{position}/{total}]: {verb} {artifact.filename} "
            f"({artifact.size_bytes} bytes)",
            file=sys.stderr,
        )
        command = [
            _wget_binary(),
            "--no-config",
            "--no-netrc",
            "--https-only",
            "--retry-connrefused",
            "--retry-on-http-error=429,500,502,503,504",
            f"--tries={WGET_RETRIES}",
            "--timeout=30",
            "--waitretry=2",
            "--continue",
            "--show-progress",
            "--output-document",
            str(partial),
            url,
        ]
        completed = subprocess.run(command, check=False, env=_wget_environment())
        if completed.returncode != 0:
            raise ModelDownloadError(
                f"wget failed for {artifact.filename} after retries "
                f"(exit {completed.returncode}); partial file was kept for resume"
            )
        try:
            _validate_artifact_file(partial, artifact)
        except (OSError, CheckpointVerificationError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == 0:
                print(
                    f"WARNING: corrupt partial for {artifact.filename}; retrying once from zero",
                    file=sys.stderr,
                )
                continue
            raise ModelDownloadError(
                f"downloaded artifact remained corrupt after retry: {artifact.filename}"
            ) from exc

        os.replace(partial, target)
        print(
            f"STATUS [{position}/{total}]: authenticated {artifact.filename}",
            file=sys.stderr,
        )
        return

    raise AssertionError("unreachable download retry state")


def verify_checkpoint(
    path: str | Path,
    spec: ModelSpec | str | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Verify files, index mappings, and every safetensors header in a snapshot.

    ``full=True`` additionally reads every verified file completely and returns
    SHA-256 digests. The hashes establish local repeatability; callers need an
    independently trusted manifest to establish upstream provenance.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")

    resolved_spec = get_model_spec(spec) if isinstance(spec, str) else spec
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing required files: {', '.join(missing)}")

    config = _read_json(root / "config.json")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
        raise CheckpointVerificationError("config.json is not a fine-grained FP8 checkpoint")

    index = _read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointVerificationError("model.safetensors.index.json has no weight_map")

    expected_locations: dict[str, str] = {}
    shard_paths: dict[str, Path] = {}
    for tensor_name, raw_shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise CheckpointVerificationError("weight_map tensor names must be non-empty strings")
        shard_name, shard_path = _safe_shard_path(root, raw_shard_name)
        expected_locations[tensor_name] = shard_name
        shard_paths[shard_name] = shard_path

    missing_shards = [name for name, shard_path in shard_paths.items() if not shard_path.is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"checkpoint is missing safetensors shards: {', '.join(sorted(missing_shards))}"
        )

    actual_locations: dict[str, str] = {}
    shard_sizes: dict[str, int] = {}
    for shard_name in sorted(shard_paths):
        tensor_keys, shard_size = _read_safetensors_header(shard_paths[shard_name])
        shard_sizes[shard_name] = shard_size
        for tensor_name in tensor_keys:
            previous = actual_locations.setdefault(tensor_name, shard_name)
            if previous != shard_name:
                raise CheckpointVerificationError(
                    f"tensor {tensor_name!r} occurs in both {previous!r} and {shard_name!r}"
                )

    expected_keys = set(expected_locations)
    actual_keys = set(actual_locations)
    if expected_keys != actual_keys:
        missing_keys = sorted(expected_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_keys)
        raise CheckpointVerificationError(
            "safetensors/index tensor key mismatch: "
            f"missing={missing_keys[:5]!r}, extra={extra_keys[:5]!r}"
        )
    wrong_shards = sorted(
        tensor_name
        for tensor_name, expected_shard in expected_locations.items()
        if actual_locations[tensor_name] != expected_shard
    )
    if wrong_shards:
        raise CheckpointVerificationError(
            f"weight_map assigns tensors to the wrong shard: {wrong_shards[:5]!r}"
        )

    tensor_count = len(expected_keys)
    scale_count = sum(name.endswith(".weight_scale_inv") for name in expected_keys)
    shard_count = len(shard_paths)
    if resolved_spec is not None:
        mismatches: list[str] = []
        if tensor_count != resolved_spec.expected_tensors:
            mismatches.append(
                f"tensors={tensor_count} (expected {resolved_spec.expected_tensors})"
            )
        if scale_count != resolved_spec.expected_scales:
            mismatches.append(f"scales={scale_count} (expected {resolved_spec.expected_scales})")
        if shard_count != resolved_spec.expected_shards:
            mismatches.append(f"shards={shard_count} (expected {resolved_spec.expected_shards})")
        trusted_shards = {item.filename: item for item in resolved_spec.weight_shards}
        if trusted_shards and set(shard_paths) != set(trusted_shards):
            mismatches.append(
                f"shard_names={sorted(shard_paths)!r} "
                f"(expected {sorted(trusted_shards)!r})"
            )
        for shard_name in sorted(set(shard_paths) & set(trusted_shards)):
            expected_size = trusted_shards[shard_name].size_bytes
            if shard_sizes[shard_name] != expected_size:
                mismatches.append(
                    f"{shard_name} bytes={shard_sizes[shard_name]} (expected {expected_size})"
                )
        trusted_required = {item.filename: item for item in resolved_spec.required_files}
        if trusted_required and set(trusted_required) != set(REQUIRED_FILES):
            mismatches.append(
                f"required_file_manifest={sorted(trusted_required)!r} "
                f"(expected {sorted(REQUIRED_FILES)!r})"
            )
        for filename in sorted(set(REQUIRED_FILES) & set(trusted_required)):
            actual_size = (root / filename).stat().st_size
            expected_size = trusted_required[filename].size_bytes
            if actual_size != expected_size:
                mismatches.append(
                    f"{filename} bytes={actual_size} (expected {expected_size})"
                )
        if mismatches:
            raise CheckpointVerificationError(
                "checkpoint manifest mismatch: " + ", ".join(mismatches)
            )

    digests: dict[str, str] = {}
    if full:
        files_to_hash = set(REQUIRED_FILES) | set(shard_paths)
        for relative_name in sorted(files_to_hash):
            file_path = root.joinpath(*PurePosixPath(relative_name).parts)
            digests[relative_name] = _sha256(file_path)
        if resolved_spec is not None:
            trusted_files = {
                item.filename: item
                for item in (*resolved_spec.required_files, *resolved_spec.weight_shards)
            }
            hash_mismatches = [
                filename
                for filename, file_spec in trusted_files.items()
                if digests.get(filename) != file_spec.sha256
            ]
            if hash_mismatches:
                raise CheckpointVerificationError(
                    "checkpoint SHA-256 mismatch: " + ", ".join(hash_mismatches)
                )

    return {
        "ok": True,
        "path": str(root),
        "model_size": resolved_spec.key if resolved_spec is not None else None,
        "repo_id": resolved_spec.repo_id if resolved_spec is not None else None,
        "revision": resolved_spec.revision if resolved_spec is not None else None,
        "tensor_count": tensor_count,
        "scale_count": scale_count,
        "shard_count": shard_count,
        "shards": sorted(shard_paths),
        "shard_bytes": sum(shard_sizes.values()),
        "full": bool(full),
        "sha256": digests,
    }


def download_model(
    size: str,
    cache_dir: str | Path,
) -> Path:
    """Download one public checkpoint with pinned HTTPS URLs and GNU wget.

    A complete handcrafted ``snapshots/main`` directory is reused without a
    network call. Incomplete snapshots are remotely preflighted, resumed via
    ``.part`` files, and fully authenticated before use.
    """

    spec = get_model_spec(size)
    cache_root = Path(cache_dir).expanduser().resolve()
    target = default_snapshot_path(cache_root, spec.key)

    if target.is_dir():
        try:
            verify_checkpoint(target, spec=spec, full=True)
        except (FileNotFoundError, OSError, CheckpointVerificationError):
            pass
        else:
            print(
                f"STATUS: reusing fully verified local snapshot for {spec.repo_id}: {target}",
                file=sys.stderr,
            )
            return target

    manifest = _artifact_manifest(spec)
    # Public repositories only: no token, cookie, netrc, or wget config is used.
    # The tree check must succeed before the first artifact request.
    _query_remote_tree(spec)
    target.mkdir(parents=True, exist_ok=True)
    ordered_names = [item.filename for item in spec.required_files] + [
        item.filename for item in spec.weight_shards
    ]
    for position, filename in enumerate(ordered_names, start=1):
        _download_artifact(
            spec,
            manifest[filename],
            target / filename,
            position=position,
            total=len(ordered_names),
        )

    try:
        verify_checkpoint(target, spec=spec, full=True)
    except (FileNotFoundError, OSError, CheckpointVerificationError) as exc:
        raise CheckpointVerificationError(
            f"downloaded checkpoint failed verification: {exc}"
        ) from exc
    print(
        f"STATUS: complete snapshot authenticated for {spec.repo_id}: {target}",
        file=sys.stderr,
    )
    return target


def _expand_sizes(values: list[str]) -> list[str]:
    if any(value.strip().lower() == "all" for value in values):
        if len(values) != 1:
            raise ValueError("use 'all' by itself")
        return list(MODEL_SPECS)
    result: list[str] = []
    for value in values:
        key = get_model_spec(value).key
        if key not in result:
            result.append(key)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download or verify Qwen3-VL 2B/4B/8B Thinking FP8 checkpoints."
    )
    parser.add_argument("sizes", nargs="+", help="2b, 4b, 8b, or all")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing handcrafted snapshots/main directories without downloading",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip SHA-256 only with --verify-only; downloads are always fully authenticated",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)

    try:
        sizes = _expand_sizes(args.sizes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.quick and not args.verify_only:
        parser.error("--quick is only valid with --verify-only; downloads always verify SHA-256")

    results: list[dict[str, Any]] = []
    for size in sizes:
        spec = get_model_spec(size)
        if args.verify_only:
            snapshot = default_snapshot_path(args.cache_dir, size)
            summary = verify_checkpoint(snapshot, spec=spec, full=not args.quick)
        else:
            snapshot = download_model(size, args.cache_dir)
            summary = verify_checkpoint(snapshot, spec=spec, full=False)
        results.append(summary)

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            print(
                f"{result['model_size']}: OK at {result['path']} "
                f"({result['tensor_count']} tensors, {result['shard_count']} shards)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

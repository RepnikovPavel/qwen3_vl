from __future__ import annotations

import argparse
import hashlib
import json
import operator
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch


SCHEMA_VERSION = 1
INPUT_TENSOR_NAMES = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
)
TENSOR_ENCODING = "torch-contiguous-little-endian-v1"
TOKEN_ENCODING = "qwen3-token-ids-little-endian-uint64-v1"
TOKEN_MAGIC = b"Q3VLPTOK"


def _little_endian_tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.layout != torch.strided:
        raise TypeError(f"tensor layout must be strided, got {tensor.layout}")
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    width = value.element_size()
    if sys.byteorder == "little" or width == 1 or not raw:
        return raw
    if value.is_complex():
        width //= 2
    return b"".join(
        raw[index : index + width][::-1] for index in range(0, len(raw), width)
    )


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, object]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")
    payload = _little_endian_tensor_bytes(tensor)
    return {
        "encoding": TENSOR_ENCODING,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "byte_length": len(payload),
        "bytes_sha256": hashlib.sha256(payload).hexdigest(),
    }


def fingerprint_tensors(
    mapping: Mapping[str, object],
    names: Sequence[str] = INPUT_TENSOR_NAMES,
) -> dict[str, dict[str, object] | None]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"expected a mapping, got {type(mapping).__name__}")
    result: dict[str, dict[str, object] | None] = {}
    for name in names:
        value = mapping.get(name)
        result[name] = None if value is None else tensor_fingerprint(value)
    return result


def _normalize_token_ids(token_ids: Iterable[int] | torch.Tensor) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError(
                f"token tensor must be one-dimensional, got shape {list(token_ids.shape)}"
            )
        values: Iterable[object] = token_ids.detach().to(device="cpu").tolist()
    elif isinstance(token_ids, (str, bytes, bytearray)):
        raise TypeError("token IDs must be an iterable of integers")
    else:
        values = token_ids
    normalized: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise TypeError(f"token ID at index {index} is boolean")
        try:
            token_id = operator.index(value)
        except TypeError as exc:
            raise TypeError(
                f"token ID at index {index} is not an integer: {value!r}"
            ) from exc
        if not 0 <= token_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"token ID at index {index} is outside uint64: {token_id}")
        normalized.append(token_id)
    return normalized


def encode_token_ids(token_ids: Iterable[int] | torch.Tensor) -> bytes:
    normalized = _normalize_token_ids(token_ids)
    header = struct.pack("<8sHQ", TOKEN_MAGIC, SCHEMA_VERSION, len(normalized))
    payload = b"".join(struct.pack("<Q", token_id) for token_id in normalized)
    return header + payload


def token_ids_sha256(token_ids: Iterable[int] | torch.Tensor) -> str:
    return hashlib.sha256(encode_token_ids(token_ids)).hexdigest()


def compare_token_sequences(
    reference: Iterable[int] | torch.Tensor,
    candidate: Iterable[int] | torch.Tensor,
) -> dict[str, object]:
    reference_ids = _normalize_token_ids(reference)
    candidate_ids = _normalize_token_ids(candidate)
    limit = min(len(reference_ids), len(candidate_ids))
    common_prefix = 0
    while (
        common_prefix < limit
        and reference_ids[common_prefix] == candidate_ids[common_prefix]
    ):
        common_prefix += 1
    exact = reference_ids == candidate_ids
    first_mismatch = None
    if not exact:
        first_mismatch = {
            "index": common_prefix,
            "reference_token_id": (
                reference_ids[common_prefix]
                if common_prefix < len(reference_ids)
                else None
            ),
            "candidate_token_id": (
                candidate_ids[common_prefix]
                if common_prefix < len(candidate_ids)
                else None
            ),
        }
    return {
        "exact": exact,
        "lengths": {
            "reference": len(reference_ids),
            "candidate": len(candidate_ids),
        },
        "common_prefix": common_prefix,
        "first_mismatch": first_mismatch,
    }


def build_parity_artifact(
    inputs: Mapping[str, object],
    continuation_token_ids: Iterable[int] | torch.Tensor,
    *,
    metadata: Mapping[str, object] | None = None,
    include_token_ids: bool = True,
) -> dict[str, object]:
    normalized_ids = _normalize_token_ids(continuation_token_ids)
    continuation: dict[str, object] = {
        "encoding": TOKEN_ENCODING,
        "length": len(normalized_ids),
        "sha256": token_ids_sha256(normalized_ids),
    }
    if include_token_ids:
        continuation["token_ids"] = normalized_ids
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "input_fingerprints": fingerprint_tensors(inputs),
        "continuation": continuation,
    }
    if metadata is not None:
        artifact["metadata"] = dict(metadata)
    return artifact


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _validate_tensor_fingerprint(name: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"input fingerprint {name!r} must be an object or null")
    if value.get("encoding") != TENSOR_ENCODING:
        raise ValueError(f"input fingerprint {name!r} has unsupported encoding")
    shape = value.get("shape")
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape
    ):
        raise ValueError(f"input fingerprint {name!r} has invalid shape")
    if not isinstance(value.get("dtype"), str) or not value["dtype"]:
        raise ValueError(f"input fingerprint {name!r} has invalid dtype")
    byte_length = value.get("byte_length")
    if (
        isinstance(byte_length, bool)
        or not isinstance(byte_length, int)
        or byte_length < 0
    ):
        raise ValueError(f"input fingerprint {name!r} has invalid byte_length")
    if not _is_sha256(value.get("bytes_sha256")):
        raise ValueError(f"input fingerprint {name!r} has invalid bytes_sha256")


def _validated_artifact(
    artifact: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], list[int] | None]:
    if not isinstance(artifact, Mapping):
        raise ValueError("artifact must be a JSON object")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"artifact schema_version must be {SCHEMA_VERSION}")
    fingerprints = artifact.get("input_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ValueError("artifact input_fingerprints must be an object")
    missing = [name for name in INPUT_TENSOR_NAMES if name not in fingerprints]
    if missing:
        raise ValueError(
            f"artifact is missing input fingerprints: {', '.join(missing)}"
        )
    normalized_fingerprints: dict[str, object] = {}
    for name in INPUT_TENSOR_NAMES:
        value = fingerprints[name]
        _validate_tensor_fingerprint(name, value)
        normalized_fingerprints[name] = value
    continuation = artifact.get("continuation")
    if not isinstance(continuation, Mapping):
        raise ValueError("artifact continuation must be an object")
    if continuation.get("encoding") != TOKEN_ENCODING:
        raise ValueError("artifact continuation has unsupported encoding")
    length = continuation.get("length")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("artifact continuation has invalid length")
    digest = continuation.get("sha256")
    if not _is_sha256(digest):
        raise ValueError("artifact continuation has invalid sha256")
    normalized_continuation = {
        "encoding": TOKEN_ENCODING,
        "length": length,
        "sha256": digest,
    }
    token_ids = None
    if "token_ids" in continuation:
        raw_ids = continuation["token_ids"]
        if not isinstance(raw_ids, list):
            raise ValueError("artifact continuation token_ids must be an array")
        token_ids = _normalize_token_ids(raw_ids)
        if len(token_ids) != length:
            raise ValueError("artifact continuation length does not match token_ids")
        if token_ids_sha256(token_ids) != digest:
            raise ValueError("artifact continuation sha256 does not match token_ids")
    return normalized_fingerprints, normalized_continuation, token_ids


def _fingerprint_differences(
    reference: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    differences: dict[str, object] = {}
    for name in INPUT_TENSOR_NAMES:
        if reference[name] != candidate[name]:
            differences[name] = {
                "reference": reference[name],
                "candidate": candidate[name],
            }
    return differences


def compare_artifacts(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    require_token_ids: bool = False,
) -> dict[str, object]:
    reference_inputs, reference_continuation, reference_ids = _validated_artifact(
        reference
    )
    candidate_inputs, candidate_continuation, candidate_ids = _validated_artifact(
        candidate
    )
    if require_token_ids and (reference_ids is None or candidate_ids is None):
        raise ValueError("both artifacts must contain continuation token_ids")
    input_differences = _fingerprint_differences(reference_inputs, candidate_inputs)
    digest_match = (
        reference_continuation["encoding"] == candidate_continuation["encoding"]
        and reference_continuation["length"] == candidate_continuation["length"]
        and reference_continuation["sha256"] == candidate_continuation["sha256"]
    )
    token_comparison = None
    if reference_ids is not None and candidate_ids is not None:
        token_comparison = compare_token_sequences(reference_ids, candidate_ids)
        continuation_match = digest_match and bool(token_comparison["exact"])
    else:
        continuation_match = digest_match
    input_match = not input_differences
    return {
        "match": input_match and continuation_match,
        "input_match": input_match,
        "input_differences": input_differences,
        "continuation_match": continuation_match,
        "continuation_digest_match": digest_match,
        "continuation": {
            "reference": reference_continuation,
            "candidate": candidate_continuation,
            "token_comparison": token_comparison,
        },
    }


def _read_artifact(path: str) -> dict[str, object]:
    if path == "-":
        value = json.load(sys.stdin)
    else:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact {path!r} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Qwen3 parity artifacts")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--require-token-ids", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reference == "-" and args.candidate == "-":
        print("parity error: only one artifact can be read from stdin", file=sys.stderr)
        return 2
    try:
        result = compare_artifacts(
            _read_artifact(args.reference),
            _read_artifact(args.candidate),
            require_token_ids=args.require_token_ids,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"parity error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        separators=(",", ":") if args.compact else None,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

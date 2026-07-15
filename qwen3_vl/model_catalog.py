#!/usr/bin/env python3
"""Small, explicit catalog of the supported Qwen3-VL FP8 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WeightShardSpec:
    """Trusted metadata for one upstream safetensors weight shard."""

    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SnapshotFileSpec:
    """Trusted metadata for one non-weight file in a pinned snapshot."""

    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable metadata used to locate and validate one supported model."""

    key: str
    parameters_b: int
    repo_id: str
    cache_name: str
    expected_tensors: int
    expected_scales: int
    expected_shards: int
    revision: str = "main"
    weight_shards: tuple[WeightShardSpec, ...] = ()
    required_files: tuple[SnapshotFileSpec, ...] = ()

    @property
    def immutable_revision(self) -> str:
        """Descriptive alias emphasizing that catalog revisions are pinned."""

        return self.revision


_COMMON_REQUIRED_FILES = {
    ".gitattributes": SnapshotFileSpec(
        filename=".gitattributes",
        size_bytes=1_519,
        sha256="11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "chat_template.json": SnapshotFileSpec(
        filename="chat_template.json",
        size_bytes=5_410,
        sha256="efad756db13feb8e8c5fe1341713bf18b82e91916b978d36b7c043f2b1bedf5f",
    ),
    "generation_config.json": SnapshotFileSpec(
        filename="generation_config.json",
        size_bytes=242,
        sha256="a01c0121bc17301c42e8fd1895c79c7558f7321c55110d1037478c3f3cec2049",
    ),
    "preprocessor_config.json": SnapshotFileSpec(
        filename="preprocessor_config.json",
        size_bytes=336,
        sha256="6a970fd06f30e6943b3e2c14d5d3b42d49b06cf99b99103d56689bef462d90f8",
    ),
    "tokenizer.json": SnapshotFileSpec(
        filename="tokenizer.json",
        size_bytes=10_179_867,
        sha256="ba85e4e5222d9f53d4bd00b303ef7e9743c8ac3d07e3c23f8498dbe17baa9a2d",
    ),
    "tokenizer_config.json": SnapshotFileSpec(
        filename="tokenizer_config.json",
        size_bytes=10_781,
        sha256="7b501e639b4d107a23effbe30390ee33d553f722467f7ac8e2744d7ff5d3a7d5",
    ),
    "video_preprocessor_config.json": SnapshotFileSpec(
        filename="video_preprocessor_config.json",
        size_bytes=331,
        sha256="e203bc065dfcd75226838b8e937d624bec8f0eb6ef6630a397e9a675f2873ea6",
    ),
    "vocab.json": SnapshotFileSpec(
        filename="vocab.json",
        size_bytes=4_957_462,
        sha256="7a0cfa95c65792d7510205839f80cfd8a3c8f6b1fdad5132d95cee481800374d",
    ),
}


def _required_snapshot_files(
    *,
    readme_size: int,
    readme_sha256: str,
    config_size: int,
    config_sha256: str,
    index_size: int,
    index_sha256: str,
) -> tuple[SnapshotFileSpec, ...]:
    """Build the complete ordered manifest while sharing identical files."""

    return (
        _COMMON_REQUIRED_FILES[".gitattributes"],
        SnapshotFileSpec("README.md", readme_size, readme_sha256),
        _COMMON_REQUIRED_FILES["chat_template.json"],
        SnapshotFileSpec("config.json", config_size, config_sha256),
        _COMMON_REQUIRED_FILES["generation_config.json"],
        SnapshotFileSpec("model.safetensors.index.json", index_size, index_sha256),
        _COMMON_REQUIRED_FILES["preprocessor_config.json"],
        _COMMON_REQUIRED_FILES["tokenizer.json"],
        _COMMON_REQUIRED_FILES["tokenizer_config.json"],
        _COMMON_REQUIRED_FILES["video_preprocessor_config.json"],
        _COMMON_REQUIRED_FILES["vocab.json"],
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "2b": ModelSpec(
        key="2b",
        parameters_b=2,
        repo_id="Qwen/Qwen3-VL-2B-Thinking-FP8",
        cache_name="models--Qwen--Qwen3-VL-2B-Thinking-FP8",
        expected_tensors=822,
        expected_scales=196,
        expected_shards=1,
        revision="bc71e10812c1bba5532bd2eca46a4166f3b7fffd",
        weight_shards=(
            WeightShardSpec(
                filename="model-00001-of-00001.safetensors",
                size_bytes=3_468_553_776,
                sha256="327bc29770e4e1f1f73e14b849b0ed6856639d3fa11f4525aacf353a1394ab98",
            ),
        ),
        required_files=_required_snapshot_files(
            readme_size=10_431,
            readme_sha256="94d0ac4fb881d9ad279d564b49b75326ad4aa96f71fdb7a7f00c0b74483e994b",
            config_size=11_960,
            config_sha256="9d3050de5482d3ba017c945bea8cf94fc0bee7f0bd272e3abeeeb502774c9601",
            index_size=76_965,
            index_sha256="59beb5fdb25290b2efa4c7c16c3b081d45bff7635c5703caa89078ec71c68d46",
        ),
    ),
    "4b": ModelSpec(
        key="4b",
        parameters_b=4,
        repo_id="Qwen/Qwen3-VL-4B-Thinking-FP8",
        cache_name="models--Qwen--Qwen3-VL-4B-Thinking-FP8",
        expected_tensors=966,
        expected_scales=252,
        expected_shards=2,
        revision="219b8e195ea30e383c55c954278767990974bba9",
        weight_shards=(
            WeightShardSpec(
                filename="model-00001-of-00002.safetensors",
                size_bytes=5_366_863_440,
                sha256="c29f37dacc3e40f1579166d1f74bf406d2b2800fad149e9e8ac5219e801d4862",
            ),
            WeightShardSpec(
                filename="model-00002-of-00002.safetensors",
                size_bytes=654_372_016,
                sha256="be6fe6b2564e3211acf15427244662973e37c9b872cfdf130ed9d1917bb20dfa",
            ),
        ),
        required_files=_required_snapshot_files(
            readme_size=10_500,
            readme_sha256="7a39e7ad54461a451040e1630bb09c0b43e6237d4ac5d8ce118bd8d7f4e718b1",
            config_size=12_037,
            config_sha256="48bc97743d741920de3455f7a6f0a807b0c3ae692e662a0864f188ca528f92b2",
            index_size=91_517,
            index_sha256="94c659b1e64a2e3f23e83e6392ee0961152902e7871da74e867c1bd826e2a761",
        ),
    ),
    "8b": ModelSpec(
        key="8b",
        parameters_b=8,
        repo_id="Qwen/Qwen3-VL-8B-Thinking-FP8",
        cache_name="models--Qwen--Qwen3-VL-8B-Thinking-FP8",
        expected_tensors=1002,
        expected_scales=252,
        expected_shards=2,
        revision="a6638e84662f85a17bb8688224541e153d4f6c71",
        weight_shards=(
            WeightShardSpec(
                filename="model-00001-of-00002.safetensors",
                size_bytes=5_363_407_552,
                sha256="48c85cea60c7b445addf0ae150041b97298fb20a63e2149d9612ede9f0a1e2db",
            ),
            WeightShardSpec(
                filename="model-00002-of-00002.safetensors",
                size_bytes=5_226_891_960,
                sha256="642f60b06b2cf0896ef177fdfd295eb98aec998a0f4b9ae06422a3789ac20f3c",
            ),
        ),
        required_files=_required_snapshot_files(
            readme_size=10_501,
            readme_sha256="d998a4eb874a06c2cb6d59c8a3e5ae3d0d31f94a1fee3ed3a2cc2165bd74abf2",
            config_size=12_005,
            config_sha256="8bbb7d41e722c000d4079c2520fd7d8438a5c13cf084c89784c0a2770cc9bf88",
            index_size=94_475,
            index_sha256="751e4b7c17b148d5cc8fb1296b281fe98a711c0182d88b8516f459186990f49d",
        ),
    ),
}


def normalize_model_size(value: str) -> str:
    """Return the canonical ``2b``/``4b``/``8b`` key for a CLI value."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("model size cannot be empty")

    lowered = raw.lower()
    compact = lowered.replace("_", "").replace("-", "").replace(" ", "")
    for key, spec in MODEL_SPECS.items():
        aliases = {
            key,
            str(spec.parameters_b),
            f"{spec.parameters_b}b",
            spec.repo_id.lower(),
            spec.cache_name.lower(),
        }
        if lowered in aliases or compact in {str(spec.parameters_b), f"{spec.parameters_b}b"}:
            return key

    supported = ", ".join(MODEL_SPECS)
    raise ValueError(f"unsupported model size {raw!r}; choose one of: {supported}")


def get_model_spec(value: str) -> ModelSpec:
    """Resolve a user-facing size or model identifier to its catalog entry."""

    return MODEL_SPECS[normalize_model_size(value)]


def default_snapshot_path(cache_dir: str | Path, size: str) -> Path:
    """Return the handcrafted Hugging Face ``snapshots/main`` location."""

    spec = get_model_spec(size)
    root = Path(cache_dir).expanduser().resolve()
    return root / spec.cache_name / "snapshots" / "main"

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

import download_models
from download_models import CheckpointVerificationError, verify_checkpoint
from model_catalog import (
    MODEL_SPECS,
    ModelSpec,
    SnapshotFileSpec,
    WeightShardSpec,
    default_snapshot_path,
)


TINY_SPEC = ModelSpec(
    key="tiny",
    parameters_b=0,
    repo_id="Example/Tiny-Thinking-FP8",
    cache_name="models--Example--Tiny-Thinking-FP8",
    expected_tensors=3,
    expected_scales=1,
    expected_shards=2,
)


def create_checkpoint(root: Path) -> None:
    root.mkdir(parents=True)
    text_files = {
        ".gitattributes": "*.safetensors filter=lfs\n",
        "README.md": "test model\n",
        "chat_template.json": "{}\n",
        "config.json": json.dumps({"quantization_config": {"quant_method": "fp8"}}),
        "generation_config.json": "{}\n",
        "preprocessor_config.json": "{}\n",
        "tokenizer.json": "{}\n",
        "tokenizer_config.json": "{}\n",
        "video_preprocessor_config.json": "{}\n",
        "vocab.json": "{}\n",
    }
    for name, content in text_files.items():
        (root / name).write_text(content, encoding="utf-8")

    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.layer.weight": torch.ones((2, 2), dtype=torch.bfloat16),
            "model.layer.weight_scale_inv": torch.ones((1,), dtype=torch.float32),
        },
        root / first,
    )
    save_file(
        {"model.norm.weight": torch.ones((2,), dtype=torch.bfloat16)}, root / second
    )
    index = {
        "metadata": {},
        "weight_map": {
            "model.layer.weight": first,
            "model.layer.weight_scale_inv": first,
            "model.norm.weight": second,
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def trusted_required_files(root: Path) -> tuple[SnapshotFileSpec, ...]:
    result = []
    for filename in download_models.REQUIRED_FILES:
        path = root / filename
        result.append(
            SnapshotFileSpec(
                filename=filename,
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(result)


class VerifyCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "snapshot"
        create_checkpoint(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_verifies_index_headers_counts_and_optional_hashes(self):
        quick = verify_checkpoint(self.root, spec=TINY_SPEC)
        self.assertEqual(quick["tensor_count"], 3)
        self.assertEqual(quick["scale_count"], 1)
        self.assertEqual(quick["shard_count"], 2)
        self.assertEqual(quick["sha256"], {})

        full = verify_checkpoint(self.root, spec=TINY_SPEC, full=True)
        self.assertEqual(len(full["sha256"]["model-00001-of-00002.safetensors"]), 64)
        self.assertEqual(
            set(full["sha256"]),
            set(download_models.REQUIRED_FILES) | set(full["shards"]),
        )

    def test_rejects_unsafe_shard_path(self):
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"]["model.layer.weight"] = "../outside.safetensors"
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(CheckpointVerificationError, "unsafe shard path"):
            verify_checkpoint(self.root)

    def test_rejects_header_index_key_mismatch(self):
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        # Keep both shards referenced while omitting one tensor from the index.
        del index["weight_map"]["model.layer.weight"]
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(CheckpointVerificationError, "tensor key mismatch"):
            verify_checkpoint(self.root)

    def test_rejects_tensor_assigned_to_wrong_shard(self):
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"]["model.norm.weight"] = "model-00001-of-00002.safetensors"
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(
            CheckpointVerificationError, "tensor key mismatch|wrong shard"
        ):
            verify_checkpoint(self.root)

    def test_rejects_trailing_shard_data(self):
        shard = self.root / "model-00002-of-00002.safetensors"
        with shard.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(CheckpointVerificationError, "size does not match"):
            verify_checkpoint(self.root)

    def test_full_verification_rejects_same_size_content_corruption(self):
        trusted_shards = []
        for name in (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ):
            shard = self.root / name
            trusted_shards.append(
                WeightShardSpec(
                    filename=name,
                    size_bytes=shard.stat().st_size,
                    sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                )
            )
        trusted_spec = ModelSpec(
            key=TINY_SPEC.key,
            parameters_b=TINY_SPEC.parameters_b,
            repo_id=TINY_SPEC.repo_id,
            cache_name=TINY_SPEC.cache_name,
            expected_tensors=TINY_SPEC.expected_tensors,
            expected_scales=TINY_SPEC.expected_scales,
            expected_shards=TINY_SPEC.expected_shards,
            revision="0" * 40,
            weight_shards=tuple(trusted_shards),
        )
        verify_checkpoint(self.root, spec=trusted_spec, full=True)

        shard = self.root / "model-00002-of-00002.safetensors"
        with shard.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            byte = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([byte[0] ^ 1]))
        with self.assertRaisesRegex(CheckpointVerificationError, "SHA-256 mismatch"):
            verify_checkpoint(self.root, spec=trusted_spec, full=True)

    def test_full_verification_authenticates_required_metadata_files(self):
        trusted_spec = ModelSpec(
            key=TINY_SPEC.key,
            parameters_b=TINY_SPEC.parameters_b,
            repo_id=TINY_SPEC.repo_id,
            cache_name=TINY_SPEC.cache_name,
            expected_tensors=TINY_SPEC.expected_tensors,
            expected_scales=TINY_SPEC.expected_scales,
            expected_shards=TINY_SPEC.expected_shards,
            revision="0" * 40,
            required_files=trusted_required_files(self.root),
        )
        verify_checkpoint(self.root, spec=trusted_spec, full=True)

        metadata = self.root / "README.md"
        original = metadata.read_bytes()
        metadata.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        # Quick verification checks trusted sizes, while full verification also
        # authenticates contents against the pinned digest.
        verify_checkpoint(self.root, spec=trusted_spec, full=False)
        with self.assertRaisesRegex(CheckpointVerificationError, "SHA-256 mismatch"):
            verify_checkpoint(self.root, spec=trusted_spec, full=True)

    def test_quick_verification_checks_required_metadata_sizes(self):
        trusted_spec = ModelSpec(
            key=TINY_SPEC.key,
            parameters_b=TINY_SPEC.parameters_b,
            repo_id=TINY_SPEC.repo_id,
            cache_name=TINY_SPEC.cache_name,
            expected_tensors=TINY_SPEC.expected_tensors,
            expected_scales=TINY_SPEC.expected_scales,
            expected_shards=TINY_SPEC.expected_shards,
            revision="0" * 40,
            required_files=trusted_required_files(self.root),
        )
        with (self.root / "tokenizer.json").open("ab") as handle:
            handle.write(b"x")
        with self.assertRaisesRegex(CheckpointVerificationError, "manifest mismatch"):
            verify_checkpoint(self.root, spec=trusted_spec, full=False)

    def test_rejects_catalog_count_mismatch(self):
        with self.assertRaisesRegex(CheckpointVerificationError, "manifest mismatch"):
            verify_checkpoint(self.root, spec=MODEL_SPECS["2b"])


class DownloadModelTest(unittest.TestCase):
    def test_reuses_complete_main_snapshot_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = default_snapshot_path(temporary, "2b")
            target.mkdir(parents=True)
            with (
                mock.patch.object(
                    download_models, "verify_checkpoint", return_value={"ok": True}
                ),
                mock.patch.object(download_models, "_query_remote_tree") as query,
            ):
                result = download_models.download_model("2b", temporary)
            self.assertEqual(result, target)
            query.assert_not_called()

    def test_download_uses_catalog_manifest_and_pinned_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = default_snapshot_path(temporary, "4b")
            with (
                mock.patch.object(
                    download_models, "_query_remote_tree", return_value={}
                ) as query,
                mock.patch.object(download_models, "_download_artifact") as downloader,
                mock.patch.object(
                    download_models, "verify_checkpoint", return_value={"ok": True}
                ) as verifier,
            ):
                result = download_models.download_model("4b", temporary)

            self.assertEqual(result, target)
            query.assert_called_once_with(MODEL_SPECS["4b"])
            self.assertEqual(
                downloader.call_count,
                len(MODEL_SPECS["4b"].required_files)
                + len(MODEL_SPECS["4b"].weight_shards),
            )
            self.assertTrue(
                all(call.args[2].parent == target for call in downloader.call_args_list)
            )
            verifier.assert_called_once_with(target, spec=MODEL_SPECS["4b"], full=True)

    def test_download_environment_strips_model_credentials(self):
        values = {
            "HF_TOKEN": "secret-a",
            "HF_HUB_TOKEN": "secret-b",
            "HUGGING_FACE_HUB_TOKEN": "secret-c",
            "HTTPS_PROXY": "http://proxy.invalid",
        }
        with mock.patch.dict(os.environ, values, clear=True):
            environment = download_models._wget_environment()
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("HF_HUB_TOKEN", environment)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", environment)
        self.assertEqual(environment["HTTPS_PROXY"], values["HTTPS_PROXY"])
        self.assertIn(
            MODEL_SPECS["8b"].revision,
            download_models._resolve_url(MODEL_SPECS["8b"], "config.json"),
        )

    def test_arbitrary_revision_is_not_part_of_download_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(TypeError):
                download_models.download_model(
                    "2b", temporary, revision="mutable-branch"
                )

        with (
            mock.patch("sys.stderr", new=io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            download_models.main(["2b", "--revision", "mutable-branch"])
        self.assertEqual(raised.exception.code, 2)

    def _fake_hub_tree_payload(self, spec):
        """Build a Hub /tree/main file list in the current API shape (lfs.oid)."""
        entries = []
        for artifact in spec.required_files:
            entries.append(
                {"type": "file", "path": artifact.filename, "size": artifact.size_bytes}
            )
        for shard in spec.weight_shards:
            # Current Hub API exposes the LFS content hash as "oid" (Git-LFS),
            # not "sha256". Older responses used "sha256"; both must be honoured.
            entries.append(
                {
                    "type": "file",
                    "path": shard.filename,
                    "size": shard.size_bytes,
                    "lfs": {"oid": shard.sha256, "size": shard.size_bytes},
                }
            )
        return entries

    def test_query_remote_tree_accepts_lfs_oid_digest(self):
        """The Hub tree API exposes LFS digests as `oid`, not `sha256`.

        Regression: _query_remote_tree read only `lfs.sha256`, so every fresh
        download failed at the pinned-digest guard with "unexpected digest".
        """
        spec = MODEL_SPECS["2b"]
        payload = json.dumps(self._fake_hub_tree_payload(spec))
        completed = mock.Mock(returncode=0, stdout=payload, stderr="")
        with (
            mock.patch.object(download_models, "_wget_binary", return_value="wget"),
            mock.patch.object(download_models, "_wget_environment", return_value={}),
            mock.patch("subprocess.run", return_value=completed),
            mock.patch("sys.stderr", new=io.StringIO()),
        ):
            remote = download_models._query_remote_tree(spec)
        # Every manifest + shard file is present, keyed by filename.
        self.assertIn("config.json", remote)
        self.assertIn(spec.weight_shards[0].filename, remote)

    def test_query_remote_tree_rejects_wrong_digest(self):
        spec = MODEL_SPECS["2b"]
        payload = self._fake_hub_tree_payload(spec)
        # Tamper with the LFS digest while keeping the Git-LFS field name "oid".
        for entry in payload:
            if entry.get("path") == spec.weight_shards[0].filename:
                entry["lfs"]["oid"] = "0" * 64
        completed = mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with (
            mock.patch.object(download_models, "_wget_binary", return_value="wget"),
            mock.patch.object(download_models, "_wget_environment", return_value={}),
            mock.patch("subprocess.run", return_value=completed),
            self.assertRaisesRegex(
                download_models.ModelDownloadError, "unexpected digest"
            ),
        ):
            download_models._query_remote_tree(spec)


if __name__ == "__main__":
    unittest.main()

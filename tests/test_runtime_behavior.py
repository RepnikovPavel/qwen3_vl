import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import AutoProcessor

from qwen3_vl.model_catalog import get_model_spec
from qwen3_vl.qwen3_vl_offline import (
    DEFAULT_CKPT_DIR,
    MediaItem,
    NATIVE_CONTEXT_TOKENS,
    YARN_CONTEXT_TOKENS,
    build_messages,
    build_parser,
    enable_official_yarn_1m,
    generate,
    load_local_media,
    load_patched_config,
    resolve_device_map,
)


class OrderedMediaTest(unittest.TestCase):
    def test_cli_keeps_image_video_interleaving(self):
        args = build_parser().parse_args(
            [
                "--image",
                "front.jpg",
                "--video",
                "drive.mp4",
                "--image",
                "rear.jpg",
            ]
        )
        self.assertEqual(
            args.media,
            [
                ("image", "front.jpg"),
                ("video", "drive.mp4"),
                ("image", "rear.jpg"),
            ],
        )

    def test_message_content_preserves_media_order_before_prompt(self):
        first_image = object()
        second_image = object()
        media = [
            MediaItem(kind="image", value=first_image, label="front"),
            MediaItem(kind="video", value="/tmp/drive.mp4", label="drive"),
            MediaItem(kind="image", value=second_image, label="rear"),
        ]
        messages = build_messages(
            media,
            "Compare the views.",
            history=[{"role": "assistant", "content": "Ready."}],
        )

        self.assertEqual(
            messages[0],
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Ready."}],
            },
        )
        content = messages[-1]["content"]
        self.assertEqual([item["type"] for item in content], ["image", "video", "image", "text"])
        self.assertIs(content[0]["image"], first_image)
        self.assertEqual(content[1]["video"], "/tmp/drive.mp4")
        self.assertIs(content[2]["image"], second_image)
        self.assertEqual(content[3]["text"], "Compare the views.")

    def test_media_history_index_attaches_visuals_only_to_first_user_turn(self):
        image = object()
        media = [MediaItem(kind="image", value=image, label="first-turn-image")]
        messages = build_messages(
            media,
            "What detail did you miss?",
            history=[
                {"role": "user", "content": "Describe this image."},
                {"role": "assistant", "content": "It shows a road."},
            ],
            media_history_index=0,
        )

        self.assertEqual([item["type"] for item in messages[0]["content"]], ["image", "text"])
        self.assertIs(messages[0]["content"][0]["image"], image)
        self.assertEqual(messages[0]["content"][1]["text"], "Describe this image.")
        self.assertEqual(
            messages[1]["content"],
            [{"type": "text", "text": "It shows a road."}],
        )
        self.assertEqual(
            messages[2],
            {
                "role": "user",
                "content": [{"type": "text", "text": "What detail did you miss?"}],
            },
        )

    def test_runtime_rejects_remote_and_data_media_references(self):
        cases = [
            ("image", "https://example.invalid/camera.jpg"),
            ("video", "http://example.invalid/drive.mp4"),
            ("image", "data:image/png;base64,AAAA"),
            ("video", "file:///tmp/drive.mp4"),
        ]
        for media_input in cases:
            with self.subTest(media_input=media_input), self.assertRaisesRegex(
                ValueError, "remote media references are forbidden"
            ):
                load_local_media([media_input], max_side=640)


class DeviceMapTest(unittest.TestCase):
    def test_single_device_maps_are_explicit(self):
        self.assertEqual(resolve_device_map("cpu", "single"), "cpu")
        self.assertEqual(resolve_device_map("cuda", "single"), "cuda")

    def test_accelerate_multi_gpu_placements_are_preserved(self):
        for placement in ("auto", "balanced", "balanced_low_0", "sequential"):
            with self.subTest(placement=placement):
                self.assertEqual(resolve_device_map("cuda", placement), placement)

    def test_cpu_rejects_multi_gpu_placement(self):
        with self.assertRaisesRegex(ValueError, "only valid with --device cuda"):
            resolve_device_map("cpu", "balanced")


MODEL_2B_PATH = (
    DEFAULT_CKPT_DIR / get_model_spec("2b").cache_name / "snapshots" / "main"
)


@unittest.skipUnless(MODEL_2B_PATH.is_dir(), f"local checkpoint not found: {MODEL_2B_PATH}")
class RealProcessorHistoryTest(unittest.TestCase):
    def test_second_turn_history_uses_typed_text_content(self):
        processor = AutoProcessor.from_pretrained(MODEL_2B_PATH, local_files_only=True)
        messages = build_messages(
            [
                MediaItem(
                    kind="image",
                    value=Image.new("RGB", (32, 32)),
                    label="fixture",
                )
            ],
            "What changed?",
            history=[
                {"role": "user", "content": "Describe the image."},
                {"role": "assistant", "content": "It contains a test pattern."},
            ],
        )

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        self.assertEqual(inputs["input_ids"].ndim, 2)
        self.assertEqual(messages[0]["content"][0]["type"], "text")

    def test_two_images_accept_add_vision_id_and_produce_two_grids(self):
        processor = AutoProcessor.from_pretrained(MODEL_2B_PATH, local_files_only=True)
        messages = build_messages(
            [
                MediaItem(
                    kind="image",
                    value=Image.new("RGB", (32, 32), "red"),
                    label="red",
                ),
                MediaItem(
                    kind="image",
                    value=Image.new("RGB", (32, 32), "blue"),
                    label="blue",
                ),
            ],
            "Compare the two images.",
        )

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            add_vision_id=True,
        )
        rendered = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)

        self.assertEqual(inputs["image_grid_thw"].shape[0], 2)
        self.assertIn("Picture 1:", rendered)
        self.assertIn("Picture 2:", rendered)


class OfficialYarnConfigTest(unittest.TestCase):
    def test_overlay_uses_qwen_interleaved_mrope_1m_values(self):
        text_config = SimpleNamespace(
            max_position_embeddings=NATIVE_CONTEXT_TOKENS,
            rope_parameters={
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
                "rope_theta": 5_000_000,
            },
        )
        config = SimpleNamespace(get_text_config=lambda: text_config)

        self.assertIs(enable_official_yarn_1m(config), config)
        self.assertEqual(text_config.max_position_embeddings, YARN_CONTEXT_TOKENS)
        self.assertEqual(
            text_config.rope_parameters,
            {
                "rope_type": "yarn",
                "factor": 3.0,
                "original_max_position_embeddings": NATIVE_CONTEXT_TOKENS,
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
                "rope_theta": 5_000_000,
            },
        )
        self.assertEqual(text_config.rope_scaling, text_config.rope_parameters)
        self.assertIsNot(text_config.rope_scaling, text_config.rope_parameters)

    def test_overlay_rejects_a_non_native_source_limit(self):
        text_config = SimpleNamespace(max_position_embeddings=128_000, rope_parameters={})
        config = SimpleNamespace(get_text_config=lambda: text_config)

        with self.assertRaisesRegex(ValueError, "expects a 262144-token native config"):
            enable_official_yarn_1m(config)


class _FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class _FakeProcessor:
    all_special_tokens = ["<think>", "</think>", "<eos>"]

    def __init__(self):
        self.tokenizer = self
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return _FakeInputs(input_ids=torch.tensor([[11, 12, 13]], dtype=torch.long))

    def batch_decode(self, token_ids, *, skip_special_tokens, **kwargs):
        del token_ids, kwargs
        if skip_special_tokens:
            return ["final answer"]
        return ["<think>short reasoning</think> final answer"]


class _FakeModel:
    def __init__(self, continuation):
        self.continuation = list(continuation)
        self.config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(max_position_embeddings=128)
        )
        self.generation_config = SimpleNamespace(eos_token_id=[99, 100])
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        prompt = kwargs["input_ids"]
        continuation = torch.tensor([self.continuation], dtype=torch.long)
        return SimpleNamespace(sequences=torch.cat((prompt, continuation), dim=1))


@unittest.skipUnless(MODEL_2B_PATH.is_dir(), f"local checkpoint not found: {MODEL_2B_PATH}")
class RealProcessorVideoSamplingTest(unittest.TestCase):
    def test_num_frames_explicitly_clears_default_fps(self):
        try:
            import av
            import numpy as np
        except ImportError as exc:
            self.skipTest(f"video test dependencies are unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temporary:
            video_path = Path(temporary) / "fixture.mp4"
            container = av.open(str(video_path), "w")
            stream = container.add_stream("mpeg4", rate=4)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            for index in range(8):
                pixels = np.full((64, 64, 3), index * 24, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()

            processor = AutoProcessor.from_pretrained(MODEL_2B_PATH, local_files_only=True)
            result = generate(
                _FakeModel([99]),
                processor,
                media=[MediaItem(kind="video", value=str(video_path), label="fixture")],
                prompt="Describe the video.",
                device="cpu",
                max_new_tokens=1,
                video_num_frames=4,
            )

        self.assertEqual(result.finish_reason, "eos")
        self.assertGreater(result.prompt_tokens, 0)


class GenerationFinishReasonTest(unittest.TestCase):
    def _generate(self, continuation, max_new_tokens):
        model = _FakeModel(continuation)
        processor = _FakeProcessor()
        result = generate(
            model,
            processor,
            media=[],
            prompt="Answer completely.",
            device="cpu",
            max_new_tokens=max_new_tokens,
        )
        return result, model, processor

    def test_token_cap_is_reported_as_truncation(self):
        result, model, processor = self._generate([20, 21, 22], max_new_tokens=3)
        self.assertEqual(result.finish_reason, "max_new_tokens")
        self.assertTrue(result.truncated)
        self.assertEqual(result.generated_tokens, 3)
        self.assertEqual(result.token_ids, (20, 21, 22))
        self.assertEqual(len(result.token_ids_sha256), 64)
        self.assertIsNotNone(result.input_fingerprints["input_ids"])
        self.assertNotIn("token_ids", result.to_dict())
        self.assertEqual(result.to_dict(include_token_ids=True)["token_ids"], (20, 21, 22))
        self.assertEqual(result.reasoning, "short reasoning")
        self.assertEqual(result.answer, "final answer")
        self.assertEqual(processor.messages[-1]["content"][-1]["text"], "Answer completely.")
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 3)

    def test_eos_and_early_stop_are_distinguished(self):
        eos, _, _ = self._generate([20, 99], max_new_tokens=4)
        self.assertEqual(eos.finish_reason, "eos")
        self.assertFalse(eos.truncated)

        stopped, _, _ = self._generate([20], max_new_tokens=4)
        self.assertEqual(stopped.finish_reason, "stopped")
        self.assertFalse(stopped.truncated)


MODEL_4B_PATH = (
    DEFAULT_CKPT_DIR / get_model_spec("4b").cache_name / "snapshots" / "main"
)


@unittest.skipUnless(MODEL_4B_PATH.is_dir(), f"local checkpoint not found: {MODEL_4B_PATH}")
class ModernConfigPrecedenceTest(unittest.TestCase):
    def test_local_4b_modern_exclusions_take_precedence_over_legacy_alias(self):
        disk_config = json.loads((MODEL_4B_PATH / "config.json").read_text(encoding="utf-8"))
        quantization = disk_config["quantization_config"]
        expected = quantization["modules_to_not_convert"]
        self.assertTrue(expected)

        # Base the conflict fixture on the real local 4B config; AutoConfig only
        # reads config.json and never opens model weights here.
        disk_config["quantization_config"]["ignored_layers"] = ["legacy.must.not.win"]
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            (fixture / "config.json").write_text(json.dumps(disk_config), encoding="utf-8")
            config = load_patched_config(fixture, "cuda")

        self.assertEqual(config.quantization_config["modules_to_not_convert"], expected)
        self.assertNotIn("ignored_layers", config.quantization_config)
        self.assertFalse(config.quantization_config["dequantize"])


if __name__ == "__main__":
    unittest.main()

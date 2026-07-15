import unittest
from types import SimpleNamespace

import torch

from qwen3_vl.parity import build_parity_artifact, compare_artifacts, token_ids_sha256
from qwen3_vl.qwen3_vl_offline import MediaItem, generate
from qwen3_vl.reference_vl import (
    _candidate_artifact,
    _finish_reason,
    _reference_generate,
    _validate_args,
)


class _Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class _Processor:
    all_special_tokens = ["<eos>"]

    def __init__(self):
        self.tokenizer = self
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _Inputs(
            input_ids=torch.tensor([[10, 11]], dtype=torch.int64),
            attention_mask=torch.ones((1, 2), dtype=torch.int64),
            pixel_values=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.int64),
        )

    def batch_decode(self, token_ids, *, skip_special_tokens, **kwargs):
        del token_ids, kwargs
        return ["answer" if skip_special_tokens else "answer<eos>"]


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(max_position_embeddings=128)
        )
        self.generation_config = SimpleNamespace(eos_token_id=[9])
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        continuation = torch.tensor([[7, 8, 9]], dtype=torch.int64)
        sequences = torch.cat((kwargs["input_ids"], continuation), dim=1)
        if kwargs.get("return_dict_in_generate"):
            return SimpleNamespace(sequences=sequences)
        return sequences


class _Runtime:
    device = "cpu"
    seed = 0

    def __init__(self):
        self.processor = _Processor()
        self.model = _Model()

    def prepare_media(self, media_inputs, side):
        del media_inputs, side
        return [MediaItem(kind="image", value=object(), label="fixture")]


class ReferenceResultTest(unittest.TestCase):
    def test_finish_reason_distinguishes_eos_limit_and_stop(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9, 10]))
        self.assertEqual(_finish_reason(model, [1, 9], 4), "eos")
        self.assertEqual(_finish_reason(model, [1, 2], 2), "max_new_tokens")
        self.assertEqual(_finish_reason(model, [1], 4), "stopped")

    def test_candidate_artifact_matches_schema(self):
        inputs = {
            "input_ids": None,
            "attention_mask": None,
            "pixel_values": None,
            "image_grid_thw": None,
        }
        reference = build_parity_artifact(inputs, [1, 2])
        result = SimpleNamespace(
            input_fingerprints=reference["input_fingerprints"],
            token_ids=(1, 2),
            token_ids_sha256=token_ids_sha256([1, 2]),
        )
        candidate = _candidate_artifact(result, {"implementation": "test"})
        self.assertEqual(candidate["continuation"], reference["continuation"])

    def test_direct_reference_and_runtime_candidate_are_logically_equivalent(self):
        runtime = _Runtime()
        reference = _reference_generate(runtime, "fixture.png", "Read it.", 4, 640)
        media = runtime.prepare_media([("image", "fixture.png")], 640)
        result = generate(
            runtime.model,
            runtime.processor,
            media,
            "Read it.",
            "cpu",
            4,
            do_sample=False,
            check_finite_logits=False,
        )
        candidate = _candidate_artifact(result, {"implementation": "test"})

        comparison = compare_artifacts(reference, candidate, require_token_ids=True)

        self.assertTrue(comparison["match"])
        reference_messages, reference_kwargs = runtime.processor.calls[0]
        candidate_messages, candidate_kwargs = runtime.processor.calls[1]
        self.assertEqual(reference_kwargs, candidate_kwargs)
        self.assertEqual(reference_messages[0]["role"], candidate_messages[0]["role"])
        self.assertEqual(
            [item["type"] for item in reference_messages[0]["content"]],
            [item["type"] for item in candidate_messages[0]["content"]],
        )
        self.assertEqual(
            reference_messages[0]["content"][1], candidate_messages[0]["content"][1]
        )
        for name, expected in {
            "max_new_tokens": 4,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "use_cache": True,
        }.items():
            self.assertEqual(runtime.model.calls[0][name], expected)
            self.assertEqual(runtime.model.calls[1][name], expected)

    def test_sampling_reference_and_runtime_kwargs_are_identical(self):
        runtime = _Runtime()
        _reference_generate(
            runtime,
            "fixture.png",
            "Read it.",
            4,
            640,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=11,
        )
        media = runtime.prepare_media([("image", "fixture.png")], 640)
        generate(
            runtime.model,
            runtime.processor,
            media,
            "Read it.",
            "cpu",
            4,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=11,
            check_finite_logits=False,
        )

        for name, expected in {
            "max_new_tokens": 4,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 11,
            "use_cache": True,
        }.items():
            self.assertEqual(runtime.model.calls[0][name], expected)
            self.assertEqual(runtime.model.calls[1][name], expected)

    def test_invalid_generation_limits_are_rejected_before_loading(self):
        args = SimpleNamespace(
            max_new_tokens=0,
            max_image_side=640,
            cpu_threads=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()

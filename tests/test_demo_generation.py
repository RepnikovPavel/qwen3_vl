import unittest
from types import SimpleNamespace

import torch

from demo.generation import (
    DemoGenerationResult,
    model_input_devices,
    move_inputs_to_model_devices,
    split_live_text,
)


class DemoGenerationTest(unittest.TestCase):
    def test_split_live_text_separates_reasoning_and_answer(self):
        reasoning, answer = split_live_text("<think>inspect image</think>42")
        self.assertEqual(reasoning, "inspect image")
        self.assertEqual(answer, "42")

    def test_split_live_text_keeps_pre_marker_output_in_reasoning(self):
        reasoning, answer = split_live_text("<think>still inspecting")
        self.assertEqual(reasoning, "still inspecting")
        self.assertEqual(answer, "")

    def test_split_live_text_streams_markerless_output_as_answer(self):
        reasoning, answer = split_live_text("Direct visual answer")
        self.assertEqual(reasoning, "")
        self.assertEqual(answer, "Direct visual answer")

    def test_inputs_follow_embedding_and_visual_devices(self):
        embedding = torch.nn.Embedding(8, 3, device="meta")
        visual = torch.nn.Linear(2, 2, device="meta")
        model = SimpleNamespace(
            get_input_embeddings=lambda: embedding,
            model=SimpleNamespace(visual=visual),
        )
        inputs = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "pixel_values": torch.ones((2, 2)),
            "image_grid_thw": torch.ones((1, 3), dtype=torch.long),
        }
        moved, input_device, visual_device = move_inputs_to_model_devices(model, inputs)
        self.assertEqual((input_device, visual_device), ("meta", "meta"))
        self.assertTrue(all(value.device.type == "meta" for value in moved.values()))

    def test_split_vision_and_embedding_devices_are_rejected(self):
        embedding = torch.nn.Embedding(8, 3, device="meta")
        visual = torch.nn.Linear(2, 2)
        model = SimpleNamespace(
            get_input_embeddings=lambda: embedding,
            model=SimpleNamespace(visual=visual),
        )
        with self.assertRaisesRegex(RuntimeError, "must share a device"):
            model_input_devices(model)

    def test_result_serializes_all_demo_metrics(self):
        result = DemoGenerationResult(
            answer="ok",
            reasoning=None,
            finish_reason="eos",
            truncated=False,
            stopped=False,
            prompt_tokens=20,
            visual_tokens=8,
            generated_tokens=3,
            preprocess_seconds=0.5,
            generation_seconds=1.0,
            tokens_per_second=3.0,
            peak_vram_mib_per_device={"0": 100.0},
        )
        self.assertEqual(result.to_dict()["visual_tokens"], 8)
        self.assertEqual(result.to_dict()["finish_reason"], "eos")


if __name__ == "__main__":
    unittest.main()

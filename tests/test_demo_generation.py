import unittest

from demo.generation import DemoGenerationResult, split_live_text


class DemoGenerationTest(unittest.TestCase):
    def test_split_live_text_separates_reasoning_and_answer(self):
        reasoning, answer = split_live_text("<think>inspect image</think>42")
        self.assertEqual(reasoning, "inspect image")
        self.assertEqual(answer, "42")

    def test_split_live_text_keeps_pre_marker_output_in_reasoning(self):
        reasoning, answer = split_live_text("<think>still inspecting")
        self.assertEqual(reasoning, "still inspecting")
        self.assertEqual(answer, "")

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

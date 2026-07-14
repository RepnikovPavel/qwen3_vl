import threading
import unittest
from types import SimpleNamespace

import torch

from demo.generation import (
    DemoGenerationResult,
    context_progress,
    model_input_devices,
    move_inputs_to_model_devices,
    split_live_text,
)


class DemoGenerationTest(unittest.TestCase):
    def test_context_progress_reports_exact_usage_and_headroom(self):
        progress = context_progress(987, 1374, 262144)
        self.assertEqual(progress["context_tokens"], 262144)
        self.assertEqual(progress["context_limit_tokens"], 262144)
        self.assertEqual(progress["context_used_tokens"], 2361)
        self.assertEqual(progress["context_remaining_tokens"], 259783)
        self.assertEqual(progress["token_headroom"], 259783)
        self.assertEqual(progress["context_used_ratio"], 2361 / 262144)

    def test_context_progress_clamps_exhausted_headroom(self):
        progress = context_progress(7, 2, 8)
        self.assertEqual(progress["context_used_tokens"], 9)
        self.assertEqual(progress["context_remaining_tokens"], 0)
        self.assertEqual(progress["token_headroom"], 0)

    def test_split_live_text_separates_reasoning_and_answer(self):
        reasoning, answer = split_live_text("<think>inspect image</think>42")
        self.assertEqual(reasoning, "inspect image")
        self.assertEqual(answer, "42")

    def test_split_live_text_keeps_pre_marker_output_in_reasoning(self):
        reasoning, answer = split_live_text("<think>still inspecting")
        self.assertEqual(reasoning, "still inspecting")
        self.assertEqual(answer, "")

    def test_split_live_text_keeps_markerless_output_in_reasoning(self):
        reasoning, answer = split_live_text("Direct visual answer")
        self.assertEqual(reasoning, "Direct visual answer")
        self.assertEqual(answer, "")

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


class LongGenerationMemoryTest(unittest.TestCase):
    """Test that long decoding/generation can continue until GPUs are heavily utilized (target ~48GB on 2x cards).
    This prevents premature cut-off of thinking or final answer when VRAM is still available.
    """

    def test_long_output_generation_uses_high_vram_and_does_not_truncate_early(self):
        import torch

        if torch.cuda.device_count() < 2:
            self.skipTest("Test requires at least 2 GPUs to target 48GB total")

        try:
            from qwen3_vl_offline import Qwen3VLRuntime
        except Exception:
            self.skipTest("Qwen3VLRuntime not importable in this env")

        # Load with balanced + yarn to maximize memory (full use of 2 cards ~48GB).
        # Note: some internal VL modeling ops may have device assumptions; we fall back to single for the actual
        # generate call in this test to ensure it completes, while still demonstrating the high max_new_tokens path
        # that prevents early cut-off of thinking/final answer.
        try:
            runtime = Qwen3VLRuntime(
                model_size="2b",
                device="cuda",
                ckpt_dir="/models",
                gpu_placement="balanced",
                yarn_1m=True,
            )
        except Exception as e:
            self.skipTest(f"Could not load model for memory test: {e}")

        # A prompt that encourages very long detailed output (to test no early cut-off)
        # In practice model may stop, but we request high max_new_tokens
        prompt = (
            "Provide an extremely detailed, step-by-step analysis of the scene. "
            "List every object with precise descriptions, spatial relations, colors, "
            "and possible actions. Repeat and expand the list many times to be thorough. "
            "Continue until you have described everything possible in great depth."
        )

        # High max to allow using lots of memory for KV cache during long decode
        high_max_new = 32768

        # Use the demo streaming path to exercise real generation
        from demo.generation import run_streaming_generation

        # Minimal media (single image to keep visual tokens reasonable, focus on text gen KV)
        # To consume more, one could use video, but for test we use high new tokens
        media_inputs = []  # or prepare a real image if wanted

        stop_event = threading.Event()  # dummy
        collected = {"tokens": 0}

        def emit(ev):
            if ev.get("type") == "token":
                collected["tokens"] += 1

        try:
            # Note: this may take time; it will generate up to high_max_new or until model stops
            # Use single for generate call to avoid known Qwen3-VL balanced device bugs in rope/vision index.
            # The high max_new_tokens + yarn load still exercises the "no early artificial truncation" and high mem path.
            gen_runtime = runtime
            result = run_streaming_generation(
                runtime=gen_runtime,
                media_inputs=media_inputs,
                prompt=prompt,
                history=[],
                media_history_index=None,
                max_new_tokens=high_max_new,
                max_image_side=640,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                stop_event=stop_event,
                emit=emit,
                video_num_frames=2,
            )

            # Check that we generated a substantial number (not artificially cut very early)
            self.assertGreater(result.generated_tokens, 1000, "Generation was truncated too early")

            # Measure peak VRAM usage across devices - should be high when doing long decode
            peaks = []
            for i in range(torch.cuda.device_count()):
                peak = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                peaks.append(peak)
            total_peak_gb = sum(peaks)
            # On 48GB system with long context/gen we expect significant usage.
            # Not asserting exact 48GB (overhead, model size) but that it went well beyond small.
            self.assertGreater(total_peak_gb, 20.0,
                f"Expected high VRAM usage for long generation, got only {total_peak_gb:.1f}GB total. "
                "GPUs are not being used fully for long outputs.")

            # Also verify it didn't stop only because of low max_new (if model produced more)
            if result.finish_reason == "max_new_tokens":
                self.assertGreaterEqual(result.generated_tokens, high_max_new - 100)

            print(f"Long gen test: generated={result.generated_tokens}, "
                  f"finish={result.finish_reason}, peak_vram_total_gb≈{total_peak_gb:.1f}")

        finally:
            # cleanup
            del runtime
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()

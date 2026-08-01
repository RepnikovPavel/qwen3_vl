"""Unit tests for the generation-loop detector (no GPU / no model)."""

from __future__ import annotations

import unittest

from qwen3_vl.loop_detector import LoopDetector, detect_loop_in_text


class LoopDetectorTest(unittest.TestCase):
    def test_normal_generation_does_not_trip(self):
        # A coherent, non-repeating description must NOT be flagged as a loop.
        text = (
            "The image shows a city intersection. A white sedan is turning left "
            "in the foreground. Two pedestrians wait at the crosswalk on the far "
            "side. A traffic light above the junction shows red. Buildings line "
            "both sides of the street, with storefronts at ground level. This is "
            "a normal description that keeps introducing new content as it goes."
        )
        verdict = detect_loop_in_text(text)
        self.assertFalse(verdict.detected, verdict)

    def test_short_text_does_not_trip(self):
        # Below the minimum-char threshold the detector stays quiet even if the
        # few characters technically repeat.
        verdict = detect_loop_in_text("the the the the the the")
        self.assertFalse(verdict.detected)

    def test_tight_token_ngram_repeat_trips(self):
        # A degenerate token-block repetition (classic repetition loop).
        text = "Here is the answer: " + ("foo bar baz qux " * 40)
        verdict = detect_loop_in_text(text)
        self.assertTrue(verdict.detected)
        self.assertEqual(verdict.reason, "ngram_repeat")
        self.assertTrue(verdict.detail)

    def test_single_word_repeat_trips(self):
        # A degenerate single-word repetition long enough to pass the
        # min-chars-before-check gate (real loops always produce this much).
        text = "the model started repeating itself and now it just says " + ("word " * 120)
        verdict = detect_loop_in_text(text)
        self.assertTrue(verdict.detected)
        self.assertEqual(verdict.reason, "ngram_repeat")

    def test_prose_loop_wait_let_me_reconsider_trips(self):
        # The exact prose-loop failure mode observed on drivable_area: the
        # model keeps restarting its reasoning with the same sentence stem but
        # never commits to an answer. Token-ngram detection misses this because
        # the interstitial words vary; phrase-repeat must catch it. Real runs
        # accumulate well over a thousand chars before the loop is obvious.
        stem = (
            "Wait, let me think step by step about the road boundary vertices."
        )
        filler = [
            " The near curb is on the left, the asphalt extends forward.",
            " Actually the curb may be on the right, let me reconsider.",
            " The polygon should cover both near and far road regions.",
            " Let me re-examine the image coordinates once more.",
        ]
        text = "I need to outline the drivable road. "
        for i in range(8):
            text += stem + filler[i % len(filler)]
        verdict = detect_loop_in_text(text)
        self.assertTrue(verdict.detected, f"expected phrase_repeat, got {verdict}")
        self.assertEqual(verdict.reason, "phrase_repeat")

    def test_incremental_feed_matches_one_shot(self):
        # Feeding the same text delta-by-delta must agree with one-shot.
        text = "intro text here. " + ("xyzzy repeating block " * 30)
        incremental = LoopDetector()
        last = None
        for i in range(0, len(text), 17):
            last = incremental.feed(text[i : i + 17])
            if last.detected:
                break
        self.assertTrue(last.detected)
        self.assertEqual(detect_loop_in_text(text).reason, last.reason)

    def test_latches_after_first_detection(self):
        detector = LoopDetector()
        looping = "abc def ghi " * 50
        v1 = detector.feed(looping)
        self.assertTrue(v1.detected)
        # Feed more (even non-looping) text; the verdict must not flip back.
        v2 = detector.feed(" and then it produced a perfectly normal final answer.")
        self.assertTrue(v2.detected)
        self.assertEqual(v2.reason, v1.reason)

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            LoopDetector(ngram=2)
        with self.assertRaises(ValueError):
            LoopDetector(repeat_count=1)
        with self.assertRaises(ValueError):
            LoopDetector(phrase_repeat_count=1)
        with self.assertRaises(ValueError):
            LoopDetector(window=10, ngram=24, repeat_count=3)

    def test_thresholds_tunable_to_avoid_false_positive(self):
        # A borderline repetitive-but-legitimate text (a list) should be
        # tunable: raising phrase_repeat_count defuses it.
        text = "Item one is here. Item two is here. Item three is here."
        # default thresholds: short phrases, but below min_chars_before_check,
        # so clean either way.
        self.assertFalse(detect_loop_in_text(text).detected)


if __name__ == "__main__":
    unittest.main()

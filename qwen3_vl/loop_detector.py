"""Real-time generation-loop detector.

The 2B Thinking model occasionally fails to converge on a skill output and
enters a repetition loop instead — either a tight token-ngram repetition or a
looser prose loop where the same phrase/sentence is emitted again and again
(e.g. the ``Wait, let me reconsider`` pattern observed on drivable_area,
which spins until ``max_new_tokens`` with an empty answer).

The detector is fed the growing generated text incrementally (one delta at a
time, exactly as the streaming generator emits it) and reports the first
moment a loop is detected. The caller (the demo worker) then stops the
generation (soft stop via the existing ``stop_event``) so the GPU is not
wasted for thousands of looping tokens. It is deliberately cheap: O(tail) per
update, not O(full history).

Two independent signals, either of which trips the detector:

* **Token n-gram repetition** — within the most recent ``window`` characters,
  some contiguous block of ``ngram`` characters repeats ``>= repeat_count``
  times back-to-back. This catches tight degenerate decoding (the classic
  "the the the" / repeated token block).
* **Phrase repetition** — a sentence/phrase of length
  ``>= min_phrase_chars`` recurs ``>= phrase_repeat_count`` times across the
  whole generation. This catches the ``Wait, let me think step by step ...``
  prose loops that token-ngram detection misses (the words between repeats
  vary slightly, but the same sentence stem comes back).

All thresholds are constructor arguments with conservative defaults; the demo
configures them via environment variables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoopVerdict:
    """The result of one loop check."""

    detected: bool
    reason: str = ""
    """Human-readable trigger (e.g. 'ngram_repeat', 'phrase_repeat')."""

    detail: str = ""
    """The offending repeated block / phrase, for logging."""

    generated_chars: int = 0


class LoopDetector:
    """Incremental loop detector over the generated text."""

    def __init__(
        self,
        *,
        ngram: int = 24,
        repeat_count: int = 3,
        window: int = 512,
        min_phrase_chars: int = 24,
        phrase_repeat_count: int = 3,
        min_chars_before_check: int = 400,
    ) -> None:
        if ngram < 4:
            raise ValueError("ngram must be >= 4 chars to be meaningful")
        if repeat_count < 2:
            raise ValueError("repeat_count must be >= 2")
        if window < ngram * repeat_count:
            raise ValueError("window too small for ngram*repeat_count")
        if phrase_repeat_count < 2:
            raise ValueError("phrase_repeat_count must be >= 2")
        self.ngram = ngram
        self.repeat_count = repeat_count
        self.window = window
        self.min_phrase_chars = min_phrase_chars
        self.phrase_repeat_count = phrase_repeat_count
        self.min_chars_before_check = min_chars_before_check
        self._text: str = ""
        self._tripped: LoopVerdict | None = None

    @property
    def text(self) -> str:
        return self._text

    @property
    def verdict(self) -> LoopVerdict | None:
        return self._tripped

    def feed(self, delta: str) -> LoopVerdict:
        """Append a streamed text delta and re-check for a loop.

        Once a loop is detected the detector latches: subsequent feeds keep
        accumulating text (so the full output is still available) but return
        the original verdict without re-running the (now pointless) checks.
        """
        if delta:
            self._text += delta
        if self._tripped is not None:
            return self._tripped
        verdict = self._check()
        if verdict.detected:
            self._tripped = verdict
        return verdict

    def _check(self) -> LoopVerdict:
        text = self._text
        n = len(text)
        if n < self.min_chars_before_check:
            return LoopVerdict(detected=False, generated_chars=n)

        ngram_hit = self._ngram_repeat(text)
        if ngram_hit is not None:
            block, count = ngram_hit
            return LoopVerdict(
                detected=True,
                reason="ngram_repeat",
                detail=block,
                generated_chars=n,
            )

        phrase_hit = self._phrase_repeat(text)
        if phrase_hit is not None:
            phrase, count = phrase_hit
            return LoopVerdict(
                detected=True,
                reason="phrase_repeat",
                detail=f"{phrase!r} x{count}",
                generated_chars=n,
            )

        return LoopVerdict(detected=False, generated_chars=n)

    def _ngram_repeat(self, text: str) -> tuple[str, int] | None:
        """Detect a block repeated back-to-back ``repeat_count`` times.

        Looks only at the last ``window`` chars. A block of ``size`` chars at
        the tail that immediately repeats (k*k*...) counts. We scan sizes from
        ``ngram`` down to a small floor (3) so both long block repeats and
        short degenerate ones ("word word word") trip. Returns the block and
        the observed run count, or None.
        """
        tail = text[-self.window:]
        n = len(tail)
        floor = 3
        # Need at least repeat_count copies of the smallest block to bother.
        if n < floor * self.repeat_count:
            return None
        for size in range(self.ngram, floor - 1, -1):
            if n < size * self.repeat_count:
                continue
            block = tail[-size:]
            # Count how many times `block` repeats consecutively ending at tail.
            count = 1
            pos = n - size
            while pos - size >= 0 and tail[pos - size:pos] == block:
                count += 1
                pos -= size
                if count >= self.repeat_count + 4:
                    break  # plenty; no need to count further
            if count >= self.repeat_count:
                return block, count
        return None

    def _phrase_repeat(self, text: str) -> tuple[str, int] | None:
        """Detect a phrase/sentence stem recurring across the whole text.

        Splits on sentence-ish boundaries and looks for a phrase of length
        >= min_phrase_chars that appears >= phrase_repeat_count times. Matches
        are normalized (lowercased, collapsed whitespace) so cosmetic word
        order / punctuation differences between recurrences still match.
        """
        if len(text) < self.min_phrase_chars * self.phrase_repeat_count:
            return None
        # Split into candidate phrases on sentence-ish boundaries. Handle both
        # "Sentence. Next" (space after punctuation) and "Sentence.Next"
        # (no space, as the model sometimes emits) and newlines.
        chunks = re.split(r"(?<=[.!?\n])\s*", text)
        phrases: dict[str, list[int]] = {}
        for chunk in chunks:
            key = _normalize_phrase(chunk)
            if len(key) < self.min_phrase_chars:
                continue
            # Truncate to a fixed stem so a long recurring sentence is matched
            # even if its tail differs between recurrences.
            stem = key[:80]
            phrases.setdefault(stem, [0])
            phrases[stem][0] += 1
        for stem, counts in phrases.items():
            if counts[0] >= self.phrase_repeat_count:
                return stem, counts[0]
        return None


_PHRASE_NOISE = re.compile(r"\s+")


def _normalize_phrase(value: str) -> str:
    """Lowercase and collapse whitespace for phrase comparison."""
    return _PHRASE_NOISE.sub(" ", value).strip().lower()


def detect_loop_in_text(text: str, **kwargs: object) -> LoopVerdict:
    """One-shot check of a complete text. Convenience for tests/offline use."""
    detector = LoopDetector(**_coerce_kwargs(kwargs))
    return detector.feed(text)


def _coerce_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    # Filter to only the LoopDetector constructor args, ignoring extras.
    import inspect

    params = set(inspect.signature(LoopDetector.__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in params}

from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class DemoGenerationResult:
    answer: str
    reasoning: str | None
    finish_reason: str
    truncated: bool
    stopped: bool
    prompt_tokens: int
    visual_tokens: int
    generated_tokens: int
    preprocess_seconds: float
    generation_seconds: float
    tokens_per_second: float
    peak_vram_mib_per_device: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_live_text(value: str) -> tuple[str, str]:
    marker = "</think>"
    if marker in value:
        reasoning, answer = value.split(marker, 1)
        return reasoning.rsplit("<think>", 1)[-1].lstrip("\r\n"), answer.lstrip("\r\n")
    if "<think>" in value:
        return value.rsplit("<think>", 1)[-1].lstrip("\r\n"), ""
    return "", value


def model_input_devices(model: Any) -> tuple[Any, Any]:
    embedding = model.get_input_embeddings()
    embedding_device = next(embedding.parameters()).device
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        visual = getattr(model, "visual", None)
    if visual is None:
        raise RuntimeError("Qwen3-VL visual module was not found")
    visual_device = next(visual.parameters()).device
    if visual_device != embedding_device:
        raise RuntimeError(
            f"vision and input embeddings must share a device; got "
            f"{visual_device} and {embedding_device}"
        )
    return embedding_device, visual_device


def move_inputs_to_model_devices(model: Any, inputs: Any) -> tuple[Any, str, str]:
    import torch

    embedding_device, visual_device = model_input_devices(model)
    visual_keys = {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            target = visual_device if key in visual_keys else embedding_device
            inputs[key] = value.to(target)
    return inputs, str(embedding_device), str(visual_device)


def run_streaming_generation(
    runtime: Any,
    media_inputs: Sequence[tuple[str, Any]],
    prompt: str,
    history: Sequence[dict[str, str]],
    media_history_index: int | None,
    max_new_tokens: int,
    max_image_side: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_event: threading.Event,
    emit: Callable[[dict[str, Any]], None],
) -> DemoGenerationResult:
    import torch
    from transformers import (
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )

    from qwen3_vl_offline import (
        FiniteLogitsProcessor,
        _split_reasoning,
        build_messages,
    )

    class StopRequested(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return stop_event.is_set()

    class CountingStreamer(TextIteratorStreamer):
        def __init__(self, tokenizer):
            super().__init__(
                tokenizer,
                skip_prompt=True,
                timeout=0.5,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self.generated_tokens = 0

        def put(self, value):
            is_prompt = self.skip_prompt and self.next_tokens_are_prompt
            if not is_prompt:
                self.generated_tokens += int(value.numel())
            super().put(value)

    torch.manual_seed(runtime.seed)
    torch.cuda.manual_seed_all(runtime.seed)
    preprocessing_started = time.perf_counter()
    media = runtime.prepare_media(media_inputs, max_image_side)
    messages = build_messages(media, prompt, history, media_history_index)
    processor_kwargs: dict[str, Any] = {}
    if any(item.kind == "video" for item in media):
        processor_kwargs = {"num_frames": 32, "fps": None}
    inputs = runtime.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        add_vision_id=len(media) > 1,
        processor_kwargs=processor_kwargs,
    )
    prompt_tokens = int(inputs["input_ids"].shape[1])
    visual_ids = {
        int(value)
        for value in (
            getattr(runtime.model.config, "image_token_id", None),
            getattr(runtime.model.config, "video_token_id", None),
        )
        if value is not None
    }
    visual_tokens = sum(
        int((inputs["input_ids"] == token_id).sum().item()) for token_id in visual_ids
    )
    context_tokens = int(runtime.model.config.get_text_config().max_position_embeddings)
    if prompt_tokens + max_new_tokens > context_tokens:
        raise ValueError(
            f"prompt ({prompt_tokens}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"the model context limit ({context_tokens})"
        )
    inputs, input_device, visual_device = move_inputs_to_model_devices(
        runtime.model, inputs
    )
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
        torch.cuda.reset_peak_memory_stats(index)
    preprocess_seconds = time.perf_counter() - preprocessing_started
    emit(
        {
            "type": "prompt",
            "prompt_tokens": prompt_tokens,
            "visual_tokens": visual_tokens,
            "context_tokens": context_tokens,
            "preprocess_seconds": round(preprocess_seconds, 3),
            "input_device": input_device,
            "visual_device": visual_device,
        }
    )

    tokenizer = getattr(runtime.processor, "tokenizer", runtime.processor)
    streamer = CountingStreamer(tokenizer)
    generated: dict[str, Any] = {}
    failure: dict[str, BaseException] = {}
    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        use_cache=True,
        logits_processor=[FiniteLogitsProcessor()],
        stopping_criteria=StoppingCriteriaList([StopRequested()]),
        streamer=streamer,
        return_dict_in_generate=True,
    )

    def generate() -> None:
        try:
            with torch.inference_mode():
                generated["output"] = runtime.model.generate(**kwargs)
        except BaseException as exc:
            failure["error"] = exc
            streamer.end()

    generation_started = time.perf_counter()
    thread = threading.Thread(target=generate, daemon=True)
    thread.start()
    accumulated = ""
    previous_reasoning = ""
    previous_answer = ""
    last_stats = 0.0
    while True:
        try:
            piece = next(streamer)
        except queue.Empty:
            if not thread.is_alive():
                break
            continue
        except StopIteration:
            break
        accumulated += piece
        reasoning, answer = split_live_text(accumulated)
        if len(reasoning) >= len(previous_reasoning):
            delta = reasoning[len(previous_reasoning) :]
            if delta:
                emit({"type": "token", "phase": "reasoning", "text": delta})
            previous_reasoning = reasoning
        if len(answer) >= len(previous_answer):
            delta = answer[len(previous_answer) :]
            if delta:
                emit({"type": "token", "phase": "answer", "text": delta})
            previous_answer = answer
        now = time.perf_counter()
        if now - last_stats >= 0.25:
            elapsed = max(now - generation_started, 1e-9)
            emit(
                {
                    "type": "stats_live",
                    "generated_tokens": streamer.generated_tokens,
                    "tokens_per_second": round(streamer.generated_tokens / elapsed, 2),
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
            last_stats = now
    thread.join()
    if "error" in failure:
        raise failure["error"]
    output = generated["output"]
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
    generation_seconds = time.perf_counter() - generation_started
    continuation = output.sequences[:, prompt_tokens:]
    generated_tokens = int(continuation.shape[1])
    token_ids = tuple(int(value) for value in continuation[0].tolist())
    eos_value = runtime.model.generation_config.eos_token_id
    eos_ids = (
        {int(eos_value)}
        if isinstance(eos_value, int)
        else {int(value) for value in eos_value or []}
    )
    stopped = stop_event.is_set()
    if stopped:
        finish_reason = "stopped"
    elif token_ids and token_ids[-1] in eos_ids:
        finish_reason = "eos"
    elif generated_tokens >= max_new_tokens:
        finish_reason = "max_new_tokens"
    else:
        finish_reason = "stopped"
    clean_text = runtime.processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    raw_text = runtime.processor.batch_decode(
        continuation,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    reasoning, answer = _split_reasoning(
        raw_text, clean_text, tokenizer.all_special_tokens
    )
    peak = {
        str(index): round(torch.cuda.max_memory_allocated(index) / 1024**2, 1)
        for index in range(torch.cuda.device_count())
    }
    return DemoGenerationResult(
        answer=answer,
        reasoning=reasoning,
        finish_reason=finish_reason,
        truncated=finish_reason == "max_new_tokens",
        stopped=stopped,
        prompt_tokens=prompt_tokens,
        visual_tokens=visual_tokens,
        generated_tokens=generated_tokens,
        preprocess_seconds=preprocess_seconds,
        generation_seconds=generation_seconds,
        tokens_per_second=(
            generated_tokens / generation_seconds if generation_seconds else 0.0
        ),
        peak_vram_mib_per_device=peak,
    )

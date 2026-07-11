#!/usr/bin/env python3
"""Small local FastAPI Web UI for Qwen3-VL image/video chat."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import tempfile
import threading
from pathlib import Path
from typing import Any, Sequence

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from qwen3_vl_offline import DEFAULT_CKPT_DIR, Qwen3VLRuntime


MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_HISTORY_MESSAGES = 40
MAX_WEB_NEW_TOKENS = 40_960
MAX_PROMPT_CHARACTERS = 200_000


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3-VL local demo</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { max-width: 980px; margin: 0 auto; padding: 24px; background: #101218; color: #eef1f7; }
    h1 { margin-bottom: 4px; } .muted { color: #aeb6c7; }
    .panel { background: #191d27; border: 1px solid #303748; border-radius: 14px; padding: 18px; margin: 16px 0; }
    textarea, input, select, button { font: inherit; }
    textarea, input, select { box-sizing: border-box; width: 100%; color: inherit; background: #10141d; border: 1px solid #3b4458; border-radius: 8px; padding: 10px; }
    textarea { min-height: 100px; resize: vertical; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    label { display: grid; gap: 6px; color: #c8cfdd; }
    button { border: 0; border-radius: 9px; padding: 11px 16px; cursor: pointer; background: #5c78ff; color: white; font-weight: 650; }
    button.secondary { background: #343b4d; } button:disabled { opacity: .55; cursor: wait; }
    .actions { display: flex; gap: 10px; margin-top: 14px; }
    .turn { border-left: 3px solid #5c78ff; padding: 8px 12px; margin: 10px 0; white-space: pre-wrap; }
    .turn.user { border-color: #38bda4; } pre { white-space: pre-wrap; overflow-wrap: anywhere; }
    #error { color: #ff8e98; } details { margin-top: 10px; }
  </style>
</head>
<body>
  <h1>Qwen3-VL local demo</h1>
  <div id="status" class="muted">Loading status…</div>
  <div id="history" class="panel"></div>
  <form id="form" class="panel">
    <label>Images and/or video (multiple files are kept in selection order)
      <input id="files" name="files" type="file" accept="image/*,video/*" multiple>
      <span class="muted">Selected files remain attached to the first turn. Changing them starts a new chat.</span>
    </label>
    <label style="margin-top:12px">Prompt
      <textarea id="prompt" name="prompt">Describe the visual content completely and precisely.</textarea>
    </label>
    <div class="grid" style="margin-top:12px">
      <label>Maximum new tokens<input id="maxTokens" type="number" value="2048" min="1" max="40960"></label>
      <label>Pre-resize maximum image side<input id="maxSide" type="number" value="640" min="64" max="4096"></label>
      <label>Decoding
        <select id="sample"><option value="true">Qwen Thinking sampling preset</option><option value="false">Greedy (may loop)</option></select>
      </label>
      <label>Temperature<input id="temperature" type="number" value="0.6" min="0.01" max="5" step="0.05"></label>
      <label>Top-p<input id="topP" type="number" value="0.95" min="0.01" max="1" step="0.01"></label>
      <label>Top-k<input id="topK" type="number" value="20" min="1" max="1000"></label>
      <label>Video frames<input id="videoFrames" type="number" value="32" min="2" max="512"></label>
    </div>
    <div class="actions"><button id="submit" type="submit">Run</button><button id="clear" class="secondary" type="button">Clear chat</button></div>
    <div id="error"></div>
  </form>
  <div id="metrics" class="panel muted">No generation yet.</div>
<script>
let history = [];
const historyEl = document.getElementById('history');
function renderHistory() {
  historyEl.innerHTML = history.length ? '' : '<span class="muted">No chat turns yet.</span>';
  for (const item of history) {
    const div = document.createElement('div'); div.className = 'turn ' + item.role;
    const title = document.createElement('strong'); title.textContent = item.role === 'user' ? 'You' : 'Qwen3-VL';
    const body = document.createElement('div'); body.textContent = item.content;
    div.append(title, body); historyEl.appendChild(div);
  }
}
renderHistory();
fetch('/api/status').then(r => r.json()).then(s => {
  document.getElementById('status').textContent = `${s.model} · ${s.device_mode} · loaded in ${s.load_seconds.toFixed(2)}s`;
}).catch(e => document.getElementById('status').textContent = 'Status unavailable: ' + e);
document.getElementById('clear').onclick = () => { history = []; renderHistory(); document.getElementById('metrics').textContent = 'No generation yet.'; };
document.getElementById('files').onchange = () => {
  if (history.length) {
    history = []; renderHistory();
    document.getElementById('metrics').textContent = 'Visual selection changed; chat history was cleared.';
  }
};
document.getElementById('form').onsubmit = async (event) => {
  event.preventDefault(); const button = document.getElementById('submit'); const error = document.getElementById('error');
  button.disabled = true; error.textContent = '';
  const data = new FormData(); const files = document.getElementById('files').files;
  for (const file of files) data.append('files', file, file.name);
  data.append('prompt', document.getElementById('prompt').value);
  data.append('history_json', JSON.stringify(history));
  data.append('max_new_tokens', document.getElementById('maxTokens').value);
  data.append('max_image_side', document.getElementById('maxSide').value);
  data.append('do_sample', document.getElementById('sample').value);
  data.append('temperature', document.getElementById('temperature').value);
  data.append('top_p', document.getElementById('topP').value);
  data.append('top_k', document.getElementById('topK').value);
  data.append('video_num_frames', document.getElementById('videoFrames').value);
  try {
    const response = await fetch('/api/infer', {method:'POST', body:data}); const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Inference failed');
    history = payload.history; renderHistory();
    const r = payload.result;
    document.getElementById('metrics').innerHTML = `<strong>${r.finish_reason}</strong> · prompt ${r.prompt_tokens} tokens · generated ${r.generated_tokens} tokens · ${r.generation_seconds.toFixed(3)}s · ${r.tokens_per_second.toFixed(2)} tok/s` + (r.truncated ? '<br><b>Truncated: increase Maximum new tokens.</b>' : '') + (r.reasoning ? `<details><summary>Thinking</summary><pre>${escapeHtml(r.reasoning)}</pre></details>` : '');
  } catch (e) { error.textContent = e.message; } finally { button.disabled = false; }
};
function escapeHtml(value) { const div = document.createElement('div'); div.textContent = value; return div.innerHTML; }
</script>
</body></html>"""


def _validated_history(value: str) -> list[dict[str, str]]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("history_json is not valid JSON") from exc
    if not isinstance(parsed, list) or len(parsed) > MAX_HISTORY_MESSAGES:
        raise ValueError(f"history must be a list of at most {MAX_HISTORY_MESSAGES} messages")
    result: list[dict[str, str]] = []
    if len(parsed) % 2:
        raise ValueError("history must contain complete user/assistant pairs")
    for index, item in enumerate(parsed):
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            raise ValueError("history contains an invalid role")
        expected_role = "user" if index % 2 == 0 else "assistant"
        if item["role"] != expected_role:
            raise ValueError("history roles must alternate user/assistant")
        content = item.get("content")
        if not isinstance(content, str) or len(content) > 100_000:
            raise ValueError("history contains invalid content")
        result.append({"role": item["role"], "content": content})
    return result


async def _uploads_to_media(files: Sequence[UploadFile]) -> tuple[list[tuple[str, Any]], list[Path]]:
    media: list[tuple[str, Any]] = []
    temporary_paths: list[Path] = []
    total_bytes = 0
    for upload in files:
        chunks = bytearray()
        while chunk := await upload.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"uploads exceed the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit"
                )
            chunks.extend(chunk)
        data = bytes(chunks)
        content_type = (upload.content_type or "").lower()
        if content_type.startswith("image/"):
            with Image.open(io.BytesIO(data)) as source:
                media.append(("image", source.copy()))
        elif content_type.startswith("video/"):
            suffix = Path(upload.filename or "video.mp4").suffix or ".mp4"
            handle = tempfile.NamedTemporaryFile(prefix="qwen3-vl-", suffix=suffix, delete=False)
            handle.write(data)
            handle.close()
            path = Path(handle.name)
            temporary_paths.append(path)
            media.append(("video", str(path)))
        else:
            raise ValueError(f"unsupported upload type: {content_type or upload.filename}")
    return media, temporary_paths


def create_app(runtime: Qwen3VLRuntime) -> FastAPI:
    app = FastAPI(title="Qwen3-VL local demo", docs_url="/docs", redoc_url=None)
    inference_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return INDEX_HTML

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/api/status")
    async def status():
        return {
            "model": runtime.spec.repo_id,
            "device_mode": "gpu_fp8" if runtime.device == "cuda" else "cpu_fp32",
            "load_seconds": runtime.load_seconds,
            "fp8_modules": len(runtime.fp8_names),
        }

    @app.post("/api/infer")
    async def infer(
        prompt: str = Form(...),
        history_json: str = Form("[]"),
        max_new_tokens: int = Form(2048),
        max_image_side: int = Form(640),
        do_sample: bool = Form(True),
        temperature: float = Form(0.6),
        top_p: float = Form(0.95),
        top_k: int = Form(20),
        video_num_frames: int = Form(32),
        files: list[UploadFile] | None = File(None),
    ):
        temporary_paths: list[Path] = []
        try:
            if not 1 <= max_new_tokens <= MAX_WEB_NEW_TOKENS:
                raise ValueError(
                    f"max_new_tokens must be between 1 and {MAX_WEB_NEW_TOKENS}"
                )
            if not 64 <= max_image_side <= 4096:
                raise ValueError("max_image_side must be between 64 and 4096")
            if not 2 <= video_num_frames <= 512:
                raise ValueError("video_num_frames must be between 2 and 512")
            if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARACTERS:
                raise ValueError(
                    f"prompt must contain 1 to {MAX_PROMPT_CHARACTERS} characters"
                )
            if not 0 < temperature <= 5:
                raise ValueError("temperature must be in (0, 5]")
            if not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
            if not 1 <= top_k <= 1000:
                raise ValueError("top_k must be between 1 and 1000")
            history = _validated_history(history_json)
            media, temporary_paths = await _uploads_to_media(files or [])
            media_history_index = 0 if media and history else None

            def run():
                with inference_lock:
                    return runtime.infer(
                        media_inputs=media,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        max_image_side=max_image_side,
                        history=history,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        video_num_frames=video_num_frames,
                        media_history_index=media_history_index,
                    )[0]

            result = await asyncio.to_thread(run)
            new_history = history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": result.answer},
            ]
            if len(new_history) > MAX_HISTORY_MESSAGES:
                # Preserve the first visual turn and the newest complete pairs.
                new_history = new_history[:2] + new_history[-(MAX_HISTORY_MESSAGES - 2) :]
            return {"result": result.to_dict(), "history": new_history}
        except (ValueError, FileNotFoundError, RuntimeError, OSError, ArithmeticError) as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local Qwen3-VL Web UI")
    parser.add_argument("--model", choices=("2b", "4b", "8b"), default="2b")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-path")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--kernel-dir")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = Qwen3VLRuntime(
        model_size=args.model,
        device=args.device,
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir,
        kernel_dir=args.kernel_dir,
        cpu_threads=args.cpu_threads,
    )
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

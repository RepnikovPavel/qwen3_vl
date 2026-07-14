from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

from demo.generation import DemoGenerationResult, run_streaming_generation
from demo.grounding_3d_viz import draw_3d_bboxes, generate_camera_params, parse_bbox_3d_from_text
from demo.grounding_viz import draw_grounding, parse_grounding
from demo.model_manager import PLACEMENTS, DemoBusyError, DemoModelManager
from demo.sessions import SessionStore
from demo.tasks import (
    DemoTaskError,
    build_structured_result,
    public_presets,
    resolve_task,
)
from model_catalog import normalize_model_size


MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_UPLOAD_FILES = 16
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_DIMENSION = 16_384
MAX_VIDEO_DIMENSION = 8_192
MAX_VIDEO_SECONDS = 7_200
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}
LOGGER = logging.getLogger("qwen3_vl.demo")


def _sse(value: dict[str, Any]) -> str:
    return f"data: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}\n\n"


class Generation:
    def __init__(
        self,
        session_id: str,
        prompt: str,
        model_id: str,
        placement: str,
        task: str,
    ):
        self.session_id = session_id
        self.prompt = prompt
        self.model_id = model_id
        self.placement = placement
        self.task = task
        self.stop_event = threading.Event()
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.reasoning = ""
        self.answer = ""
        self.stats: dict[str, Any] = {}
        self.result: dict[str, Any] | None = None
        self.structured: dict[str, Any] | None = None
        self.error: str | None = None
        self.done = False
        self.worker_thread: threading.Thread | None = None
        self._sequence = 0
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=4096)
        self._condition = threading.Condition()

    def emit(self, event: dict[str, Any]) -> None:
        with self._condition:
            event_type = event.get("type")
            if event_type == "token":
                phase = event.get("phase")
                if phase == "reasoning":
                    self.reasoning += str(event.get("text", ""))
                elif phase == "answer":
                    self.answer += str(event.get("text", ""))
            elif event_type in {"prompt", "stats_live", "loading"}:
                self.stats.update(event)
            self._events.append((self._sequence, event))
            self._sequence += 1
            self._condition.notify_all()

    def complete(
        self,
        result: DemoGenerationResult,
        structured: dict[str, Any] | None,
    ) -> None:
        event = {
            "type": "done",
            "result": result.to_dict(),
            "structured": structured,
        }
        with self._condition:
            self.result = event["result"]
            self.structured = structured
            self.reasoning = result.reasoning or self.reasoning
            self.answer = result.answer
            self.done = True
            self.finished_at = time.time()
            self._events.append((self._sequence, event))
            self._sequence += 1
            self._condition.notify_all()

    def fail(self, message: str) -> None:
        event = {"type": "error", "message": message}
        with self._condition:
            self.error = message
            self.done = True
            self.finished_at = time.time()
            self._events.append((self._sequence, event))
            self._sequence += 1
            self._condition.notify_all()

    def public_status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "session_id": self.session_id,
                "model_id": self.model_id,
                "placement": self.placement,
                "task": self.task,
                "done": self.done,
                "stop_requested": self.stop_event.is_set(),
                "error": self.error,
                "created_at": self.created_at,
            }

    def stream(self) -> Iterator[str]:
        with self._condition:
            cursor = self._sequence
            snapshot = {
                "type": "snapshot",
                "session_id": self.session_id,
                "prompt": self.prompt,
                "model_id": self.model_id,
                "placement": self.placement,
                "task": self.task,
                "reasoning": self.reasoning,
                "answer": self.answer,
                "stats": self.stats,
                "result": self.result,
                "structured": self.structured,
                "done": self.done,
                "error": self.error,
            }
        yield _sse(snapshot)
        while True:
            with self._condition:
                replay = None
                if self._events and self._events[0][0] > cursor:
                    replay = {
                        "type": "snapshot",
                        "session_id": self.session_id,
                        "prompt": self.prompt,
                        "model_id": self.model_id,
                        "placement": self.placement,
                        "task": self.task,
                        "reasoning": self.reasoning,
                        "answer": self.answer,
                        "stats": self.stats,
                        "result": self.result,
                        "structured": self.structured,
                        "done": self.done,
                        "error": self.error,
                    }
                    cursor = self._sequence
                    available = []
                else:
                    available = [
                        event for sequence, event in self._events if sequence >= cursor
                    ]
                    if available:
                        cursor = self._sequence
                done = self.done
                if replay is None and not available and not done:
                    self._condition.wait(timeout=10)
                    if self._events and self._events[0][0] > cursor:
                        replay = {
                            "type": "snapshot",
                            "session_id": self.session_id,
                            "prompt": self.prompt,
                            "model_id": self.model_id,
                            "placement": self.placement,
                            "task": self.task,
                            "reasoning": self.reasoning,
                            "answer": self.answer,
                            "stats": self.stats,
                            "result": self.result,
                            "structured": self.structured,
                            "done": self.done,
                            "error": self.error,
                        }
                        cursor = self._sequence
                    else:
                        available = [
                            event
                            for sequence, event in self._events
                            if sequence >= cursor
                        ]
                        if available:
                            cursor = self._sequence
                    done = self.done
            if replay is not None:
                yield _sse(replay)
            for event in available:
                yield _sse(event)
            if done and replay is None and not available:
                return
            if replay is None and not available:
                yield ": ping\n\n"


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    placement: str = "single"
    keep_model_loaded: bool = False


class RetentionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keep_model_loaded: bool = False


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str | None = None
    title: str | None = None


class RenameSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DemoBusyError):
        return HTTPException(409, str(exc))
    if isinstance(
        exc, (ValueError, TypeError, KeyError, FileNotFoundError, DemoTaskError)
    ):
        return HTTPException(400, str(exc).strip("'"))
    LOGGER.exception("demo request failed", exc_info=exc)
    return HTTPException(500, f"{type(exc).__name__}: request failed")


def _remove_paths(paths: list[Path] | None, media_root: Path) -> None:
    for path in paths or []:
        try:
            resolved = path.resolve()
            if resolved.is_relative_to(media_root):
                resolved.unlink(missing_ok=True)
                try:
                    resolved.parent.rmdir()
                except OSError:
                    pass
        except OSError:
            pass


def _inspect_image(path: Path, original_name: str) -> dict[str, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            if (
                width < 1
                or height < 1
                or width > MAX_IMAGE_DIMENSION
                or height > MAX_IMAGE_DIMENSION
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ValueError(f"image dimensions are too large: {original_name}")
            image.verify()
        return {"width": width, "height": height}
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"invalid image: {original_name}") from exc


def _inspect_video(
    path: Path, original_name: str, mime_type: str
) -> dict[str, int | float]:
    with path.open("rb") as source:
        header = source.read(16)
    if mime_type in {"video/mp4", "video/quicktime"}:
        valid_header = len(header) >= 12 and header[4:8] == b"ftyp"
    else:
        valid_header = header.startswith(b"\x1aE\xdf\xa3")
    if not valid_header:
        raise ValueError(f"invalid video: {original_name}")
    try:
        import av
    except ImportError:
        return {}
    try:
        with av.open(path) as container:
            stream = next(
                (item for item in container.streams if item.type == "video"), None
            )
            if stream is None:
                raise ValueError(f"video stream not found: {original_name}")
            width = int(stream.codec_context.width or 0)
            height = int(stream.codec_context.height or 0)
            if (
                width < 1
                or height < 1
                or width > MAX_VIDEO_DIMENSION
                or height > MAX_VIDEO_DIMENSION
            ):
                raise ValueError(f"video dimensions are invalid: {original_name}")
            duration = (
                float(container.duration / av.time_base) if container.duration else 0.0
            )
            if duration > MAX_VIDEO_SECONDS:
                raise ValueError(f"video is too long: {original_name}")
            return {"width": width, "height": height, "duration_seconds": duration}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid video: {original_name}") from exc


async def _save_uploads(
    uploads: list[UploadFile],
    session_id: str,
    store: SessionStore,
    media_root: Path,
) -> list[dict[str, Any]]:
    if len(uploads) > MAX_UPLOAD_FILES:
        raise ValueError(f"at most {MAX_UPLOAD_FILES} files may be uploaded")
    session_dir = media_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[Path, str, str, str, int, str, dict[str, Any] | None]] = []
    total_bytes = 0
    try:
        for upload in uploads:
            mime_type = (upload.content_type or "").lower()
            if mime_type in ALLOWED_IMAGE_TYPES:
                media_type = "image"
                suffix = ALLOWED_IMAGE_TYPES[mime_type]
            elif mime_type in ALLOWED_VIDEO_TYPES:
                media_type = "video"
                suffix = ALLOWED_VIDEO_TYPES[mime_type]
            else:
                raise ValueError(
                    f"unsupported upload type: {mime_type or upload.filename}"
                )
            original_name = Path(upload.filename or f"upload{suffix}").name
            if original_name in {"", ".", ".."}:
                original_name = f"upload{suffix}"
            path = session_dir / f"{uuid.uuid4().hex}{suffix}"
            digest = hashlib.sha256()
            size = 0
            with path.open("xb") as output:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise ValueError(
                            f"uploads exceed the {MAX_UPLOAD_BYTES // 1024**2} MiB limit"
                        )
                    digest.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise ValueError(f"empty upload: {original_name}")
            metadata = None
            if media_type == "image":
                metadata = await asyncio.to_thread(_inspect_image, path, original_name)
            else:
                metadata = await asyncio.to_thread(
                    _inspect_video,
                    path,
                    original_name,
                    mime_type,
                )
            saved.append(
                (
                    path,
                    media_type,
                    original_name,
                    mime_type,
                    size,
                    digest.hexdigest(),
                    metadata,
                )
            )
        result = []
        for path, media_type, original_name, mime_type, size, digest, metadata in saved:
            result.append(
                store.register_media(
                    session_id,
                    stored_path=path,
                    media_type=media_type,
                    original_name=original_name,
                    mime_type=mime_type,
                    size_bytes=size,
                    sha256=digest,
                    metadata=metadata,
                )
            )
        return result
    except BaseException:
        cleanup_paths = store.reset_conversation(session_id)
        for cleanup_path in cleanup_paths or []:
            cleanup_path.unlink(missing_ok=True)
        for path, *_ in saved:
            path.unlink(missing_ok=True)
        for path in session_dir.glob("*"):
            if path.is_file() and not any(path == item[0] for item in saved):
                path.unlink(missing_ok=True)
        try:
            session_dir.rmdir()
        except OSError:
            pass
        raise


_COORD_SKILLS = {
    "2d_grounding", "3d_grounding", "spatial_understanding",
    "omni_recognition", "ocr_spotting", "computer_use", "mobile_agent",
}


def _build_skill_overlays(
    skill_key: str,
    answer: str,
    session_id: str,
    store: SessionStore,
    media_root: Path,
) -> tuple[list[dict[str, Any]] | None, dict[str, int] | None]:
    """Parse a coordinate skill answer into normalized (0..1) overlays.

    Returns (overlays, image_size). overlays items:
      {"kind": "box"|"point"|"poly", "pts": [[x,y],...], "label": "...", "extra": {...}}
    All coords normalized to 0..1 relative to the ORIGINAL image, so the client
    can draw them at any zoom level. For 3D, bbox_3d is projected via camera
    intrinsics (fov=60 fallback, like the cookbook) then normalized.
    """
    from skills import SKILLS
    from skill_parsers import parse_skill

    if skill_key not in _COORD_SKILLS:
        return None, None
    spec = SKILLS[skill_key]
    scale = spec.coord_scale or 1000
    try:
        parsed = parse_skill(skill_key, answer)
    except Exception:
        parsed = []
    if not isinstance(parsed, list) or not parsed:
        return None, None

    # Find the original image + its size from the session's first media item.
    session = store.get_session(session_id)
    image_size: dict[str, int] | None = None
    orig_path: Path | None = None
    if session and session.get("media"):
        first_id = session["media"][0].get("id")
        if first_id:
            item = store.get_media(first_id, include_stored_path=True)
            if item and item.get("media_type") == "image" and item.get("metadata"):
                md = item["metadata"]
                if isinstance(md, dict) and "width" in md and "height" in md:
                    image_size = {"width": int(md["width"]), "height": int(md["height"])}
                    orig_path = Path(item["stored_path"])
    if image_size is None:
        return None, None
    w, h = image_size["width"], image_size["height"]

    overlays: list[dict[str, Any]] = []

    if skill_key == "3d_grounding":
        # Project 3D boxes to 2D corners, then normalize.
        from PIL import Image as _PILImage
        from demo.grounding_3d_viz import generate_camera_params, convert_3dbbox
        try:
            if orig_path is not None:
                cam = generate_camera_params(_PILImage.open(orig_path).convert("RGB"), fov=60.0)
            else:
                cam = {"fx": w / 2, "fy": h / 2, "cx": w / 2, "cy": h / 2}
        except Exception:
            cam = {"fx": w / 2, "fy": h / 2, "cx": w / 2, "cy": h / 2}
        for item in parsed:
            bbox_3d = item.get("bbox_3d")
            if not bbox_3d or len(bbox_3d) < 9:
                continue
            corners = convert_3dbbox(list(bbox_3d), cam)
            if len(corners) < 8:
                continue
            poly = [[c[0] / w, c[1] / h] for c in corners]
            overlays.append({
                "kind": "poly", "pts": poly,
                "label": item.get("label", ""),
                "extra": {"bbox_3d": list(bbox_3d)},
            })
        return overlays, image_size

    # 2D / point skills: bbox_2d [x1,y1,x2,y2] or point_2d [x,y], scale 0..N.
    for item in parsed:
        label = item.get("label") or item.get("name") or item.get("text_content") or ""
        extra = {k: v for k, v in item.items()
                 if k not in {"bbox_2d", "point_2d", "label", "name", "text_content"}}
        if "bbox_2d" in item and len(item["bbox_2d"]) >= 4:
            x1, y1, x2, y2 = item["bbox_2d"][:4]
            overlays.append({
                "kind": "box",
                "pts": [[x1 / scale, y1 / scale], [x2 / scale, y2 / scale]],
                "label": str(label), "extra": extra,
            })
        elif "point_2d" in item and len(item["point_2d"]) >= 2:
            x, y = item["point_2d"][:2]
            overlays.append({
                "kind": "point",
                "pts": [[x / scale, y / scale]],
                "label": str(label), "extra": extra,
            })
    return overlays, image_size


def create_app(
    manager: DemoModelManager,
    store: SessionStore,
    state_dir: str | os.PathLike[str],
) -> FastAPI:
    state_root = Path(state_dir).expanduser().resolve()
    media_root = (state_root / "media").resolve()
    media_root.mkdir(parents=True, exist_ok=True)
    generations: dict[str, Generation] = {}
    generations_lock = threading.RLock()
    session_chat_locks: dict[str, asyncio.Lock] = {}

    def prune_generations() -> None:
        threshold = time.time() - 300
        with generations_lock:
            expired = [
                key
                for key, generation in generations.items()
                if generation.done
                and generation.finished_at is not None
                and generation.finished_at < threshold
            ]
            for key in expired:
                generations.pop(key, None)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start()
        yield
        with generations_lock:
            active = list(generations.values())
        for generation in active:
            generation.stop_event.set()
        deadline = time.monotonic() + 30
        for generation in active:
            worker_thread = generation.worker_thread
            if worker_thread is not None and worker_thread.is_alive():
                await asyncio.to_thread(
                    worker_thread.join,
                    max(0.0, deadline - time.monotonic()),
                )
        manager.close()

    app = FastAPI(
        title="Qwen3-VL FP8 demo",
        version="1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.store = store
    app.state.generations = generations

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        models = manager.models()
        ready = manager.status()["visible_gpus"] > 0 and any(
            item["available"] and manager.checkpoint_ready(item["id"])
            for item in models
        )
        payload = {"status": "ready" if ready else "not_ready", "models": models}
        return JSONResponse(payload, status_code=200 if ready else 503)

    @app.get("/api/models")
    def models():
        return {"models": manager.models()}

    @app.get("/api/tasks")
    def tasks():
        return public_presets()

    @app.get("/api/skills")
    def skills():
        from skills import public_skills

        return public_skills()

    @app.get("/api/status")
    def status():
        prune_generations()
        value = manager.status()
        with generations_lock:
            value["active_generations"] = [
                generation.public_status()
                for generation in generations.values()
                if not generation.done
            ]
        return value

    @app.get("/api/memory")
    def memory():
        return manager.memory()

    @app.post("/api/load")
    def load(request: LoadRequest):
        runtime = None
        acquired = False
        try:
            manager.acquire()
            acquired = True
            manager.set_keep_model_loaded(
                request.keep_model_loaded,
                unload_if_idle=False,
            )
            runtime = manager.load(request.model_id, request.placement, yarn_1m=True)
            response = {
                "ok": True,
                "model_id": runtime.model_size,
                "repo_id": runtime.spec.repo_id,
                "placement": runtime.gpu_placement,
                "load_seconds": runtime.load_seconds,
                "keep_model_loaded": request.keep_model_loaded,
            }
            runtime = None
            try:
                response["unloaded"] = manager.release(auto_unload=True)
            finally:
                acquired = False
            return response
        except Exception as exc:
            raise _http_error(exc) from exc
        finally:
            if acquired:
                runtime = None
                try:
                    manager.release(auto_unload=True)
                except Exception:
                    LOGGER.exception("model cleanup failed")

    @app.post("/api/retention")
    def retention(request: RetentionRequest):
        try:
            manager.set_keep_model_loaded(request.keep_model_loaded)
            return manager.status()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/unload")
    def unload():
        try:
            with manager.operation():
                unloaded = manager.unload()
            return {"ok": True, "unloaded": unloaded}
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/sessions")
    def list_sessions():
        return {"sessions": store.list_sessions()}

    @app.post("/api/sessions")
    def create_session(request: CreateSessionRequest):
        try:
            return store.create_session(request.model_id, request.title)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        try:
            session = store.get_session(session_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        if session is None:
            raise HTTPException(404, "session not found")
        with generations_lock:
            generation = generations.get(session_id)
        session["generation"] = generation.public_status() if generation else None
        return session

    @app.patch("/api/sessions/{session_id}")
    def rename_session(session_id: str, request: RenameSessionRequest):
        try:
            renamed = store.rename_session(session_id, request.title)
        except Exception as exc:
            raise _http_error(exc) from exc
        if not renamed:
            raise HTTPException(404, "session not found")
        return {"ok": True}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        with generations_lock:
            generation = generations.get(session_id)
        if generation is not None and not generation.done:
            raise HTTPException(
                409, "stop the active generation before deleting the session"
            )
        try:
            paths = store.delete_session(session_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        if paths is None:
            raise HTTPException(404, "session not found")
        _remove_paths(paths, media_root)
        with generations_lock:
            generations.pop(session_id, None)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/reset")
    def reset_session(session_id: str):
        with generations_lock:
            generation = generations.get(session_id)
        if generation is not None and not generation.done:
            raise HTTPException(
                409, "stop the active generation before resetting the session"
            )
        try:
            paths = store.reset_conversation(session_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        if paths is None:
            raise HTTPException(404, "session not found")
        _remove_paths(paths, media_root)
        with generations_lock:
            generations.pop(session_id, None)
        return {"ok": True}

    @app.get("/api/media/{media_id}")
    def media(media_id: str):
        try:
            item = store.get_media(media_id, include_stored_path=True)
        except Exception as exc:
            raise _http_error(exc) from exc
        if item is None:
            raise HTTPException(404, "media not found")
        path = item["stored_path"].resolve()
        if not path.is_relative_to(media_root) or not path.is_file():
            raise HTTPException(404, "media file not found")
        return FileResponse(path, media_type=item["mime_type"])

    # ------------------------------ 2D Grounding (reproduces 2d_grounding.ipynb) ------------------------------

    class GroundingResponse(BaseModel):
        text: str
        annotated_media_id: str | None = None
        parsed: list[dict[str, Any]] = []
        width: int | None = None
        height: int | None = None
        tokens_per_second: float | None = None

    @app.post("/api/grounding", response_model=GroundingResponse)
    async def grounding(
        image: UploadFile = File(...),
        prompt: str = Form('Locate every instance that belongs to the following categories: "car, person, vehicle". Report bbox coordinates in JSON format.'),
        max_new_tokens: int = Form(200000),
        max_image_side: int = Form(0),
        model_size: str = Form("2b"),
    ):
        """2D Grounding mode.

        Replicates the key flows from the official 2d_grounding.ipynb:
        - natural language prompts for bbox / point grounding
        - JSON output with bbox_2d / point_2d + optional extra fields (label, color, type, role, ...)
        - server-side visualization (boxes + points drawn on the image)
        """
        if not (image.content_type or "").startswith("image/"):
            raise HTTPException(400, "2D grounding currently supports single images")

        # Align to cookbooks + force <think> separation so final answer is clean JSON for drawing.
        if "Report bbox coordinates in JSON format" not in prompt:
            prompt = prompt.rstrip(". ") + (
                ' First write step-by-step reasoning inside <think> </think> tags. '
                'After </think> output ONLY JSON like [{"bbox_2d": [x1, y1, x2, y2], "label": "car"}].'
            )

        media_root = Path(os.environ.get("DEMO_STATE_DIR", "/state")) / "media"
        media_root.mkdir(parents=True, exist_ok=True)

        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        tmp_path = media_root / f"ground_{uuid.uuid4().hex}{suffix}"
        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image too large")
        tmp_path.write_bytes(data)

        # obtain runtime. Use single-GPU placement: 2B/8B FP8 fit on one 4090,
        # and transformers' MRoPE get_rope_index breaks under multi-GPU (balanced).
        try:
            with manager.operation():
                rt = manager.load(model_size, "single", yarn_1m=True)
        except DemoBusyError:
            raise HTTPException(503, "model is busy")
        except Exception as exc:
            raise HTTPException(500, f"failed to load model: {exc}")

        try:
            media = rt.prepare_media([("image", str(tmp_path))], max_image_side)

            # Build a minimal chat turn (same as regular path)
            from qwen3_vl_offline import build_messages  # type: ignore

            messages = build_messages(media, prompt, [], None)
            inputs = rt.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            from demo.generation import move_inputs_to_model_devices
            inputs, _, _ = move_inputs_to_model_devices(rt.model, inputs)

            import torch
            with torch.inference_mode():
                out = rt.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
            cont = out.sequences[:, inputs["input_ids"].shape[1] :]
            raw = rt.processor.batch_decode(
                cont, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            # If Thinking model used <think>, prefer the part after it for the "final" text
            # (this makes returned text + client viz cleaner, matching chat flow split).
            clean_text = raw.strip()
            if "</think>" in raw:
                try:
                    clean_text = raw.split("</think>", 1)[1].strip()
                except Exception:
                    clean_text = raw.strip()

            parsed = parse_grounding(clean_text)
            # Use exactly the image that was fed to the model (post any resize in prepare_media)
            # so that bbox coordinates match the displayed image.
            if media and len(media) > 0 and media[0].kind == "image" and hasattr(media[0], "value") and media[0].value is not None:
                base_for_draw = media[0].value
            else:
                base_for_draw = Image.open(tmp_path)
            orig = base_for_draw.convert("RGB")
            annotated = draw_grounding(orig, parsed)
            ann_name = f"ground_{uuid.uuid4().hex}.png"
            ann_path = media_root / ann_name
            annotated.save(ann_path)

            return GroundingResponse(
                text=clean_text,
                annotated_media_id=ann_name,
                parsed=parsed,
                width=orig.width,
                height=orig.height,
            )
        except Exception as exc:
            raise HTTPException(500, f"grounding inference failed: {type(exc).__name__}: {exc}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ------------------------------ 3D Grounding (reproduces 3d_grounding.ipynb) ------------------------------

    class Grounding3DResponse(BaseModel):
        text: str
        annotated_media_id: str | None = None
        parsed: list[dict[str, Any]] = []
        cam_params: dict[str, float] | None = None
        width: int | None = None
        height: int | None = None

    @app.post("/api/grounding_3d", response_model=Grounding3DResponse)
    async def grounding_3d(
        image: UploadFile = File(...),
        prompt: str = Form('Find all cars in this image. For each car, provide its 3D bounding box. The output format required is JSON.'),
        max_new_tokens: int = Form(200000),
        max_image_side: int = Form(0),
        model_size: str = Form("2b"),
        fov: float = Form(60.0),
    ):
        """3D Grounding mode.

        Replicates flows from 3d_grounding.ipynb:
        - prompts for 3D bbox output
        - camera param generation (default or from image)
        - server-side projection + drawing of 3D wireframes
        """
        if not (image.content_type or "").startswith("image/"):
            raise HTTPException(400, "3D grounding supports single images")

        # Align to cookbook 3d prompts
        if "JSON" not in prompt and "bbox_3d" not in prompt.lower():
            prompt = prompt.rstrip(". ") + ". The output format required is JSON."

        media_root = Path(os.environ.get("DEMO_STATE_DIR", "/state")) / "media"
        media_root.mkdir(parents=True, exist_ok=True)

        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        tmp_path = media_root / f"ground3d_{uuid.uuid4().hex}{suffix}"
        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image too large")
        tmp_path.write_bytes(data)

        try:
            with manager.operation():
                rt = manager.load(model_size, "single", yarn_1m=True)
        except DemoBusyError:
            raise HTTPException(503, "model busy")
        except Exception as exc:
            raise HTTPException(500, f"failed to load model: {exc}")

        try:
            media = rt.prepare_media([("image", str(tmp_path))], max_image_side)

            from qwen3_vl_offline import build_messages  # type: ignore
            messages = build_messages(media, prompt, [], None)
            inputs = rt.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            from demo.generation import move_inputs_to_model_devices
            inputs, _, _ = move_inputs_to_model_devices(rt.model, inputs)

            import torch
            with torch.inference_mode():
                out = rt.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True)
            cont = out.sequences[:, inputs["input_ids"].shape[1]:]
            raw = rt.processor.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            clean_text = raw.strip()
            if "</think>" in raw:
                try:
                    clean_text = raw.split("</think>", 1)[1].strip()
                except Exception:
                    clean_text = raw.strip()

            parsed = parse_bbox_3d_from_text(clean_text)
            orig = Image.open(tmp_path).convert("RGB")
            cam = generate_camera_params(orig, fov=fov)
            annotated = draw_3d_bboxes(orig, cam, parsed)
            ann_name = f"ground3d_{uuid.uuid4().hex}.png"
            ann_path = media_root / ann_name
            annotated.save(ann_path)

            return Grounding3DResponse(
                text=clean_text,
                annotated_media_id=ann_name,
                parsed=parsed,
                cam_params=cam,
                width=orig.width,
                height=orig.height,
            )
        except Exception as exc:
            raise HTTPException(500, f"3d grounding failed: {type(exc).__name__}: {exc}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @app.post("/api/chat")
    async def chat(
        session_id: str = Form(...),
        model_id: str = Form(...),
        placement: str = Form("single"),
        task: str = Form("describe"),
        skill: str | None = Form(None),
        custom_prompt: str | None = Form(None),
        max_new_tokens: int | None = Form(None),
        max_image_side: int | None = Form(None),
        do_sample: bool = Form(True),
        temperature: float = 0.6,
        top_p: float = Form(0.95),
        top_k: int = Form(20),
        video_num_frames: int = Form(32),
        keep_model_loaded: bool = Form(False),
        files: list[UploadFile] | None = File(None),
    ):
        acquired = False
        uploaded_this_request = False
        handed_to_worker = False
        try:
            session_lock = session_chat_locks.setdefault(session_id, asyncio.Lock())
            async with session_lock:
                resolved = resolve_task(
                    task,
                    custom_prompt=custom_prompt,
                    max_new_tokens=max_new_tokens,
                    max_image_side=max_image_side,
                )
                # Chat prompt ALWAYS comes from the textarea (custom_prompt). Skill templates
                # are never injected at launch time. Task only selects mode (max tokens, output kind, viz).
                # For grounding we append strict JSON instruction so final answer is parseable for drawing.
                user_prompt = (custom_prompt or "").strip()
                effective_prompt = user_prompt or resolved["prompt"]
                if resolved.get("task") == "grounding_2d" and effective_prompt:
                    # Use phrasing from the official cookbooks to force trained JSON output.
                    # Explicitly ask for <think> tags so thinking and final JSON are separated.
                    if "Report bbox coordinates in JSON format" not in effective_prompt:
                        effective_prompt = effective_prompt.rstrip() + (
                            ' First write your step-by-step reasoning inside <think> and </think> tags. '
                            'After the </think> output ONLY a JSON array like [{"bbox_2d": [x1, y1, x2, y2], "label": "car"}]. '
                            'No text after the closing think tag.'
                        )
                if resolved.get("task") == "grounding_3d" and effective_prompt:
                    if "provide its 3D bounding box" not in effective_prompt.lower() and "bbox_3d" not in effective_prompt.lower():
                        effective_prompt = effective_prompt.rstrip() + ' Provide 3D bounding boxes in JSON format.'
                model_key = normalize_model_size(model_id)
                if placement not in PLACEMENTS:
                    raise ValueError(f"unsupported placement: {placement}")
                if not 0 < temperature <= 5:
                    raise ValueError("temperature must be in (0, 5]")
                if not 0 < top_p <= 1:
                    raise ValueError("top_p must be in (0, 1]")
                if not 1 <= top_k <= 1000:
                    raise ValueError("top_k must be between 1 and 1000")
                if not 2 <= video_num_frames <= 256:
                    raise ValueError("video_num_frames must be between 2 and 256")
                session = store.get_session(session_id)
                if session is None:
                    raise HTTPException(404, "session not found")
                if session["model_id"] is not None:
                    session_model = normalize_model_size(session["model_id"])
                    if session_model != model_key:
                        raise HTTPException(
                            409,
                            "session model differs from the requested FP8 model; start a new session",
                        )
                with generations_lock:
                    existing = generations.get(session_id)
                if existing is not None and not existing.done:
                    raise HTTPException(
                        409, "generation already active for this session"
                    )
                uploads = files or []
                if uploads:
                    if session["messages"] or session["media"]:
                        raise ValueError(
                            "media can only be uploaded to an empty session"
                        )
                    await _save_uploads(uploads, session_id, store, media_root)
                    uploaded_this_request = True
                manager.acquire()
                acquired = True
                manager.set_keep_model_loaded(
                    keep_model_loaded,
                    unload_if_idle=False,
                )
                generation = Generation(
                    session_id,
                    effective_prompt,
                    model_key,
                    placement,
                    resolved["task"],
                )
                with generations_lock:
                    generations[session_id] = generation

            def worker() -> None:
                runtime = None
                completed: (
                    tuple[
                        DemoGenerationResult,
                        dict[str, Any] | None,
                    ]
                    | None
                ) = None
                failure: str | None = None
                try:
                    generation.emit({"type": "loading", "state": "loading_model"})
                    runtime = manager.load(model_key, placement, yarn_1m=True)
                    generation.emit(
                        {
                            "type": "loading",
                            "state": "generating",
                            "load_seconds": round(float(runtime.load_seconds), 3),
                        }
                    )
                    current = store.get_session(session_id)
                    if current is None:
                        raise RuntimeError("session was deleted")
                    history = [
                        {"role": item["role"], "content": item["content"]}
                        for item in current["messages"]
                        if item["role"] in {"user", "assistant", "system"}
                    ]
                    internal_media = [
                        store.get_media(item["id"], include_stored_path=True)
                        for item in current["media"]
                    ]
                    media_inputs = [
                        (item["media_type"], str(item["stored_path"]))
                        for item in internal_media
                        if item is not None
                    ]
                    media_history_index = 0 if media_inputs and history else None
                    result = run_streaming_generation(
                        runtime,
                        media_inputs,
                        effective_prompt,
                        history,
                        media_history_index,
                        resolved["max_new_tokens"],
                        resolved["max_image_side"],
                        do_sample,
                        temperature,
                        top_p,
                        top_k,
                        generation.stop_event,
                        generation.emit,
                        video_num_frames=video_num_frames,
                    )

                    # Ensure clean separation: if the final result.answer contains the think marker (or the streamed
                    # accumulation had it), carve out only the post-</think> part for "Final answer".
                    # This prevents the entire thinking from leaking into the final answer pane.
                    final_reasoning = result.reasoning
                    final_answer = result.answer
                    combined = (final_answer or "") + " " + (final_reasoning or "")
                    if "</think>" in combined:
                        # prefer the last </think>
                        marker_pos = combined.rfind("</think>")
                        before = combined[:marker_pos]
                        after = combined[marker_pos + 8 :]
                        if not final_reasoning:
                            final_reasoning = before.rsplit("<think>", 1)[-1].strip() or final_reasoning
                        if after.strip():
                            final_answer = after.strip()

                    structured = build_structured_result(
                        resolved["task"], final_answer
                    )
                    # For coordinate-bearing skills, parse model output into
                    # normalized overlays (0..1) the client can draw on a canvas.
                    overlays = None
                    image_size = None
                    if skill and final_answer:
                        overlays, image_size = _build_skill_overlays(
                            skill, final_answer, session_id, store, media_root
                        )
                    if overlays is not None:
                        structured = dict(structured) if structured else {}
                        structured["overlays"] = overlays
                        structured["image_size"] = image_size
                    assistant_content = final_answer
                    if not assistant_content and not final_reasoning:
                        assistant_content = (
                            "[stopped]" if result.stopped else "[empty response]"
                        )
                    # Record the prompt the user actually typed in chat (not the internal format suffix we added for grounding).
                    store.append_turn(
                        session_id,
                        user_prompt or effective_prompt,
                        assistant_content,
                        reasoning=final_reasoning,
                        metrics={
                            "task": resolved["task"],
                            "generation": result.to_dict(),
                            "structured": structured,
                        },
                    )
                    # Use cleaned for the event result shown to client
                    cleaned_result = result
                    try:
                        from dataclasses import replace
                        cleaned_result = replace(result, reasoning=final_reasoning, answer=final_answer)
                    except Exception:
                        pass
                    completed = (cleaned_result, structured)
                except Exception as exc:
                    LOGGER.exception("generation failed")
                    failure = f"{type(exc).__name__}: generation failed"
                finally:
                    runtime = None
                    try:
                        manager.release(auto_unload=True)
                    except Exception as exc:
                        LOGGER.exception("model cleanup failed")
                        failure = f"{type(exc).__name__}: model cleanup failed"
                    # Extra: ensure CUDA memory is released even after errors
                    try:
                        import gc
                        import torch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                if failure is not None:
                    generation.fail(failure)
                elif completed is not None:
                    generation.complete(*completed)

            worker_thread = threading.Thread(target=worker, daemon=True)
            generation.worker_thread = worker_thread
            worker_thread.start()
            acquired = False
            handed_to_worker = True
            return StreamingResponse(
                generation.stream(),
                media_type="text/event-stream",
                headers={
                    "X-Session-Id": session_id,
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _http_error(exc) from exc
        finally:
            if acquired:
                manager.release(auto_unload=True)
            if uploaded_this_request and not handed_to_worker:
                cleanup_paths = store.reset_conversation(session_id)
                _remove_paths(cleanup_paths, media_root)
            # Extra safety for GPU memory after any error path
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    @app.post("/api/stop/{session_id}")
    def stop(session_id: str):
        with generations_lock:
            generation = generations.get(session_id)
        if generation is None or generation.done:
            return {"ok": False, "reason": "no active generation"}
        generation.stop_event.set()
        return {"ok": True}

    @app.get("/api/stream/{session_id}")
    def stream(session_id: str):
        with generations_lock:
            generation = generations.get(session_id)
        if generation is None:
            raise HTTPException(404, "no generation for this session")
        return StreamingResponse(
            generation.stream(),
            media_type="text/event-stream",
            headers={
                "X-Session-Id": session_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def build_app_from_env() -> FastAPI:
    # Prefer the mounted persistent state dir when running in our standard container setup.
    default_state = "/state" if Path("/state").exists() else "/tmp/qwen3-vl-demo-state"
    state_dir = Path(os.environ.get("DEMO_STATE_DIR", default_state))
    manager = DemoModelManager(
        os.environ.get("CKPTDIR", os.environ.get("HF_HOME", "~/.cache/huggingface")),
        os.environ.get("QWEN3_FP8_KERNEL_DIR"),
        idle_seconds=int(os.environ.get("DEMO_IDLE_SECONDS", "600")),
    )
    store = SessionStore(state_dir / "sessions.sqlite")
    return create_app(manager, store, state_dir)


app = build_app_from_env()


def run_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `qwen3-vl web [--host H] [--port P] [--model 2b|8b]`.

    Bridges the top-level CLI to the env-var-driven demo server: CLI flags take
    precedence but fall back to DEMO_HOST/PORT/CKPTDIR env vars when absent.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="qwen3-vl web",
        description="Start the local Qwen3-VL Web UI (single-GPU FP8 by default).",
    )
    parser.add_argument("--host", default=os.environ.get("DEMO_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "7860"))
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_MODEL", "2b"),
        help="checkpoint preflight size hint (2b/4b/8b); actual model is loaded on demand",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("DEMO_HOST", args.host)
    os.environ.setdefault("PORT", str(args.port))
    main()
    return 0


def main() -> None:
    uvicorn.run(
        app,
        host=os.environ.get("DEMO_HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7860")),
        workers=1,
        timeout_keep_alive=300,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()

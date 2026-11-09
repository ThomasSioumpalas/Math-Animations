"""HTTP API + static frontend.  Run:  uv run uvicorn server.main:app --reload"""
from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, jobs, llm, prompts
from .kit import texclean
from .storage import DEFAULT_FOLDER_ID, SID_RE, new_id, new_session_id, store


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    store.cleanup()
    store.start_cleanup_thread()
    print(f"[startup] providers={[f'{p[0]}:{p[3]}' for p in config.ACTIVE_PROVIDERS]} latex={config.LATEX_AVAILABLE} "
          f"ttl={config.SESSION_TTL_SECONDS}s", flush=True)
    yield


app = FastAPI(title="Manim Studio", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")


# ---------------------------------------------------------------- session -----
def _sid_from(request: Request) -> str:
    """Session id from header (iframe-safe, set by the frontend from localStorage),
    cookie, or ?sid= (only needed for <video src> inside iframes that block cookies)."""
    for sid in (request.headers.get("x-session-id", ""), request.cookies.get(config.COOKIE_NAME, ""),
                request.query_params.get("sid", "")):
        if SID_RE.match(sid):
            return sid
    return ""


def session(request: Request, response: Response) -> str:
    """Anonymous per-browser session. Every call refreshes the TTL."""
    sid = _sid_from(request) or new_session_id()
    store.touch(sid)
    https = config.COOKIE_SECURE or request.headers.get("x-forwarded-proto") == "https"
    # SameSite=None lets the cookie work when the app is embedded (e.g. the HF Space page).
    response.set_cookie(config.COOKIE_NAME, sid, max_age=7 * 24 * 3600, httponly=True,
                        samesite="none" if https else "lax", secure=https, path="/")
    response.headers["X-Session-Id"] = sid
    return sid


def existing_session(request: Request) -> str:
    sid = _sid_from(request)
    if not sid or not store.session_dir(sid).exists():
        raise HTTPException(404, "Session not found or expired")
    return sid


def _state_view(sid: str) -> dict:
    state = store.get(sid)
    return {
        "session_id": sid,
        # Changes when the server lost this session (TTL or a restart wiping the disk): the
        # open page then hands back the videos it still holds (POST /api/restore).
        "created_at": state["created_at"],
        "folders": state["folders"],
        # chats are fetched per video (keeps the frequently polled session payload small)
        "videos": sorted(({k: v for k, v in vid.items() if k != "chat"} for vid in state["videos"]
                          if not vid.get("restoring")),
                         key=lambda v: v["created_at"], reverse=True),
        "jobs": [j.public() for j in jobs.for_session(sid)],
        "expires_at": store.expires_at(state),
        "ttl_seconds": config.SESSION_TTL_SECONDS,
        "now": time.time(),
    }


# ---------------------------------------------------------------- models ------
class FolderIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class VideoPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    folder_id: str | None = None


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    thinking: str = "high"  # standard | high | max


# chat thinking level -> (reasoning_effort, extra self-check instructions)
CHAT_THINKING = {"standard": ("medium", False), "high": ("high", False), "max": ("high", True)}


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=3)
    folder_id: str = DEFAULT_FOLDER_ID
    quality: str = "low"


_ID = r"^[A-Za-z0-9_-]{1,64}$"


class RestoreFolder(BaseModel):
    id: str = Field(pattern=_ID)
    name: str = Field(min_length=1, max_length=60)
    created_at: float = 0


class RestoreMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=40_000)
    at: float = 0
    model: str | None = Field(default=None, max_length=200)
    thinking: str | None = Field(default=None, max_length=20)


class RestoreVideo(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{6,64}$")
    title: str = Field(min_length=1, max_length=80)
    prompt: str = Field(max_length=20_000)
    folder_id: str = DEFAULT_FOLDER_ID
    quality: str = Field(default="low", max_length=20)
    created_at: float = 0
    attempts: int = 1
    model: str = Field(default="", max_length=200)
    layout_notes: list[str] = Field(default_factory=list, max_length=50)
    code: str = Field(default="", max_length=400_000)
    chat: list[RestoreMessage] = Field(default_factory=list, max_length=60)


class RestoreIn(BaseModel):
    folders: list[RestoreFolder] = Field(default_factory=list, max_length=100)
    videos: list[RestoreVideo] = Field(default_factory=list, max_length=100)


RESTORE_MAX_BYTES = 150 * 1024 * 1024  # a 6-minute 720p video is ~40 MB


# ---------------------------------------------------------------- routes ------
@app.get("/api/health")
def health():
    return {"ok": True, "providers": llm.status(), "latex": config.LATEX_AVAILABLE}


@app.get("/api/config")
def public_config():
    chain = config.ACTIVE_PROVIDERS
    names = {p[0]: p[3] for p in chain}
    teacher = next((names[t] for t in config.TEACHER_ORDER if t in names), chain[0][3] if chain else "none")
    return {"model": teacher, "allow_hd": config.ALLOW_HD, "max_prompt_chars": config.MAX_PROMPT_CHARS,
            "ttl_seconds": config.SESSION_TTL_SECONDS, "latex": config.LATEX_AVAILABLE}


@app.get("/api/session")
def get_session(sid: str = Depends(session)):
    return _state_view(sid)


@app.post("/api/folders")
def create_folder(body: FolderIn, sid: str = Depends(session)):
    with store.edit(sid) as state:
        if len(state["folders"]) >= config.MAX_FOLDERS_PER_SESSION:
            raise HTTPException(400, "Too many folders")
        folder = {"id": new_id(6), "name": body.name.strip(), "created_at": time.time()}
        state["folders"].append(folder)
    return folder


@app.patch("/api/folders/{folder_id}")
def rename_folder(folder_id: str, body: FolderIn, sid: str = Depends(session)):
    with store.edit(sid) as state:
        f = next((f for f in state["folders"] if f["id"] == folder_id), None)
        if not f:
            raise HTTPException(404, "Folder not found")
        f["name"] = body.name.strip()
    return f


@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str, sid: str = Depends(session)):
    """Deleting a folder moves its videos to General (nothing is lost by accident)."""
    if folder_id == DEFAULT_FOLDER_ID:
        raise HTTPException(400, "The General folder can't be deleted")
    with store.edit(sid) as state:
        state["folders"] = [f for f in state["folders"] if f["id"] != folder_id]
        for v in state["videos"]:
            if v["folder_id"] == folder_id:
                v["folder_id"] = DEFAULT_FOLDER_ID
    return {"ok": True}


@app.patch("/api/videos/{video_id}")
def update_video(video_id: str, body: VideoPatch, sid: str = Depends(session)):
    with store.edit(sid) as state:
        v = next((v for v in state["videos"] if v["id"] == video_id), None)
        if not v:
            raise HTTPException(404, "Video not found")
        if body.folder_id is not None:
            if body.folder_id not in {f["id"] for f in state["folders"]}:
                raise HTTPException(400, "Folder not found")
            v["folder_id"] = body.folder_id
        if body.title is not None:
            v["title"] = body.title.strip()
    return v


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str, sid: str = Depends(session)):
    with store.edit(sid) as state:
        before = len(state["videos"])
        state["videos"] = [v for v in state["videos"] if v["id"] != video_id]
        if len(state["videos"]) == before:
            raise HTTPException(404, "Video not found")
    for ext in ("mp4", "py"):
        store.video_path(sid, video_id, ext).unlink(missing_ok=True)
    return {"ok": True}


def _video_meta(sid: str, video_id: str) -> dict:
    v = next((v for v in store.get(sid)["videos"] if v["id"] == video_id), None)
    if not v:
        raise HTTPException(404, "Video not found")
    return v


def _safe_filename(title: str) -> str:
    return (re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_") or "animation")[:60]


@app.get("/api/videos/{video_id}/file")
def video_file(video_id: str, download: bool = False, sid: str = Depends(existing_session)):
    v = _video_meta(sid, video_id)
    path = store.video_path(sid, video_id, "mp4")
    if not path.is_file():
        raise HTTPException(404, "Video file missing")
    name = f"{_safe_filename(v['title'])}.mp4"
    return FileResponse(path, media_type="video/mp4", filename=name if download else None,
                        content_disposition_type="attachment" if download else "inline",
                        headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/videos/{video_id}/code")
def video_code(video_id: str, download: bool = False, sid: str = Depends(existing_session)):
    v = _video_meta(sid, video_id)
    path = store.video_path(sid, video_id, "py")
    code = path.read_text("utf-8") if path.is_file() else ""
    headers = {"Content-Disposition": f'attachment; filename="{_safe_filename(v["title"])}.py"'} if download else {}
    return PlainTextResponse(code, headers=headers)


CHAT_HISTORY_LIMIT = 16  # messages sent back to the model as context


@app.get("/api/videos/{video_id}/chat")
def video_chat(video_id: str, sid: str = Depends(session)):
    return {"messages": _video_meta(sid, video_id).get("chat", [])}


@app.post("/api/videos/{video_id}/chat")
def ask_about_video(video_id: str, body: ChatIn, sid: str = Depends(session)):
    """Answer a question about a generated video, using its prompt and code as context."""
    v = _video_meta(sid, video_id)
    code_path = store.video_path(sid, video_id, "py")
    code = code_path.read_text("utf-8") if code_path.is_file() else ""
    history = v.get("chat", [])[-CHAT_HISTORY_LIMIT:]
    question = body.message.strip()
    effort, deep = CHAT_THINKING.get(body.thinking, CHAT_THINKING["high"])
    messages = [{"role": "system", "content": prompts.chat_system_prompt(v["title"], v["prompt"], code, deep)}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": question})
    try:
        reply = llm.chat(messages, temperature=0.3 if deep else 0.4, effort=effort, prefer=config.TEACHER_ORDER)
    except llm.LLMError as e:
        raise HTTPException(503, str(e))
    now = time.time()
    with store.edit(sid) as state:
        target = next((x for x in state["videos"] if x["id"] == video_id), None)
        if target is None:
            raise HTTPException(404, "Video not found")
        target.setdefault("chat", []).extend([
            {"role": "user", "content": question, "at": now},
            {"role": "assistant", "content": texclean.plain_chat(reply.text.strip()), "at": time.time(), "model": reply.model,
             "thinking": body.thinking if body.thinking in CHAT_THINKING else "high"},
        ])
        target["chat"] = target["chat"][-60:]
        messages_out = target["chat"]
    return {"messages": messages_out}


@app.post("/api/restore")
def restore(body: RestoreIn, sid: str = Depends(session)):
    """Put back folders and videos that an open page still holds after the server lost them
    (a restart wipes the disk; a phone can freeze a tab past the TTL). Only adds what is
    missing and never overwrites. New videos stay hidden until their file arrives via
    PUT /api/restore/videos/{id}/file; the reply lists the ones that need it."""
    need: list[str] = []
    with store.edit(sid) as state:
        folder_ids = {f["id"] for f in state["folders"]}
        for f in body.folders:
            if f.id not in folder_ids and len(state["folders"]) < config.MAX_FOLDERS_PER_SESSION:
                state["folders"].append(f.model_dump())
                folder_ids.add(f.id)
        videos = {v["id"]: v for v in state["videos"]}
        for v in body.videos:
            current = videos.get(v.id)
            if current is None:
                if len(state["videos"]) >= config.MAX_VIDEOS_PER_SESSION:
                    continue
                entry = v.model_dump(exclude={"code"}, exclude_none=True)
                if entry["folder_id"] not in folder_ids:
                    entry["folder_id"] = DEFAULT_FOLDER_ID
                entry["restoring"] = True
                state["videos"].append(entry)
                store.video_path(sid, v.id, "py").write_text(v.code, "utf-8")
                need.append(v.id)
            elif current.get("restoring"):
                need.append(v.id)
    return {"need_files": need}


@app.put("/api/restore/videos/{video_id}/file")
async def restore_video_file(video_id: str, request: Request, sid: str = Depends(session)):
    """The MP4 of a video announced by POST /api/restore (raw request body)."""
    if not _video_meta(sid, video_id).get("restoring"):
        raise HTTPException(409, "This video doesn't need restoring")
    if int(request.headers.get("content-length") or 0) > RESTORE_MAX_BYTES:
        raise HTTPException(413, "Video file too large")
    path = store.video_path(sid, video_id, "mp4")
    part = path.with_suffix(".part")
    size, head = 0, b""
    try:
        with part.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > RESTORE_MAX_BYTES:
                    raise HTTPException(413, "Video file too large")
                if len(head) < 12:
                    head += chunk[:12 - len(head)]
                f.write(chunk)
        if head[4:8] != b"ftyp":  # every MP4 starts with an ftyp box
            raise HTTPException(400, "Not an MP4 file")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    with store.edit(sid) as state:
        v = next((x for x in state["videos"] if x["id"] == video_id), None)
        if v is not None:
            v.pop("restoring", None)
    return {"ok": True}


@app.post("/api/generate")
def generate(body: GenerateIn, sid: str = Depends(session)):
    prompt = body.prompt.strip()
    if len(prompt) > config.MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt is too long (max {config.MAX_PROMPT_CHARS} characters)")
    quality = body.quality if body.quality in {"low", "medium"} else "low"
    if quality == "medium" and not config.ALLOW_HD:
        quality = "low"
    if err := jobs.check_limits(sid):
        raise HTTPException(429, err)
    folder_ids = {f["id"] for f in store.get(sid)["folders"]}
    job = jobs.submit(sid, prompt, body.folder_id if body.folder_id in folder_ids else DEFAULT_FOLDER_ID, quality)
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, sid: str = Depends(session)):
    job = jobs.get(job_id, sid)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, sid: str = Depends(session)):
    if not jobs.get(job_id, sid):
        raise HTTPException(404, "Job not found")
    if not jobs.cancel(job_id, sid):
        raise HTTPException(409, "This animation has already finished")
    return {"ok": True}


@app.exception_handler(ValueError)
async def bad_id(_: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


# ---------------------------------------------------------------- frontend ----
def mount_frontend(application: FastAPI) -> None:
    """Serve the built React app at "/". Must be the LAST mount (it catches every path)."""
    if config.STATIC_DIR.is_dir():
        application.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="frontend")
    else:
        @application.get("/")
        def no_frontend():
            return PlainTextResponse("Frontend not built. Run: cd frontend && npm install && npm run build\n"
                                     "Or use the Vite dev server: cd frontend && npm run dev")


# hf_space/app.py mounts Gradio first, then calls mount_frontend itself.
if not os.getenv("DEFER_FRONTEND_MOUNT"):
    mount_frontend(app)

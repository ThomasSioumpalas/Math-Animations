"""HTTP API + static frontend.  Run:  uv run uvicorn server.main:app --reload"""
from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, jobs, llm
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
        "folders": state["folders"],
        "videos": sorted(state["videos"], key=lambda v: v["created_at"], reverse=True),
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


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=3)
    folder_id: str = DEFAULT_FOLDER_ID
    quality: str = "low"


# ---------------------------------------------------------------- routes ------
@app.get("/api/health")
def health():
    return {"ok": True, "providers": llm.status(), "latex": config.LATEX_AVAILABLE}


@app.get("/api/config")
def public_config():
    chain = config.ACTIVE_PROVIDERS
    return {"model": chain[0][3] if chain else "none", "allow_hd": config.ALLOW_HD, "max_prompt_chars": config.MAX_PROMPT_CHARS,
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

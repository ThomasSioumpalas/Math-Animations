"""In-memory job queue: prompt -> LLM code -> render -> (fix -> re-render)* -> video.

Jobs live in memory only (lost on restart, which is fine for a session-scoped app).
"""
from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from . import config, lesson, llm, prompts, sandbox
from .storage import new_id, store


@dataclass
class Job:
    id: str
    sid: str
    prompt: str
    folder_id: str
    quality: str
    status: str = "queued"          # queued | writing | rendering | fixing | done | failed | cancelled
    message: str = "Waiting in queue…"
    attempt: int = 0
    max_attempts: int = 1 + config.MAX_FIX_ATTEMPTS
    code: str = ""
    error: str = ""
    video_id: str | None = None
    title: str = ""
    provider: str = ""
    model: str = ""
    layout_notes: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("sid")
        d["position"] = queue_position(self) if self.status == "queued" else 0
        return d


_jobs: dict[str, Job] = {}
_order: list[str] = []
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=max(1, config.RENDER_WORKERS), thread_name_prefix="render")

ACTIVE = {"queued", "writing", "rendering", "fixing"}
_procs: dict[str, object] = {}   # job id -> running render subprocess (so cancel can kill it)


class _Cancelled(Exception):
    pass


def _check(job: Job) -> None:
    if job.status == "cancelled":
        raise _Cancelled


def _track(job: Job, proc) -> None:
    """Remember a job's render process; if it was cancelled a moment ago, stop it right away."""
    _procs[job.id] = proc
    if job.status == "cancelled":
        sandbox.kill(proc)


def cancel(job_id: str, sid: str) -> bool:
    """Stop a job now: it is marked cancelled at once (the UI updates immediately), a running
    render is killed, and a pending AI request is abandoned as soon as it returns."""
    job = get(job_id, sid)
    if not job or job.status not in ACTIVE:
        return False
    _set(job, status="cancelled", message="Cancelled.", finished_at=time.time())
    proc = _procs.pop(job.id, None)
    if proc is not None:
        sandbox.kill(proc)
    return True


def queue_position(job: Job) -> int:
    with _lock:
        queued = [j for j in (_jobs[i] for i in _order) if j.status == "queued"]
    return next((i + 1 for i, j in enumerate(queued) if j.id == job.id), 0)


def get(job_id: str, sid: str) -> Job | None:
    j = _jobs.get(job_id)
    return j if j and j.sid == sid else None


def for_session(sid: str) -> list[Job]:
    cutoff = time.time() - 600  # keep finished jobs visible for 10 minutes
    with _lock:
        return [j for j in (_jobs[i] for i in _order)
                if j.sid == sid and j.status != "cancelled"
                and (j.status in ACTIVE or (j.finished_at or 0) > cutoff)]


def check_limits(sid: str) -> str | None:
    hour_ago = time.time() - 3600
    with _lock:
        mine = [j for j in _jobs.values() if j.sid == sid]
    if sum(j.status in ACTIVE for j in mine) >= config.MAX_ACTIVE_JOBS_PER_SESSION:
        return "You already have animations in progress. Wait for one to finish."
    if config.MAX_JOBS_PER_SESSION_PER_HOUR and sum(j.created_at > hour_ago for j in mine) >= config.MAX_JOBS_PER_SESSION_PER_HOUR:
        return "Hourly limit reached for this browser. Please try again later."
    if len(store.get(sid)["videos"]) >= config.MAX_VIDEOS_PER_SESSION:
        return "Storage full: delete some videos first."
    return None


def submit(sid: str, prompt: str, folder_id: str, quality: str) -> Job:
    job = Job(id=new_id(), sid=sid, prompt=prompt, folder_id=folder_id, quality=quality)
    with _lock:
        _jobs[job.id] = job
        _order.append(job.id)
        _gc()
    store.pin(sid)
    _pool.submit(_run, job)
    return job


def _gc() -> None:
    """Drop old finished jobs so memory doesn't grow forever (called under _lock)."""
    cutoff = time.time() - 3600
    for jid in list(_order):
        j = _jobs[jid]
        if j.status not in ACTIVE and (j.finished_at or 0) < cutoff:
            _order.remove(jid)
            del _jobs[jid]


def _set(job: Job, **kw) -> None:
    for k, v in kw.items():
        setattr(job, k, v)


def _ask(job: Job, user_content: str, temperature: float) -> str:
    reply = llm.chat([
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user", "content": user_content},
    ], temperature=temperature, prefer=job.provider or None,
        should_stop=lambda: job.status == "cancelled", effort=config.VIDEO_REASONING_EFFORT,
        effort_for=config.VIDEO_EFFORT_BY_PROVIDER)
    _check(job)
    job.provider, job.model = reply.provider, reply.model
    return sandbox.extract_code(reply.text)


def _plan(job: Job) -> dict | None:
    """The teacher model's lesson plan, or None when the request needs custom code."""
    messages = [{"role": "system", "content": prompts.LESSON_SYSTEM},
                {"role": "user", "content": prompts.lesson_user_prompt(job.prompt)}]
    for attempt in range(2):
        reply = llm.chat(messages, temperature=0.4, prefer=config.TEACHER_ORDER,
                         should_stop=lambda: job.status == "cancelled", effort=config.PLAN_REASONING_EFFORT,
                         effort_for=config.PLAN_EFFORT_BY_PROVIDER)
        _check(job)
        job.provider, job.model = reply.provider, reply.model
        try:
            plan = lesson.normalize(lesson.parse(reply.text))
        except lesson.PlanError as e:
            print(f"[plan] {job.id}: unusable plan ({e})", flush=True)
            messages[1]["content"] = prompts.lesson_retry_prompt(job.prompt, reply.text, str(e))
            continue
        return None if plan["kind"] != "lesson" else plan
    return None


def _run_lesson(job: Job) -> bool:
    """Prompt -> lesson plan -> one render. False means: use the code-writing path instead."""
    _set(job, status="writing", message="The AI teacher is planning the explanation…", attempt=1)
    plan = _plan(job)
    if plan is None:
        print(f"[plan] {job.id}: no lesson plan for this request, writing code instead", flush=True)
        job.provider = job.model = ""
        return False
    code = lesson.to_code(plan)
    if problems := sandbox.validate(code):
        print(f"[plan] {job.id}: generated scene rejected: {problems}", flush=True)
        return False
    job.code, job.title = code, sandbox.extract_title(code, job.prompt)
    final = _final_render(job, code, lesson.estimate_seconds(plan), label="Animating the explanation")
    if final.ok:
        job.layout_notes = []
        _save(job, final.video)
        return True
    print(f"[plan] {job.id}: lesson render failed, writing code instead: "
          f"{final.error.strip().splitlines()[-1][:300] if final.error.strip() else '?'}", flush=True)
    job.provider = job.model = ""
    return False


def _run(job: Job) -> None:
    """Lesson plan first (fast: a short JSON plan, then one render). Otherwise, or if that fails:
    write code -> quick check (no frames drawn, seconds) -> fix -> ... -> ONE real render.
    Crashes and the length limit always get fix rounds; layout-only problems get at most
    MAX_LAYOUT_FIX_ROUNDS, because most are prevented by the kit and each round costs a full AI call."""
    work = config.WORK_DIR / job.id
    best_code, best_notes, best_len = "", [], 0.0   # a version that works but still has minor layout problems
    layout_rounds = 0
    try:
        _check(job)  # cancelled while still queued
        if config.LESSON_PLANS and _run_lesson(job):
            return
        _set(job, status="writing", message="AI is planning the storyboard and writing the Manim code…", attempt=1)
        code = _ask(job, prompts.user_prompt(job.prompt), 0.4)

        for attempt in range(1, job.max_attempts + 1):
            job.code, job.attempt = code, attempt
            job.title = sandbox.extract_title(code, job.prompt)
            problems = sandbox.validate(code)
            if problems:
                error = "Code rejected by the safety/static checker:\n" + "\n".join(problems)
            else:
                _check(job)
                _set(job, status="rendering",
                     message="Checking the animation…" if attempt == 1 else f"Checking the fixed version ({attempt}/{job.max_attempts})…")
                result = sandbox.check(code, work, on_start=lambda p: _track(job, p))
                _procs.pop(job.id, None)
                _check(job)
                if result.ok and result.warnings and (not best_code or len(result.warnings) <= len(best_notes)):
                    best_code, best_notes, best_len = code, result.warnings, result.duration
                if result.ok and (not result.warnings or layout_rounds >= config.MAX_LAYOUT_FIX_ROUNDS
                                  or attempt == job.max_attempts):
                    final = _final_render(job, code, result.duration)
                    if final.ok:
                        job.layout_notes = list(result.warnings)
                        _save(job, final.video)
                        return
                    error = final.error   # rare: the full render failed although the check passed
                elif result.ok:
                    layout_rounds += 1
                    error = ("The animation works, but the automatic layout check found these problems. "
                             "Fix them by repositioning, resizing or fading out objects (use the studio_kit "
                             "zones):\n- " + "\n- ".join(result.warnings))
                else:
                    error = result.error
            job.error = error
            if attempt == job.max_attempts:
                break
            _check(job)
            _set(job, status="fixing", message=f"Improving the video (attempt {attempt + 1}/{job.max_attempts})…")
            code = _ask(job, prompts.fix_prompt(job.prompt, code, error), 0.2)

        _check(job)
        if best_code:  # nothing perfect: use the best working version (its notes are shown with the video)
            job.code, job.layout_notes = best_code, best_notes
            job.title = sandbox.extract_title(best_code, job.prompt)
            final = _final_render(job, best_code, best_len)
            if final.ok:
                _save(job, final.video)
                return
        _set(job, status="failed", message="Could not produce a working animation. Try rephrasing or simplifying.")
    except _Cancelled:
        pass  # status is already "cancelled"
    except llm.LLMError as e:
        if job.status != "cancelled":
            _set(job, status="failed", message=str(e), error=str(e))
    except Exception as e:  # pragma: no cover - last resort
        if job.status != "cancelled":
            _set(job, status="failed", message="Unexpected server error.", error=repr(e))
    finally:
        _procs.pop(job.id, None)
        job.finished_at = job.finished_at or time.time()
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(config.WORK_DIR / f"{job.id}_final", ignore_errors=True)
        store.unpin(job.sid)


def _final_render(job: Job, code: str, video_seconds: float = 0.0, label: str = ""):
    """The one real render, at the quality the user picked (default 720p)."""
    _check(job)
    res = {"low": "480p", "medium": "720p", "high": "1080p"}.get(job.quality, job.quality)
    base = f"{label} ({res})" if label else f"All checks passed. Rendering the {res} video"
    _set(job, status="rendering", message=base + "…")

    def progress(done: float) -> None:   # seconds of video rendered so far (from the renderer)
        if video_seconds > 0 and job.status == "rendering":
            pct = int(min(99, 100 * done / video_seconds))
            _set(job, message=f"{base}… {pct}%")

    result = sandbox.render(code, config.WORK_DIR / f"{job.id}_final", job.quality,
                            on_start=lambda p: _track(job, p), timeout=config.FINAL_RENDER_TIMEOUT_SECONDS,
                            on_progress=progress)
    _procs.pop(job.id, None)
    _check(job)
    if not result.ok and "timed out" not in result.error:
        # The check run already proved the code works, so this is almost always the machine (encoder,
        # memory), not the code: render again with Manim's plain settings instead of asking the AI.
        print(f"[render] {job.id}: final render failed, retrying with safe settings: "
              f"{result.error.strip().splitlines()[-1][:300] if result.error.strip() else '?'}", flush=True)
        _set(job, message=base + "… (retrying)")
        result = sandbox.render(code, config.WORK_DIR / f"{job.id}_final", job.quality,
                                on_start=lambda p: _track(job, p), timeout=config.FINAL_RENDER_TIMEOUT_SECONDS,
                                mode="render_safe", on_progress=progress)
        _procs.pop(job.id, None)
        _check(job)
    return result


def _save(job: Job, video_path) -> None:
    vid = new_id()
    shutil.move(str(video_path), store.video_path(job.sid, vid, "mp4"))
    store.video_path(job.sid, vid, "py").write_text(job.code, "utf-8")
    with store.edit(job.sid) as state:
        folder_ids = {f["id"] for f in state["folders"]}
        state["videos"].append({
            "id": vid,
            "title": job.title,
            "prompt": job.prompt,
            "folder_id": job.folder_id if job.folder_id in folder_ids else "general",
            "quality": job.quality,
            "created_at": time.time(),
            "attempts": job.attempt,
            "model": f"{job.provider} · {job.model}" if job.provider else "",
            "layout_notes": job.layout_notes,
        })
    _set(job, status="done", message="Done!", video_id=vid, error="", finished_at=time.time())

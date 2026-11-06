"""Database-free, per-browser storage.

Every browser gets a random session id in an HttpOnly cookie. Everything that
browser makes lives in   DATA_DIR/sessions/<sid>/
    state.json          folders + video metadata
    videos/<vid>.mp4    rendered video
    videos/<vid>.py     the Manim source that produced it

A background thread deletes a session folder once it hasn't been seen for
SESSION_TTL_SECONDS (default 1 hour). Closing the tab by mistake and coming back
within that hour brings everything back, because the cookie is still there.
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import config

SID_RE = re.compile(r"^[A-Za-z0-9_-]{24,64}$")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
DEFAULT_FOLDER_ID = "general"


def new_id(n: int = 9) -> str:
    return secrets.token_urlsafe(n)


def new_session_id() -> str:
    return secrets.token_urlsafe(24)


def _empty_state() -> dict:
    now = time.time()
    return {
        "created_at": now,
        "last_seen": now,
        "folders": [{"id": DEFAULT_FOLDER_ID, "name": "General", "created_at": now}],
        "videos": [],
    }


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        # Sessions with a running job must not be garbage-collected mid-render.
        self._pinned: dict[str, int] = {}

    # -- paths -------------------------------------------------------------
    def session_dir(self, sid: str) -> Path:
        if not SID_RE.match(sid):
            raise ValueError("bad session id")
        return self.root / sid

    def videos_dir(self, sid: str) -> Path:
        d = self.session_dir(sid) / "videos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def video_path(self, sid: str, vid: str, ext: str = "mp4") -> Path:
        if not ID_RE.match(vid):
            raise ValueError("bad video id")
        return self.videos_dir(sid) / f"{vid}.{ext}"

    def _lock(self, sid: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(sid, threading.RLock())

    # -- state -------------------------------------------------------------
    def _read(self, sid: str) -> dict:
        p = self.session_dir(sid) / "state.json"
        try:
            return json.loads(p.read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return _empty_state()

    def _write(self, sid: str, state: dict) -> None:
        d = self.session_dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=1), "utf-8")
        tmp.replace(d / "state.json")

    def get(self, sid: str) -> dict:
        with self._lock(sid):
            return self._read(sid)

    @contextmanager
    def edit(self, sid: str) -> Iterator[dict]:
        """Read-modify-write a session's state under its lock."""
        with self._lock(sid):
            state = self._read(sid)
            yield state
            state["last_seen"] = time.time()
            self._write(sid, state)

    def touch(self, sid: str) -> dict:
        with self.edit(sid) as state:
            pass
        return state

    # -- lifecycle ---------------------------------------------------------
    def pin(self, sid: str) -> None:
        with self._locks_guard:
            self._pinned[sid] = self._pinned.get(sid, 0) + 1

    def unpin(self, sid: str) -> None:
        with self._locks_guard:
            n = self._pinned.get(sid, 0) - 1
            if n <= 0:
                self._pinned.pop(sid, None)
            else:
                self._pinned[sid] = n

    def expires_at(self, state: dict) -> float:
        return state.get("last_seen", time.time()) + config.SESSION_TTL_SECONDS

    def cleanup(self) -> int:
        """Delete sessions not seen for SESSION_TTL_SECONDS. Returns how many."""
        removed = 0
        now = time.time()
        for d in self.root.iterdir():
            if not d.is_dir() or not SID_RE.match(d.name):
                continue
            with self._locks_guard:
                if d.name in self._pinned:
                    continue
            with self._lock(d.name):
                state = self._read(d.name)
                expired = now - state.get("last_seen", 0) > config.SESSION_TTL_SECONDS
                if expired:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            if expired:
                with self._locks_guard:
                    self._locks.pop(d.name, None)
        return removed

    def start_cleanup_thread(self) -> None:
        def loop() -> None:
            while True:
                try:
                    n = self.cleanup()
                    if n:
                        print(f"[cleanup] removed {n} expired session(s)", flush=True)
                except Exception as e:  # never let the janitor die
                    print(f"[cleanup] error: {e}", flush=True)
                time.sleep(config.CLEANUP_INTERVAL_SECONDS)

        threading.Thread(target=loop, name="session-cleanup", daemon=True).start()


store = SessionStore(config.SESSIONS_DIR)

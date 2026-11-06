"""Static checks + isolated subprocess rendering for AI-generated Manim code.

Layers of defence (none is perfect alone, together they are reasonable for a
free demo; see PLAN.md for the production upgrade):
  1. AST allow-list: only whitelisted imports, no dunder access, no file / exec
     builtins, no file-reading Manim/NumPy APIs.
  2. render_runner.py installs a Python audit hook that blocks processes,
     sockets, and file access outside the job's work dir.
  3. The subprocess gets a scrubbed environment (no API keys), a timeout,
     and (on Linux) CPU / memory / file-size limits.
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import config

SCENE_NAME = "GeneratedScene"

ALLOWED_MODULES = {
    "manim", "numpy", "math", "cmath", "random", "itertools", "functools", "operator",
    "colorsys", "typing", "dataclasses", "collections", "fractions", "decimal",
    "statistics", "string", "enum", "copy", "abc", "__future__", "scipy", "networkx",
    "textwrap",
}

BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "exit", "quit",
    "help", "__builtins__", "__loader__", "__spec__",
    # Manim / NumPy objects that read arbitrary files from disk
    "Code", "ImageMobject", "SVGMobject", "SVGPathMobject", "VMobjectFromSVGPath",
    "SoundFile", "add_sound",
    # modules that could be reached through re-exports
    "os", "sys", "subprocess", "shutil", "pathlib", "Path", "socket", "importlib",
    "ctypes", "builtins", "pickle", "marshal", "io", "tempfile", "glob", "threading",
    "multiprocessing", "signal", "urllib", "http", "requests", "httpx",
    # custom TeX templates can name an arbitrary compiler executable
    "TexTemplate", "TexTemplateLibrary", "TexFontTemplates",
}

BLOCKED_ATTRS = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "importlib", "ctypes",
    "builtins", "file_ops", "ctypeslib", "lib",
    # config fields that point manim at other executables / directories
    "tex_template", "tex_template_file", "tex_compiler", "media_dir", "video_dir",
    "images_dir", "tex_dir", "text_dir", "partial_movie_dir", "output_file", "log_dir",
    "assets_dir", "input_file", "custom_folders", "plugins", "upto_animation_number",
    # numpy file I/O
    "load", "loadtxt", "genfromtxt", "fromfile", "save", "savez", "savez_compressed",
    "savetxt", "tofile", "memmap", "DataSource", "fromregex",
    # misc escape hatches
    "add_sound", "f_globals", "f_locals", "f_back", "gi_frame", "cr_frame", "tb_frame",
}

DUNDER_OK = {"__init__", "__name__"}

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
TITLE_RE = re.compile(r"^\s*#\s*Title:\s*(.+)$", re.M | re.I)


def extract_code(text: str) -> str:
    """Pull the python code out of an LLM reply (fenced or raw)."""
    blocks = FENCE_RE.findall(text)
    if blocks:
        code = max(blocks, key=len)
    else:
        code = text.strip().removeprefix("```python").removeprefix("```").removesuffix("```")
    return code.strip() + "\n"


def extract_title(code: str, fallback: str) -> str:
    m = TITLE_RE.search(code)
    title = (m.group(1).strip() if m else fallback.strip().splitlines()[0]) or "Untitled"
    return title[:80]


def validate(code: str) -> list[str]:
    """Return a list of problems; empty means the code passed the static checks."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]

    problems: list[str] = []

    def bad(node: ast.AST, msg: str) -> None:
        problems.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    scene_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_MODULES:
                    bad(node, f"import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if node.level or mod not in ALLOWED_MODULES:
                bad(node, f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            bad(node, f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if (a.startswith("__") and a.endswith("__") and a not in DUNDER_OK) or a in BLOCKED_ATTRS:
                bad(node, f"attribute '.{a}' is not allowed")
        elif isinstance(node, ast.keyword) and node.arg in {"tex_template", "file_name", "filename"}:
            bad(node, f"keyword argument '{node.arg}=' is not allowed")
        elif isinstance(node, ast.ClassDef) and node.name == SCENE_NAME:
            scene_found = True
    if not scene_found:
        problems.append(f"no class named {SCENE_NAME} was defined")
    return problems[:15]


# ---------------------------------------------------------------------------
@dataclass
class RenderResult:
    ok: bool
    video: Path | None
    error: str
    seconds: float


QUALITY = {"low": "low_quality", "medium": "medium_quality", "high": "high_quality"}
SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASS|LLM_|GROQ|OPENROUTER|HF_|AWS|CREDENTIAL", re.I)


def _child_env(work: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV_RE.search(k)}
    env.update(
        HOME=str(work), TMPDIR=str(work / "tmp"), TEMP=str(work / "tmp"), TMP=str(work / "tmp"),
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        MPLCONFIGDIR=str(work / "tmp"),
        # kpathsea: stop LaTeX from \input-ing files outside the work dir
        openin_any="p", openout_any="p", TEXMFOUTPUT=str(work), shell_escape="f",
    )
    return env


def _limits() -> None:  # runs in the child before exec (Linux only)
    import resource

    os.setsid()
    mem = config.RENDER_MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    cpu = config.RENDER_TIMEOUT_SECONDS + 10
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (300 * 1024 * 1024, 300 * 1024 * 1024))


def render(code: str, work: Path, quality: str = "low") -> RenderResult:
    """Render GeneratedScene from `code` inside `work`. Never raises."""
    t0 = time.time()
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    (work / "scene.py").write_text(code, "utf-8")

    cmd = [
        sys.executable, "-I", str(Path(__file__).with_name("render_runner.py")),
        str(work), SCENE_NAME, QUALITY.get(quality, "low_quality"),
        str(config.MAX_VIDEO_SECONDS), str(config.SESSIONS_DIR),
    ]
    posix = os.name == "posix"
    try:
        proc = subprocess.Popen(
            cmd, cwd=work, env=_child_env(work), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            preexec_fn=_limits if posix else None,
            creationflags=0 if posix else subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            out, _ = proc.communicate(timeout=config.RENDER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            if posix:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.communicate()
            return RenderResult(False, None,
                                f"Render timed out after {config.RENDER_TIMEOUT_SECONDS}s. "
                                "Make the animation shorter / simpler.", time.time() - t0)
    except Exception as e:
        return RenderResult(False, None, f"Could not start renderer: {e}", time.time() - t0)

    log = out.decode("utf-8", "replace")
    m = re.search(r"^__VIDEO__=(.+)$", log, re.M)
    if proc.returncode == 0 and m and Path(m.group(1).strip()).is_file():
        return RenderResult(True, Path(m.group(1).strip()), "", time.time() - t0)
    if proc.returncode and proc.returncode < 0:
        log += f"\n(renderer killed by signal {-proc.returncode}; probably out of memory or CPU time)"
    return RenderResult(False, None, _trim_log(log), time.time() - t0)


def _trim_log(log: str, limit: int = 3500) -> str:
    """Keep the useful end of a traceback, drop progress-bar noise."""
    lines = [l for l in log.splitlines() if l.strip() and "%|" not in l and "it/s]" not in l]
    text = "\n".join(lines)
    return text[-limit:] if len(text) > limit else text

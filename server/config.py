"""All settings come from environment variables so the same image runs locally,
on Render, or on Hugging Face Spaces without code changes."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a local, git-ignored .env (real env vars win)."""
    if not path.is_file():
        return
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


# --- storage -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()
SESSIONS_DIR = DATA_DIR / "sessions"
WORK_DIR = DATA_DIR / "work"
STATIC_DIR = Path(os.getenv("STATIC_DIR", ROOT / "frontend" / "dist")).resolve()

# A session (and all its videos) is deleted this long after the browser was last seen.
SESSION_TTL_SECONDS = _int("SESSION_TTL_SECONDS", 3600)
CLEANUP_INTERVAL_SECONDS = _int("CLEANUP_INTERVAL_SECONDS", 300)
COOKIE_NAME = "mstudio_sid"
COOKIE_SECURE = _bool("COOKIE_SECURE", False)  # set true behind https

# --- LLM provider chain ------------------------------------------------------
# Free providers, tried in order. When one is rate-limited or out of daily quota, the next one
# is used automatically. A provider is enabled when its key is set (OVHcloud needs no key).
# Each entry: (name, base_url, api_key, model)
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
# Several Gemini models in a row: each has its own free quota, and when the newest one is
# overloaded (free tier gets the lowest priority) the next one usually answers in seconds.
_GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite").split(",")
    if m.strip()]
_OVH_BASE = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
_OVH_KEY = "anonymous" if _bool("OVH_ENABLED", True) else ""   # keyless: 2 requests/min per IP per model
PROVIDERS = [
    # First: gpt-oss-120b (OVHcloud, no key, no token budget). Best speed/accuracy in evals/model_eval.py
    # (math 9/10, median 12 s). When it is rate limited (HTTP 429), errors or stays silent for
    # LLM_FIRST_TOKEN_SECONDS, the chain moves on: Gemini -> NVIDIA -> Qwen -> GLM.
    ("gpt-oss-120b", _OVH_BASE, _OVH_KEY, os.getenv("OVH_FAST_MODEL", "gpt-oss-120b")),
] + [(f"Gemini {m.removeprefix('gemini-')}", _GEMINI_BASE, _GEMINI_KEY, m) for m in _GEMINI_MODELS] + [
    ("NVIDIA", "https://integrate.api.nvidia.com/v1",
     os.getenv("NVIDIA_API_KEY", ""), os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")),
    # Last-resort backups: Qwen 3.5 (9/10 but ~6x slower), GLM (one request at a time).
    ("Qwen 3.5", _OVH_BASE, _OVH_KEY, os.getenv("OVH_MODEL", "Qwen3.5-397B-A17B")),
    ("Z.ai", "https://api.z.ai/api/paas/v4",
     os.getenv("ZAI_API_KEY", ""), os.getenv("ZAI_MODEL", "glm-4.7-flash")),
]
# Optional extra OpenAI-compatible provider, tried last (e.g. a paid key later).
if os.getenv("CUSTOM_LLM_BASE_URL") and os.getenv("CUSTOM_LLM_MODEL"):
    PROVIDERS.append(("Custom", os.environ["CUSTOM_LLM_BASE_URL"].rstrip("/"),
                      os.getenv("CUSTOM_LLM_API_KEY", "none"), os.environ["CUSTOM_LLM_MODEL"]))
ACTIVE_PROVIDERS = [p for p in PROVIDERS if p[2]]

LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 420)          # whole answer
LLM_FIRST_TOKEN_SECONDS = _int("LLM_FIRST_TOKEN_SECONDS", 75)   # no output by then -> next provider
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 32768)   # includes thinking tokens; all providers accept 32768
# Reasoning ("thinking") effort sent to every provider (OpenAI-style `reasoning_effort`).
# Gemini Flash accepts none/minimal/low/medium/high; "high" is its maximum.
VIDEO_REASONING_EFFORT = os.getenv("VIDEO_REASONING_EFFORT", "high")   # writing + fixing videos
# Per-model exceptions for writing videos. gpt-oss-120b on the free OVH endpoint thinks slowly
# (~30 tokens/s): on "high" a video took 275 s and once more than 420 s (timed out), on "medium"
# 152-168 s with code that passed every check first time (2026-11-09 measurements).
VIDEO_EFFORT_BY_PROVIDER = {"gpt-oss-120b": os.getenv("OSS_VIDEO_EFFORT", "medium")}
# Lesson plans (server/lesson.py): the teacher model writes a short JSON plan that the kit animates,
# instead of a whole Manim file. Requests the visuals can't show fall back to code writing.
LESSON_PLANS = _bool("LESSON_PLANS", True)
# The "teacher" models: they write lesson plans and answer the chat under each video. Gemini Flash
# wrote valid plans in 4-29 s; gpt-oss on the free OVH endpoint needed 40-180 s (2026-11-09), so it
# stays first only for writing code (PROVIDERS order). Missing or busy teachers fall back to the chain.
TEACHER_ORDER = [s.strip() for s in os.getenv(
    "TEACHER_MODELS", "Gemini 3.6-flash,Gemini 3.7-flash,Gemini 3.8-flash,Gemini 3.5-flash,Gemini 3.5-flash-lite").split(",")
    if s.strip()]
# A teacher that stays silent this long is skipped (overloaded free tiers often hang instead of failing).
TEACHER_SILENCE_SECONDS = _int("TEACHER_SILENCE_SECONDS", 35)
PLAN_REASONING_EFFORT = os.getenv("PLAN_REASONING_EFFORT", "medium")
PLAN_EFFORT_BY_PROVIDER = {"gpt-oss-120b": os.getenv("OSS_PLAN_EFFORT", "medium")}
LLM_DAILY_CALL_LIMIT = _int("LLM_DAILY_CALL_LIMIT", 0)  # 0 = no site-wide cap

# --- rendering -----------------------------------------------------------------
MAX_FIX_ATTEMPTS = _int("MAX_FIX_ATTEMPTS", 3)            # versions the AI may write (first + fixes)
MAX_LAYOUT_FIX_ROUNDS = _int("MAX_LAYOUT_FIX_ROUNDS", 1)  # extra AI rounds spent on layout-only problems
RENDER_TIMEOUT_SECONDS = _int("RENDER_TIMEOUT_SECONDS", 600)          # check runs (no frames drawn)
FINAL_RENDER_TIMEOUT_SECONDS = _int("FINAL_RENDER_TIMEOUT_SECONDS", 2400)  # the one 720p/1080p render
RENDER_WORKERS = _int("RENDER_WORKERS", 1)               # parallel renders (CPU bound)
MAX_VIDEO_SECONDS = _int("MAX_VIDEO_SECONDS", 360)
RENDER_MEMORY_MB = _int("RENDER_MEMORY_MB", 3072)        # Linux only (RLIMIT_AS)
ALLOW_HD = _bool("ALLOW_HD", True)
LATEX_AVAILABLE = shutil.which("latex") is not None and shutil.which("dvisvgm") is not None

# --- abuse limits --------------------------------------------------------------
MAX_PROMPT_CHARS = _int("MAX_PROMPT_CHARS", 8000)
MAX_VIDEOS_PER_SESSION = _int("MAX_VIDEOS_PER_SESSION", 40)
MAX_FOLDERS_PER_SESSION = _int("MAX_FOLDERS_PER_SESSION", 30)
MAX_JOBS_PER_SESSION_PER_HOUR = _int("MAX_JOBS_PER_SESSION_PER_HOUR", 0)  # 0 = unlimited
MAX_ACTIVE_JOBS_PER_SESSION = _int("MAX_ACTIVE_JOBS_PER_SESSION", 2)

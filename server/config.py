"""All settings come from environment variables so the same image runs locally,
on Render, or on Hugging Face Spaces without code changes."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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

# --- LLM (any OpenAI-compatible endpoint: Groq, OpenRouter, Together, Ollama...) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 180)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 8000)
LLM_DAILY_CALL_LIMIT = _int("LLM_DAILY_CALL_LIMIT", 500)  # whole site, protects the free key

# --- rendering -----------------------------------------------------------------
MAX_FIX_ATTEMPTS = _int("MAX_FIX_ATTEMPTS", 2)          # LLM repair rounds after a failed render
RENDER_TIMEOUT_SECONDS = _int("RENDER_TIMEOUT_SECONDS", 300)
RENDER_WORKERS = _int("RENDER_WORKERS", 1)               # parallel renders (CPU bound)
MAX_VIDEO_SECONDS = _int("MAX_VIDEO_SECONDS", 120)
RENDER_MEMORY_MB = _int("RENDER_MEMORY_MB", 3072)        # Linux only (RLIMIT_AS)
ALLOW_HD = _bool("ALLOW_HD", True)
LATEX_AVAILABLE = shutil.which("latex") is not None and shutil.which("dvisvgm") is not None

# --- abuse limits --------------------------------------------------------------
MAX_PROMPT_CHARS = _int("MAX_PROMPT_CHARS", 8000)
MAX_VIDEOS_PER_SESSION = _int("MAX_VIDEOS_PER_SESSION", 40)
MAX_FOLDERS_PER_SESSION = _int("MAX_FOLDERS_PER_SESSION", 30)
MAX_JOBS_PER_SESSION_PER_HOUR = _int("MAX_JOBS_PER_SESSION_PER_HOUR", 15)
MAX_ACTIVE_JOBS_PER_SESSION = _int("MAX_ACTIVE_JOBS_PER_SESSION", 2)

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
PROVIDERS = [
    ("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
     os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_MODEL", "gemini-3.8-flash")),
    ("NVIDIA", "https://integrate.api.nvidia.com/v1",
     os.getenv("NVIDIA_API_KEY", ""), os.getenv("NVIDIA_MODEL", "google/gemma-4-31b-it")),
    ("Z.ai", "https://api.z.ai/api/paas/v4",
     os.getenv("ZAI_API_KEY", ""), os.getenv("ZAI_MODEL", "glm-4.7-flash")),
    ("OVHcloud", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
     "anonymous" if _bool("OVH_ENABLED", True) else "", os.getenv("OVH_MODEL", "Qwen3.5-397B-A17B")),
]
# Optional extra OpenAI-compatible provider, tried last (e.g. a paid key later).
if os.getenv("CUSTOM_LLM_BASE_URL") and os.getenv("CUSTOM_LLM_MODEL"):
    PROVIDERS.append(("Custom", os.environ["CUSTOM_LLM_BASE_URL"].rstrip("/"),
                      os.getenv("CUSTOM_LLM_API_KEY", "none"), os.environ["CUSTOM_LLM_MODEL"]))
ACTIVE_PROVIDERS = [p for p in PROVIDERS if p[2]]

LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 240)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 16000)
LLM_DAILY_CALL_LIMIT = _int("LLM_DAILY_CALL_LIMIT", 0)  # 0 = no site-wide cap

# --- rendering -----------------------------------------------------------------
MAX_FIX_ATTEMPTS = _int("MAX_FIX_ATTEMPTS", 2)          # LLM repair rounds after a failed render
RENDER_TIMEOUT_SECONDS = _int("RENDER_TIMEOUT_SECONDS", 600)
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

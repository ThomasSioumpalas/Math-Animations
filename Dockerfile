# Manim Studio: one container = FastAPI API + built React frontend + Manim renderer.
# Works on Hugging Face Spaces (port 7860), Render, Railway, Fly, or any Docker host ($PORT).

# ---- 1. build the React frontend ------------------------------------------------
FROM node:24-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. runtime: python + manim system deps -------------------------------------
FROM python:3.12-slim
ARG WITH_LATEX=1
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    DATA_DIR=/data \
    STATIC_DIR=/app/frontend/dist \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libcairo2-dev libpango1.0-dev ffmpeg fonts-dejavu-core \
    && if [ "$WITH_LATEX" = "1" ]; then apt-get install -y --no-install-recommends \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-fonts-recommended texlive-science dvisvgm ; fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY server/ ./server/
COPY --from=web /web/dist ./frontend/dist

# Hugging Face Spaces runs containers as uid 1000; do the same everywhere.
RUN useradd -m -u 1000 app && mkdir -p /data && chown -R app:app /data /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]

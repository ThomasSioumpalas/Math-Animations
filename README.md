# Math Animations + Manim Studio

Two things live in this repo:

1. **`scenes/`**: hand-written Manim animations that explain linear algebra for machine learning
   (chapter 2 of *Deep Learning*, Goodfellow et al.). No LaTeX needed.
2. **Manim Studio** (`server/` + `frontend/`): a website where you describe an animation in plain
   English, an LLM (GPT-OSS-120B via Groq by default, swappable for Kimi/Qwen/DeepSeek) writes the Manim code, and the server renders
   the video. Videos go into folders in a side panel and are kept for 1 hour after you leave,
   with no account or database. See **[PLAN.md](PLAN.md)** for the design and roadmap.

## Setup (Windows / PowerShell, with uv)

```powershell
uv python install 3.12
uv sync
uv run manim checkhealth
```

## Run the hand-written scenes

```powershell
uv run manim -pql -a scenes\column_space.py   # all scenes in a file (-pql = preview, low quality)
uv run manim -pqh scenes\inverse.py UndoA     # one scene in HD
```

| File | Topic |
|---|---|
| `scenes/column_space.py` | Column space, span, linear dependence (§2.4) |
| `scenes/inverse.py` | How the inverse A⁻¹ works, determinant (§2.3) |
| `scenes/norms.py` | Norms and the dot product (§2.5) |
| `scenes/eigen.py` | Eigendecomposition (§2.7) |
| `scenes/special_matrices.py` | Special matrices and vectors (§2.6) |
| `scenes/ml_students.py` | Why ML needs all of this: predicting grades with X w = y |

## Run Manim Studio locally

Get a free API key at <https://console.groq.com/keys> (or OpenRouter, see `.env.example`).

```powershell
# terminal 1: backend on :8000
$env:LLM_API_KEY = "gsk_..."
uv run uvicorn server.main:app --reload --reload-dir server

# terminal 2: frontend dev server on :5173 (proxies /api to :8000)
cd frontend; npm install; npm run dev
```

Open <http://localhost:5173>. Or build once (`cd frontend; npm run build`) and use
<http://localhost:8000>, where FastAPI serves the built site.

**With Docker** (includes LaTeX, the same image as production):

```powershell
docker build -t manim-studio .
docker run -p 7860:7860 -e LLM_API_KEY=gsk_... manim-studio   # → http://localhost:7860
```

## Deploy (public URL)

**Live:** <https://tacious-manim-studio.hf.space>, a free Hugging Face Space.

Free HF accounts can't run Docker Spaces anymore (PRO only, since 2026), but they can run up to
2 **Gradio Spaces on ZeroGPU** hardware. `hf_space/app.py` uses that: it declares an unused
`@spaces.GPU` function so ZeroGPU boots, mounts a stub Gradio page at `/gradio`, and serves our
FastAPI app + React build on port 7860. Renders run on the Space's CPUs. System packages
(cairo, pango, ffmpeg, LaTeX) come from `hf_space/packages.txt`.

Redeploy after changes:

```powershell
cd frontend; npm run build; cd ..
$env:HF_TOKEN = "hf_..."; $env:LLM_API_KEY = "gsk_..."
uv run --with huggingface_hub python deploy/hf_deploy.py Tacious/manim-studio
```

Other hosts: the `Dockerfile` runs anywhere that runs Docker (`--docker` flag for HF PRO, `render.yaml` for
Render's free plan, which works but is slow and wipes videos when it sleeps after 15 minutes).

## Project layout

```
server/
  main.py           FastAPI routes + serves frontend/dist
  storage.py        cookie sessions, per-session folders on disk, 1 h TTL cleanup
  jobs.py           queue: LLM → safety check → render → auto-fix → save
  llm.py            OpenAI-compatible client (Groq / OpenRouter / Ollama…)
  prompts.py        system / fix prompts for the model
  sandbox.py        static code checks + sandboxed subprocess render
  render_runner.py  runs inside the subprocess (audit hook, length cap)
frontend/src/       React + TypeScript UI (sidebar, composer, job + video views)
Dockerfile          one image: built frontend + API + Manim + LaTeX
hf_space/           entry point + apt/pip deps for the free Gradio/ZeroGPU Space
deploy/hf_deploy.py one-command Hugging Face Spaces deploy
```

"""One-command deploy to a FREE Hugging Face Space (Gradio SDK on ZeroGPU hardware).

Free HF accounts can no longer run Docker Spaces, but they can run up to 2 Gradio Spaces
on ZeroGPU. hf_space/app.py runs our FastAPI app inside that Space (see its docstring).

    cd frontend; npm run build; cd ..            # the Space has no Node, so ship the built UI
    $env:HF_TOKEN = "hf_..."                     # https://huggingface.co/settings/tokens (write)
    # AI keys are read from your local .env (GEMINI_API_KEY, NVIDIA_API_KEY, ZAI_API_KEY)
    uv run --with huggingface_hub python deploy/hf_deploy.py <hf-username>/manim-studio

Add --docker to deploy the Dockerfile instead (needs HF PRO).
"""
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
GRADIO_VERSION = "6.31.0"

FRONTMATTER = {
    "gradio": f"""sdk: gradio
sdk_version: {GRADIO_VERSION}
python_version: "3.12"
app_file: app.py""",
    "docker": """sdk: docker
app_port: 7860""",
}

README = """---
title: Manim Studio
emoji: 🎬
colorFrom: indigo
colorTo: yellow
{sdk}
pinned: false
short_description: Describe any visualization, get a Manim video
---

# Manim Studio

Type a prompt; an LLM writes Manim code; the server renders it to video.
Source: https://github.com/ThomasSioumpalas/Math-Animations
"""

VARIABLES = {
    "COOKIE_SECURE": "true",
    "RENDER_WORKERS": "2",
}
SECRETS = ("GEMINI_API_KEY", "NVIDIA_API_KEY", "ZAI_API_KEY")
REMOVED = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")  # old Groq setup


def _dotenv() -> dict:
    env = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text("utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = "docker" if "--docker" in sys.argv else "gradio"
    if len(args) != 1 or "/" not in args[0]:
        sys.exit("usage: hf_deploy.py <hf-username>/<space-name> [--docker]")
    repo_id = args[0]
    token = os.environ.get("HF_TOKEN") or sys.exit("Set HF_TOKEN (write token).")
    if mode == "gradio" and not (ROOT / "frontend" / "dist" / "index.html").is_file():
        sys.exit("Build the frontend first: cd frontend; npm run build")
    api = HfApi(token=token)

    if mode == "gradio":
        api.create_repo(repo_id, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
    else:
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"space ready ({mode})")

    local = _dotenv()
    for name in SECRETS:
        if value := (os.getenv(name) or local.get(name)):
            api.add_space_secret(repo_id, name, value)
            print(f"secret {name} set")
        else:
            print(f"{name} not found locally (Space keeps its current value, if any)")
    for name in REMOVED:
        for remove in (api.delete_space_secret, api.delete_space_variable):
            try:
                remove(repo_id, name)
            except Exception:
                pass
    for k, v in VARIABLES.items():
        api.add_space_variable(repo_id, k, v)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "README.md").write_text(README.format(sdk=FRONTMATTER[mode]), "utf-8")
        if mode == "gradio":
            for name in ("app.py", "requirements.txt", "packages.txt"):
                (stage / name).write_bytes((ROOT / "hf_space" / name).read_bytes())
            patterns = ["server/*.py", "server/kit/*.py", "server/kit/fonts/*", "frontend/dist/**"]
        else:
            patterns = ["Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", ".python-version",
                        "server/*.py", "server/kit/*.py", "server/kit/fonts/*", "frontend/*.json", "frontend/*.ts", "frontend/index.html",
                        "frontend/public/**", "frontend/src/**"]
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=ROOT, allow_patterns=patterns,
                          ignore_patterns=["**/node_modules/**", "**/__pycache__/**"],
                          commit_message="Deploy Manim Studio (code)")
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=stage,
                          commit_message="Deploy Manim Studio (space config)")

    owner, name = repo_id.split("/")
    print(f"\nBuilding... (first build takes ~5-15 min because of LaTeX)\n"
          f"  Space page : https://huggingface.co/spaces/{repo_id}\n"
          f"  Public URL : https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()

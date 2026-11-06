"""Entry point for the free Hugging Face *Gradio + ZeroGPU* Space.

Free HF accounts can't run Docker Spaces anymore, but they can run up to 2 Gradio
Spaces on ZeroGPU hardware. So this file:
  1. declares one (unused) @spaces.GPU function, because ZeroGPU refuses to start without one,
  2. builds a tiny Gradio page (mounted at /gradio, just to satisfy the SDK),
  3. serves our real FastAPI app + React frontend on port 7860 with uvicorn.
Rendering runs on the Space's CPUs; the GPU is never requested, so no GPU quota is used.
"""
import os

os.environ.setdefault("DEFER_FRONTEND_MOUNT", "1")
# HF sets GRADIO_SSR_MODE=True, which makes Gradio start a Node.js server on port 7860
# as soon as a Blocks app is mounted. We serve our own React UI, so turn SSR off.
os.environ["GRADIO_SSR_MODE"] = "False"
os.environ.setdefault("DATA_DIR", "/tmp/manim-studio")
os.environ.setdefault("STATIC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist"))

import spaces  # noqa: E402  (must be imported before anything CUDA-related)
import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from server.main import app, mount_frontend  # noqa: E402


@spaces.GPU(duration=10)
def gpu_health() -> str:
    """Never called by the website; it only exists so ZeroGPU lets the Space boot."""
    return "ok"


with gr.Blocks(title="Manim Studio backend") as demo:
    gr.Markdown("This is the Manim Studio backend. The app itself is at [/](/).")

app = gr.mount_gradio_app(app, demo, path="/gradio", ssr_mode=False)
mount_frontend(app)  # last: catches every remaining path

def _diagnose_port(port: int) -> None:
    """Log who already listens on `port` (Linux /proc), to debug Space startup."""
    try:
        hexport = f"{port:04X}"
        inodes = {l.split()[9] for f in ("/proc/net/tcp", "/proc/net/tcp6") if os.path.exists(f)
                  for l in open(f).read().splitlines()[1:] if l.split()[1].endswith(":" + hexport) and l.split()[3] == "0A"}
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                fds = os.listdir(f"/proc/{pid}/fd")
                if any(os.readlink(f"/proc/{pid}/fd/{fd}") in {f"socket:[{i}]" for i in inodes} for fd in fds):
                    print(f"[hf] port {port} held by pid {pid}:", open(f"/proc/{pid}/cmdline").read().replace("\0", " "), flush=True)
            except OSError:
                pass
    except Exception as e:
        print("[hf] diagnose failed:", e, flush=True)


def _zerogpu_startup() -> None:
    """ZeroGPU registers @spaces.GPU functions inside a patched gr.Blocks.launch(). We never
    call launch() (uvicorn serves the app), so run that startup hook ourselves."""
    try:
        from spaces import zero

        if hasattr(zero, "startup"):
            zero.startup()
            print("[hf] ZeroGPU startup report sent", flush=True)
    except Exception as e:  # not on ZeroGPU, or the API changed
        print(f"[hf] ZeroGPU startup skipped: {e!r}", flush=True)


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    _zerogpu_startup()
    _diagnose_port(port)
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")

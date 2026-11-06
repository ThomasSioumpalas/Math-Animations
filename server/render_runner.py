"""Executed in a separate, scrubbed subprocess by sandbox.render().

    python -I render_runner.py <work_dir> <SceneName> <quality> <max_seconds> <sessions_dir>

Imports Manim, installs an audit hook (no processes except LaTeX, no sockets,
no file writes outside work_dir, no reads of /proc, /sys or other users' data),
then executes scene.py and renders the scene. Prints __VIDEO__=<path> on success.
"""
import os
import sys
import traceback


def _norm(p) -> str:
    return os.path.normcase(os.path.realpath(os.fsdecode(p)))


def _inside(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip(os.sep) + os.sep)


def install_hook(work: str, data_dir: str) -> None:
    deny_read = [_norm(d) for d in ("/proc", "/sys", "/root", "/run/secrets", "/var/run/secrets")]
    deny_read.append(data_dir)
    harmless = {_norm(d) for d in ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", os.devnull)}
    tex_exes = {"latex", "dvisvgm", "xelatex", "lualatex", "pdflatex"}
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    path_events = {"os.remove", "os.rmdir", "os.rename", "shutil.rmtree", "shutil.move",
                   "os.mkdir", "os.chmod", "os.chown", "os.symlink", "os.link", "os.truncate",
                   "os.utime"}
    blocked = ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
               "os.kill", "os.killpg", "socket.", "pty.spawn", "webbrowser.", "urllib.",
               "http.client.", "ftplib.", "smtplib.", "sys.addaudithook", "os.putenv",
               "os.unsetenv", "winreg.", "ctypes.dlopen", "_winapi.CreateProcess")

    def deny(msg: str):
        raise PermissionError(f"[sandbox] {msg}")

    def hook(event: str, args) -> None:
        if event == "open":
            path, mode, flags = args
            if path is None or isinstance(path, int):
                return
            p = _norm(path)
            if _inside(p, work) or p in harmless:
                return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & write_flags)
            if writing:
                deny(f"writing outside the work directory is blocked: {path}")
            if any(_inside(p, d) for d in deny_read) or os.path.basename(p).startswith(".env"):
                deny(f"reading {path} is blocked")
        elif event in path_events:
            if args and isinstance(args[0], (str, bytes, os.PathLike)) and not _inside(_norm(args[0]), work):
                deny(f"{event} outside the work directory is blocked")
        elif event in ("os.listdir", "os.scandir"):
            if args and isinstance(args[0], (str, bytes, os.PathLike)):
                p = _norm(args[0])
                if not _inside(p, work) and any(_inside(p, d) for d in deny_read):
                    deny(f"listing {args[0]} is blocked")
        elif event == "subprocess.Popen":
            exe, argv = args[0], args[1]
            first = exe or (argv if isinstance(argv, (str, bytes)) else (argv[0] if argv else ""))
            name = os.path.splitext(os.path.basename(os.fsdecode(first).split()[0]))[0].lower()
            if name not in tex_exes:
                deny(f"starting the process '{name}' is blocked")
        elif event.startswith(blocked):
            deny(f"{event} is blocked")

    sys.addaudithook(hook)


def main() -> int:
    work, scene_name, quality, max_seconds, sessions_dir = sys.argv[1:6]
    work = _norm(work)
    data_dir = _norm(os.path.dirname(os.path.abspath(sessions_dir)))
    max_seconds = float(max_seconds)
    os.chdir(work)

    with open(os.path.join(work, "scene.py"), encoding="utf-8") as f:
        source = f.read()

    import manim  # noqa: F401  (heavy import before the hook)
    from manim import Scene, config

    def apply_config() -> None:
        config.media_dir = os.path.join(work, "media")
        config.quality = quality
        config.disable_caching = True
        config.write_to_movie = True
        config.save_last_frame = False
        config.preview = False
        config.progress_bar = "none"
        config.verbosity = "WARNING"
        config.output_file = "video"
        config.format = None

    apply_config()

    # Hard cap on video length: abort as soon as the scene runs too long.
    for name in ("play", "wait"):
        original = getattr(Scene, name)

        def capped(self, *a, _orig=original, **kw):
            result = _orig(self, *a, **kw)
            if getattr(self.renderer, "time", 0) > max_seconds:
                raise RuntimeError(f"Animation is longer than the {int(max_seconds)}s limit. Make it shorter.")
            return result

        setattr(Scene, name, capped)

    code = compile(source, os.path.join(work, "scene.py"), "exec")
    install_hook(work, data_dir)

    try:
        ns: dict = {"__name__": "__generated__"}
        exec(code, ns)
        cls = ns.get(scene_name)
        if not (isinstance(cls, type) and issubclass(cls, Scene)):
            print(f"Error: scene.py must define `class {scene_name}(Scene)`.")
            return 2
        apply_config()  # undo any config tweaks made at import time
        scene = cls()
        scene.render()
        print(f"__VIDEO__={scene.renderer.file_writer.movie_file_path}", flush=True)
        return 0
    except BaseException:
        tb = traceback.format_exc()
        # Hide the runner's own frames; the LLM only needs its scene.py frames.
        lines = [l for l in tb.splitlines() if "render_runner.py" not in l]
        print("\n".join(lines), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

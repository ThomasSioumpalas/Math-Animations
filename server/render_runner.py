"""Executed in a separate, scrubbed subprocess by sandbox.render().

    python -I render_runner.py <work_dir> <SceneName> <quality> <max_seconds> <sessions_dir>

Imports Manim, installs an audit hook (no processes except LaTeX, no sockets,
no file writes outside work_dir, no reads of /proc, /sys or other users' data),
then executes scene.py and renders the scene. Prints __VIDEO__=<path> on success.
In "check" mode (6th argument) it runs the whole scene with Manim's skip mode: every line of code and
every animation's start and end state run (so errors and the layout checks are found), but no frames
are drawn or encoded. That takes seconds instead of minutes. Prints __CHECKED__ on success.
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
    mode = sys.argv[6] if len(sys.argv) > 6 else "render"
    work = _norm(work)
    data_dir = _norm(os.path.dirname(os.path.abspath(sessions_dir)))
    max_seconds = float(max_seconds)
    os.chdir(work)

    with open(os.path.join(work, "scene.py"), encoding="utf-8") as f:
        source = f.read()

    import manim  # noqa: F401  (heavy imports before the hook)
    import scipy.linalg  # noqa: F401  (used by studio_kit's space view)
    from manim import Scene, config

    def apply_config() -> None:
        config.media_dir = os.path.join(work, "media")
        config.quality = quality
        config.disable_caching = True
        config.write_to_movie = mode != "check"
        config.save_last_frame = False
        config.preview = False
        config.progress_bar = "none"
        config.verbosity = "WARNING"
        config.output_file = "video"
        config.format = None
        # Speed (Manim >= 0.22): encode finished animations in background threads while the next
        # ones render (the default waits for each one), and use a fast x264 preset ("animation" tune
        # suits flat colors and sharp lines; crf 21 keeps the picture crisp).
        for key, value in (("max_inflight_encoders", 4),
                           ("video_encoder_options", {"crf": "21", "preset": "veryfast", "tune": "animation"})):
            try:
                setattr(config, key, value)
            except Exception:
                pass

    apply_config()

    # Hard cap on video length: abort as soon as the scene runs too long.
    for name in ("play", "wait"):
        original = getattr(Scene, name)

        def capped(self, *a, _orig=original, **kw):
            result = _orig(self, *a, **kw)
            t = getattr(self.renderer, "time", 0)
            if t > max_seconds:
                raise RuntimeError(f"Animation is longer than the {int(max_seconds)}s limit. Make it shorter.")
            if mode != "check":
                print(f"__T__={t:.1f}", flush=True)   # render progress (seconds of video done)
            return result

        setattr(Scene, name, capped)

    # Layout check: after every play(), find visible text that leaves the frame or overlaps
    # other text. Findings are reported back so the AI can fix them in another round.
    import json
    from manim import MarkupText, SingleStringMathTex, Text

    text_types = (Text, MarkupText, SingleStringMathTex)
    issues: dict[str, str] = {}
    half_w, half_h = config.frame_width / 2, config.frame_height / 2

    def _label(m) -> str:
        s = getattr(m, "original_text", None) or getattr(m, "text", None) or getattr(m, "tex_string", None) or type(m).__name__
        s = " ".join(str(s).split())
        return s if len(s) <= 40 else s[:37] + "..."

    def _visible(m) -> bool:
        return any(sm.get_fill_opacity() > 0.15 or sm.get_stroke_opacity() > 0.15
                   for sm in m.family_members_with_points())

    from manim import SurroundingRectangle

    rect_first_seen: dict[int, float] = {}

    def _name(m) -> str:
        if getattr(m, "name", None) and type(m).__name__ == "NumberMatrix":
            return f'matrix "{m.name}"'
        if type(m).__name__ == "NumberMatrix":
            return f"matrix {m.values.shape[0]}×{m.values.shape[1]}"
        if type(m).__name__ == "DataPlot":
            return "plot"
        return f'text "{_label(m)}"'

    def lint(scene) -> None:
        cam = scene.camera
        if type(cam).__name__.startswith("ThreeD"):
            return
        if hasattr(cam, "frame"):  # MovingCamera: use the camera frame's current bounds
            cx, cy = cam.frame.get_center()[:2]
            hw, hh = cam.frame.width / 2, cam.frame.height / 2
        else:
            cx, cy, hw, hh = 0.0, 0.0, half_w, half_h
        kit = sys.modules.get("studio_kit")
        block_types = tuple(c for c in (getattr(kit, "NumberMatrix", None), getattr(kit, "DataPlot", None)) if c)
        t = getattr(scene.renderer, "time", 0)

        # "Blocks": kit matrices/plots as a whole, plus every text not inside one.
        blocks, inside = [], set()
        rects = []
        for top in scene.mobjects:
            for m in top.get_family():
                if id(m) in inside:
                    continue
                if block_types and isinstance(m, block_types):
                    inside.update(id(sub) for sub in m.get_family()[1:])
                    if _visible(m):
                        blocks.append(m)
                elif isinstance(m, text_types):
                    inside.update(id(sub) for sub in m.get_family()[1:])
                    if m.width > 1e-3 and _visible(m):
                        blocks.append(m)
                elif isinstance(m, SurroundingRectangle) and _visible(m) and m.get_fill_opacity() < 0.3:
                    # (filled rectangles are panel backgrounds, not highlights)
                    rects.append(m)

        boxes = []
        for m in blocks:
            l, r, b, tp = m.get_left()[0], m.get_right()[0], m.get_bottom()[1], m.get_top()[1]
            boxes.append((m, l, r, b, tp))
            if l < cx - hw - 0.05 or r > cx + hw + 0.05 or b < cy - hh - 0.05 or tp > cy + hh + 0.05:
                issues.setdefault(f"off:{_name(m)}", f"{_name(m)} goes off-screen at t={t:.0f}s")
        for i in range(len(boxes)):
            m1, l1, r1, b1, t1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                m2, l2, r2, b2, t2 = boxes[j]
                ow, oh = min(r1, r2) - max(l1, l2), min(t1, t2) - max(b1, b2)
                if ow <= 0 or oh <= 0:
                    continue
                smaller = min((r1 - l1) * (t1 - b1), (r2 - l2) * (t2 - b2))
                if ow * oh > 0.08 * smaller:
                    key = f"ov:{_name(m1)}|{_name(m2)}"
                    issues.setdefault(key, f"{_name(m1)} overlaps {_name(m2)} at t={t:.0f}s")

        # Highlight rectangles must be removed after use.
        live = {id(r) for r in rects}
        for k in list(rect_first_seen):
            if k not in live:
                del rect_first_seen[k]
        for r in rects:
            first = rect_first_seen.setdefault(id(r), t)
            if t - first > 12:
                issues.setdefault(f"rect:{first:.0f}", f"a highlight rectangle created at t={first:.0f}s is still "
                                                       f"on screen at t={t:.0f}s (FadeOut highlights right after use)")

        # Content checks: dots without their axes, result matrices left as "?".
        on_screen = {id(m) for top in scene.mobjects for m in top.get_family()}
        for top in scene.mobjects:
            for m in top.get_family():
                plot_id = getattr(m, "_kit_plot_id", None)
                if plot_id is not None and _visible(m) and plot_id not in on_screen:
                    issues.setdefault("orphan-dots", f"data points are on screen without their axes at t={t:.0f}s "
                                                     "(keep the DataPlot on screen while its dots are shown: "
                                                     "clear_stage(self, plot) keeps both)")
                if type(m).__name__ == "NumberMatrix" and _visible(m) and _has_placeholder(m):
                    placeholder_live[id(m)] = (m, t)
        # A matrix that left the screen while still showing "?" was never filled in.
        for key, (m, seen) in list(placeholder_live.items()):
            if key not in on_screen or not _visible(m):
                del placeholder_live[key]
                if _has_placeholder(m):
                    issues.setdefault(f"ph:{key}", f"{_name(m)} was removed at t={t:.0f}s still showing '?' cells: "
                                                   "fill it with self.play(M.set_values(...)) or show_matrix_product")

    def _has_placeholder(m) -> bool:
        return any(str(getattr(c, "original_text", None) or getattr(c, "text", "")).strip() == "?" for c in m.cells)

    def final_check() -> None:
        for key, (m, seen) in placeholder_live.items():
            if _has_placeholder(m):
                issues.setdefault(f"ph:{key}", f"{_name(m)} still shows '?' cells at the end of the video: "
                                               "fill it with self.play(M.set_values(...)) or show_matrix_product")

    placeholder_live: dict[int, tuple] = {}
    lint_time = [0.0]
    import time as _time
    original_play = Scene.play

    def play_and_lint(self, *a, **kw):
        result = original_play(self, *a, **kw)
        t0 = _time.perf_counter()
        try:
            lint(self)
        except Exception:
            pass
        lint_time[0] += _time.perf_counter() - t0
        return result

    Scene.play = play_and_lint

    # studio_kit (layout-safe helpers) is importable by the generated code.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kit"))

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
        if mode == "check":  # run everything, draw nothing
            scene.renderer.skip_animations = True
            scene.renderer._original_skipping_status = True
        scene.render()
        final_check()
        print(f"[lint] {lint_time[0]:.1f}s spent in layout checks", flush=True)
        if issues:
            print("__LAYOUT__=" + json.dumps(list(issues.values())[:12]), flush=True)
        if mode == "check":
            print(f"__CHECKED__ duration={getattr(scene.renderer, 'time', 0):.1f}", flush=True)
        else:
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

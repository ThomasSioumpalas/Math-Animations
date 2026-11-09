"""Render the hand-written test plans through the real sandbox (the videos land in data/test_renders/).
    uv run python tools/render_plan.py all                 quick: check run only (no frames drawn)
    uv run python tools/render_plan.py decomp --render     a real 480p render
    uv run python tools/render_plan.py decomp --render --parallel medium"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from lesson_plans import PLANS  # noqa: E402

from server import config, lesson, sandbox  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
names = list(PLANS) if not args or args[0] == "all" else args[0].split(",")
quality = args[1] if len(args) > 1 else "low"
out_dir = config.DATA_DIR / "test_renders"
out_dir.mkdir(parents=True, exist_ok=True)
failed = 0
for name in names:
    plan = lesson.normalize(PLANS[name])
    problems = lesson.check(plan)
    code = lesson.to_code(plan)
    (out_dir / f"{name}.py").write_text(code, "utf-8")
    print(f"== {name}: checker {problems or 'ok'} | about {lesson.estimate_seconds(plan):.0f}s of video", flush=True)
    t0 = time.time()
    c = sandbox.check(code, config.WORK_DIR / f"plan_{name}")
    print(f"   check run ok={c.ok} in {time.time() - t0:.0f}s, video {c.duration:.0f}s, layout notes: {c.warnings}", flush=True)
    if not c.ok:
        failed += 1
        print(c.error[-3000:])
        continue
    if "--render" in sys.argv:
        t0 = time.time()
        if "--parallel" in sys.argv:
            starts = lesson.split_parts(plan, config.RENDER_PARALLEL)
            r = sandbox.render_parallel([lesson.to_code(plan, (k, starts)) for k in range(len(starts))],
                                        config.WORK_DIR / f"plan_{name}_r", quality, timeout=1800)
        else:
            r = sandbox.render(code, config.WORK_DIR / f"plan_{name}_r", quality, timeout=1800)
        print(f"   render ok={r.ok} in {time.time() - t0:.0f}s, skipped beats: {r.skipped}", flush=True)
        if r.ok:
            shutil.copy(r.video, out_dir / f"{name}.mp4")
            print("   saved", out_dir / f"{name}.mp4")
        else:
            failed += 1
            print(r.error[-3000:])
sys.exit(1 if failed else 0)

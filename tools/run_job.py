"""Run one generation job synchronously through the real pipeline and save the result.
Uses the real AI providers from .env.
    uv run python tools/run_job.py <out.mp4> "<prompt>" [low|medium]"""
import shutil
import sys
import threading
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from server import jobs  # noqa: E402
from server.storage import new_session_id, store  # noqa: E402

out, prompt = sys.argv[1], sys.argv[2]
sid = new_session_id()
store.touch(sid)
job = jobs.Job(id="test" + str(int(time.time())), sid=sid, prompt=prompt, folder_id="general", quality=(sys.argv[3] if len(sys.argv) > 3 else "low"))
store.pin(sid)
t = threading.Thread(target=jobs._run, args=(job,))
t0 = time.time()
t.start()
last = None
while t.is_alive():
    cur = (job.status, job.attempt, job.provider)
    if cur != last and job.status == "fixing" and job.error:
        print("   reason:", job.error.strip().splitlines()[-1][:220], flush=True)
    if cur != last:
        print(f"[{time.time() - t0:5.0f}s] {job.status} attempt={job.attempt} provider={job.provider} {job.model} | {job.message}", flush=True)
        last = cur
    time.sleep(2)
print(f"[{time.time() - t0:5.0f}s] FINAL {job.status} attempts={job.attempt} provider={job.provider} {job.model}")
print("layout notes:", job.layout_notes)
if job.error and job.status != "done":
    print("error:", job.error[-1500:])
with open(out.replace(".mp4", ".py"), "w", encoding="utf-8") as f:
    f.write(job.code)
if job.video_id:
    shutil.copy(store.video_path(sid, job.video_id, "mp4"), out)
    print("saved", out)

"""Benchmark the free AI models on the two jobs the app needs.

Part 1, math reasoning: every model solves the same problems (taken from the example library);
each answer is checked automatically against a ground truth computed with numpy.

Part 2, translation into a video: every model writes a video for the same prompts through the real
pipeline (static checks, sandboxed render at 480p, automatic fix rounds); we record first-try
success, fix rounds needed, remaining layout problems and time.

    uv run python evals/model_eval.py math                  # part 1, all models
    uv run python evals/model_eval.py video                 # part 2, all models (slow: renders)
    uv run python evals/model_eval.py math --models gpt-oss-120b,glm-4.7-flash
    uv run python evals/model_eval.py math --skip-gemini    # keep Gemini's 20-requests/day quota

Results are printed as a table and saved to evals/results/<part>-<timestamp>.json.
Note: Gemini's free tier allows about 20 requests per day per model, shared with the live site.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import config, llm  # noqa: E402

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
NVIDIA = "https://integrate.api.nvidia.com/v1"
ZAI = "https://api.z.ai/api/paas/v4"
OVH = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"

# name -> (base_url, api_key, model id, seconds between calls)
MODELS = {
    "gemini-3.8-flash": (GEMINI, os.getenv("GEMINI_API_KEY", ""), "gemini-3.8-flash", 5),
    "gemini-3.6-flash": (GEMINI, os.getenv("GEMINI_API_KEY", ""), "gemini-3.6-flash", 5),
    "nemotron-3-ultra": (NVIDIA, os.getenv("NVIDIA_API_KEY", ""), "nvidia/nemotron-3-ultra-550b-a55b", 2),
    "gpt-oss-20b@nvidia": (NVIDIA, os.getenv("NVIDIA_API_KEY", ""), "openai/gpt-oss-20b", 2),
    "glm-4.7-flash": (ZAI, os.getenv("ZAI_API_KEY", ""), "glm-4.7-flash", 2),
    "qwen3.5-397b": (OVH, "anonymous", "Qwen3.5-397B-A17B", 31),     # OVH keyless: 2 requests/min
    "gpt-oss-120b": (OVH, "anonymous", "gpt-oss-120b", 31),
    "gpt-oss-20b@ovh": (OVH, "anonymous", "gpt-oss-20b", 31),
}


# ------------------------------------------------------------------ part 1: math
def _problems() -> list[dict]:
    X = np.array([[2, 1], [4, 5], [6, 3], [8, 7]], float)
    Z = (X - X.mean(0)) / X.std(0)
    pts = np.array([[-3, -2], [-2, -1], [-1, -1], [1, 0], [2, 2], [3, 2]], float)
    P, Q = np.array([0.7, 0.2, 0.1]), np.array([1 / 3] * 3)
    Xl = np.array([[1, 0], [1, 1], [1, 2]], float)
    return [
        {"id": "eigenvalues", "q": "Find both eigenvalues of A = [[2, 1], [1, 2]]. Answer as a JSON list sorted ascending.",
         "a": sorted(np.linalg.eigvalsh([[2, 1], [1, 2]]).tolist())},
        {"id": "det3x3", "q": "Compute det([[2, 0, 1], [1, 3, 2], [1, 1, 1]]). Answer as a number.",
         "a": float(np.linalg.det([[2, 0, 1], [1, 3, 2], [1, 1, 1]]))},
        {"id": "standardize_xtx", "q": "X = [[2, 1], [4, 5], [6, 3], [8, 7]] (rows are samples). Standardize each column "
                                       "(subtract the column mean, divide by the population standard deviation, i.e. divide "
                                       "by n). For the standardized matrix Z compute the off-diagonal entry of ZᵀZ. "
                                       "Answer as a number.", "a": float((Z.T @ Z)[0, 1])},
        {"id": "bayes", "q": "1% of people have a disease. A test detects it 90% of the time and gives a false positive "
                             "for 5% of healthy people. What is P(disease | positive test)? Answer as a probability "
                             "between 0 and 1.", "a": 0.9 * 0.01 / (0.9 * 0.01 + 0.05 * 0.99)},
        {"id": "entropy_bits", "q": "Compute the Shannon entropy, in bits, of the distribution P = [0.7, 0.2, 0.1]. "
                                    "Answer as a number.", "a": float(-(P * np.log2(P)).sum())},
        {"id": "kl_bits", "q": "Compute the KL divergence D(P‖Q) in bits for P = [0.7, 0.2, 0.1] and Q = [1/3, 1/3, 1/3]. "
                               "Answer as a number.", "a": float((P * np.log2(P / Q)).sum())},
        {"id": "least_squares", "q": "Fit y = w0 + w1·x by least squares to the points (0, 1), (1, 2), (2, 2). "
                                     "Answer as a JSON list [w0, w1].",
         "a": (np.linalg.pinv(Xl) @ np.array([1, 2, 2], float)).tolist()},
        {"id": "pca_top", "q": "The centered points (-3, -2), (-2, -1), (-1, -1), (1, 0), (2, 2), (3, 2) are the rows of X. "
                               "What is the largest eigenvalue of XᵀX? Answer as a number.",
         "a": float(np.linalg.eigvalsh(pts.T @ pts).max())},
        {"id": "singular_values", "q": "Find the singular values of A = [[3, 0], [4, 5]]. Answer as a JSON list sorted "
                                       "descending.", "a": np.linalg.svd([[3, 0], [4, 5]], compute_uv=False).tolist()},
        {"id": "unbiased_var", "q": "Compute the unbiased sample variance (divide by n − 1) of the data {2, 4, 9}. "
                                    "Answer as a number.", "a": float(np.var([2, 4, 9], ddof=1))},
    ]


MATH_SYSTEM = ("You are a careful mathematician. Solve the problem. You may reason first, but the LAST line of "
               "your reply must be exactly: ANSWER: <json> (a JSON number or a JSON list of numbers, no units).")


def _parse_answer(text: str):
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if not m:
        return None
    raw = m[-1].strip().strip("`").strip()
    try:
        return json.loads(raw)
    except ValueError:
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", raw)
        vals = [eval(n, {"__builtins__": {}}) if "/" in n else float(n) for n in nums]  # simple fractions only
        return vals[0] if len(vals) == 1 else vals or None


def _close(got, want, tol=0.01) -> bool:
    try:
        g, w = np.atleast_1d(np.asarray(got, float)), np.atleast_1d(np.asarray(want, float))
        return g.shape == w.shape and bool(np.all(np.abs(g - w) <= tol * np.maximum(1, np.abs(w))))
    except (TypeError, ValueError):
        return False


def run_math(models: list[str]) -> dict:
    problems = _problems()
    results = {}
    for name in models:
        base, key, model, pause = MODELS[name]
        if not key:
            print(f"skip {name}: no API key")
            continue
        rows = []
        for i, p in enumerate(problems):
            t0 = time.time()
            try:
                text = llm._call(name, base, key, model, [{"role": "system", "content": MATH_SYSTEM},
                                                          {"role": "user", "content": p["q"]}], 0.2, effort="high")
                got = _parse_answer(text)
                ok = _close(got, p["a"])
                err = ""
            except Exception as e:  # rate limits, timeouts, ...
                got, ok, err = None, False, str(e)[:120]
            rows.append({"id": p["id"], "ok": ok, "got": got, "want": p["a"], "seconds": round(time.time() - t0, 1),
                         "error": err})
            print(f"  {name:20s} {p['id']:18s} {'OK ' if ok else 'XX '} {time.time() - t0:5.1f}s"
                  f"{'  ' + err if err else ''}", flush=True)
            llm._cooldown_until.clear()  # the benchmark talks to each model directly
            if i < len(problems) - 1:
                time.sleep(pause)
        results[name] = rows
    return results


# ------------------------------------------------------------------ part 2: video translation
VIDEO_PROMPTS = [
    ("matrix_vector", "Explain the matrix-vector product with A = [[2, 1], [1, 3]] and x = [3, -1], as a "
                      "transformation of the plane: show where the basis vectors land, then show Ax as a combination "
                      "of the columns. Explain every step for a total beginner, with captions."),
    ("bayes", "Explain Bayes' rule with a medical test: 1% prevalence, 90% detection, 5% false positives. Show the "
              "numbers for 10,000 people as an area diagram and compute P(sick | positive). Explain every step for "
              "a total beginner, with captions."),
    ("standardize", "Standardize the matrix X = [[2, 1], [4, 5], [6, 3], [8, 7]] column by column and compute XᵀX of "
                    "the result, showing every calculation and the points moving on a plot. Explain every step for a "
                    "total beginner, with captions."),
]


def run_video(models: list[str]) -> dict:
    from server import jobs  # noqa: WPS433 (import here: needs the full app)
    from server.storage import new_session_id, store

    results = {}
    for name in models:
        base, key, model, _ = MODELS[name]
        if not key:
            print(f"skip {name}: no API key")
            continue
        # Route every request of this job to this one model only.
        config.ACTIVE_PROVIDERS[:] = [(name, base, key, model)]
        rows = []
        for pid, prompt in VIDEO_PROMPTS:
            sid = new_session_id()
            store.touch(sid)
            job = jobs.Job(id=f"eval{int(time.time())}", sid=sid, prompt=prompt, folder_id="general", quality="low")
            store.pin(sid)
            t0 = time.time()
            jobs._run(job)
            rows.append({"id": pid, "status": job.status, "attempts": job.attempt,
                         "first_try": job.status == "done" and job.attempt == 1 and not job.layout_notes,
                         "layout_notes": len(job.layout_notes), "seconds": round(time.time() - t0),
                         "error": (job.error or "")[-200:] if job.status != "done" else ""})
            print(f"  {name:20s} {pid:14s} {job.status:7s} attempts={job.attempt} notes={len(job.layout_notes)} "
                  f"{time.time() - t0:5.0f}s", flush=True)
            llm._cooldown_until.clear()
        results[name] = rows
    return results


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["math", "video"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--skip-gemini", action="store_true")
    ap.add_argument("--first-token", type=int, default=150, help="seconds a model may think before its first output")
    args = ap.parse_args()
    config.LLM_FIRST_TOKEN_SECONDS = args.first_token
    models = [m for m in args.models.split(",") if m in MODELS and not (args.skip_gemini and m.startswith("gemini"))]
    results = run_math(models) if args.part == "math" else run_video(models)

    print("\nSUMMARY")
    for name, rows in results.items():
        if args.part == "math":
            score = sum(r["ok"] for r in rows)
            errors = sum(bool(r["error"]) for r in rows)
            secs = np.median([r["seconds"] for r in rows if not r["error"]] or [0])
            print(f"  {name:20s} {score:2d}/{len(rows)} correct   median {secs:5.1f}s   "
                  f"{errors} failed calls   wrong: {[r['id'] for r in rows if not r['ok'] and not r['error']]}")
        else:
            done = sum(r["status"] == "done" for r in rows)
            first = sum(r["first_try"] for r in rows)
            print(f"  {name:20s} {done}/{len(rows)} videos   {first} perfect on first try   "
                  f"avg attempts {np.mean([r['attempts'] for r in rows]):.1f}   "
                  f"avg {np.mean([r['seconds'] for r in rows]):.0f}s")
    out = ROOT / "evals" / "results" / f"{args.part}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=str), "utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

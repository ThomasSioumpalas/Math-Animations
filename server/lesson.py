"""Lesson plans: the teacher model's JSON -> a tiny scene file that studio_lesson plays.

The teacher model only decides WHAT to explain (steps, captions, which visual each caption goes
with). The animation itself is our tested kit code, so there is nothing to debug: no check run,
no fix rounds, just one render. See server/kit/studio_lesson.py for the vocabulary.
"""
from __future__ import annotations

import json
import math
import pprint
import re

MAX_STEPS = 8
MAX_BEATS_PER_STEP = 10
MAX_BEATS = 50
SLOW_ACTIONS = {"transform": 4.0, "compose": 10.0, "multiply": 14.0, "eigenvectors": 2.5, "reset": 3.0,
                "orbit": 5.0, "transpose": 2.5, "column_op": 4.0, "dot": 3.5, "bars": 2.2, "curve": 2.0}


class PlanError(ValueError):
    pass


_FRACTION = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\w.])")


def _repair(raw: str) -> str:
    """Fix the slips models make in JSON, outside of strings only: fractions written as numbers
    (1/6 -> 0.166667) and trailing commas."""
    out, i, n = [], 0, len(raw)
    while i < n:
        if raw[i] == '"':   # copy a whole string literal untouched
            j = i + 1
            while j < n and raw[j] != '"':
                j += 2 if raw[j] == "\\" else 1
            out.append(raw[i:j + 1])
            i = j + 1
            continue
        j = raw.find('"', i)
        j = n if j < 0 else j
        chunk = _FRACTION.sub(lambda m: f"{float(m.group(1)) / float(m.group(2) or 1):.6g}"
                              if float(m.group(2)) else m.group(0), raw[i:j])
        out.append(re.sub(r",\s*([\]}])", r"\1", chunk))
        i = j
    return "".join(out)


def parse(text: str) -> dict:
    """The JSON object in a model reply (fenced or bare)."""
    raw = text[text.find("{"): text.rfind("}") + 1]
    if not raw:
        raise PlanError("the reply contains no JSON object")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair(raw))
    except json.JSONDecodeError as e:
        raise PlanError(f"invalid JSON: {e}") from None


def _clean(x, depth: int = 0):
    """Only plain JSON values, bounded in size (the plan is embedded in generated code)."""
    if depth > 8:
        return None
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x if math.isfinite(x) else 0
    if isinstance(x, str):
        return x[:600]
    if isinstance(x, list):
        return [_clean(v, depth + 1) for v in x[:80]]
    if isinstance(x, dict):
        return {str(k)[:40]: _clean(v, depth + 1) for k, v in list(x.items())[:40]}
    return None


def _view_type(step: dict) -> str:
    v = step.get("view")
    if isinstance(v, str):
        return v.lower()
    if isinstance(v, dict):
        return str(v.get("type") or "same").lower()
    return "same"


def normalize(plan) -> dict:
    """A safe, bounded plan, or {"kind": "custom"} when the teacher said the visuals can't show it."""
    if not isinstance(plan, dict):
        raise PlanError("the plan is not a JSON object")
    plan = _clean(plan)
    if str(plan.get("kind") or "lesson").lower() != "lesson":
        return {"kind": "custom"}
    steps, total = [], 0
    for st in plan.get("steps") or []:
        if not isinstance(st, dict) or len(steps) >= MAX_STEPS:
            continue
        beats = [b for b in st.get("beats") or [] if isinstance(b, dict) and (b.get("say") or b.get("do"))]
        beats = beats[:MAX_BEATS_PER_STEP][: max(0, MAX_BEATS - total)]
        if beats:
            total += len(beats)
            steps.append({"title": str(st.get("title") or "")[:70], "view": st.get("view") or "same", "beats": beats})
    if not steps:
        raise PlanError("the plan has no steps with beats")
    # A 3-D video runs in a ThreeDScene; flat grids there are drawn as 3-D grids instead.
    if any(_view_type(st) == "space" for st in steps):
        for st in steps:
            if _view_type(st) == "plane":
                st["view"] = {"type": "space"}
    return {"kind": "lesson",
            "title": str(plan.get("title") or "")[:70],
            "question": str(plan.get("question") or "")[:300],
            "answer": str(plan.get("answer") or "")[:400],
            "steps": steps}


def uses_3d(plan: dict) -> bool:
    return any(_view_type(st) == "space" for st in plan.get("steps", []))


def to_code(plan: dict) -> str:
    """The scene file: the plan as a Python literal plus one call into studio_lesson."""
    title = re.sub(r"\s+", " ", plan.get("title") or "Explanation").strip()[:70] or "Explanation"
    head = [f"# Title: {title}",
            "# Made from a lesson plan: the teacher model wrote the steps and captions in PLAN,",
            "# studio_lesson animates them (each caption is on screen while its picture plays).",
            f"# Question: {re.sub(r'\s+', ' ', plan.get('question', ''))}",
            f"# Answer: {re.sub(r'\s+', ' ', plan.get('answer', ''))}",
            "# STORYBOARD:"]
    head += [f"# {i}. {re.sub(r'\s+', ' ', st['title'] or _view_type(st))}" for i, st in enumerate(plan["steps"], 1)]
    body = pprint.pformat(plan, width=110, sort_dicts=False)
    cls = "ThreeDScene" if uses_3d(plan) else "Scene"
    return ("\n".join(head) + "\n\nfrom manim import *\nfrom studio_kit import *\nfrom studio_lesson import play_lesson\n\n"
            f"PLAN = {body}\n\n\nclass GeneratedScene({cls}):\n    def construct(self):\n        play_lesson(self, PLAN)\n")


def estimate_seconds(plan: dict) -> float:
    """Rough video length (for the progress bar)."""
    def read(text: str) -> float:
        return max(1.6, 0.3 * len(str(text).split())) + 0.6

    total = 2.0 + read(plan.get("question", "")) + read(plan.get("answer", "")) + 1.5
    for st in plan.get("steps", []):
        total += 1.5 + (2.0 if _view_type(st) != "same" else 0.0)
        for b in st["beats"]:
            act = SLOW_ACTIONS.get(str(b.get("do") or "").lower(), 1.0 if b.get("do") else 0.0)
            total += max(read(b.get("say", "")), act + 0.5)
    return total

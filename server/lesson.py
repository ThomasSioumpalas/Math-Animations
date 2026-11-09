"""Lesson plans: the teacher model's JSON -> a tiny scene file that studio_lesson plays.

The teacher model only decides WHAT to explain (steps, captions, which visual each caption goes
with). The animation itself is our tested kit code, so there is nothing to debug: no check run,
no fix rounds, just one render. See server/kit/studio_lesson.py for the vocabulary.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pprint
import re
from pathlib import Path

MAX_STEPS = 8
MAX_BEATS_PER_STEP = 10
MAX_BEATS = 50
SLOW_ACTIONS = {"transform": 4.0, "compose": 10.0, "multiply": 14.0, "eigenvectors": 2.5, "reset": 3.0,
                "orbit": 5.0, "transpose": 2.5, "column_op": 4.0, "dot": 3.5, "bars": 2.2, "curve": 2.0,
                "kmeans": 7.0, "descent": 6.0, "descent2d": 8.0, "forward": 4.0, "project": 2.5, "series": 2.0,
                "histogram": 1.6, "boundary": 1.6, "knn": 1.6, "pca": 1.6, "fit_curve": 1.6, "residuals": 2.0,
                "heatmap": 2.0, "data": 1.2}


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
        t = x.strip()
        if t.startswith("[") and t.endswith("]") and len(t) < 4000:   # a matrix or list written as text
            try:
                v = json.loads(t)
                if isinstance(v, list):
                    return _clean(v, depth)
            except ValueError:
                pass
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


ALIASES = {
    "text": "note", "formula": "note", "equation": "note", "show_text": "note", "write": "note", "label": "note",
    "show_formula": "note", "show_note": "note", "compute": "calc", "calculate": "calc", "calculation": "calc",
    "eigen": "eigenvectors", "eigenvector": "eigenvectors", "show_eigenvectors": "eigenvectors",
    "apply": "transform", "transformation": "transform", "apply_matrix": "transform", "transform_space": "transform",
    "add_vector": "vector", "show_vector": "vector", "arrow": "vector", "det": "determinant",
    "show_determinant": "determinant", "volume": "determinant", "square": "unit_square", "cube": "unit_cube",
    "add_unit_square": "unit_square", "add_unit_cube": "unit_cube", "scatter": "points", "plot_points": "points",
    "show_points": "points", "dataset": "data", "table": "data", "show_data": "data", "heat_map": "heatmap",
    "color_matrix": "heatmap", "colors": "heatmap", "regression": "fit_line", "best_fit": "fit_line",
    "line_fit": "fit_line", "plot": "curve", "function": "curve", "graph": "curve", "polynomial": "fit_curve",
    "hist": "histogram", "line_chart": "series", "loss_curve": "series", "gradient_descent": "descent",
    "k_means": "kmeans", "clustering": "kmeans", "classify": "boundary", "decision_boundary": "boundary",
    "nearest_neighbors": "knn", "neighbors": "knn", "principal_components": "pca", "projection": "project",
}


def _fix_action(do: str, view: str) -> str:
    do = ALIASES.get(do, do)
    if do == "matrix" and view in ("plane", "space"):
        return "show_matrix"
    if do == "show_matrix" and view == "matrices":
        return "matrix"
    if do == "descent" and view == "contour":
        return "descent2d"
    return do


def normalize(plan) -> dict:
    """A safe, bounded plan, or {"kind": "custom"} when the teacher said the visuals can't show it."""
    if not isinstance(plan, dict):
        raise PlanError("the plan is not a JSON object")
    plan = _clean(plan)
    if str(plan.get("kind") or "lesson").lower() != "lesson":
        return {"kind": "custom"}
    steps, total, view = [], 0, "text"
    for st in plan.get("steps") or []:
        if not isinstance(st, dict) or len(steps) >= MAX_STEPS:
            continue
        beats = [b for b in st.get("beats") or [] if isinstance(b, dict) and (b.get("say") or b.get("do"))]
        beats = beats[:MAX_BEATS_PER_STEP][: max(0, MAX_BEATS - total)]
        vt = _view_type(st)
        view = vt if vt != "same" else view if steps else "text"
        for b in beats:
            if b.get("do"):
                b["do"] = _fix_action(str(b["do"]).strip().lower(), view)
            if b.get("do") == "note" and not b.get("text"):
                b["text"] = b.get("value") or b.get("content") or b.get("formula") or ""
        if beats:
            total += len(beats)
            steps.append({"title": str(st.get("title") or "")[:70], "view": st.get("view") or "same", "beats": beats})
    if not steps:
        raise PlanError("the plan has no steps with beats")
    # A 3-D video runs in a ThreeDScene; its flat views (2-D grids, plots, tables) look the same there.
    return {"kind": "lesson",
            "title": str(plan.get("title") or "")[:70],
            "question": str(plan.get("question") or "")[:300],
            "answer": str(plan.get("answer") or "")[:400],
            "steps": steps}


# ------------------------------------------------------------------ plan checker --
# Dry-runs a plan without drawing: catches what would otherwise be skipped silently while rendering
# (an undefined name, an action in the wrong view, a template that can't be computed) or would leave
# the screen empty. The problems go back to the teacher model once to be fixed.
def _load_math():
    spec = importlib.util.spec_from_file_location("lesson_math", Path(__file__).with_name("kit") / "lesson_math.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LM = _load_math()
VIEWS = {"plane", "space", "matrices", "plot", "text", "contour", "network"}
GRID = {"plane", "space"}
PLOTS = {"plot", "contour"}
ACTIONS = {   # action: (views it works in, required parameters)
    "show_matrix": (GRID, ("matrix",)), "vector": (GRID, ("coords",)), "combination": ({"plane"}, ()),
    "unit_square": (GRID, ()), "unit_cube": (GRID, ()), "determinant": (GRID, ()), "transform": (GRID, ("matrix",)),
    "landings": (GRID, ()), "eigenvectors": ({"plane"}, ("matrix",)), "eigen_stretch": ({"plane"}, ()),
    "off_span": ({"plane"}, ("coords",)), "reset": (GRID, ()), "compose": ({"plane"}, ("steps",)),
    "orbit": ({"space"}, ()), "points": ({"plane"} | PLOTS, ()),
    "data": ({"matrices"}, ("columns",)), "matrix": ({"matrices"}, ("name",)), "heatmap": ({"matrices"}, ("name",)),
    "multiply": ({"matrices"}, ("a", "b")), "dot": ({"matrices"}, ("a", "b")), "transpose": ({"matrices"}, ("name",)),
    "highlight": ({"matrices"}, ("name",)), "set_values": ({"matrices"}, ("name", "values")),
    "column_op": ({"matrices"}, ("name", "op")),
    "curve": ({"plot"}, ("f",)), "area": ({"plot"}, ("from", "to")), "bars": ({"plot"}, ("values",)),
    "move_points": ({"plot"}, ()), "fit_line": ({"plot"}, ()), "vline": (PLOTS, ("x",)),
    "point": (PLOTS, ("x", "y")), "segment": (PLOTS, ("from", "to")),
    # machine-learning pictures (server/kit/lesson_ml.py)
    "boundary": ({"plot"}, ()), "knn": ({"plot"}, ("query",)), "residuals": ({"plot"}, ()),
    "fit_curve": ({"plot"}, ("degree",)), "pca": ({"plot"}, ()), "project": ({"plot"}, ()),
    "kmeans": ({"plot"}, ()), "histogram": ({"plot"}, ("values",)), "series": ({"plot"}, ()),
    "descent": ({"plot"}, ("f",)), "descent2d": ({"contour"}, ()),
    "forward": ({"network"}, ()), "activate": ({"network"}, ("values",)), "focus": ({"network"}, ()),
    "note": (VIEWS, ("text",)), "calc": (VIEWS, ("what",)),
}
TALK = {"", "none", "wait", "say"}
TEXT_ONLY = {"note", "calc"}          # they add a line of text, not a picture
REF_KEYS = ("matrix", "vector", "coords", "a", "b", "x", "y", "values", "probs")
ON_SCREEN = ("heatmap", "transpose", "highlight", "set_values", "column_op")


def _actions_for(view: str) -> str:
    return ", ".join(sorted(a for a, (views, _) in ACTIONS.items() if view in views))


def _simulate(b: dict, do: str, named: dict, shown: set) -> None:
    """Record the names (and values) a beat defines, exactly like the renderer will."""
    name = b.get("name")
    if do == "data":
        cols = b.get("columns") or {}
        if isinstance(cols, list):
            cols = {str(c.get("name")): c.get("values") for c in cols if isinstance(c, dict)}
        vals = [LM.np.asarray(v, float).ravel() for v in cols.values()]
        if not vals or len({len(v) for v in vals}) != 1:
            raise ValueError("data needs columns of equal length, e.g. {\"X\": [1, 2], \"Y\": [3, 4]}")
        for k, v in zip(cols, vals):
            named[str(k)] = v
        dname = str(name or "data")
        named[dname] = LM.np.column_stack(vals)
        shown.add(dname)
    elif do == "matrix":
        named[name] = (LM.compute(b["compute"], b, named) if b.get("compute")
                       else LM.np.atleast_2d(LM.as_array(b.get("values", b.get("matrix")), named)))
        shown.add(name)
    elif do in ("show_matrix", "vector") and isinstance(name, str):
        named[name] = LM.as_array(b.get("matrix") if do == "show_matrix" else b.get("coords"), named)
    elif do == "multiply":
        named[str(name or "C")] = LM.np.atleast_2d(LM.as_array(b["a"], named)) @ LM.np.atleast_2d(LM.as_array(b["b"], named))
        shown.clear()
        shown.add(str(name or "C"))
    elif do == "transpose" and isinstance(name, str):
        named[f"{name}ᵀ"] = LM.np.atleast_2d(LM.as_array(name, named)).T
        shown.add(f"{name}ᵀ")
    elif do == "set_values" and isinstance(name, str):
        named[name] = LM.np.atleast_2d(LM.as_array(b["values"], named))


def check(plan: dict, bad: set | None = None) -> list[str]:
    """Problems that would make the video skip something or show nothing; [] means fine.
    `bad` collects (step index, beat index) of beats whose action would fail."""
    problems: list[str] = []
    bad = set() if bad is None else bad
    named: dict = {}
    view, shown, text_steps, series_seen = "", set(), 0, False
    for si, st in enumerate(plan.get("steps", []), 1):
        vt = _view_type(st)
        where_step = f'step {si} ("{st.get("title", "")}")'
        if vt not in VIEWS | {"same"}:
            problems.append(f"{where_step}: unknown view '{vt}' (use plane, space, matrices, plot, text or same)")
            vt = "text"
        new = vt != "same" or not view
        if new:
            view, shown = ("text" if vt == "same" else vt), set()
            text_steps += view == "text"
        pictures, talk = 0, 0
        series_seen = series_seen if not new else False
        for bi, b in enumerate(st.get("beats", []), 1):
            where = f"{where_step}, beat {bi}"
            do = str(b.get("do") or "").lower()
            ok = True
            if do in TALK:
                talk += 1
                if talk == 3:
                    problems.append(f"{where}: three captions in a row with nothing new on screen; give them an "
                                    "action that shows what they say")
            else:
                talk = 0
                views, required = ACTIONS.get(do, (None, ()))
                if views is None:
                    problems.append(f"{where}: unknown action '{do}'. In a {view} view you can use: {_actions_for(view)}")
                    ok = False
                elif view not in views:
                    problems.append(f"{where}: '{do}' does not work in a {view} view, only in: {', '.join(sorted(views))}")
                    ok = False
                else:
                    missing = [k for k in required if b.get(k) in (None, "", [], {})]
                    if do == "points" and not (b.get("data") or b.get("coords") or
                                               (b.get("x") is not None and b.get("y") is not None)):
                        missing.append("data (or x and y)")
                    if do == "matrix" and b.get("values") is None and b.get("matrix") is None and not b.get("compute"):
                        missing.append("values (or compute)")
                    if do == "series" and b.get("ys") is None and b.get("y") is None:
                        missing.append("ys (or y)")
                    if do in ("boundary", "knn", "residuals", "fit_curve", "pca", "project", "kmeans",
                              "move_points", "fit_line") and not series_seen:
                        missing.append("points shown first in this view (with 'points')")
                    if missing:
                        problems.append(f"{where}: '{do}' is missing {', '.join(missing)}")
                        ok = False
                if ok:
                    refs = [b.get(k) for k in REF_KEYS if isinstance(b.get(k), str)]
                    if isinstance(b.get("of"), str):
                        refs.append(b["of"])
                    elif isinstance(b.get("of"), list):
                        refs += [c for c in b["of"] if isinstance(c, str)]
                    for r in refs:
                        if r not in named:
                            problems.append(f"{where}: '{r}' is used before it is defined (define it first with "
                                            "data / matrix / show_matrix / vector, or write the numbers)")
                            ok = False
                    if do in ON_SCREEN and b.get("name") not in shown:
                        problems.append(f"{where}: matrix '{b.get('name')}' is not on screen in this view; show it "
                                        "first with 'matrix' or 'data'")
                        ok = False
                    if do == "dot" and not {b.get("a"), b.get("b")} <= shown:
                        problems.append(f"{where}: dot needs both matrices on screen in this view")
                        ok = False
                if ok and do == "points":
                    series_seen = True
                if ok:
                    try:
                        _simulate(b, do, named, shown)
                        if view in GRID and do in ("show_matrix", "transform", "eigenvectors", "vector", "off_span"):
                            dim = 2 if view == "plane" else 3
                            v = LM.as_array(b.get("coords") if do in ("vector", "off_span") else b.get("matrix"), named)
                            want = (dim,) if do in ("vector", "off_span") else (dim, dim)
                            if v.shape != want and v.size != dim:
                                problems.append(f"{where}: a {view} view needs {'x'.join(map(str, want))} values, got "
                                                f"shape {tuple(v.shape)}" + (" (use a space view for 3x3)" if dim == 2 else ""))
                    except Exception as e:
                        problems.append(f"{where}: {e}")
                if not ok:
                    bad.add((si - 1, bi - 1))
                if ok and do not in TEXT_ONLY:
                    pictures += 1
            for key in ("say", "text"):
                try:
                    for expr, names in LM.template_names(b.get(key) or ""):
                        for n in names - set(named):
                            problems.append(f"{where}: the template {{{expr}}} uses '{n}', which is not defined yet")
                        if not names - set(named):
                            LM.evaluate(expr, named)
                except Exception as e:
                    problems.append(f"{where}: a {{...}} template in the {key} cannot be computed ({e})")
        if new and view != "text" and pictures == 0:
            problems.append(f"{where_step}: this {view} view never draws anything; add the action that shows "
                            "what its captions talk about")
    if text_steps > 1:
        problems.append(f"{text_steps} steps use a text view; use at most one and show things with data tables, "
                        "matrices, plots or grids instead")
    problems += _missed_facts(plan, named)
    for si, st in enumerate(plan.get("steps", []), 1):
        for bi, b in enumerate(st.get("beats", []), 1):
            for key in ("say", "text"):
                for m, expr, stated, actual in _claims(b.get(key), named):
                    problems.append(f'step {si}, beat {bi}: you wrote "{m.group(0)}" but {expr} = {LM.fmt(actual)}. '
                                    f"Write {{{expr}}} instead of a number, and make the sentence agree with it")
    for m, expr, stated, actual in _claims(plan.get("answer"), named):
        problems.append(f'the answer says "{m.group(0)}" but {expr} = {LM.fmt(actual)}; write {{{expr}}}')
    for key in ("question", "answer"):
        try:
            for expr, names in LM.template_names(plan.get(key) or ""):
                for n in names - set(named):
                    problems.append(f"the {key} uses '{n}' in {{{expr}}}, which the video never defines")
        except Exception as e:
            problems.append(f"a {{...}} template in the {key} cannot be computed ({e})")
    return problems[:12]


_CLAIM = re.compile(r"((?:det|corr|cov|mean|var|std|norm|trace|rank)\([^()]{1,30}\)|\b[A-Za-z]\w{0,3}\[\d\]\[\d\])"
                    r"\s*(=|≈|is about|is roughly|is)\s*([−-]?\d+(?:\.\d+)?)")


def _claims(text: str, named: dict):
    """(matched text, expression, stated number, exact value) for every number the text states about
    something the renderer can compute, e.g. 'corr(X, Z) ≈ 0' or 'R[0][1] = 0.6'."""
    for m in _CLAIM.finditer(str(text or "")):
        expr, stated = m.group(1), float(m.group(3).replace("−", "-"))
        try:
            names = {n for _, ns in LM.template_names("{" + expr + "}") for n in ns}
            if names - set(named):
                continue
            actual = LM.np.asarray(LM.evaluate(expr, named), float)
        except Exception:
            continue
        if actual.ndim == 0 and abs(float(actual) - stated) > max(0.02, 0.05 * abs(float(actual))):
            yield m, expr, stated, float(actual)


def correct_claims(plan: dict) -> int:
    """Replace stated numbers that are wrong with the exact values (the last line of defence)."""
    named, shown, fixed = {}, set(), 0
    for st in plan.get("steps", []):
        for b in st.get("beats", []):
            try:
                _simulate(b, str(b.get("do") or "").lower(), named, shown)
            except Exception:
                pass
    targets = [(plan, "answer")] + [(b, k) for st in plan.get("steps", []) for b in st.get("beats", []) for k in ("say", "text")]
    for obj, key in targets:
        text = obj.get(key)
        if not isinstance(text, str):
            continue
        for m, expr, stated, actual in reversed(list(_claims(text, named))):
            text = text[: m.start(3)] + LM.fmt(actual) + text[m.end(3):]
            fixed += 1
        obj[key] = text
    return fixed


_SINGULAR_WORDS = ("det(", "singular", "flat", "redundant", "collaps", "squash", "squish", "rank", "dependent",
                   "combination", "no inverse", "zero volume", "zero area", "determinant of 0", "determinant is 0")


def _missed_facts(plan: dict, named: dict) -> list[str]:
    """Key facts the numbers reveal but the plan never says (models often skip working them out)."""
    text = " ".join([plan.get("answer", "")] + [str(b.get(k) or "") for st in plan.get("steps", [])
                                                for b in st.get("beats", []) for k in ("say", "text")]).lower()
    out = []
    for name, value in named.items():
        try:
            A = LM.np.atleast_2d(LM.np.asarray(value, float))
        except (TypeError, ValueError):
            continue
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 2:
            continue
        if abs(LM.np.linalg.det(A)) < 1e-9 * max(1.0, float(LM.np.abs(A).max()) ** A.shape[0]) \
                and not any(w in text for w in _SINGULAR_WORDS):
            rank = int(LM.np.linalg.matrix_rank(A, tol=1e-9))
            out.append(f"key fact missing: {name} is singular, det({name}) = 0 and its rank is {rank} < {A.shape[0]}. "
                       f"As a transformation it flattens space (area/volume becomes 0, no inverse); for a "
                       f"correlation/covariance matrix it means one variable is a combination of the others. "
                       f"Say this in a caption (quote {{det({name})}}) and in the answer.")
    return out[:2]


def prune(plan: dict) -> tuple[dict, int]:
    """Drop beats whose action would fail, WITH their captions (a caption about a picture that never
    appears is worse than no caption), then steps left empty. Repeats until nothing more breaks."""
    dropped = 0
    for _ in range(4):
        bad: set = set()
        check(plan, bad)
        if not bad:
            break
        for si, st in enumerate(plan["steps"]):
            keep = [b for bi, b in enumerate(st["beats"]) if (si, bi) not in bad]
            dropped += len(st["beats"]) - len(keep)
            st["beats"] = keep
        plan["steps"] = [st for st in plan["steps"] if st["beats"]]
    return plan, dropped


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

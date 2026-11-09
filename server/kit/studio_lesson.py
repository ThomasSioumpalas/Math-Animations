"""Play a LESSON PLAN (JSON written by the teacher model) as an animation.

The teacher model decides WHAT to explain: steps, each a list of beats, and a beat is one
caption plus at most one visual action. This module decides HOW it looks, with the tested kit
(Plane2D, Space3D, NumberMatrix, DataPlot). Rules that make the video match the words:
  * a beat's caption appears first and stays on screen while its action plays;
  * the video opens with the user's question and closes with the answer over the final picture;
  * calculations ("calc") are computed here with numpy, never by the model;
  * a beat whose action fails is skipped (its caption still shows) instead of crashing the video.

    from studio_lesson import play_lesson
    class GeneratedScene(Scene):
        def construct(self):
            play_lesson(self, PLAN)
"""
from __future__ import annotations

import ast
import sys

import numpy as np
from manim import (
    BLACK, BLUE, DEGREES, DOWN, GOLD, GREEN, GREY_B, LEFT, ORANGE, PURPLE, RED, RIGHT, TEAL, UP, UR, WHITE, YELLOW,
    AnimationGroup, Create, DashedLine, Dot, FadeIn, FadeOut, GrowFromEdge, Indicate, LaggedStart, Line, ManimColor,
    Rectangle, Text, ThreeDScene, VGroup, interpolate_color,
)
from studio_kit import (
    DataPlot, Narrator, NumberMatrix, _caption_chunks, _wrap, animate_transpose, boxed, clear_stage, fit_to,
    fmt, hud, set_title, show_dot_product, show_matrix_product, transform_column, work_text,
)
from studio_space import Plane2D, Space3D

import lesson_math as LM
from lesson_ml import MLActions

COLORS = {"yellow": YELLOW, "blue": BLUE, "green": GREEN, "red": RED, "orange": ORANGE, "purple": PURPLE,
          "pink": "#FF86C8", "teal": TEAL, "white": WHITE, "grey": GREY_B, "gray": GREY_B, "gold": GOLD}
SERIES = [YELLOW, "#FF86C8", ORANGE, TEAL, PURPLE, BLUE, GOLD]
NEG_COLOR, POS_COLOR = ManimColor("#3B6FD8"), ManimColor("#E5533D")


def _diverging(t: float):
    """Blue (-1) -> white (0) -> red (+1)."""
    t = float(np.clip(t, -1, 1))
    return interpolate_color(ManimColor(WHITE), POS_COLOR if t >= 0 else NEG_COLOR, abs(t))


def play_lesson(scene, plan: dict) -> None:
    _Lesson(scene, plan).run()


def _warn(msg: str) -> None:
    print(f"[lesson] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ formulas --
_FUNCS = {"exp": np.exp, "log": np.log, "ln": np.log, "log2": np.log2, "log10": np.log10, "sqrt": np.sqrt,
          "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh, "arctan": np.arctan, "atan": np.arctan,
          "abs": np.abs, "floor": np.floor, "max": np.maximum, "min": np.minimum, "maximum": np.maximum,
          "minimum": np.minimum, "sign": np.sign}
_CONSTS = {"pi": np.pi, "e": np.e}


def formula(src: str):
    """A safe f(x) from text like '1/(1+exp(-x))' or 'x^2 - 3' (no eval: the tree is walked by hand)."""
    tree = ast.parse(str(src).replace("^", "**").replace("·", "*").replace("−", "-"), mode="eval")
    ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
           ast.Pow: np.power, ast.Mod: np.mod}

    def ev(n, x):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id == "x":
                return x
            if n.id in _CONSTS:
                return _CONSTS[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left, x), ev(n.right, x))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand, x)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*(ev(a, x) for a in n.args))
        raise ValueError(f"unsupported formula part: {ast.dump(n)[:60]}")

    def f(x):
        with np.errstate(all="ignore"):
            return np.asarray(ev(tree.body, np.asarray(x, float)), float) * np.ones_like(np.asarray(x, float))
    f(np.array([0.5, 1.0]))  # fail now, not mid-animation
    return f


# ------------------------------------------------------------------ text helpers --
def _vec(v) -> str:
    return "[" + ", ".join(fmt(x) for x in np.asarray(v, float).ravel()) + "]"


def _matrix_text(M) -> str:
    M = np.atleast_2d(np.asarray(M, float))
    return "[" + ", ".join(_vec(r) for r in M) + "]"


def _p(x) -> str:
    """A number inside a product: negative numbers in brackets."""
    s = fmt(x)
    return f"({s})" if s.startswith("-") else s


def _nice_range(lo: float, hi: float, max_ticks: int = 8):
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -5.0, 5.0
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    span = hi - lo
    step = next((s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000)
                 if span / s <= max_ticks), span / max_ticks)
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step), float(step)


# ------------------------------------------------------------------ the player --
class _Lesson(MLActions):
    def __init__(self, scene, plan: dict):
        self.s, self.plan = scene, plan if isinstance(plan, dict) else {}
        self.n = Narrator(scene)
        self.three_d = isinstance(scene, ThreeDScene)
        self.view = ""            # plane | space | matrices | plot | text
        self.stage = None         # Plane2D / Space3D
        self.plot = None          # DataPlot
        self.named: dict[str, np.ndarray] = {}   # matrices and vectors by name (for calc / references)
        self.mats: dict[str, NumberMatrix] = {}  # NumberMatrix objects on screen, by id
        self.zones: dict[str, str] = {}          # zone -> matrix id occupying it
        self.vecs: dict[str, object] = {}        # vector arrows on the grid, by name
        self.series: dict[str, tuple] = {}       # plot points: id -> (dots, data)
        self.graphs: dict[str, object] = {}      # plot curves: id -> graph
        self.fits: dict[str, list] = {}          # plot points id -> its fitted line + label
        self.carry = {"square": False, "det": False}   # grid features a new grid shows again
        self.display: dict[str, object] = {}     # matrices view: id -> what is on screen for it (table = matrix + headers)
        self.board: list = []                    # text view: formula lines stacked in the middle of the screen
        self._ml_reset()
        self.note = None
        self.temp: list = []                     # highlights that go at the next beat
        self.loose: list = []                    # extra objects in the matrices view (e.g. a product's  A · B →)
        self.color_i = 0

    # ---------------------------------------------------------------- flow
    def run(self) -> None:
        steps = [st for st in self.plan.get("steps") or [] if isinstance(st, dict)]
        title = str(self.plan.get("title") or "").strip()
        if title:
            set_title(self.s, title)
        question = LM.fill(str(self.plan.get("question") or "").strip(), self.named)
        if question:
            self._question_card(question)
        for k, step in enumerate(steps):
            self._start_step(step, steps[k:])
            for beat in step.get("beats") or []:
                if isinstance(beat, dict):
                    self._beat(beat)
        self._clear_temp()
        answer = LM.fill(str(self.plan.get("answer") or "").strip(), self.named)
        if answer:
            self._answer_card(answer)

    def _question_card(self, question: str) -> None:
        """The question, big and centered, before the first picture (not a small caption on an empty screen)."""
        card = Text(_wrap(question, 46), font_size=36, line_spacing=1.0)
        fit_to(card, "main")
        hud(self.s, card)
        self.s.play(FadeIn(card, shift=UP * 0.2), run_time=0.8)
        self.s.wait(max(2.0, 0.28 * len(question.split())))
        self.s.play(FadeOut(card), run_time=0.5)

    def _answer_card(self, answer: str) -> None:
        """The answer to the user's question, big and centered over the dimmed last picture. (The last
        picture may be a side case, so the answer gets its own card instead of a caption over it.)"""
        self._clear_note()
        self.n.clear()
        title = getattr(self.s, "_kit_title", None)
        dim = [m for m in self.s.mobjects if m is not title]
        card = Text(_wrap(answer, 46), font_size=34, line_spacing=1.0)
        card.add_background_rectangle(color=BLACK, opacity=0.88, buff=0.3)
        fit_to(card, "main")
        hud(self.s, card)
        set_title(self.s, "The answer")
        self.s.play(*(m.animate.fade(0.82) for m in dim), FadeIn(card), run_time=1.0)
        self.s.wait(max(3.0, 0.32 * len(answer.split())) + 1.0)

    def _start_step(self, step: dict, rest: list) -> None:
        self._clear_temp()
        self._clear_note()
        view = step.get("view") or {}
        if isinstance(view, str):
            view = {"type": view}
        vtype = str(view.get("type") or "same").lower()
        title = str(step.get("title") or "").strip()
        if vtype != "same" or not self.view:
            if vtype == "same":
                vtype = "text"
            self.n.clear()
            clear_stage(self.s)
            if title:   # the new title first, so it names the picture being built
                set_title(self.s, title)
                title = ""
            try:
                self._new_view(vtype, view, rest)
            except Exception as e:  # never lose the whole video to one view
                _warn(f"view '{vtype}' failed: {type(e).__name__}: {e}")
                self.view, self.stage, self.plot = "text", None, None
        if title:
            set_title(self.s, title)

    def _span(self, rest: list) -> list[dict]:
        """The beats of this step and of the following steps that keep its view ("same")."""
        beats = []
        for k, st in enumerate(rest):
            v = st.get("view") or {}
            vtype = (v if isinstance(v, str) else v.get("type") or "same") if v else "same"
            if k > 0 and str(vtype).lower() != "same":
                break
            beats += [b for b in st.get("beats") or [] if isinstance(b, dict)]
        return beats

    def _new_view(self, vtype: str, view: dict, rest: list) -> None:
        self.mats, self.zones, self.vecs, self.series, self.graphs, self.loose = {}, {}, {}, {}, {}, []
        self.fits, self.display, self.board = {}, {}, []
        self._ml_reset()
        self.stage = self.plot = None
        if self.three_d and vtype != "space":   # flat views in a 3-D video: look straight at the screen again
            try:
                self.s.stop_ambient_camera_rotation()
            except Exception:
                pass
            self.s.set_camera_orientation(phi=0, theta=-90 * DEGREES, gamma=0)
        beats = self._span(rest)
        if vtype in ("plane", "space"):
            dim = 3 if vtype == "space" else 2
            mats, vecs = self._geometry_of(beats, dim)
            if dim == 3 and not self.three_d:
                raise ValueError("space view needs a ThreeDScene")
            self.stage = (Space3D if dim == 3 else Plane2D)(self.s, matrices=mats, vectors=vecs)
            self.view = vtype
            # The unit square and the area readout come back on every new grid once they were used:
            # captions keep referring to them ("watch the readout").
            if self.carry["square"] or self.carry["det"]:
                (self.stage.add_unit_cube if dim == 3 else self.stage.add_unit_square)()
            if self.carry["det"]:
                self.stage.show_determinant()
        elif vtype == "plot":
            self._build_plot(view, beats)
            self.view = "plot"
        elif vtype == "contour":
            self._build_contour(view)
            self.view = "contour"
        elif vtype == "network":
            self._build_network(view)
            self.view = "network"
        elif vtype == "matrices":
            self.view = "matrices"
        else:
            self.view = "text"

    def _geometry_of(self, beats, dim):
        """Matrices (including running products) and vectors used in a span: the grid is sized so
        every arrow stays on screen through every transformation."""
        mats, vecs, M = [], [], np.eye(dim)
        local: dict[str, np.ndarray] = {}

        def res(x):  # a value, or a name defined earlier in this span / in the video
            if isinstance(x, str):
                return np.asarray(local[x] if x in local else self.named[x], float)
            return np.asarray(x, float)

        for b in beats:
            do = str(b.get("do") or "").lower()
            try:
                if do == "show_matrix" and isinstance(b.get("name"), str) and not isinstance(b.get("matrix"), str):
                    local[b["name"]] = np.asarray(b.get("matrix"), float)
                if do in ("transform", "compose", "show_matrix", "eigenvectors"):
                    seq = ([res(s.get("matrix")) for s in b.get("steps") or []] if do == "compose"
                           else [res(b.get("matrix"))])
                    for A in seq:
                        if A.shape == (dim, dim):
                            mats.append(A)
                            if do in ("transform", "compose"):
                                M = A @ M
                                mats.append(M.copy())
                elif do in ("vector", "off_span"):
                    v = res(b.get("coords"))
                    if v.size == dim:
                        vecs.append(v.ravel())
                        if isinstance(b.get("name"), str):
                            local[b["name"]] = v.ravel()
                elif do == "points":
                    vecs += [np.asarray(p, float)[:dim] for p in (b.get("coords") or b.get("data") or [])][:40]
            except Exception:
                pass
        return mats, vecs

    # ---------------------------------------------------------------- beats
    def _beat(self, b: dict) -> None:
        self._clear_temp()
        do = str(b.get("do") or "").strip().lower()
        self._predefine(b, do)   # values this beat creates can already be quoted in its caption
        say = LM.fill(str(b.get("say") or "").strip(), self.named)
        read = 0.0
        if say:
            chunks = _caption_chunks(say, self.n.cpl)
            for c in chunks[:-1]:
                self.n._say_one(c, None, WHITE)
            self._caption(chunks[-1])
            read = max(1.6, 0.3 * len(chunks[-1].split()))
        t0 = self.s.renderer.time
        if do and do not in ("none", "wait", "say"):
            handler = getattr(self, f"_do_{do}", None)
            if handler is None:
                _warn(f"unknown action '{do}' (caption shown without it)")
            else:
                try:
                    handler(b)
                except Exception as e:
                    _warn(f"action '{do}' skipped: {type(e).__name__}: {e}")
        spent = self.s.renderer.time - t0
        self.s.wait(max(0.5, read - spent))

    def _predefine(self, b: dict, do: str) -> None:
        name = b.get("name")
        try:
            if do == "data":
                self._define_data(b)
            elif do == "matrix" and isinstance(name, str):
                self.named[name] = self._matrix_values(b)
            elif do in ("show_matrix", "vector") and isinstance(name, str):
                v = b.get("matrix") if do == "show_matrix" else b.get("coords")
                self.named[name] = self._arr(v)
            elif do == "multiply" and isinstance(b.get("name"), str):
                self.named[name] = np.atleast_2d(self._arr(b.get("a"))) @ np.atleast_2d(self._arr(b.get("b")))
        except Exception:
            pass

    def _caption(self, text: str) -> None:
        """Show a caption WITHOUT waiting: the beat's action plays while it is read."""
        new = Text(_wrap(text, self.n.cpl), font_size=self.n.font_size, line_spacing=0.9)
        new.add_background_rectangle(color=BLACK, opacity=0.82, buff=0.14)
        fit_to(new, "caption")
        hud(self.s, new)
        anims = [FadeIn(new)] + ([FadeOut(self.n.current)] if self.n.current is not None else [])
        self.s.play(*anims, run_time=0.6)
        self.n.current = new

    def _clear_temp(self) -> None:
        live = [m for m in self.temp if m in self.s.mobjects]
        if live:
            self.s.play(*(FadeOut(m) for m in live), run_time=0.4)
        self.temp = []

    def _clear_note(self) -> None:
        if self.note is not None and self.note in self.s.mobjects:
            self.s.play(FadeOut(self.note), run_time=0.4)
        self.note = None

    # ---------------------------------------------------------------- values
    def _arr(self, x) -> np.ndarray:
        if isinstance(x, str):
            if x in self.named:
                return np.asarray(self.named[x], float)
            raise KeyError(f"nothing named {x!r} yet")
        return np.asarray(x, float)

    def _color(self, b: dict, default=None):
        c = COLORS.get(str(b.get("color") or "").lower())
        if c is not None:
            return c
        if default is not None:
            return default
        c = SERIES[self.color_i % len(SERIES)]
        self.color_i += 1
        return c

    def _need(self, *views):
        if self.view not in views and not (self.view == "contour" and "plot" in views):
            raise ValueError(f"needs a {' or '.join(views)} view (current: {self.view or 'none'})")

    # ---------------------------------------------------------------- notes & calculations
    def _show_note(self, text: str, color=WHITE) -> None:
        self._clear_note()
        if self.view in ("plane", "space") and self.stage is not None:
            self.stage.clear_annotations()   # landing / stretch labels would sit under the panel
            mob = boxed(text, color=color, font_size=26)
            if mob.width > 6.4:
                mob.scale_to_fit_width(6.4)
            self.note = self.stage.panel(mob, corner=UR)
            return
        if self.view == "text":   # nothing else on screen: formulas big, in the middle, stacking downwards
            mob = Text(text, font_size=36, color=color)
            if mob.width > 12.6:
                mob.scale_to_fit_width(12.6)
            anims = []
            if len(self.board) >= 5:
                old = self.board.pop(0)
                anims = [FadeOut(old)] + [m.animate.shift(UP * 0.9) for m in self.board]
            mob.move_to([0, 2.35 - 0.9 * len(self.board), 0])
            hud(self.s, mob)
            self.s.play(*anims, FadeIn(mob, shift=UP * 0.15), run_time=0.7)
            self.board.append(mob)
            return
        mob = work_text(text, color=color)
        hud(self.s, mob)
        old = []
        if getattr(self, "status", None) is not None:   # a status line sits where the note goes
            if self.plot is not None and self.status in self.plot:
                self.plot.remove(self.status)
                self.s.add(self.status)
            old = [self.status]
        self.status = None
        self.s.play(FadeIn(mob), *(FadeOut(m) for m in old), run_time=0.6)
        self.note = mob

    def _do_note(self, b):
        self._show_note(LM.fill(str(b.get("text") or "").strip(), self.named), self._color(b, WHITE))

    def _do_calc(self, b):
        self._show_note(self._calc_text(b), COLORS["yellow"] if b.get("highlight") else WHITE)

    def _calc_text(self, b) -> str:
        what = str(b.get("what") or "").lower()
        name = str(b.get("name") or "")

        def mat(key="matrix"):
            v = b.get(key) if b.get(key) is not None else b.get("of")
            return np.atleast_2d(self._arr(v))

        def label(key, fallback):
            v = b.get(key)
            return v if isinstance(v, str) and len(v) <= 4 else fallback

        if what in ("det", "determinant"):
            A, n = mat(), label("matrix", name or "A")
            d = float(np.linalg.det(A))
            if A.shape == (2, 2):
                a, bb, c, dd = A.ravel()
                return f"det({n}) = a·d − b·c = {_p(a)}·{_p(dd)} − {_p(bb)}·{_p(c)} = {fmt(d)}"
            return f"det({n}) = {fmt(d)}"
        if what in ("matvec", "apply"):
            A, v = mat(), self._arr(b.get("vector")).ravel()
            n, vn = label("matrix", "A"), label("vector", "v")
            r = A @ v
            if A.shape[1] <= 3:
                parts = " + ".join(f"{_p(x)}·{_vec(A[:, j])}" for j, x in enumerate(v))
                return f"{n}·{vn} = {parts} = {_vec(r)}"
            return f"{n}·{vn} = {_vec(r)}"
        if what in ("matmul", "product"):
            A, B = np.atleast_2d(self._arr(b.get("a"))), np.atleast_2d(self._arr(b.get("b")))
            return f"{label('a', 'A')}·{label('b', 'B')} = {_matrix_text(A @ B)}"
        if what == "dot":
            u, v = self._arr(b.get("a")).ravel(), self._arr(b.get("b")).ravel()
            terms = " + ".join(f"{_p(x)}·{_p(y)}" for x, y in zip(u, v))
            return f"{label('a', 'u')}·{label('b', 'v')} = {terms} = {fmt(float(u @ v))}"
        if what in ("norm", "length"):
            v = self._arr(b.get("vector")).ravel()
            inside = " + ".join(f"{_p(x)}²" for x in v)
            return f"‖{label('vector', 'v')}‖ = √({inside}) = {fmt(float(np.linalg.norm(v)))}"
        if what == "inverse":
            A, n = mat(), label("matrix", "A")
            d = float(np.linalg.det(A))
            if abs(d) < 1e-12:
                return f"det({n}) = 0, so {n} has no inverse"
            inv = np.linalg.inv(A)
            if A.shape == (2, 2):
                a, bb, c, dd = A.ravel()
                return (f"{n}⁻¹ = 1/{fmt(d)} · [[{fmt(dd)}, {fmt(-bb)}], [{fmt(-c)}, {fmt(a)}]] = {_matrix_text(inv)}")
            return f"{n}⁻¹ = {_matrix_text(inv)}"
        if what in ("eigen", "eigenvalues"):
            A, n = mat(), label("matrix", "A")
            if b.get("vector") is not None:
                v = self._arr(b.get("vector")).ravel()
                Av = A @ v
                k = int(np.argmax(np.abs(v)))
                lam = Av[k] / v[k] if abs(v[k]) > 1e-12 else 0.0
                ok = np.allclose(Av, lam * v, atol=1e-6)
                return (f"{n}·{_vec(v)} = {_vec(Av)} = {fmt(lam)}·{_vec(v)}" if ok
                        else f"{n}·{_vec(v)} = {_vec(Av)}: not a multiple of {_vec(v)}")
            vals = np.linalg.eigvals(A)
            if np.all(np.abs(vals.imag) < 1e-9):
                vals = sorted(vals.real, reverse=True)
                return ", ".join(f"λ{chr(0x2081 + i)} = {fmt(x)}" for i, x in enumerate(vals))
            return f"{n} has no real eigenvalues (it rotates every direction)"
        if what in ("mean", "average", "variance", "var", "std", "sd"):
            x = self._arr(b.get("values")).ravel()
            m = float(x.mean())
            mean_txt = f"mean = ({' + '.join(fmt(v) for v in x)}) / {len(x)} = {fmt(m)}"
            if what in ("mean", "average"):
                return mean_txt
            ddof = 1 if b.get("sample") else 0
            var = float(((x - m) ** 2).sum() / (len(x) - ddof))
            dev = " + ".join(f"({fmt(v)} − {fmt(m)})²" for v in x[:6]) + (" + …" if len(x) > 6 else "")
            var_txt = f"variance = ({dev}) / {len(x) - ddof} = {fmt(var)}"
            if what in ("variance", "var"):
                return var_txt
            return f"std = √{fmt(var)} = {fmt(np.sqrt(var))}"
        if what in ("covariance", "cov", "correlation", "corr"):
            a, c = self._arr(b.get("a")).ravel(), self._arr(b.get("b")).ravel()
            an, cn = label("a", "X"), label("b", "Y")
            ma, mc = a.mean(), c.mean()
            terms = " + ".join(f"({fmt(x - ma)})·({fmt(y - mc)})" for x, y in zip(a[:4], c[:4]))
            terms += " + …" if len(a) > 4 else ""
            cov = float(((a - ma) * (c - mc)).mean())
            if what in ("covariance", "cov"):
                return f"cov({an}, {cn}) = ({terms}) / {len(a)} = {fmt(cov)}"
            sa, sc = float(a.std()), float(c.std())
            r = cov / (sa * sc) if sa * sc > 0 else 0.0
            return f"r({an}, {cn}) = cov({an}, {cn}) / (σ{an}·σ{cn}) = {fmt(cov)} / ({fmt(sa)}·{fmt(sc)}) = {fmt(r)}"
        if what in ("expectation", "expected"):
            x, p = self._arr(b.get("values")).ravel(), self._arr(b.get("probs")).ravel()
            terms = " + ".join(f"{fmt(v)}·{fmt(q)}" for v, q in zip(x, p))
            return f"E[X] = {terms} = {fmt(float(x @ p))}"
        if what in ("sum", "total"):
            x = self._arr(b.get("values")).ravel()
            return f"{' + '.join(fmt(v) for v in x)} = {fmt(float(x.sum()))}"
        raise ValueError(f"unknown calc '{what}'")

    # ---------------------------------------------------------------- grid views (plane / space)
    def _do_show_matrix(self, b):
        self._need("plane", "space")
        A, name = self._arr(b.get("matrix")), str(b.get("name") or "A")
        self.named[name] = A
        self.stage.show_matrix(A, name=name)

    def _do_vector(self, b):
        self._need("plane", "space")
        v, name = self._arr(b.get("coords")).ravel(), str(b.get("name") or "").strip() or None
        basis = list(getattr(self.stage, "basis", []) or [])
        if basis and v.size == len(basis) and np.allclose(self.stage.M, np.eye(len(basis))):
            for k, arrow in enumerate(basis):   # î / ĵ / k̂ are already on the grid: point at them
                if np.allclose(v, np.eye(len(basis))[k]):
                    self.s.play(Indicate(arrow, scale_factor=1.3), run_time=0.9)
                    return
        if name:
            self.named[name] = v
        arrow = self.stage.add_vector(v, label=name, color=self._color(b))
        self.vecs[name or f"_{len(self.vecs)}"] = arrow

    def _do_combination(self, b):
        self._need("plane")
        name = str(b.get("vector") or b.get("name") or "")
        arrow = self.vecs.get(name) or (list(self.vecs.values())[-1] if self.vecs else None)
        if arrow is None:
            raise KeyError("no vector on the grid yet")
        self.stage.show_combination(arrow)

    def _do_unit_square(self, b):
        self._need("plane", "space")
        self.carry["square"] = True
        if self.stage.square is None:
            (self.stage.add_unit_cube if self.view == "space" else self.stage.add_unit_square)()

    _do_unit_cube = _do_unit_square

    def _do_determinant(self, b):
        self._need("plane", "space")
        self.carry["det"] = True
        if getattr(self.stage, "det_readout", None) is None:
            self.stage.show_determinant()

    def _do_transform(self, b):
        self._need("plane", "space")
        A = self._arr(b.get("matrix"))
        self.stage.transform_space(A)
        if b.get("landings"):
            self.stage.label_landings()

    def _do_landings(self, b):
        self._need("plane", "space")
        self.stage.label_landings()

    def _do_eigenvectors(self, b):
        self._need("plane")
        self.stage.show_eigenvectors(self._arr(b.get("matrix")))

    def _do_eigen_stretch(self, b):
        self._need("plane")
        self.stage.label_eigen_stretch()

    def _do_off_span(self, b):
        self._need("plane")
        self.stage.show_off_span(self._arr(b.get("coords")).ravel())

    def _do_reset(self, b):
        self._need("plane", "space")
        self.stage.reset_space()

    def _do_compose(self, b):
        self._need("plane")
        steps = [(self._arr(st.get("matrix")), str(st.get("name") or f"M{k + 1}"))
                 for k, st in enumerate(b.get("steps") or [])]
        self.stage.compose(steps)

    def _do_orbit(self, b):
        self._need("space")
        self.stage.orbit()

    # ---------------------------------------------------------------- matrices view
    def _place(self, shown, at: str, mid: str, M: NumberMatrix | None = None) -> None:
        """Put a matrix (or a table: matrix + headers) in a zone; whatever sat there fades out first."""
        at = at if at in ("left", "right", "main") else ("left" if "left" not in self.zones else
                                                         "right" if "right" not in self.zones else "main")
        clash = {"main": ("left", "right", "main"), "left": ("left", "main"), "right": ("right", "main")}[at]
        old = []
        for z in clash:
            if z in self.zones:
                gone = self.zones.pop(z)
                old.append(self.display.pop(gone, None) or self.mats.get(gone))
                self.mats.pop(gone, None)
        old = [m for m in old if m is not None and m in self.s.mobjects]
        old += [m for m in self.loose if m in self.s.mobjects]
        self.loose = []
        if old:
            self.s.play(*(FadeOut(m) for m in old), run_time=0.4)
        fit_to(shown, at)
        self.zones[at] = mid
        self.display[mid] = shown
        self.mats[mid] = M if M is not None else shown

    def _matrix_values(self, b) -> np.ndarray:
        if b.get("compute"):
            return np.atleast_2d(LM.compute(b["compute"], b, self.named))
        return np.atleast_2d(self._arr(b.get("values") if b.get("values") is not None else b.get("matrix")))

    def _do_matrix(self, b):
        self._need("matrices")
        mid = str(b.get("name") or b.get("id") or f"M{len(self.mats) + 1}")
        vals = self._matrix_values(b)
        self.named[mid] = vals
        M = NumberMatrix(vals, name=mid)
        self._place(M, str(b.get("at") or ""), mid)
        self.s.play(FadeIn(M), run_time=0.8)

    def _define_data(self, b):
        cols = b.get("columns") or {}
        if isinstance(cols, list):   # [{"name": "X", "values": [...]}, ...]
            cols = {str(c.get("name")): c.get("values") for c in cols if isinstance(c, dict)}
        names = [str(k) for k in cols]
        vals = [np.asarray(v, float).ravel() for v in cols.values()]
        if not vals or len({len(v) for v in vals}) != 1:
            raise ValueError("data needs columns of equal length")
        for n_, v in zip(names, vals):
            self.named[n_] = v
        D = np.column_stack(vals)
        dname = str(b.get("name") or "data")
        self.named[dname] = D
        return names, D, dname

    def _do_data(self, b):
        """A dataset as a table: one column per variable, its name on top (rows = samples)."""
        self._need("matrices")
        names, D, dname = self._define_data(b)
        M = NumberMatrix(D, name=None)
        heads = VGroup(*(Text(n_, font_size=30, color=YELLOW) for n_ in names))
        for j, h in enumerate(heads):
            h.next_to(M.col(j), UP, buff=0.45)
        top = max(h.get_center()[1] for h in heads)
        for h in heads:
            h.set_y(top)
        table = VGroup(M, heads)
        self._place(table, str(b.get("at") or ""), dname, M)
        self.s.play(FadeIn(M), FadeIn(heads, shift=DOWN * 0.1), run_time=0.9)

    def _legend(self, m: float) -> VGroup:
        n = 9
        bar = VGroup(*(Rectangle(width=0.32, height=0.26, fill_color=_diverging(1 - 2 * k / (n - 1)),
                                 fill_opacity=0.95, stroke_width=0) for k in range(n))).arrange(DOWN, buff=0)
        labels = VGroup(Text(fmt(m), font_size=22).next_to(bar[0], RIGHT, buff=0.12),
                        Text("0", font_size=22).next_to(bar[n // 2], RIGHT, buff=0.12),
                        Text(fmt(-m), font_size=22).next_to(bar[-1], RIGHT, buff=0.12))
        return VGroup(bar, labels)

    def _do_heatmap(self, b):
        """Color every cell of a matrix on screen by its value: blue (negative), white (0), red (positive)."""
        self._need("matrices")
        mid = str(b.get("name") or b.get("id") or "")
        M = self._on_screen(mid)
        V = M.values
        rows, cols = V.shape
        m = 1.0 if np.all(np.abs(V) <= 1 + 1e-9) else float(np.abs(V).max() or 1)
        cx = [M.cell(0, j).get_center()[0] for j in range(cols)]
        hw = [max(M.cell(i, j).width for i in range(rows)) / 2 for j in range(cols)]
        gap = ((cx[1] - hw[1]) - (cx[0] + hw[0])) if cols > 1 else 0.3
        bounds = []
        for j in range(cols):
            left = (cx[j - 1] + hw[j - 1] + cx[j] - hw[j]) / 2 if j > 0 else cx[j] - hw[j] - gap / 2
            right = (cx[j] + hw[j] + cx[j + 1] - hw[j + 1]) / 2 if j < cols - 1 else cx[j] + hw[j] + gap / 2
            bounds.append((left, right))
        cy = [M.cell(i, 0).get_center()[1] for i in range(rows)]
        pitch = (cy[0] - cy[1]) if rows > 1 else M.cell(0, 0).height * 2.2
        rects, recolor = VGroup(), []
        for i in range(rows):
            for j in range(cols):
                t = V[i, j] / m
                left, right = bounds[j]
                rects.add(Rectangle(width=(right - left) * 0.97, height=pitch * 0.94, fill_color=_diverging(t),
                                    fill_opacity=0.92, stroke_width=0).move_to([(left + right) / 2, cy[i], 0]))
                recolor.append(M.cell(i, j).animate.set_color(BLACK if abs(t) < 0.6 else WHITE))
        legend = self._legend(m)
        legend.next_to(M, RIGHT, buff=0.5)
        if legend.get_right()[0] > 6.9 or any(z == "right" and v != mid for z, v in self.zones.items()):
            legend.rotate(0).next_to(M, DOWN, buff=0.35)
        self.s.play(LaggedStart(*(FadeIn(r) for r in rects), lag_ratio=0.05), *recolor, FadeIn(legend), run_time=1.6)
        self.s.remove(*rects)
        M.add_to_back(rects)
        self.loose.append(legend)

    def _on_screen(self, key) -> NumberMatrix:
        if isinstance(key, str) and key in self.mats:
            return self.mats[key]
        raise KeyError(f"matrix {key!r} is not on screen")

    def _do_multiply(self, b):
        self._need("matrices")
        A, B = np.atleast_2d(self._arr(b.get("a"))), np.atleast_2d(self._arr(b.get("b")))
        an, bn = str(b.get("a")) if isinstance(b.get("a"), str) else "A", str(b.get("b")) if isinstance(b.get("b"), str) else "B"
        name = str(b.get("name") or b.get("result") or "C")
        gone = [m for m in list(self.display.values()) + list(self.mats.values()) + self.loose if m in self.s.mobjects]
        if gone:
            self.s.play(*(FadeOut(m) for m in gone), run_time=0.4)
        self.mats, self.zones, self.loose, self.display = {}, {}, [], {}
        before = {id(m) for m in self.s.mobjects}
        C = show_matrix_product(self.s, A, B, name=name, a_name=an, b_name=bn)
        self.named[name] = A @ B
        self.loose = [m for m in self.s.mobjects if id(m) not in before and m is not C]
        self.mats[name], self.zones["main"] = C, name

    def _do_dot(self, b):
        self._need("matrices")
        show_dot_product(self.s, self._on_screen(b.get("a")), int(b.get("row") or 0),
                         self._on_screen(b.get("b")), int(b.get("col") or 0))

    def _do_transpose(self, b):
        self._need("matrices")
        mid = str(b.get("name") or b.get("id") or "")
        src = self._on_screen(mid)
        T = NumberMatrix(src.values.T, name=f"{mid}ᵀ")
        tid = f"{mid}ᵀ"
        self.named[tid] = src.values.T
        if "left" not in self.zones or self.zones.get("left") != mid:  # source to the left first
            self.s.play(src.animate.move_to(fit_to(src.copy(), "left").get_center()), run_time=0.6)
            for z in [z for z, v in self.zones.items() if v == mid]:
                self.zones.pop(z)
            self.zones["left"] = mid
        self._place(T, "right", tid)
        self.s.play(animate_transpose(src, T), run_time=1.6)

    def _do_highlight(self, b):
        self._need("matrices")
        M = self._on_screen(b.get("name") or b.get("id"))
        color = self._color(b, YELLOW)
        if b.get("cell") is not None:
            i, j = b.get("cell")
            box = M.highlight_cell(int(i), int(j), color)
        elif b.get("row") is not None:
            box = M.highlight_row(int(b.get("row")), color)
        else:
            box = M.highlight_col(int(b.get("col") or 0), color)
        self.s.play(Create(box), run_time=0.6)
        self.temp.append(box)

    def _do_set_values(self, b):
        self._need("matrices")
        mid = str(b.get("name") or b.get("id") or "")
        M = self._on_screen(mid)
        vals = np.atleast_2d(self._arr(b.get("values")))
        self.named[mid] = vals
        self.s.play(M.set_values(vals, color=self._color(b, GREEN)), run_time=1.5)

    def _do_column_op(self, b):
        self._need("matrices")
        mid = str(b.get("name") or b.get("id") or "")
        M = self._on_screen(mid)
        j = int(b.get("col") or 0)
        col = M.values[:, j]
        op = str(b.get("op") or "").lower()
        if op in ("center", "subtract_mean"):
            m = col.mean()
            new, form = col - m, f"x − {fmt(m)}   (the mean of column {j + 1})"
        elif op in ("standardize", "zscore"):
            m, sd = col.mean(), col.std()
            new, form = (col - m) / (sd or 1), f"(x − {fmt(m)}) / {fmt(sd)}   (mean and std of column {j + 1})"
        elif op == "scale":
            k = float(b.get("factor") or 1)
            new, form = col * k, f"x · {fmt(k)}"
        else:
            raise ValueError(f"unknown column op '{op}'")
        transform_column(self.s, M, j, new, form)
        self.named[mid] = M.values.copy()

    # ---------------------------------------------------------------- plot view
    def _build_plot(self, view: dict, beats: list) -> None:
        xs, ys, bars = [], [0.0], 0
        x_range = view.get("x_range")
        series: dict = {}
        for b in beats:
            do = str(b.get("do") or "").lower()
            try:
                if do in ("points", "move_points"):
                    key = str(b.get("name") or b.get("id") or "")
                    if do == "points" and b.get("x") is not None and b.get("y") is not None:
                        d = self._points_data(b)
                        if isinstance(b.get("x"), str):
                            view.setdefault("x_label", b["x"])
                        if isinstance(b.get("y"), str):
                            view.setdefault("y_label", b["y"])
                    elif b.get("data") is not None:
                        d = np.atleast_2d(np.asarray(b.get("data"), float))
                    else:   # an op on earlier points: compute where they will go
                        d = series.get(key, next(iter(series.values()), None)) if series else None
                        op = str(b.get("op") or "").lower()
                        d = d - d.mean(axis=0) if op in ("center", "subtract_mean") else (
                            (d - d.mean(axis=0)) / np.where(d.std(axis=0) > 0, d.std(axis=0), 1))
                    if d.shape[1] == 1:
                        d = np.hstack([d, np.zeros_like(d)])
                    series[key] = d
                    xs += list(d[:, 0]); ys += list(d[:, 1])
                elif do in ("histogram", "series", "knn"):
                    ex, ey = self._ml_extent(b, do)
                    xs += list(ex)
                    ys += list(ey)
                elif do == "bars":
                    bars = max(bars, len(b.get("values") or []))
                    ys += [float(v) for v in b.get("values") or []]
                elif do in ("vline",):
                    xs.append(float(b.get("x")))
                elif do == "point":
                    xs.append(float(b.get("x"))); ys.append(float(b.get("y")))
                elif do == "segment":
                    for k in ("from", "to"):
                        p = b.get(k) or [0, 0]
                        xs.append(float(p[0])); ys.append(float(p[1]))
            except Exception:
                pass
        if bars:
            xlo, xhi, xstep = 0.0, bars + 1.0, 1.0
        elif x_range and len(x_range) >= 2:
            xlo, xhi, xstep = _nice_range(*x_range[:2])
        elif xs:
            pad = 0.1 * (max(xs) - min(xs) or 2)
            lo, hi = min(xs), max(xs)
            if (0 < lo < 0.5 * hi) or (lo * 0.5 > hi < 0):
                lo, hi = min(lo, 0.0), max(hi, 0.0)
            xlo, xhi, xstep = _nice_range(lo - pad, hi + pad)
        else:
            xlo, xhi, xstep = -5.0, 5.0, 1.0
        for b in beats:  # curves decide the y range over the chosen x range
            if str(b.get("do") or "").lower() in ("curve", "area", "descent") and b.get("f"):
                try:
                    yy = formula(b["f"])(np.linspace(xlo, xhi, 200))
                    yy = yy[np.isfinite(yy)]
                    if yy.size:
                        lo, hi = np.percentile(yy, 2), np.percentile(yy, 98)
                        ys += [float(lo), float(hi)]
                except Exception:
                    pass
        y_range = view.get("y_range")
        if y_range and len(y_range) >= 2:
            ylo, yhi, ystep = _nice_range(*y_range[:2], max_ticks=6)
        else:
            data_ys = ys[1:] or [0.0]   # ys[0] is a 0 placeholder: only keep 0 when it is near the data
            lo, hi = min(data_ys), max(data_ys)
            if bars or (0 <= lo < 0.5 * hi) or (lo * 0.5 > hi < 0) or lo <= 0 <= hi:
                lo, hi = min(lo, 0.0), max(hi, 0.0)
            pad = (0.35 if bars else 0.12) * (hi - lo or 1)   # bars: room for value labels + markers
            ylo, yhi, ystep = _nice_range(lo - (pad if lo < 0 else 0), hi + pad, max_ticks=6)
        self.plot = DataPlot(x_range=(xlo, xhi, xstep), y_range=(ylo, yhi, ystep), width=8.6, height=3.9,
                             x_label=str(view.get("x_label") or "x"), y_label=str(view.get("y_label") or "y"),
                             x_numbers=not bars)
        fit_to(self.plot, "main")
        self.s.play(Create(self.plot.axes), FadeIn(self.plot.ticks), FadeIn(self.plot.x_label),
                    FadeIn(self.plot.y_label), run_time=1.2)
        self.s.remove(*self.plot.submobjects)
        self.s.add(self.plot)

    def _xy(self, x, y):
        return self.plot.axes.c2p(float(x), float(y))

    def _adopt(self, *mobs) -> None:
        """Animated pieces become part of the plot (drawn once, cleared with it)."""
        self.s.remove(*mobs)
        self.plot.add(*mobs)

    def _do_curve(self, b):
        self._need("plot")
        f = formula(b.get("f"))
        (xlo, xhi), (ylo, yhi) = self.plot.axes.x_range[:2], self.plot.axes.y_range[:2]
        a, c = b.get("from"), b.get("to")
        a = xlo if a is None else max(xlo, float(a))
        c = xhi if c is None else min(xhi, float(c))
        margin = 0.02 * (yhi - ylo)

        def g(x):
            y = float(f(np.array([x]))[0])
            return float(np.clip(y if np.isfinite(y) else yhi, ylo - margin, yhi + margin))

        color = self._color(b)
        graph = self.plot.axes.plot(g, x_range=[a, c, (c - a) / 200], color=color, stroke_width=4)
        anims = [Create(graph, run_time=1.6)]
        label = str(b.get("label") or "").strip()
        if label:
            xe = a + 0.92 * (c - a)
            lab = boxed(label, color=color, font_size=24, opacity=0.6).next_to(self._xy(xe, g(xe)), UP, buff=0.15)
            if lab.get_right()[0] > 6.8:
                lab.shift(LEFT * (lab.get_right()[0] - 6.8))
            anims.append(FadeIn(lab))
        self.s.play(*anims)
        self._adopt(*(a.mobject for a in anims))
        self.graphs[str(b.get("name") or b.get("id") or label or f"f{len(self.graphs)}")] = (graph, f)

    def _do_area(self, b):
        self._need("plot")
        key = b.get("curve")
        if isinstance(key, str) and key in self.graphs:
            graph, f = self.graphs[key]
        elif b.get("f"):
            f = formula(b["f"])
            graph = self.plot.axes.plot(lambda x: float(f(np.array([x]))[0]), color=WHITE)
        elif self.graphs:
            graph, f = list(self.graphs.values())[-1]
        else:
            raise KeyError("no curve to shade")
        a, c = float(b.get("from")), float(b.get("to"))
        color = self._color(b, BLUE)
        area = self.plot.axes.get_area(graph, x_range=(a, c), color=color, opacity=0.45)
        xx = np.linspace(a, c, 801)
        value = float(np.trapezoid(f(xx), xx)) if hasattr(np, "trapezoid") else float(np.trapz(f(xx), xx))
        anims = [FadeIn(area)]
        if b.get("show_value", True):
            lab = boxed(f"area ≈ {fmt(value, 3)}", color=color, font_size=24, opacity=0.7)
            mid = (a + c) / 2
            lab.next_to(self._xy(mid, float(f(np.array([mid]))[0])), UP, buff=0.25)
            anims.append(FadeIn(lab))
        self.s.play(*anims, run_time=1.2)
        self._adopt(*(a.mobject for a in anims))

    def _do_bars(self, b):
        self._need("plot")
        vals = [float(v) for v in b.get("values") or []]
        labels = [str(x) for x in (b.get("labels") or [])]
        color = self._color(b, BLUE)
        unit_w = self._xy(1, 0)[0] - self._xy(0, 0)[0]
        group, texts = VGroup(), VGroup()
        for k, v in enumerate(vals):
            base, top = self._xy(k + 1, 0), self._xy(k + 1, v)
            h = abs(top[1] - base[1])
            bar = Rectangle(width=0.6 * unit_w, height=max(h, 0.01), fill_color=color, fill_opacity=0.75,
                            stroke_width=1.5, stroke_color=color)
            bar.move_to(base, aligned_edge=DOWN if v >= 0 else UP)
            bar._kit_neg = v < 0
            group.add(bar)
            if b.get("show_values", True):
                texts.add(Text(fmt(v, 3), font_size=20, color=color).next_to(bar, UP if v >= 0 else DOWN, buff=0.08))
            if k < len(labels):
                texts.add(Text(labels[k], font_size=20, color=GREY_B).next_to(base, DOWN, buff=0.12))
        self.s.play(AnimationGroup(*(GrowFromEdge(r, UP if r._kit_neg else DOWN) for r in group), lag_ratio=0.12),
                    run_time=1.4)
        if len(texts):
            self.s.play(FadeIn(texts), run_time=0.6)
        self._adopt(group, texts)

    def _do_points(self, b):
        if self.view == "plane":
            self.stage.add_points([np.asarray(p, float)[:2] for p in (b.get("coords") or b.get("data") or [])],
                                  color=self._color(b))
            return
        self._need("plot")
        data = self._points_data(b)
        if data.shape[1] == 1:
            data = np.hstack([data, np.zeros_like(data)])
        key = str(b.get("name") or b.get("id") or f"p{len(self.series)}")
        if b.get("labels"):   # one class label per point: colored by class, with a legend
            dots, legend = self._class_dots(key, data, b["labels"])
            data = data[: len(dots)]
            self.s.play(FadeIn(dots, lag_ratio=0.05), FadeIn(legend), run_time=1.0)
            self._adopt(legend)
        else:
            dots = self.plot.dots(data[:, :2], color=self._color(b, BLUE))
            self.s.play(FadeIn(dots, lag_ratio=0.05), run_time=1.0)
        self.series[key] = (dots, data[:, :2])

    def _points_data(self, b) -> np.ndarray:
        """[[x, y], ...] from "data", or from two columns: "x": "X", "y": "Y" (names or lists)."""
        if b.get("x") is not None and b.get("y") is not None:
            return np.column_stack([self._arr(b["x"]).ravel(), self._arr(b["y"]).ravel()])
        return np.atleast_2d(np.asarray(b.get("data"), float))

    def _series(self, b):
        key = b.get("name") or b.get("id")
        if isinstance(key, str) and key in self.series:
            return key, self.series[key]
        if self.series:
            key = list(self.series)[-1]
            return key, self.series[key]
        raise KeyError("no points on the plot yet")

    def _do_move_points(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        op = str(b.get("op") or "").lower()
        if b.get("data") is not None:
            new = np.atleast_2d(np.asarray(b.get("data"), float))[:, :2]
        elif op in ("center", "subtract_mean"):
            new = data - data.mean(axis=0)
        elif op in ("standardize", "zscore"):
            new = (data - data.mean(axis=0)) / np.where(data.std(axis=0) > 0, data.std(axis=0), 1)
        else:
            raise ValueError("move_points needs data or op")
        stale = [m for m in self.fits.pop(key, [])]
        for m in stale:
            self.plot.remove(m)
        self.s.play(self.plot.move_dots(dots, new), *(FadeOut(m) for m in stale))
        self.series[key] = (dots, new)

    def _do_fit_line(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        x, y = data[:, 0], data[:, 1]
        slope, icept = np.polyfit(x, y, 1)
        (xlo, xhi) = self.plot.axes.x_range[:2]
        color = self._color(b, ORANGE)
        line = Line(self._xy(xlo, slope * xlo + icept), self._xy(xhi, slope * xhi + icept), color=color, stroke_width=4)
        sign = "+" if icept >= 0 else "−"
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
        lab = boxed(f"y = {fmt(slope)}·x {sign} {fmt(abs(icept))}   (r = {fmt(r)})", color=color, font_size=24,
                    opacity=0.7)
        lab.next_to(line.get_end(), UP + LEFT, buff=0.1)
        if lab.get_right()[0] > 6.8:
            lab.shift(LEFT * (lab.get_right()[0] - 6.8))
        self.s.play(Create(line), FadeIn(lab), run_time=1.2)
        self._adopt(line, lab)
        self.fits[key] = [line, lab]

    def _do_vline(self, b):
        self._need("plot")
        x = float(b.get("x"))
        (ylo, yhi) = self.plot.axes.y_range[:2]
        color = self._color(b, YELLOW)
        line = DashedLine(self._xy(x, ylo), self._xy(x, yhi), color=color, dash_length=0.12)
        anims = [Create(line)]
        if b.get("label"):
            lab = boxed(str(b["label"]), color=color, font_size=22, opacity=0.7).next_to(line.get_end(), DOWN, buff=0.1)
            anims.append(FadeIn(lab))
        self.s.play(*anims, run_time=0.8)
        self._adopt(*(a.mobject for a in anims))

    def _do_point(self, b):
        self._need("plot")
        p = self._xy(b.get("x"), b.get("y"))
        color = self._color(b, YELLOW)
        dot = Dot(p, color=color, radius=0.09)
        anims = [FadeIn(dot, scale=1.5)]
        if b.get("label"):
            lab = boxed(str(b["label"]), color=color, font_size=22, opacity=0.7).next_to(dot, UP + RIGHT, buff=0.08)
            anims.append(FadeIn(lab))
        self.s.play(*anims, run_time=0.8)
        self._adopt(*(a.mobject for a in anims))

    def _do_segment(self, b):
        self._need("plot")
        p0, p1 = b.get("from") or [0, 0], b.get("to") or [0, 0]
        color = self._color(b, YELLOW)
        line = Line(self._xy(*p0[:2]), self._xy(*p1[:2]), color=color, stroke_width=4)
        anims = [Create(line)]
        if b.get("label"):
            lab = boxed(str(b["label"]), color=color, font_size=22, opacity=0.7).next_to(line.get_center(), RIGHT, buff=0.1)
            anims.append(FadeIn(lab))
        self.s.play(*anims, run_time=0.8)
        self._adopt(*(a.mobject for a in anims))

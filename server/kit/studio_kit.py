"""studio_kit: tested, layout-safe building blocks for AI-generated Manim scenes.

Generated code does `from studio_kit import *`. Everything here uses `Text` (works with or
without LaTeX) and a fixed screen layout so captions, titles, matrices and worked equations
never pile on top of each other:

    y  4 ┌──────────────────────────────────────────────┐
         │ TITLE  (set_title)                            │ y ≈ 3.4
         │ MAIN area  (fit_to(m, "main"/"left"/"right")) │ y ≈ -1.6 .. 2.9
         │ WORK line  (equations, show_dot_product)      │ y ≈ -2.25
         │ CAPTION    (Narrator.say)                     │ y ≈ -3.2 (clear of player controls)
    y -4 └──────────────────────────────────────────────┘
"""
from __future__ import annotations

import os
import re
import textwrap

import manimpango
import numpy as np
from manim import (
    BLACK, BLUE, DOWN, GREEN, GREY_B, LEFT, ORIGIN, RIGHT, UP, WHITE, YELLOW,
    AnimationGroup, Axes, Create, Dot, FadeIn, FadeOut, Line, MarkupText, Mobject, Scene,
    SurroundingRectangle, Text, ThreeDScene, Transform, TransformFromCopy, VGroup, VMobject, Write,
)

# Typeset everything in Computer Modern (the classic LaTeX look) without needing LaTeX:
# the CMU Serif fonts (SIL Open Font License) ship next to this file. Only on Linux (the server):
# on Windows, Pango needs ~7 s per new string with runtime-registered fonts, so local test renders
# keep the default font.
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT = ""
if not os.name == "nt":
    for _f in ("cmunrm.otf", "cmunbx.otf", "cmunti.otf", "cmunbi.otf"):
        try:
            manimpango.register_font(os.path.join(_FONT_DIR, _f))
        except Exception:
            pass
    FONT = "CMU Serif" if "CMU Serif" in manimpango.list_fonts() else ""
if FONT:
    Text.set_default(font=FONT)
    MarkupText.set_default(font=FONT)

# Some models write LaTeX inside plain Text ("$A$", "\lambda", "\begin{bmatrix}..."), which Text shows
# literally. Every Text is passed through _plain_math first, which turns that into Unicode math.
from texclean import plain_math as _plain_math  # noqa: E402


_text_init = Text.__init__


def _text_init_plain(self, text, *args, **kwargs):
    _text_init(self, _plain_math(text), *args, **kwargs)


Text.__init__ = _text_init_plain

# Speed: every FadeIn / FadeOut / Transform deep-copies its mobject, and each glyph of a Text keeps its
# parsed SVG path, which costs ~1-2 ms per character to copy (a caption: ~150 ms per copy). The path is
# never changed once the glyph's points are built, so copies share it (Text copies get ~10x faster).
try:
    from manim.mobject.svg.svg_mobject import VMobjectFromSVGPath

    def _deepcopy_sharing_svg_path(self, memo):
        path = self.__dict__.get("path_obj")
        if path is not None:
            memo[id(path)] = path
        return Mobject.__deepcopy__(self, memo)

    VMobjectFromSVGPath.__deepcopy__ = _deepcopy_sharing_svg_path
except ImportError:  # pragma: no cover - other Manim layouts
    pass

__all__ = [
    "ZONES", "fmt", "fit_to", "set_title", "Narrator", "NumberMatrix", "show_dot_product",
    "transform_column", "animate_transpose", "work_text", "DataPlot", "clear_stage",
    "show_matrix_product", "hud", "boxed", "Plane2D", "Space3D", "I_COLOR", "J_COLOR", "K_COLOR",
]

# id(DataPlot) -> dot groups made by plot.dots(); kept at module level so it is never deep-copied
_PLOT_DOTS: dict[int, list] = {}

# ----------------------------------------------------------------- layout --
# name: (center, max_width, max_height)
ZONES = {
    "title": (np.array([0.0, 3.4, 0.0]), 13.0, 0.8),
    "main": (np.array([0.0, 0.65, 0.0]), 13.0, 4.4),
    "left": (np.array([-3.6, 0.65, 0.0]), 6.4, 4.4),
    "right": (np.array([3.6, 0.65, 0.0]), 6.4, 4.4),
    "work": (np.array([0.0, -2.25, 0.0]), 13.0, 0.75),
    # Kept above the bottom ~4% of the frame: video players draw their control bar there.
    "caption": (np.array([0.0, -3.2, 0.0]), 13.4, 1.0),
}


def fmt(x: float, decimals: int = 2) -> str:
    """Format a number for display: integers without decimals, no '-0.00'."""
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.{decimals}f}"
    return "0" if s.strip("-0.") == "" else s


def fit_to(mob: Mobject, zone: str = "main") -> Mobject:
    """Scale `mob` down (never up) to fit the zone and move it to the zone's center."""
    center, w, h = ZONES[zone]
    if mob.width > w:
        mob.scale_to_fit_width(w)
    if mob.height > h:
        mob.scale_to_fit_height(h)
    return mob.move_to(center)


def hud(scene: Scene, *mobs: Mobject):
    """Pin mobjects to the screen. Needed for text and panels in 3-D scenes (otherwise they
    rotate with the camera); does nothing in 2-D. Call it BEFORE FadeIn/Write."""
    if isinstance(scene, ThreeDScene) and mobs:
        scene.add_fixed_in_frame_mobjects(*mobs)
        scene.remove(*mobs)
    return mobs[0] if len(mobs) == 1 else VGroup(*mobs)


def boxed(text: str, color=WHITE, font_size: int = 28, opacity: float = 0.8) -> Text:
    """Text on a dark box, readable on top of grids and moving shapes."""
    t = Text(text, font_size=font_size, color=color)
    t.add_background_rectangle(color=BLACK, opacity=opacity, buff=0.1)
    return t


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


def _caption_chunks(text: str, width: int, max_lines: int = 2) -> list[str]:
    """Split a caption into pieces of at most `max_lines` wrapped lines, at sentence ends
    (then at commas/semicolons for one very long sentence)."""
    def lines(s):
        return len(_wrap(s, width).split("\n"))

    if lines(text) <= max_lines:
        return [text]
    pieces = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if lines(sentence) > max_lines:   # one long sentence: break it at commas / semicolons
            pieces += [p for p in re.split(r"(?<=[,;:])\s+", sentence) if p]
        else:
            pieces.append(sentence)
    chunks, cur = [], ""
    for p in pieces:
        trial = f"{cur} {p}".strip()
        if cur and lines(trial) > max_lines:
            chunks.append(cur)
            cur = p
        else:
            cur = trial
    if cur:
        chunks.append(cur)
    return chunks or [text]


def set_title(scene: Scene, text: str, font_size: int = 34, color=WHITE) -> Text:
    """Show (or replace) the title in the TITLE zone."""
    new = fit_to(boxed(text, color=color, font_size=font_size), "title")
    hud(scene, new)
    old = getattr(scene, "_kit_title", None)
    scene.play(*([FadeOut(old)] if old else []), FadeIn(new, shift=DOWN * 0.2))
    scene._kit_title = new
    return new


class Narrator:
    """One caption at a time in the CAPTION zone. `say` replaces the previous caption and
    waits long enough to read it (about 0.3 s per word)."""

    def __init__(self, scene: Scene, font_size: int = 26, chars_per_line: int = 72):
        self.scene, self.font_size, self.cpl = scene, font_size, chars_per_line
        self.current: Text | None = None

    def say(self, text: str, wait: float | None = None, color=WHITE) -> Text:
        """Show a caption. Text longer than two lines is shown as several two-line captions, one
        after another (squeezing it into the caption zone would make the letters too small)."""
        chunks = _caption_chunks(text, self.cpl)
        for chunk in chunks[:-1]:
            self._say_one(chunk, None, color)
        return self._say_one(chunks[-1], wait, color)

    def _say_one(self, text: str, wait: float | None, color) -> Text:
        new = Text(_wrap(text, self.cpl), font_size=self.font_size, color=color, line_spacing=0.9)
        new.add_background_rectangle(color=BLACK, opacity=0.82, buff=0.14)
        fit_to(new, "caption")
        hud(self.scene, new)
        anims = [FadeIn(new)]
        if self.current is not None:
            anims.insert(0, FadeOut(self.current))
        self.scene.play(*anims, run_time=0.6)
        self.current = new
        self.scene.wait(wait if wait is not None else max(1.5, 0.3 * len(text.split())))
        return new

    def clear(self) -> None:
        if self.current is not None:
            self.scene.play(FadeOut(self.current), run_time=0.4)
            self.current = None


def clear_stage(scene: Scene, *keep: Mobject, run_time: float = 0.8) -> None:
    """Fade out everything on screen except the title, the current caption and `keep`.
    Call it between storyboard steps so old matrices, plots, dots and highlights never pile up.
    `keep` may be a Plane2D / Space3D: its grid, vectors, their labels and the matrix panel stay."""
    protected = set()
    for m in keep:
        if m is None:
            continue
        if not isinstance(m, Mobject) and hasattr(m, "_kept_mobjects"):   # a Plane2D / Space3D
            mobs = m._kept_mobjects()
        elif isinstance(m, Mobject):
            mobs = [m]
        else:
            continue
        for k in mobs:   # whole families: Manim may have split a group into its parts on screen
            protected.update(id(x) for x in k.get_family())
        for g in _PLOT_DOTS.get(id(m), []):  # a kept plot keeps its dots
            protected.add(id(g))
            protected.update(id(d) for d in g)
    title = getattr(scene, "_kit_title", None)
    if title is not None:
        protected.add(id(title))
    for m in scene.mobjects:
        if isinstance(m, Text) and abs(m.get_center()[1] - ZONES["caption"][0][1]) < 0.7:
            protected.add(id(m))  # the Narrator caption
    gone = [m for m in scene.mobjects if id(m) not in protected]
    if gone:
        scene.play(*(FadeOut(m) for m in gone), run_time=run_time)


# Models often call these as methods (self.clear_stage(), self.set_title(...)); support both.
Scene.clear_stage = lambda self, *keep, **kw: clear_stage(self, *keep, **kw)
Scene.set_title = lambda self, text, **kw: set_title(self, text, **kw)


def work_text(text: str, font_size: int = 28, color=WHITE) -> Text:
    """A one-line worked equation placed in the WORK zone (fade it in/out yourself).
    In a 3-D scene call hud(self, eq) before showing it."""
    return fit_to(boxed(text, color=color, font_size=font_size), "work")


# ----------------------------------------------------------------- matrices --
def _bracket(top, bottom, side: int, tick: float = 0.15) -> VMobject:
    """`[` (side=-1) or `]` (side=+1) between two points."""
    t = tick * -side
    return VMobject(stroke_width=3).set_points_as_corners(
        [top + RIGHT * t, top, bottom, bottom + RIGHT * t])


class NumberMatrix(VGroup):
    """A matrix of numbers with real brackets and addressable cells.

        X = NumberMatrix([[2, 4], [6, 8]], name="X")
        X.cell(0, 1)                        # the Text showing 4
        X.row(1), X.col(0)                  # VGroups of cells
        X.highlight_col(0)                  # a SurroundingRectangle (play Create on it)
        self.play(X.set_values(new_vals))   # morph all numbers; columns re-space automatically
        self.play(X.set_cell(0, 1, 3.5))    # morph one number
        X.values                            # the numbers as a numpy array (kept in sync)
    Use placeholder="?" for a result matrix that is filled in step by step.
    """

    def __init__(self, values, name: str | None = None, decimals: int = 2, font_size: int = 30,
                 h_buff: float = 0.45, v_buff: float = 0.3, color=WHITE, placeholder: str | None = None,
                 col_colors=None):
        super().__init__()
        self.values = np.array(values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        self.decimals, self.font_size, self.color = decimals, font_size, color
        self.h_buff, self.v_buff, self.name = h_buff, v_buff, name
        rows, cols = self.values.shape
        self._strings = [[placeholder if placeholder is not None else fmt(v, decimals) for v in row]
                         for row in self.values]
        self._colors = [[(col_colors[j % len(col_colors)] if col_colors else color) for j in range(cols)]
                        for _ in range(rows)]
        # A placeholder matrix reserves the width of its final numbers (the `values` passed in),
        # so filling it in later never makes it grow into its neighbours.
        self._reserve = ([[fmt(v, decimals) for v in row] for row in self.values]
                         if placeholder is not None else None)
        grid, brackets, label = self._build()
        self._grid = grid
        self.cells = VGroup(*(t for row in grid for t in row))
        self.brackets = brackets
        self.label = label
        self.add(self.cells, self.brackets)
        if label is not None:
            self.add(label)

    def _build(self, anchor=None):
        """Lay out Text cells for the current strings; returns (grid, brackets, label).
        The cell block is centered on `anchor` (default: origin)."""
        rows, cols = len(self._strings), len(self._strings[0])
        grid = [[Text(self._strings[i][j], font_size=self.font_size, color=self._colors[i][j])
                 for j in range(cols)] for i in range(rows)]
        ref_h = Text("0", font_size=self.font_size).height
        col_w = [max(max(grid[i][j].width for i in range(rows)), ref_h * 0.7) for j in range(cols)]
        if self._reserve is not None:
            for j in range(cols):
                col_w[j] = max(col_w[j], *(Text(self._reserve[i][j], font_size=self.font_size).width
                                           for i in range(rows)))
        x = 0.0
        for j in range(cols):
            for i in range(rows):
                grid[i][j].move_to([x + col_w[j] / 2, -i * (ref_h + self.v_buff + 0.1), 0])
            x += col_w[j] + self.h_buff
        block = VGroup(*(t for r in grid for t in r))
        # Center on the row/column grid (not glyph extents) so the block never drifts.
        width = x - self.h_buff
        height = (rows - 1) * (ref_h + self.v_buff + 0.1)
        offset = np.array([width / 2, -height / 2, 0])
        block.shift((anchor if anchor is not None else ORIGIN) - offset)
        pad = 0.18
        top = grid[0][0].get_center()[1] + ref_h / 2 + pad
        bot = grid[-1][0].get_center()[1] - ref_h / 2 - pad
        lx = block.get_center()[0] - width / 2 - pad
        rx = block.get_center()[0] + width / 2 + pad
        lx = min(lx, block.get_left()[0] - pad)
        rx = max(rx, block.get_right()[0] + pad)
        brackets = VGroup(_bracket(np.array([lx, top, 0]), np.array([lx, bot, 0]), -1),
                          _bracket(np.array([rx, top, 0]), np.array([rx, bot, 0]), +1))
        label = None
        if self.name:
            label = Text(f"{self.name} =", font_size=self.font_size, color=self.color)
            label.next_to(brackets[0], LEFT, buff=0.2)
        return grid, brackets, label

    @property
    def shape(self):
        return self.values.shape

    def cell(self, i: int, j: int) -> Text:
        return self._grid[i][j]

    def row(self, i: int) -> VGroup:
        return VGroup(*self._grid[i])

    def col(self, j: int) -> VGroup:
        return VGroup(*(r[j] for r in self._grid))

    def highlight_cell(self, i, j, color=YELLOW) -> SurroundingRectangle:
        return SurroundingRectangle(self.cell(i, j), color=color, buff=0.08)

    def highlight_row(self, i, color=YELLOW) -> SurroundingRectangle:
        return SurroundingRectangle(self.row(i), color=color, buff=0.1)

    def highlight_col(self, j, color=YELLOW) -> SurroundingRectangle:
        return SurroundingRectangle(self.col(j), color=color, buff=0.1)

    def _relayout(self, lag_ratio: float = 0.0) -> Transform:
        """One Transform of the whole matrix to a fresh layout of the current strings, keeping it
        centered where it is and at its current scale. Animating the matrix as a single object
        keeps it intact in the scene (per-cell animations make Manim split the group apart)."""
        center = VGroup(self.cells, self.brackets).get_center()
        old_h = abs(self.brackets[0].get_top()[1] - self.brackets[0].get_bottom()[1])
        grid, brackets, label = self._build()
        new_h = abs(brackets[0].get_top()[1] - brackets[0].get_bottom()[1])
        cells = VGroup(*(t for r in grid for t in r))
        parts = [cells, brackets] + ([label] if self.label is not None and label is not None else [])
        target = VGroup(*parts)
        if new_h > 0 and old_h > 0:
            target.scale(old_h / new_h, about_point=ORIGIN)
        target.shift(center - VGroup(cells, brackets).get_center())
        bg = getattr(self, "_kit_panel_bg", None)
        if bg is not None:  # a matrix shown on a panel: its dark box is its first part and resizes with it
            target.add_to_back(type(bg)(target, color=BLACK, fill_opacity=bg.get_fill_opacity(), buff=0.18))
            # it hugs a screen corner: keep that outer edge and the top in place (grow inwards, not off-screen)
            if getattr(self, "_kit_panel_side", "right") == "left":
                target.shift(RIGHT * (self.get_left()[0] - target.get_left()[0]))
            else:
                target.shift(RIGHT * (self.get_right()[0] - target.get_right()[0]))
            target.shift(UP * (self.get_top()[1] - target.get_top()[1]))
        texts = [(self._grid[i][j], self._strings[i][j]) for i in range(len(grid)) for j in range(len(grid[0]))]
        return _MatrixTransform(self, target, texts)

    def set_cell(self, i: int, j: int, value: float, color=None) -> Transform:
        """Animation: change one number (the matrix re-spaces if the width changes)."""
        self.values[i, j] = value
        self._strings[i][j] = fmt(value, self.decimals)
        if color is not None:
            self._colors[i][j] = color
        return self._relayout()

    def set_values(self, new_values, color=None, lag_ratio: float = 0.0) -> Transform:
        """Animation: change every number (shape must match)."""
        new_values = np.array(new_values, dtype=float).reshape(self.values.shape)
        for i in range(new_values.shape[0]):
            for j in range(new_values.shape[1]):
                s = fmt(new_values[i, j], self.decimals)
                if s != self._strings[i][j]:
                    self._strings[i][j] = s
                    if color is not None:
                        self._colors[i][j] = color
        self.values = new_values
        return self._relayout(lag_ratio)


class _MatrixTransform(Transform):
    """Transform of a whole NumberMatrix that records each cell's new text only when it actually
    plays, so the layout checker sees what is really on screen (an unplayed set_values changes nothing)."""

    def __init__(self, matrix, target, texts, **kw):
        super().__init__(matrix, target, **kw)
        self._texts = texts

    def begin(self):
        super().begin()
        for cell, text in self._texts:
            cell.text = text
            cell.original_text = text


def animate_transpose(source: NumberMatrix, target: NumberMatrix) -> AnimationGroup:
    """Animation: copies of source cell (i, j) fly to target cell (j, i), brackets appear.
    Build `target = NumberMatrix(source.values.T, ...)` and position it first."""
    rows, cols = source.shape
    moves = [TransformFromCopy(source.cell(i, j), target.cell(j, i)) for i in range(rows) for j in range(cols)]
    extras = [Create(target.brackets)] + ([FadeIn(target.label)] if target.label else [])
    return AnimationGroup(*moves, *extras, lag_ratio=0.03)


def transform_column(scene: Scene, M: NumberMatrix, j: int, new_values, formula: str,
                     narrator: Narrator | None = None, caption: str | None = None, color=GREEN) -> None:
    """Highlight column j, show `formula` on the work line, morph the column's numbers to
    `new_values`, then clean up. Use it for per-column operations like (x - mean) / std."""
    box = M.highlight_col(j)
    eq = work_text(formula)
    scene.play(Create(box), FadeIn(eq))
    if narrator and caption:
        narrator.say(caption)
    full = M.values.copy()
    full[:, j] = np.asarray(new_values, dtype=float).reshape(-1)
    for i in range(M.shape[0]):
        M._colors[i][j] = color
    scene.play(FadeOut(box), M.set_values(full), run_time=1.5)
    scene.wait(0.8)
    scene.play(FadeOut(eq))


def show_dot_product(scene: Scene, A: NumberMatrix, i: int, B: NumberMatrix, j: int,
                     C: NumberMatrix | None = None, narrator: Narrator | None = None,
                     caption: str | None = None, color=YELLOW, result_color=GREEN) -> float:
    """Worked example of C[i, j] = row i of A · column j of B.
    Highlights the row and column, writes every product term on the work line,
    then moves the result into C[i, j] (if C is given). Returns the value."""
    a, b = A.values[i, :], B.values[:, j]
    value = float(a @ b)
    rbox, cbox = A.highlight_row(i, color), B.highlight_col(j, color)
    terms = " + ".join(f"({fmt(x)})·({fmt(y)})" for x, y in zip(a, b))
    eq = work_text(f"{terms} = {fmt(value)}")
    scene.play(Create(rbox), Create(cbox))
    if narrator and caption:
        narrator.say(caption)
    scene.play(Write(eq), run_time=1.2)
    scene.wait(0.8)
    if C is not None:
        scene.play(C.set_cell(i, j, value, result_color), run_time=0.9)
        flash = C.highlight_cell(i, j, result_color)
        scene.play(Create(flash), run_time=0.4)
        scene.play(FadeOut(rbox), FadeOut(cbox), FadeOut(eq), FadeOut(flash))
        return value
    scene.play(FadeOut(rbox), FadeOut(cbox), FadeOut(eq))
    return value


def show_matrix_product(scene: Scene, A, B, name: str = "C", a_name: str | None = None,
                        b_name: str | None = None, narrator: Narrator | None = None,
                        font_size: int = 28, explain_each: bool = True) -> NumberMatrix:
    """Full worked matrix product on its own stage: lays out  A · B = C  in the MAIN zone, then
    computes every entry of C with show_dot_product (row · column, every term written out).
    A and B may be arrays or NumberMatrix objects. Returns C (a NumberMatrix on screen).
    Call clear_stage(self) before it if the screen is busy."""
    Av = A.values if isinstance(A, NumberMatrix) else np.asarray(A, dtype=float)
    Bv = B.values if isinstance(B, NumberMatrix) else np.asarray(B, dtype=float)
    Am = NumberMatrix(Av, name=a_name, font_size=font_size)
    Bm = NumberMatrix(Bv, name=b_name, font_size=font_size)
    Cm = NumberMatrix(Av @ Bv, name=name, font_size=font_size, placeholder="?")  # reserves final widths
    dot = Text("·", font_size=font_size + 10)
    eq = Text("→", font_size=font_size + 4)
    row = VGroup(Am, dot, Bm, eq, Cm).arrange(RIGHT, buff=0.35)
    fit_to(row, "main")
    scene.play(FadeIn(Am), FadeIn(dot), FadeIn(Bm), FadeIn(eq), FadeIn(Cm))
    if narrator:
        narrator.say(f"Each entry ({name}) row i, column j = row i of the left matrix · column j of the "
                     f"right matrix: multiply matching numbers and add them up.")
    for i in range(Av.shape[0]):
        for j in range(Bv.shape[1]):
            cap = f"Entry ({i + 1},{j + 1}): row {i + 1} · column {j + 1}" if explain_each else None
            show_dot_product(scene, Am, i, Bm, j, Cm, narrator if explain_each else None, cap)
    return Cm


# ----------------------------------------------------------------- plots ----
class DataPlot(VGroup):
    """Axes with plain-text tick labels plus helpers to draw data rows as dots.

        plot = DataPlot(x_range=(-3, 13, 2), y_range=(-3, 13, 2), x_label="x₁", y_label="x₂")
        fit_to(plot, "right")
        dots = plot.dots(X)                       # VGroup of Dots, one per row of X
        self.play(plot.move_dots(dots, X_std))    # animate every point to new coordinates
    """

    def __init__(self, x_range=(-5, 5, 1), y_range=(-5, 5, 1), width: float = 5.5, height: float = 4.0,
                 x_label: str = "x", y_label: str = "y", font_size: int = 20, x_numbers: bool = True):
        super().__init__()
        self.axes = Axes(x_range=list(x_range), y_range=list(y_range), x_length=width, y_length=height,
                         tips=False, axis_config={"include_numbers": False, "stroke_color": GREY_B})
        self.add(self.axes)
        x0 = 0 if x_range[0] <= 0 <= x_range[1] else x_range[0]
        y0 = 0 if y_range[0] <= 0 <= y_range[1] else y_range[0]
        ticks = VGroup()
        for v in (np.arange(x_range[0], x_range[1] + 1e-9, x_range[2]) if x_numbers else []):
            if abs(v - x0) > 1e-9:
                ticks.add(Text(fmt(v), font_size=font_size, color=GREY_B).next_to(self.axes.c2p(v, y0), DOWN, buff=0.1))
        for v in np.arange(y_range[0], y_range[1] + 1e-9, y_range[2]):
            if abs(v - y0) > 1e-9:
                ticks.add(Text(fmt(v), font_size=font_size, color=GREY_B).next_to(self.axes.c2p(x0, v), LEFT, buff=0.1))
        self.ticks = ticks
        self.x_label = Text(x_label, font_size=font_size + 4).next_to(self.axes.x_axis, RIGHT, buff=0.15)
        self.y_label = Text(y_label, font_size=font_size + 4).next_to(self.axes.y_axis, UP, buff=0.15)
        self.add(ticks, self.x_label, self.y_label)

    @classmethod
    def fitting(cls, *datasets, x_label: str = "x", y_label: str = "y", **kw) -> "DataPlot":
        """Axes whose ranges contain every point of every dataset (rows = points), e.g.
        DataPlot.fitting(X, X_centered, X_std) so all stages of a transformation fit."""
        pts = np.vstack([np.asarray(d, dtype=float)[:, :2] for d in datasets])
        ranges = []
        for k in range(2):
            lo, hi = min(pts[:, k].min(), 0.0), max(pts[:, k].max(), 0.0)
            span = max(hi - lo, 1.0)
            step = next(s for s in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if span / s <= 8)
            ranges.append((np.floor(lo / step) * step - step, np.ceil(hi / step) * step + step, step))
        return cls(x_range=ranges[0], y_range=ranges[1], x_label=x_label, y_label=y_label, **kw)

    def point(self, x: float, y: float):
        return self.axes.c2p(x, y)

    def dots(self, data, color=BLUE, radius: float = 0.08) -> VGroup:
        data = np.asarray(data, dtype=float)
        group = VGroup(*(Dot(self.point(r[0], r[1]), color=color, radius=radius) for r in data))
        # Pair dots with their axes by id only: Manim deep-copies mobjects when animating, and a
        # reference to the whole plot on every dot would be copied each time (very slow).
        group._kit_plot_id = id(self)
        for d in group:
            d._kit_plot_id = id(self)
        _PLOT_DOTS.setdefault(id(self), []).append(group)
        return group

    def move_dots(self, dots: VGroup, new_data, run_time: float = 2.0) -> AnimationGroup:
        new_data = np.asarray(new_data, dtype=float)
        return AnimationGroup(*(d.animate(run_time=run_time).move_to(self.point(r[0], r[1]))
                                for d, r in zip(dots, new_data)))

    def line_to(self, x0, y0, x1, y1, color=YELLOW) -> Line:
        return Line(self.point(x0, y0), self.point(x1, y1), color=color)


# ----------------------------------------------------------------- space view (2-D / 3-D) --
# Linear algebra as the transformation of space itself (see studio_space.py).
from studio_space import I_COLOR, J_COLOR, K_COLOR, Plane2D, Space3D  # noqa: E402

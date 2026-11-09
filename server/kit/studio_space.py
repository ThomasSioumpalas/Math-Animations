"""studio_space: linear algebra shown as the transformation of space itself.

Visual conventions (the same in every video, so viewers learn them once):
  - black background; a faint grey copy of the original grid stays put as a reference
  - the transformable grid: blue lines, white axes, extending well past the screen edges
  - î green, ĵ red, k̂ blue: arrows from the origin; a matrix's column j is colored like basis j,
    because column j is exactly where basis vector j lands
  - grid lines stay parallel and evenly spaced and the origin stays fixed during every transformation
  - area / volume: a translucent yellow unit square / unit cube; its signed area/volume is the determinant
  - other vectors are yellow by default; every label sits on a dark box and follows its arrow
  - all methods ANIMATE BY THEMSELVES (never wrap them in self.play)

    plane = Plane2D(self, matrices=[A], vectors=[v])   # 2-D, in a normal Scene
    space = Space3D(self, matrices=[A])                 # 3-D, in a ThreeDScene
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from manim import (
    BLACK, DEGREES, DOWN, RIGHT, UL, UP, UR, WHITE, YELLOW,
    Animation, Arrow, BackgroundRectangle, Create, Dot, FadeIn, FadeOut, GrowArrow, Line,
    MarkupText, Mobject, Polygon, Scene, SingleStringMathTex, SurroundingRectangle, Text, ThreeDScene,
    VGroup, VMobject,
)

from studio_kit import NumberMatrix, boxed, fmt, hud

_TEXTY = (Text, MarkupText, SingleStringMathTex, NumberMatrix)

I_COLOR, J_COLOR, K_COLOR = "#83C167", "#FC6255", "#58C4DD"   # î green, ĵ red, k̂ blue
GRID_COLOR = "#29ABCA"
GHOST_COLOR = "#888888"
AREA_COLOR = YELLOW
EIGEN_COLORS = ["#5CD0B3", "#C55FDB"]                          # teal, purple
SAFE_X, SAFE_TOP, SAFE_BOTTOM = 6.9, 3.0, -2.62                  # clear of the title and caption zones


# ----------------------------------------------------------------------------- helpers
def _shown(method):
    """Public stage methods first bring the grid back if clear_stage(self) faded it out."""
    def wrapper(self, *args, **kwargs):
        self._ensure_shown()
        return method(self, *args, **kwargs)
    wrapper.__doc__, wrapper.__name__ = method.__doc__, method.__name__
    return wrapper


def _bbox(m):
    """(center x, center y, half width, half height) from the raw points: much faster than Manim's
    get_center()/width, which walk the family in Python for every coordinate. None if m has no points."""
    pts = [sm.points for sm in m.get_family() if len(sm.points)]
    if not pts:
        return None
    a = np.vstack(pts)
    lo, hi = a.min(axis=0), a.max(axis=0)
    return ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2)


def _segments(pairs, **style) -> VMobject:
    """Many straight strokes as ONE shape (one draw call per frame instead of one per line)."""
    pts = []
    for a, b in pairs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        pts += [a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]
    vm = VMobject(**style)
    vm.set_points(np.array(pts) if pts else np.zeros((0, 3)))
    return vm


def _dashed(a, b, dash_length: float = 0.15, **style) -> VMobject:
    """A dashed line as ONE shape (Manim's DashedLine is one shape per dash)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    n = max(2, int(np.ceil(length * 0.5 / dash_length)))
    period = (b - a) / n
    return _segments([(a + k * period, a + (k + 0.5) * period) for k in range(n)], **style)


def _text_boxes(m) -> list:
    """Bounding boxes of the text-like parts of m (each Text/MathTex/matrix of a VGroup separately)."""
    if isinstance(m, _TEXTY):
        return [_bbox(m)]
    out = []
    for sm in m.submobjects:
        out += _text_boxes(sm)
    return out


def _path(A: np.ndarray):
    """Matrix path M(t) from the identity (t=0) to A (t=1), as a viewer expects to see it:
    shears and stretches slide straight; rotations (and rotation-heavy maps) turn rigidly instead
    of shrinking through the middle; flips and collapses pass honestly through det = 0."""
    A = np.asarray(A, float)
    n = A.shape[0]
    eye = np.eye(n)

    def straight(t):
        return (1 - t) * eye + t * A

    det = float(np.linalg.det(A))
    if det <= 1e-9:
        return straight
    gram = A.T @ A
    conformal = np.allclose(gram, np.trace(gram) / n * eye, atol=1e-6)
    worst = min(float(np.linalg.det(straight(t))) for t in np.linspace(0, 1, 41))
    if conformal or worst < 0.25 * min(1.0, det):
        U, P = scipy.linalg.polar(A)
        L = np.real(scipy.linalg.logm(U))
        return lambda t: np.real(scipy.linalg.expm(t * L)) @ ((1 - t) * eye + t * P)
    return straight


def _vec_text(v) -> str:
    return "[" + ", ".join(fmt(x) for x in np.asarray(v, float).ravel()) + "]"


def _top_limit(scene) -> float:
    return SAFE_TOP if getattr(scene, "_kit_title", None) is not None else 3.82


class _SpaceMorph(Animation):
    """Moves everything in a stage's layer along a matrix path.
    2-D arrows are redrawn every frame (tips stay crisp); 3-D arrows move as shapes and are rebuilt at the end."""

    def __init__(self, stage, A, **kw):
        self.stage, self.A = stage, np.asarray(A, float)
        kw.setdefault("run_time", 3)
        super().__init__(stage.layer, **kw)

    def create_starting_mobject(self) -> Mobject:
        return Mobject()  # we snapshot what we need ourselves (copying the whole grid is slow)

    def begin(self) -> None:
        st = self.stage
        self.path = _path(self.A)
        shapes = st._shape_family()
        if not st.redraw_arrows:
            for a, _, _ in st._arrows.values():
                shapes += [m for m in a.get_family() if len(m.points)]
        self.snap = [(m, m.points.copy()) for m in shapes]
        self.arrows = [(a, s.copy(), e.copy()) for a, s, e in st._arrows.values()] if st.redraw_arrows else []
        self.anchors = [(d, c.copy()) for d, c in st._anchors.values()]
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        st = self.stage
        M = self.path(self.rate_func(alpha))
        M3 = st._embed(M)
        for m, pts in self.snap:
            m.points = st.origin + (pts - st.origin) @ M3.T
        for a, s, e in self.arrows:
            st._draw_arrow(a, M @ s, M @ e)
        for d, c in self.anchors:
            d.move_to(st.c2p(M @ c))

    def finish(self) -> None:
        super().finish()
        st = self.stage
        for key, (a, s, e) in list(st._arrows.items()):
            st._arrows[key] = (a, self.A @ s, self.A @ e)
            st._draw_arrow(a, self.A @ s, self.A @ e)
        for key, (d, c) in list(st._anchors.items()):
            st._anchors[key] = (d, self.A @ c)
        st.M = self.A @ st.M
        for lab in st._labels:   # new coordinates in the labels now (a check run skips Manim's own update)
            lab.update(0)


class _Stage:
    """Shared machinery for Plane2D and Space3D."""

    dim = 2
    redraw_arrows = True

    def _init_common(self, scene: Scene):
        self.scene = scene
        self.M = np.eye(self.dim)            # everything applied so far
        self.layer = VGroup()                 # ONE persistent top-level group: grid, shapes, arrows, points
        self._shapes = VGroup()               # moved point by point (grid lines, squares, polygons)
        self.layer.add(self._shapes)
        self._arrows: dict[int, tuple] = {}   # id -> (arrow, start coords, end coords)
        self._anchors: dict[int, tuple] = {}  # id -> (dot, coords)  (points that move as points)
        self._notes: list[Mobject] = []       # temporary annotations, cleared before each transformation
        self._panels: list[Mobject] = []      # screen panels that labels must avoid
        self._labels: list[Mobject] = []      # labels (they avoid each other)
        self.square = None
        self.basis = []
        self.matrix_panel = None
        self.grid = VGroup()
        self._retire_others()

    def _retire_others(self):
        """One grid at a time: a new (or re-shown) stage fades out every other stage still on screen,
        with its arrows, labels and panels. Models often keep the old plane with clear_stage(self, old)
        and then build a new one, which would draw two grids on top of each other."""
        stages = getattr(self.scene, "_kit_stages", None)
        if stages is None:
            stages = self.scene._kit_stages = []
        owned: set[int] = set()
        for st in stages:
            if st is self:
                continue
            mobs = [st.background, st.layer, st.matrix_panel] + list(st._labels) + list(st._panels)
            mobs += list(getattr(st, "axis_labels", []))
            ro = getattr(st, "det_readout", None)
            mobs.append(ro[0] if isinstance(ro, tuple) else ro)
            for m in mobs:
                if isinstance(m, Mobject):
                    owned.update(id(x) for x in m.get_family())
        if self not in stages:
            stages.append(self)
        gone = [m for m in self.scene.mobjects if id(m) in owned]
        if gone:
            self.scene.play(*(FadeOut(m) for m in gone), run_time=0.6)

    def _embed(self, M):
        M3 = np.eye(3)
        M3[: self.dim, : self.dim] = M
        return M3

    def _shape_family(self):
        return [m for m in self._shapes.get_family() if len(m.points)]

    def _track_arrow(self, arrow, start, end):
        self._arrows[id(arrow)] = (arrow, np.asarray(start, float), np.asarray(end, float))
        self.layer.add(arrow)

    def _note(self, *mobs):
        self._notes.extend(mobs)
        return mobs

    def _add_static(self, mob):
        """A reference line that stays where it is during transformations (drawn under the arrows)."""
        self.layer.add_to_back(mob)
        self.layer.add_to_back(self._shapes)
        return mob

    def clear_annotations(self, run_time: float = 0.5) -> None:
        """Fade out temporary labels (landing labels, stretch labels)."""
        live = [m for m in self._notes if m in self.scene.mobjects]
        if live:
            self.scene.play(*(FadeOut(m) for m in live), run_time=run_time)
        for m in self._notes:
            m.clear_updaters()
        self._notes = []

    def coords_of(self, arrow) -> np.ndarray:
        """Current coordinates of a tracked vector's tip."""
        return self._arrows[id(arrow)][2].copy()

    def _matrix_panel(self, A, name, colors):
        mat = NumberMatrix(np.asarray(A, float), name=name, col_colors=colors, font_size=30)
        bg = SurroundingRectangle(mat, color=BLACK, fill_color=BLACK, fill_opacity=0.85, stroke_width=0, buff=0.18)
        return VGroup(bg, mat)

    def _place_panel(self, panel, corner, ignore=()):
        """Top-left or top-right corner; below any panel already showing there (panels never cover each other).
        `ignore`: panels this one replaces (it takes their place instead of stacking under them)."""
        left = float(np.asarray(corner, float)[0]) < 0
        x = (-SAFE_X - 0.1 + panel.width / 2) if left else (SAFE_X + 0.1 - panel.width / 2)
        y = _top_limit(self.scene) - panel.height / 2
        panel.move_to([x, y, 0])
        skip = set(map(id, ignore))
        live = [p for p in self._live(self._panels) if id(p) not in skip and p is not panel]
        # too wide to sit beside the panel in the other top corner? shrink it to fit (down to 60 %)
        other_side = [p for p in live if (p.get_x() < 0) != left and
                      abs(panel.get_y() - p.get_y()) < (panel.height + p.height) / 2 + 0.05]
        if other_side:
            inner = (max(p.get_right()[0] for p in other_side) if not left
                     else min(p.get_left()[0] for p in other_side))
            room = (SAFE_X + 0.1 - inner - 0.2) if not left else (inner - 0.2 + SAFE_X + 0.1)
            if 0 < room < panel.width and room / panel.width >= 0.6:
                panel.scale(room / panel.width)
                x = (-SAFE_X - 0.1 + panel.width / 2) if left else (SAFE_X + 0.1 - panel.width / 2)
                panel.move_to([x, _top_limit(self.scene) - panel.height / 2, 0])
        for _ in range(len(live)):
            hit = [p for p in live if p is not panel and
                   abs(panel.get_x() - p.get_x()) < (panel.width + p.width) / 2 + 0.05 and
                   abs(panel.get_y() - p.get_y()) < (panel.height + p.height) / 2 + 0.05]
            if not hit:
                break
            panel.set_y(min(p.get_bottom()[1] for p in hit) - 0.12 - panel.height / 2)
        return panel

    def _register_panel(self, panel):
        self._panels.append(panel)
        return panel

    def _make_panel(self, mob, corner):
        """Put `mob` on a dark box. The box goes INSIDE mob (as its first part), so FadeOut(mob) also
        removes it; a NumberMatrix's set_values / set_cell resize the box with the numbers."""
        if getattr(mob, "_kit_panel_bg", None) is None:
            bg = BackgroundRectangle(mob, color=BLACK, fill_opacity=0.85, buff=0.18)
            mob.add_to_back(bg)
            mob._kit_panel_bg = bg
        mob._kit_panel_side = "left" if float(np.asarray(corner, float)[0]) < 0 else "right"
        return self._register_panel(self._place_panel(mob, corner))

    def _live(self, mobs):
        on = set(map(id, self.scene.get_mobject_family_members()))
        return [m for m in mobs if id(m) in on]

    # -- what clear_stage keeps, and bringing the stage back
    def _kept_mobjects(self):
        """clear_stage(self, plane) keeps these: the grid with everything drawn on it, the labels that
        belong to vectors (î, ĵ, v, ...), the matrix panel and the det readout. Temporary notes
        (landing / stretch labels) and extra panels are cleared."""
        keep = [self.background, self.layer] + [m for m in self._labels if m not in self._notes]
        keep += list(getattr(self, "axis_labels", []))
        if self.matrix_panel is not None:
            keep.append(self.matrix_panel)
        ro = getattr(self, "det_readout", None)
        if ro is not None:
            keep.append(ro[0] if isinstance(ro, tuple) else ro)
        return keep

    def _add_to_back_of_scene(self):
        """Grid and arrows go BEHIND everything else (titles, captions, panels). Besides the right
        stacking, this keeps renders fast: Manim redraws every frame all objects that come after the
        first moving one, so a grid listed after a caption would be redrawn on every caption change."""
        self.scene.add(self.background, self.layer)
        self.scene.bring_to_back(self.background, self.layer)

    def _ensure_shown(self):
        """If clear_stage(self) faded the grid out and a stage method is used again, fade it back in
        (to keep it in the first place: clear_stage(self, plane))."""
        if self.grid in self.scene.get_mobject_family_members():
            return
        self._retire_others()
        labels = [m for m in self._labels if m not in self._notes] + list(getattr(self, "axis_labels", []))
        self._add_to_back_of_scene()
        self.scene.play(FadeIn(self.background), FadeIn(self.layer), *(FadeIn(m) for m in labels), run_time=0.8)


# ============================================================================= 2-D
class Plane2D(_Stage):
    """The 2-D plane as a transformable grid with î (green) and ĵ (red). Use in a normal Scene.
    Every method animates by itself."""

    dim = 2
    redraw_arrows = True

    def __init__(self, scene: Scene, matrices=(), vectors=(), unit: float | None = None,
                 show_basis: bool = True, basis_labels: bool = True, animate: bool = True):
        self._init_common(scene)
        self.matrices = [np.asarray(A, float) for A in matrices]
        self.vectors = [np.asarray(v, float) for v in vectors]
        self.origin = np.array([0.0, 0.18, 0.0])
        self.unit = float(unit) if unit else self._auto_unit(matrices, vectors)
        u = self.unit
        # Lines of one style are ONE shape with many separate strokes: drawing ~100 separate Line objects
        # per frame is what makes a grid slow to render.
        n_bg = int(np.ceil(7.8 / u))
        ks = [k for k in range(-n_bg, n_bg + 1) if k]
        self.background = VGroup(
            _segments([(self.c2p([k, -n_bg]), self.c2p([k, n_bg])) for k in ks] +
                      [(self.c2p([-n_bg, k]), self.c2p([n_bg, k])) for k in ks],
                      stroke_color=GHOST_COLOR, stroke_width=1, stroke_opacity=0.4),
            _segments([(self.c2p([0, -n_bg]), self.c2p([0, n_bg])), (self.c2p([-n_bg, 0]), self.c2p([n_bg, 0]))],
                      stroke_color=GHOST_COLOR, stroke_width=1.6, stroke_opacity=0.75))
        R = int(np.ceil(8.0 / u)) * 2  # far past the screen, so edges never show after shrinking
        ks = sorted((k for k in range(-R, R + 1) if k), key=abs)   # center first: Create grows outwards
        style = dict(stroke_color=GRID_COLOR, stroke_width=1.8, stroke_opacity=0.95)
        lines = VGroup(_segments([(self.c2p([k, -R]), self.c2p([k, R])) for k in ks], **style),
                       _segments([(self.c2p([-R, k]), self.c2p([R, k])) for k in ks], **style))
        axes = VGroup(Line(self.c2p([-R, 0]), self.c2p([R, 0]), stroke_color=WHITE, stroke_width=2.4),
                      Line(self.c2p([0, -R]), self.c2p([0, R]), stroke_color=WHITE, stroke_width=2.4))
        self.grid = VGroup(lines, axes)
        self._shapes.add(self.grid)
        self._add_to_back_of_scene()   # the layer is the single persistent group
        if animate:
            scene.play(Create(self.grid, lag_ratio=0.01), run_time=1.5)
        if show_basis:
            self.i_hat = self._arrow([0, 0], [1, 0], I_COLOR, 6)
            self.j_hat = self._arrow([0, 0], [0, 1], J_COLOR, 6)
            self.basis = [self.i_hat, self.j_hat]
            anims = [GrowArrow(self.i_hat), GrowArrow(self.j_hat)]
            if basis_labels:
                anims += [FadeIn(self._follow_label(self.i_hat, "î", I_COLOR)),
                          FadeIn(self._follow_label(self.j_hat, "ĵ", J_COLOR))]
            scene.play(*anims, run_time=1.0)

    # -- geometry
    def _auto_unit(self, matrices, vectors) -> float:
        pts = [np.array([1.0, 0]), np.array([0, 1.0])] + [np.asarray(v, float)[:2] for v in vectors]
        for A in matrices:
            A = np.asarray(A, float)
            pts += [A[:, 0], A[:, 1], A[:, 0] + A[:, 1]] + [A @ np.asarray(v, float)[:2] for v in vectors]
        pts = np.array(pts)
        need_x = np.max(np.abs(pts[:, 0])) + 0.7
        need_up = max(float(np.max(pts[:, 1])), 0.0) + 0.7
        need_down = max(float(-np.min(pts[:, 1])), 0.0) + 0.7
        unit = min(1.0, SAFE_X / need_x, (SAFE_TOP - 0.18) / need_up, (0.18 - SAFE_BOTTOM) / need_down)
        return float(max(0.35, unit))

    def _screen_reach(self) -> float:
        """Length (in plane units) from the origin to past every screen corner: static lines need no more
        (each dash is a separate shape to draw, so shorter lines render faster)."""
        return 8.8 / self.unit

    def c2p(self, coords) -> np.ndarray:
        c = np.asarray(coords, float).ravel()
        return self.origin + self.unit * np.array([c[0], c[1], 0.0])

    def _arrow(self, start, end, color, width=5):
        a = Arrow(self.c2p(start), self.c2p(end), buff=0, color=color, stroke_width=width,
                  max_tip_length_to_length_ratio=0.3, max_stroke_width_to_length_ratio=8)
        a._kit_color, a._kit_width = color, width
        self._track_arrow(a, start, end)
        return a

    def _draw_arrow(self, arrow, s, e):
        p0, p1 = self.c2p(s), self.c2p(e)
        hidden = np.linalg.norm(p1 - p0) < 0.03          # squished onto the origin
        if hidden:
            p1 = p0 + RIGHT * 0.05
        # keep the color / opacity the scene gave the arrow (e.g. eigenvectors recolored pink)
        color = arrow.get_color()
        opacity = 1.0 if getattr(arrow, "_kit_squished", False) else arrow.get_stroke_opacity()
        new = Arrow(p0, p1, buff=0, color=color, stroke_width=arrow._kit_width,
                    max_tip_length_to_length_ratio=0.3, max_stroke_width_to_length_ratio=8)
        new.set_opacity(0 if hidden else opacity)
        arrow._kit_squished = hidden
        arrow.become(new)

    def _label_spot(self, arrow, label, mid: bool = False, strict: bool = False):
        """Where a label goes: next to the arrow's tip (or beside its middle), inside the visible area,
        and not on top of a panel or another label (first free candidate wins)."""
        tip, start = arrow.get_end(), arrow.get_start()
        d = tip - start
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else RIGHT
        perp = np.array([-d[1], d[0], 0.0])
        box = _bbox(label)
        off = 0.16 + max(box[2], box[3]) if box else 0.4
        if mid:
            base = (tip + start) / 2
            cands = (base + perp * off, base - perp * off, base + perp * off * 1.8, base - perp * off * 1.8)
        else:
            cands = (tip + d * off, tip + perp * off, tip - perp * off, tip + d * off + perp * off * 1.2,
                     tip + d * off - perp * off * 1.2, tip - d * off * 2.2, tip + d * off * 2.2)
        return self._free_spot(label, cands, strict=strict, box=box)

    def _obstacles(self, label):
        """Bounding boxes of the panels, labels and any text on screen (titles, captions, notes)
        that a label must not cover."""
        on = self.scene.mobjects
        on_ids = set(map(id, on))
        label_ids = set(map(id, self._labels))
        seen, boxes = {id(label)}, []
        for m in self._panels + self._labels:
            if id(m) in seen or id(m) not in on_ids or getattr(m, "_kit_hidden", False):
                continue
            seen.add(id(m))
            boxes.append(_bbox(m))
        skip = seen | label_ids | {id(self.layer), id(self.background)}
        for m in on:
            if id(m) not in skip:
                boxes += _text_boxes(m)
        return [b for b in boxes if b and b[2] > 1e-3]

    def _free_spot(self, label, cands, extra=(), strict: bool = False, box=None):
        """The first candidate position (clipped to the visible area) where `label` covers nothing.
        None when every candidate is covered and strict=True (otherwise the first candidate)."""
        box = box or _bbox(label)
        if not box:
            return None if strict else np.array(cands[0], float)
        w, h = box[2], box[3]
        top = _top_limit(self.scene)
        others = self._obstacles(label) + [b for b in map(_bbox, extra) if b]
        first = None
        for cand in cands:
            spot = np.array(cand, float)
            spot[0] = np.clip(spot[0], -7.0 + w, 7.0 - w)
            spot[1] = np.clip(spot[1], SAFE_BOTTOM + h, top - h)
            first = spot if first is None else first
            if not any(self._overlaps(spot, w, h, o) for o in others):
                return spot
        return None if strict else first

    def _place_label(self, m, arrow, mid: bool = False):
        """Move a following label next to its arrow, or hide it while panels/text cover every free spot
        (it comes back by itself as soon as there is room)."""
        spot = self._label_spot(arrow, m, mid, strict=True)
        hidden = getattr(m, "_kit_hidden", False)
        if spot is None:
            if not hidden:
                m._kit_shown = m.copy()
                m.set_opacity(0)
                m._kit_hidden = True
            return
        if hidden:
            shown, m._kit_shown = m._kit_shown, None
            m.become(shown)
            m._kit_hidden = False
        m.move_to(spot)

    @staticmethod
    def _overlaps(c, w, h, other):
        """Does a w x h half-size box at c overlap `other` (a bbox tuple or a mobject)?"""
        ob = other if isinstance(other, tuple) else _bbox(other)
        if not ob:
            return False
        return abs(c[0] - ob[0]) < w + ob[2] + 0.05 and abs(c[1] - ob[1]) < h + ob[3] + 0.05

    def _follow_label(self, arrow, text, color, font_size=30, mid: bool = False):
        lab = boxed(text, color=color, font_size=font_size, opacity=0.7)
        self._place_label(lab, arrow, mid)
        # (Manim also runs updaters on an animation's internal copies: act on the real label only)
        lab.add_updater(lambda m: self._place_label(m, arrow, mid) if m is lab else None)
        self._labels.append(lab)
        return lab

    # -- public API
    @_shown
    def show_matrix(self, A, name: str = "A", corner=UL):
        """Matrix panel in a top corner; column 1 green (where î lands), column 2 red (where ĵ lands)."""
        if self.matrix_panel is not None and self.matrix_panel in self.scene.mobjects:
            self.scene.play(FadeOut(self.matrix_panel), run_time=0.4)
        panel = self._register_panel(self._place_panel(self._matrix_panel(A, name, [I_COLOR, J_COLOR]), corner))
        self.scene.play(FadeIn(panel), run_time=0.8)
        self.matrix_panel = panel
        return panel

    @_shown
    def panel(self, mob, corner=UR):
        """Any extra numbers or text on a dark panel in a top corner (e.g. a worked calculation).
        Panels in the same corner stack downwards. FadeOut(mob) removes the panel with its box."""
        p = self._make_panel(mob, corner)
        self.scene.play(FadeIn(p), run_time=0.6)
        return p

    @_shown
    def add_vector(self, coords, label: str | None = None, color=YELLOW, show_coords: bool | None = None):
        """A tracked vector from the origin with a label that follows its tip and shows its CURRENT
        coordinates, e.g. 'v = [3, -1]' (updated after each transformation). An unnamed vector gets
        no label unless show_coords=True (many labelled arrows would bury the picture)."""
        a = self._arrow([0, 0], coords, color, 5)
        anims = [GrowArrow(a)]
        label = str(label).strip() if label is not None else ""
        if show_coords is None:
            show_coords = bool(label)
        if "=" in label:          # the label already states the coordinates: do not repeat them
            show_coords = False
        if label or show_coords:
            def text():
                body = _vec_text(self._arrows[id(a)][2]) if show_coords else ""
                return f"{label} = {body}" if (label and body) else (label or body)
            lab = boxed(text(), color=color, font_size=28, opacity=0.75)
            state = {"text": text()}

            def upd(m):
                if m is not lab:   # an animation's copy: leave it alone
                    return
                t = text()
                if t != state["text"]:
                    state["text"] = t
                    m.become(boxed(t, color=color, font_size=28, opacity=0.75))
                    m._kit_hidden, m._kit_shown = False, None
                self._place_label(m, a)
            self._place_label(lab, a)
            lab.add_updater(upd)
            self._labels.append(lab)
            a._kit_label = lab
            anims.append(FadeIn(lab))
        self.scene.play(*anims, run_time=1.0)
        return a

    @_shown
    def show_combination(self, arrow, run_time: float = 2.0):
        """Show a vector as x·î + y·ĵ: a green piece along î, then a red piece along ĵ, tip to tail.
        The pieces are tracked, so after a transformation they show x·(new î) + y·(new ĵ)."""
        x, y = np.linalg.solve(self.M, self.coords_of(arrow))
        e1, e2 = self.M[:, 0], self.M[:, 1]
        a1 = self._arrow([0, 0], x * e1, I_COLOR, 4)
        a2 = self._arrow(x * e1, x * e1 + y * e2, J_COLOR, 4)
        l1 = self._follow_label(a1, f"{fmt(x)}·î", I_COLOR, 26, mid=True)
        l2 = self._follow_label(a2, f"{fmt(y)}·ĵ", J_COLOR, 26, mid=True)
        self.scene.play(GrowArrow(a1), FadeIn(l1), run_time=run_time / 2)
        self.scene.play(GrowArrow(a2), FadeIn(l2), run_time=run_time / 2)
        self.combination = (a1, a2, l1, l2)
        return a1, a2

    @_shown
    def add_unit_square(self):
        """Translucent yellow square spanned by î and ĵ: its (signed) area becomes the determinant."""
        e1, e2 = self.M[:, 0], self.M[:, 1]
        sq = Polygon(self.c2p([0, 0]), self.c2p(e1), self.c2p(e1 + e2), self.c2p(e2), color=AREA_COLOR,
                     stroke_width=2.5, fill_color=AREA_COLOR, fill_opacity=0.32)
        self._shapes.add(sq)
        self.square = sq
        self.scene.play(FadeIn(sq), run_time=0.8)
        return sq

    @_shown
    def add_outline(self, points, color=YELLOW, fill: float = 0.0, closed: bool = True, width: float = 3.0):
        """A shape given by points in ORIGINAL coordinates (e.g. the unit circle). It is drawn where those
        points are now and moves with the plane: a circle becomes an ellipse."""
        pts = [self.c2p(self.M @ np.asarray(q, float)[:2]) for q in points]
        shape = VMobject(stroke_color=color, stroke_width=width, fill_color=color, fill_opacity=fill)
        shape.set_points_as_corners(pts + ([pts[0]] if closed else []))
        self._shapes.add(shape)
        self.scene.play(Create(shape), run_time=1.0)
        return shape

    def current_det(self) -> float:
        """Signed area of the (moving) unit square = determinant of everything applied so far."""
        if self.square is None:
            return float(np.linalg.det(self.M))
        v = self.square.get_vertices()
        a, b = (v[1] - v[0]) / self.unit, (v[3] - v[0]) / self.unit
        return float(a[0] * b[1] - a[1] * b[0])

    @_shown
    def show_determinant(self, corner=UR):
        """Live 'det = …' readout that updates DURING transformations (red when orientation flips)."""
        if self.square is None:
            self.add_unit_square()

        def make():
            d = self.current_det()
            color = AREA_COLOR if d > 1e-3 else ("#FC6255" if d < -1e-3 else WHITE)
            note = "  (orientation flipped)" if d < -1e-3 else ("  (squished flat)" if abs(d) <= 1e-3 else "")
            return self._place_panel(boxed(f"det = {fmt(round(d, 2))}{note}", color=color, font_size=30,
                                           opacity=0.85), corner, ignore=holder)

        holder = []
        lab = make()
        holder.append(lab)
        state = {"v": round(self.current_det(), 2)}

        def upd(m):
            if m is not lab:   # an animation's copy: leave it alone
                return
            v = round(self.current_det(), 2)
            if v != state["v"]:
                state["v"] = v
                m.become(make())
        lab.add_updater(upd)
        self._register_panel(lab)
        self.scene.play(FadeIn(lab), run_time=0.6)
        self.det_readout = lab
        return lab

    @_shown
    def transform_space(self, A, run_time: float = 3.0):
        """Apply the 2x2 matrix to the whole plane: grid, square, vectors and points all move together;
        î and ĵ end exactly on A's columns. Temporary labels are cleared first."""
        self.clear_annotations()
        self.scene.play(_SpaceMorph(self, A, run_time=run_time))
        self.scene.wait(0.6)

    @_shown
    def label_landings(self):
        """Labels at the tips of î and ĵ saying where they landed (= the columns of the matrix)."""
        labs = []
        for arrow, name, color in ((self.i_hat, "î", I_COLOR), (self.j_hat, "ĵ", J_COLOR)):
            end = self._arrows[id(arrow)][2]
            lab = boxed(f"{name} lands on {_vec_text(end)}", color=color, font_size=26)
            lab.move_to(self._label_spot(arrow, lab))
            self._labels.append(lab)
            labs.append(lab)
        if len(labs) == 2 and self._overlaps(labs[0].get_center(), labs[0].width / 2, labs[0].height / 2, labs[1]):
            labs[1].shift(UP * (labs[1].height + 0.1))
        self.scene.play(*(FadeIn(m) for m in labs), run_time=0.8)
        self._note(*labs)
        return labs

    @_shown
    def show_eigenvectors(self, A, count: int = 4):
        """Dashed span lines along A's real eigenvectors, each with a family of vectors on it.
        After transform_space(A) those vectors are still on their line, only stretched by λ.
        Returns [(λ, unit eigenvector)]; an empty list means no real eigenvectors (e.g. a rotation)."""
        A = np.asarray(A, float)
        vals, vecs = np.linalg.eig(A)
        pairs, seen = [], []
        for lam, v in zip(vals, vecs.T):
            if abs(np.imag(lam)) > 1e-9:
                continue
            v = np.real(v)
            v = v / np.linalg.norm(v)
            if any(abs(abs(v @ w) - 1) < 1e-6 for w in seen):
                continue
            seen.append(v)
            pairs.append((float(np.real(lam)), v))
        self.eigen, anims = [], []
        for k, (lam, v) in enumerate(pairs):
            color = EIGEN_COLORS[k % len(EIGEN_COLORS)]
            L = self._screen_reach()
            line = _dashed(self.c2p(-L * v), self.c2p(L * v), 0.15, stroke_color=color, stroke_width=2.5)
            self._add_static(line)      # static: an eigen line maps onto itself
            anims.append(Create(line))
            reach = float(np.clip(2.2 / max(abs(lam), 0.4), 1.0, 2.5))
            arrows = []
            for s in np.linspace(-reach, reach, count + 1):
                if abs(s) < 1e-6:
                    continue
                a = self._arrow([0, 0], s * v, color, 4)
                arrows.append(a)
                anims.append(GrowArrow(a))
            self.eigen.append((lam, v, line, arrows))
        if anims:
            self.scene.play(*anims, run_time=1.5)
        return [(lam, v) for lam, v, _, _ in self.eigen]

    @_shown
    def label_eigen_stretch(self):
        """After the transformation: 'stretched by λ' labels next to each eigen line."""
        labs = []
        for k, (lam, v, line, arrows) in enumerate(getattr(self, "eigen", [])):
            word = "stretched" if abs(lam) >= 1 else "squished"
            text = f"{word} by {fmt(lam)}" if lam >= 0 else f"flipped and scaled by {fmt(lam)}"
            lab = boxed(text, color=EIGEN_COLORS[k % len(EIGEN_COLORS)], font_size=26)
            d = v if v[1] >= 0 else -v                      # prefer the upper half of the line
            d3 = np.array([d[0], d[1], 0.0])
            n3 = np.array([-d[1], d[0], 0.0])
            # beside the line (just clear of it), at several distances from the origin, both sides, both halves
            off = 0.15 + abs(n3[0]) * lab.width / 2 + abs(n3[1]) * lab.height / 2
            cands = [self.origin + s * d3 * r + side * n3 * off
                     for r in (2.2, 3.0, 1.5, 3.8) for s in (1, -1) for side in (1, -1)]
            lab.move_to(self._free_spot(lab, cands, extra=labs))
            labs.append(lab)
            self._labels.append(lab)
        if labs:
            self.scene.play(*(FadeIn(m) for m in labs), run_time=0.8)
            self._note(*labs)
        return labs

    @_shown
    def show_off_span(self, coords, color="#FF86C8"):
        """A vector together with its own static dashed span line: under a transformation it is not an
        eigenvector of, it visibly gets knocked off that line."""
        v = np.asarray(coords, float)
        u = v / np.linalg.norm(v)
        L = self._screen_reach()
        line = _dashed(self.c2p(-L * u), self.c2p(L * u), 0.12, stroke_color=color, stroke_width=2,
                       stroke_opacity=0.7)
        self._add_static(line)
        a = self._arrow([0, 0], v, color, 5)
        self.scene.play(Create(line), GrowArrow(a), run_time=1.2)
        return a

    @_shown
    def add_points(self, coords, color=YELLOW, radius: float = 0.06):
        """Many vectors shown as points (their tips); they move with the plane."""
        dots = VGroup()
        for c in coords:
            d = Dot(self.c2p(c), radius=radius, color=color)
            self._anchors[id(d)] = (d, np.asarray(c, float))
            dots.add(d)
        self.layer.add(dots)
        self.scene.play(FadeIn(dots, lag_ratio=0.02), run_time=1.0)
        return dots

    @_shown
    def compose(self, steps, verify: bool = True, before=None):
        """Apply several transformations in order, e.g. compose([(R, "R"), (S, "S")]) applies R then S.
        Shows the formula S·R with the matrix being applied highlighted, then (verify=True) undoes
        everything and applies the product matrix in one move, landing in the same place."""
        names = [n for _, n in steps][::-1]   # written right-to-left: the first applied is rightmost
        parts = [Text(n, font_size=40) for n in names]
        formula = VGroup(*parts).arrange(RIGHT, buff=0.2)
        bg = SurroundingRectangle(formula, color=BLACK, fill_color=BLACK, fill_opacity=0.85, stroke_width=0, buff=0.2)
        box = VGroup(bg, formula)
        box.move_to([0, _top_limit(self.scene) - box.height / 2, 0])
        self._register_panel(box)
        self.scene.play(FadeIn(box))
        start_M = self.M.copy()
        for k, (A, _) in enumerate(steps):
            if before is not None:
                before(k)          # e.g. show the caption of this move
            part = parts[len(parts) - 1 - k]
            hint = boxed("applying this one", font_size=22, color=YELLOW).next_to(box, DOWN, buff=0.08)
            hint.set_x(part.get_x())
            self.scene.play(part.animate.set_color(YELLOW), FadeIn(hint), run_time=0.5)
            self.transform_space(A)
            self.scene.play(part.animate.set_color(WHITE), FadeOut(hint), run_time=0.4)
        product = np.eye(2)
        for A, _ in steps:
            product = np.asarray(A, float) @ product
        if verify:
            if before is not None:
                before(len(steps))
            self.reset_space(target=start_M)
            self.show_matrix(product, name="".join(names), corner=UR)
            self.transform_space(product)
        self.scene.play(FadeOut(box))
        return product

    @_shown
    def reset_space(self, target=None, run_time: float = 2.0):
        """Animate back to the original grid (or to an earlier state `target`)."""
        target = np.eye(2) if target is None else np.asarray(target, float)
        if abs(np.linalg.det(self.M)) < 1e-9:
            raise ValueError("space was squished flat (det = 0), so it cannot be undone: build a new Plane2D")
        self.transform_space(target @ np.linalg.inv(self.M), run_time=run_time)


# ============================================================================= 3-D
class Space3D(_Stage):
    """3-D space as a lattice of lines with î (green), ĵ (red), k̂ (blue). Use in a ThreeDScene.
    Every method animates by itself; panels and readouts are pinned to the screen."""

    dim = 3
    redraw_arrows = False   # 3-D arrows move as shapes during motion and are rebuilt at the end

    def __init__(self, scene: ThreeDScene, matrices=(), vectors=(), extent: int = 2, unit: float | None = None,
                 phi: float = 68, theta: float = -42, show_basis: bool = True, animate: bool = True):
        if not isinstance(scene, ThreeDScene):
            raise TypeError("Space3D needs the scene class to subclass ThreeDScene")
        self._init_common(scene)
        self.matrices = [np.asarray(A, float) for A in matrices]
        self.vectors = [np.asarray(v, float) for v in vectors]
        self.origin = np.zeros(3)
        pts = [np.eye(3)[i] for i in range(3)] + [np.asarray(v, float) for v in vectors]
        for A in matrices:
            A = np.asarray(A, float)
            pts += [A[:, i] for i in range(3)] + [A @ np.asarray(v, float) for v in vectors]
            pts += [A @ np.array(c, float) for c in [(1, 1, 1), (extent, extent, extent)]]
        reach = max(float(np.max(np.abs(np.array(pts)))), float(extent))
        self.unit = float(unit) if unit else float(np.clip(2.6 / reach, 0.35, 1.1))
        scene.set_camera_orientation(phi=phi * DEGREES, theta=theta * DEGREES)
        n = extent
        ghost = VGroup()
        for axis in np.eye(3):
            ghost.add(Line(self.c2p(-axis * (n + 1.5)), self.c2p(axis * (n + 1.5)), stroke_color=GHOST_COLOR,
                           stroke_width=1.5, stroke_opacity=0.6))
        self.background = ghost
        lattice = VGroup()
        for a in range(-n, n + 1):
            for b in range(-n, n + 1):
                for axis in range(3):
                    p, q = np.zeros(3), np.zeros(3)
                    o1, o2 = [k for k in range(3) if k != axis]
                    p[axis], q[axis] = -n, n
                    p[o1] = q[o1] = a
                    p[o2] = q[o2] = b
                    is_axis = a == 0 and b == 0
                    lattice.add(Line(self.c2p(p), self.c2p(q), stroke_color=WHITE if is_axis else GRID_COLOR,
                                     stroke_width=2.4 if is_axis else 1.2, stroke_opacity=1 if is_axis else 0.45))
        self.grid = lattice
        self._shapes.add(lattice)
        self._add_to_back_of_scene()
        if animate:
            scene.play(Create(lattice, lag_ratio=0.005), run_time=2)
        self.axis_labels = VGroup()
        for axis, name in zip(np.eye(3), "xyz"):
            lab = Text(name, font_size=30, color=GHOST_COLOR).move_to(self.c2p(axis * (n + 1.8)))
            scene.add_fixed_orientation_mobjects(lab)
            self.axis_labels.add(lab)
        if show_basis:
            self.i_hat = self._arrow(np.zeros(3), [1, 0, 0], I_COLOR)
            self.j_hat = self._arrow(np.zeros(3), [0, 1, 0], J_COLOR)
            self.k_hat = self._arrow(np.zeros(3), [0, 0, 1], K_COLOR)
            self.basis = [self.i_hat, self.j_hat, self.k_hat]
            labs = [self._follow_label(a, name, color) for a, name, color in
                    ((self.i_hat, "î", I_COLOR), (self.j_hat, "ĵ", J_COLOR), (self.k_hat, "k̂", K_COLOR))]
            scene.play(*(FadeIn(a) for a in self.basis), *(FadeIn(m) for m in labs), run_time=1.0)

    def c2p(self, coords) -> np.ndarray:
        return self.origin + self.unit * np.asarray(coords, float).ravel()[:3]

    def _tip(self, arrow) -> np.ndarray:
        """Actual tip position from the arrow's points (works while the arrow is being moved as a shape)."""
        pts = np.vstack([m.points for m in arrow.get_family() if len(m.points)])
        return pts[np.argmax(np.linalg.norm(pts - self.origin, axis=1))]

    def _follow_label(self, arrow, text, color, font_size=34):
        lab = Text(text, font_size=font_size, color=color)

        def upd(m):
            if m is not lab:   # an animation's copy: leave it alone
                return
            tip = self._tip(arrow)
            n = np.linalg.norm(tip - self.origin)
            m.move_to(tip + (0.32 * (tip - self.origin) / n if n > 1e-6 else np.array([0, 0, 0.3])))
        upd(lab)
        lab.add_updater(upd)
        self.scene.add_fixed_orientation_mobjects(lab)
        self.scene.remove(lab)
        self._labels.append(lab)
        return lab

    @staticmethod
    def _light_arrow(p0, p1, color, width: float = 5, tip_len: float = 0.22, tip_radius: float = 0.075,
                     sides: int = 8) -> VGroup:
        """A cheap 3-D arrow: a stroke line plus a small cone made of a few shaded triangles.
        (Manim's Arrow3D is built from hundreds of surface patches and is far too slow to render.)"""
        d = p1 - p0
        length = float(np.linalg.norm(d))
        u = d / length
        tip_len = min(tip_len, 0.4 * length)
        base = p1 - u * tip_len
        helper = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(u, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        ring = [base + tip_radius * (np.cos(a) * e1 + np.sin(a) * e2)
                for a in np.linspace(0, 2 * np.pi, sides, endpoint=False)]
        cone = VGroup(*[Polygon(ring[k], ring[(k + 1) % sides], p1, stroke_width=0, fill_color=color, fill_opacity=1,
                                shade_in_3d=True) for k in range(sides)])
        shaft = Line(p0, base, stroke_color=color, stroke_width=width)
        return VGroup(shaft, cone)

    def _arrow(self, start, end, color, thickness=0.02):
        a = self._light_arrow(self.c2p(start), self.c2p(end), color)
        a._kit_color = color
        self._track_arrow(a, start, end)
        return a

    def _draw_arrow(self, arrow, s, e):
        p0, p1 = self.c2p(s), self.c2p(e)
        if np.linalg.norm(p1 - p0) < 0.03:
            arrow.set_opacity(0)
            return
        arrow.become(self._light_arrow(p0, p1, arrow[0].get_color()))   # keep a recolored arrow's color

    @_shown
    def show_matrix(self, A, name: str = "A", corner=UL):
        """Matrix panel pinned to a top corner; columns colored like î, ĵ, k̂."""
        if self.matrix_panel is not None and self.matrix_panel in self.scene.mobjects:
            self.scene.play(FadeOut(self.matrix_panel), run_time=0.4)
        panel = self._register_panel(self._place_panel(self._matrix_panel(A, name, [I_COLOR, J_COLOR, K_COLOR]),
                                                       corner))
        hud(self.scene, panel)
        self.scene.play(FadeIn(panel), run_time=0.8)
        self.matrix_panel = panel
        return panel

    @_shown
    def panel(self, mob, corner=UR):
        """Worked numbers or text on a dark panel pinned to a top corner; FadeOut(mob) removes it."""
        p = self._make_panel(mob, corner)
        hud(self.scene, p)
        self.scene.play(FadeIn(p), run_time=0.6)
        return p

    @_shown
    def add_vector(self, coords, label: str | None = None, color=YELLOW):
        a = self._arrow(np.zeros(3), coords, color, 0.02)
        anims = [FadeIn(a)]
        if label:
            anims.append(FadeIn(self._follow_label(a, label, color, 30)))
        self.scene.play(*anims, run_time=1.0)
        return a

    @_shown
    def add_unit_cube(self):
        """Translucent yellow cube spanned by î, ĵ, k̂: its (signed) volume becomes the determinant."""
        faces = VGroup()
        e = np.eye(3)
        for axis in range(3):
            o1, o2 = [k for k in range(3) if k != axis]
            for off in (0, 1):
                base = e[axis] * off
                corners = [base, base + e[o1], base + e[o1] + e[o2], base + e[o2]]
                faces.add(Polygon(*[self.c2p(self.M @ c) for c in corners], color=AREA_COLOR, stroke_width=1.5,
                                  fill_color=AREA_COLOR, fill_opacity=0.28))
        self._shapes.add(faces)
        self.square = faces
        self.scene.play(FadeIn(faces), run_time=0.8)
        return faces

    def current_det(self) -> float:
        return float(np.linalg.det(self.M))

    def _det_panel(self, corner, replaces=()):
        d = self.current_det()
        note = "  (orientation flipped)" if d < -1e-3 else ("  (squished flat)" if abs(d) <= 1e-3 else "")
        lab = boxed(f"det = {fmt(round(d, 2))}{note}", color=AREA_COLOR if d > 1e-3 else "#FC6255",
                    font_size=28, opacity=0.85)
        return self._place_panel(lab, corner, ignore=replaces)

    @_shown
    def show_determinant(self, corner=UR):
        """'det = …' readout pinned to a top corner (updated after each transformation)."""
        if self.square is None:
            self.add_unit_cube()
        lab = self._register_panel(self._det_panel(corner))
        hud(self.scene, lab)
        self.scene.play(FadeIn(lab), run_time=0.6)
        self.det_readout = (lab, corner)
        return lab

    @_shown
    def transform_space(self, A, run_time: float = 4.0, orbit: bool = True):
        """Apply the 3x3 matrix to all of space. The camera turns slowly meanwhile so depth stays visible."""
        self.clear_annotations()
        if orbit:
            self.scene.begin_ambient_camera_rotation(rate=0.12)
        self.scene.play(_SpaceMorph(self, A, run_time=run_time))
        if orbit:
            self.scene.stop_ambient_camera_rotation()
        ro = getattr(self, "det_readout", None)
        if ro is not None:
            lab, corner = ro
            new = self._register_panel(self._det_panel(corner, replaces=(lab,)))
            hud(self.scene, new)
            self.scene.play(FadeOut(lab), FadeIn(new), run_time=0.5)
            self.det_readout = (new, corner)
        self.scene.wait(0.5)

    @_shown
    def label_landings(self):
        """Where î, ĵ, k̂ landed (= the columns of the matrix), listed on screen."""
        labs = []
        for k, (arrow, name, color) in enumerate(((self.i_hat, "î", I_COLOR), (self.j_hat, "ĵ", J_COLOR),
                                                   (self.k_hat, "k̂", K_COLOR))):
            end = self._arrows[id(arrow)][2]
            lab = boxed(f"{name} lands on {_vec_text(end)}", color=color, font_size=26)
            lab.move_to([-6.9 + lab.width / 2, -1.2 - 0.5 * k, 0])
            hud(self.scene, lab)
            labs.append(lab)
        self.scene.play(*(FadeIn(m) for m in labs), run_time=0.8)
        self._note(*labs)
        return labs

    @_shown
    def orbit(self, degrees: float = 90, run_time: float = 5.0):
        """Turn the camera around the scene to show the 3-D shape."""
        self.scene.move_camera(theta=self.scene.camera.get_theta() + degrees * DEGREES, run_time=run_time)

    @_shown
    def reset_space(self, run_time: float = 3.0):
        """Animate back to the original lattice."""
        if abs(np.linalg.det(self.M)) < 1e-9:
            raise ValueError("space was squished flat (det = 0), so it cannot be undone: build a new Space3D")
        self.transform_space(np.linalg.inv(self.M), run_time=run_time)

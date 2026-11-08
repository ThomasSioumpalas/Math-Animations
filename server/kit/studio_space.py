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
    BLACK, DEGREES, DOWN, LEFT, RIGHT, UL, UP, UR, WHITE, YELLOW,
    Animation, Arrow, Create, DashedLine, Dot, FadeIn, FadeOut, GrowArrow, Line,
    Mobject, Polygon, Scene, SurroundingRectangle, Text, ThreeDScene, VGroup,
)

from studio_kit import NumberMatrix, boxed, fmt, hud

I_COLOR, J_COLOR, K_COLOR = "#83C167", "#FC6255", "#58C4DD"   # î green, ĵ red, k̂ blue
GRID_COLOR = "#29ABCA"
GHOST_COLOR = "#888888"
AREA_COLOR = YELLOW
EIGEN_COLORS = ["#5CD0B3", "#C55FDB"]                          # teal, purple
SAFE_X, SAFE_TOP, SAFE_BOTTOM = 6.9, 3.0, -2.62                  # clear of the title and caption zones


# ----------------------------------------------------------------------------- helpers
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

    def _place_panel(self, panel, corner):
        left = float(np.asarray(corner, float)[0]) < 0
        x = (-SAFE_X - 0.1 + panel.width / 2) if left else (SAFE_X + 0.1 - panel.width / 2)
        panel.move_to([x, _top_limit(self.scene) - panel.height / 2, 0])
        return panel

    def _register_panel(self, panel):
        self._panels.append(panel)
        return panel


# ============================================================================= 2-D
class Plane2D(_Stage):
    """The 2-D plane as a transformable grid with î (green) and ĵ (red). Use in a normal Scene.
    Every method animates by itself."""

    dim = 2
    redraw_arrows = True

    def __init__(self, scene: Scene, matrices=(), vectors=(), unit: float | None = None,
                 show_basis: bool = True, basis_labels: bool = True, animate: bool = True):
        self._init_common(scene)
        self.origin = np.array([0.0, 0.18, 0.0])
        self.unit = float(unit) if unit else self._auto_unit(matrices, vectors)
        u = self.unit
        n_bg = int(np.ceil(7.8 / u))
        bg = VGroup()
        for k in range(-n_bg, n_bg + 1):
            axis = k == 0
            style = dict(stroke_color=GHOST_COLOR, stroke_width=1.6 if axis else 1,
                         stroke_opacity=0.75 if axis else 0.4)
            bg.add(Line(self.c2p([k, -n_bg]), self.c2p([k, n_bg]), **style))
            bg.add(Line(self.c2p([-n_bg, k]), self.c2p([n_bg, k]), **style))
        self.background = bg
        R = int(np.ceil(8.0 / u)) * 2  # far past the screen, so edges never show after shrinking
        lines = VGroup()
        for k in range(-R, R + 1):
            if k == 0:
                continue
            style = dict(stroke_color=GRID_COLOR, stroke_width=1.8, stroke_opacity=0.95)
            lines.add(Line(self.c2p([k, -R]), self.c2p([k, R]), **style))
            lines.add(Line(self.c2p([-R, k]), self.c2p([R, k]), **style))
        axes = VGroup(Line(self.c2p([-R, 0]), self.c2p([R, 0]), stroke_color=WHITE, stroke_width=2.4),
                      Line(self.c2p([0, -R]), self.c2p([0, R]), stroke_color=WHITE, stroke_width=2.4))
        self.grid = VGroup(lines, axes)
        self._shapes.add(self.grid)
        scene.add(self.background, self.layer)  # the layer is the single persistent group
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
        new = Arrow(p0, p1, buff=0, color=arrow._kit_color, stroke_width=arrow._kit_width,
                    max_tip_length_to_length_ratio=0.3, max_stroke_width_to_length_ratio=8)
        if hidden:
            new.set_opacity(0)
        arrow.become(new)

    def _label_spot(self, arrow, label, mid: bool = False):
        """Where a label goes: next to the arrow's tip (or beside its middle), inside the visible area,
        and not on top of a panel or another label (first free candidate wins)."""
        tip, start = arrow.get_end(), arrow.get_start()
        d = tip - start
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else RIGHT
        perp = np.array([-d[1], d[0], 0.0])
        off = 0.16 + 0.5 * max(label.width, label.height)
        w, h = label.width / 2, label.height / 2
        top = _top_limit(self.scene)
        if mid:
            base = (tip + start) / 2
            cands = (base + perp * off, base - perp * off, base + perp * off * 1.8, base - perp * off * 1.8)
        else:
            cands = (tip + d * off, tip + perp * off, tip - perp * off, tip + d * off + perp * off * 1.2,
                     tip + d * off - perp * off * 1.2, tip - d * off * 2.2)
        others = [m for m in self._panels + self._labels if m is not label and m in self.scene.mobjects]
        # also any text the scene put on screen itself (titles, captions, notes)
        others += [m for m in self.scene.mobjects
                   if isinstance(m, Text) and m is not label and m not in self._labels and m.width > 1e-3]
        first = None
        for cand in cands:
            spot = cand.copy()
            spot[0] = np.clip(spot[0], -7.0 + w, 7.0 - w)
            spot[1] = np.clip(spot[1], SAFE_BOTTOM + h, top - h)
            first = spot if first is None else first
            if not any(self._overlaps(spot, w, h, p) for p in others):
                return spot
        return first

    @staticmethod
    def _overlaps(c, w, h, panel):
        return (abs(c[0] - panel.get_x()) < w + panel.width / 2 + 0.05 and
                abs(c[1] - panel.get_y()) < h + panel.height / 2 + 0.05)

    def _follow_label(self, arrow, text, color, font_size=30, mid: bool = False):
        lab = boxed(text, color=color, font_size=font_size, opacity=0.7)
        lab.move_to(self._label_spot(arrow, lab, mid))
        lab.add_updater(lambda m: m.move_to(self._label_spot(arrow, m, mid)))
        self._labels.append(lab)
        return lab

    # -- public API
    def show_matrix(self, A, name: str = "A", corner=UL):
        """Matrix panel in a top corner; column 1 green (where î lands), column 2 red (where ĵ lands)."""
        if self.matrix_panel is not None and self.matrix_panel in self.scene.mobjects:
            self.scene.play(FadeOut(self.matrix_panel), run_time=0.4)
        panel = self._register_panel(self._place_panel(self._matrix_panel(A, name, [I_COLOR, J_COLOR]), corner))
        self.scene.play(FadeIn(panel), run_time=0.8)
        self.matrix_panel = panel
        return panel

    def panel(self, mob, corner=UR):
        """Any extra numbers or text on a dark panel in a top corner (e.g. a worked calculation)."""
        bg = SurroundingRectangle(mob, color=BLACK, fill_color=BLACK, fill_opacity=0.85, stroke_width=0, buff=0.18)
        p = self._register_panel(self._place_panel(VGroup(bg, mob), corner))
        self.scene.play(FadeIn(p), run_time=0.6)
        return p

    def add_vector(self, coords, label: str | None = None, color=YELLOW, show_coords: bool = True):
        """A tracked vector from the origin with a label that follows its tip and shows its CURRENT
        coordinates, e.g. 'v = [3, -1]' (updated after each transformation)."""
        a = self._arrow([0, 0], coords, color, 5)
        anims = [GrowArrow(a)]
        if label is not None or show_coords:
            def text():
                body = _vec_text(self._arrows[id(a)][2]) if show_coords else ""
                return f"{label} = {body}" if (label and body) else (label or body)
            lab = boxed(text(), color=color, font_size=28, opacity=0.75)
            state = {"text": text()}

            def upd(m):
                t = text()
                if t != state["text"]:
                    state["text"] = t
                    m.become(boxed(t, color=color, font_size=28, opacity=0.75))
                m.move_to(self._label_spot(a, m))
            lab.move_to(self._label_spot(a, lab))
            lab.add_updater(upd)
            self._labels.append(lab)
            a._kit_label = lab
            anims.append(FadeIn(lab))
        self.scene.play(*anims, run_time=1.0)
        return a

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

    def add_unit_square(self):
        """Translucent yellow square spanned by î and ĵ: its (signed) area becomes the determinant."""
        e1, e2 = self.M[:, 0], self.M[:, 1]
        sq = Polygon(self.c2p([0, 0]), self.c2p(e1), self.c2p(e1 + e2), self.c2p(e2), color=AREA_COLOR,
                     stroke_width=2.5, fill_color=AREA_COLOR, fill_opacity=0.32)
        self._shapes.add(sq)
        self.square = sq
        self.scene.play(FadeIn(sq), run_time=0.8)
        return sq

    def current_det(self) -> float:
        """Signed area of the (moving) unit square = determinant of everything applied so far."""
        if self.square is None:
            return float(np.linalg.det(self.M))
        v = self.square.get_vertices()
        a, b = (v[1] - v[0]) / self.unit, (v[3] - v[0]) / self.unit
        return float(a[0] * b[1] - a[1] * b[0])

    def show_determinant(self, corner=UR):
        """Live 'det = …' readout that updates DURING transformations (red when orientation flips)."""
        if self.square is None:
            self.add_unit_square()

        def make():
            d = self.current_det()
            color = AREA_COLOR if d > 1e-3 else ("#FC6255" if d < -1e-3 else WHITE)
            note = "  (orientation flipped)" if d < -1e-3 else ("  (squished flat)" if abs(d) <= 1e-3 else "")
            return self._place_panel(boxed(f"det = {fmt(round(d, 2))}{note}", color=color, font_size=30,
                                           opacity=0.85), corner)

        lab = make()
        state = {"v": round(self.current_det(), 2)}

        def upd(m):
            v = round(self.current_det(), 2)
            if v != state["v"]:
                state["v"] = v
                m.become(make())
        lab.add_updater(upd)
        self._register_panel(lab)
        self.scene.play(FadeIn(lab), run_time=0.6)
        self.det_readout = lab
        return lab

    def transform_space(self, A, run_time: float = 3.0):
        """Apply the 2x2 matrix to the whole plane: grid, square, vectors and points all move together;
        î and ĵ end exactly on A's columns. Temporary labels are cleared first."""
        self.clear_annotations()
        self.scene.play(_SpaceMorph(self, A, run_time=run_time))
        self.scene.wait(0.6)

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
            line = DashedLine(self.c2p(-30 * v), self.c2p(30 * v), color=color, stroke_width=2.5, dash_length=0.15)
            anims.append(Create(line))  # static: an eigen line maps onto itself
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

    def label_eigen_stretch(self):
        """After the transformation: 'stretched by λ' labels next to each eigen line."""
        labs = []
        for k, (lam, v, line, arrows) in enumerate(getattr(self, "eigen", [])):
            word = "stretched" if abs(lam) >= 1 else "squished"
            text = f"{word} by {fmt(lam)}" if lam >= 0 else f"flipped and scaled by {fmt(lam)}"
            lab = boxed(text, color=EIGEN_COLORS[k % len(EIGEN_COLORS)], font_size=26)
            direction = v if v[1] >= 0 else -v
            spot = self.c2p(direction * 2.0) + 0.45 * UP
            spot[0] = np.clip(spot[0], -6.9 + lab.width / 2, 6.9 - lab.width / 2)
            spot[1] = np.clip(spot[1], SAFE_BOTTOM + lab.height / 2, _top_limit(self.scene) - lab.height / 2)
            lab.move_to(spot)
            labs.append(lab)
        if labs:
            self.scene.play(*(FadeIn(m) for m in labs), run_time=0.8)
            self._note(*labs)
        return labs

    def show_off_span(self, coords, color="#FF86C8"):
        """A vector together with its own static dashed span line: under a transformation it is not an
        eigenvector of, it visibly gets knocked off that line."""
        v = np.asarray(coords, float)
        u = v / np.linalg.norm(v)
        line = DashedLine(self.c2p(-30 * u), self.c2p(30 * u), color=color, stroke_width=2, dash_length=0.12,
                          stroke_opacity=0.7)
        a = self._arrow([0, 0], v, color, 5)
        self.scene.play(Create(line), GrowArrow(a), run_time=1.2)
        return a

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

    def compose(self, steps, verify: bool = True):
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
            self.reset_space(target=start_M)
            self.show_matrix(product, name="".join(names), corner=UR)
            self.transform_space(product)
        self.scene.play(FadeOut(box))
        return product

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
        scene.add(self.background, self.layer)
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
            tip = self._tip(arrow)
            n = np.linalg.norm(tip - self.origin)
            m.move_to(tip + (0.32 * (tip - self.origin) / n if n > 1e-6 else np.array([0, 0, 0.3])))
        upd(lab)
        lab.add_updater(upd)
        self.scene.add_fixed_orientation_mobjects(lab)
        self.scene.remove(lab)
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
        arrow.become(self._light_arrow(p0, p1, arrow._kit_color))

    def show_matrix(self, A, name: str = "A", corner=UL):
        """Matrix panel pinned to a top corner; columns colored like î, ĵ, k̂."""
        panel = self._register_panel(self._place_panel(self._matrix_panel(A, name, [I_COLOR, J_COLOR, K_COLOR]),
                                                       corner))
        hud(self.scene, panel)
        self.scene.play(FadeIn(panel), run_time=0.8)
        self.matrix_panel = panel
        return panel

    def panel(self, mob, corner=UR):
        bg = SurroundingRectangle(mob, color=BLACK, fill_color=BLACK, fill_opacity=0.85, stroke_width=0, buff=0.18)
        p = self._register_panel(self._place_panel(VGroup(bg, mob), corner))
        hud(self.scene, p)
        self.scene.play(FadeIn(p), run_time=0.6)
        return p

    def add_vector(self, coords, label: str | None = None, color=YELLOW):
        a = self._arrow(np.zeros(3), coords, color, 0.02)
        anims = [FadeIn(a)]
        if label:
            anims.append(FadeIn(self._follow_label(a, label, color, 30)))
        self.scene.play(*anims, run_time=1.0)
        return a

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

    def _det_panel(self, corner):
        d = self.current_det()
        note = "  (orientation flipped)" if d < -1e-3 else ("  (squished flat)" if abs(d) <= 1e-3 else "")
        lab = boxed(f"det = {fmt(round(d, 2))}{note}", color=AREA_COLOR if d > 1e-3 else "#FC6255",
                    font_size=28, opacity=0.85)
        return self._place_panel(lab, corner)

    def show_determinant(self, corner=UR):
        """'det = …' readout pinned to a top corner (updated after each transformation)."""
        if self.square is None:
            self.add_unit_cube()
        lab = self._register_panel(self._det_panel(corner))
        hud(self.scene, lab)
        self.scene.play(FadeIn(lab), run_time=0.6)
        self.det_readout = (lab, corner)
        return lab

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
            new = self._register_panel(self._det_panel(corner))
            hud(self.scene, new)
            self.scene.play(FadeOut(lab), FadeIn(new), run_time=0.5)
            self.det_readout = (new, corner)
        self.scene.wait(0.5)

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

    def orbit(self, degrees: float = 90, run_time: float = 5.0):
        """Turn the camera around the scene to show the 3-D shape."""
        self.scene.move_camera(theta=self.scene.camera.get_theta() + degrees * DEGREES, run_time=run_time)

    def reset_space(self, run_time: float = 3.0):
        """Animate back to the original lattice."""
        if abs(np.linalg.det(self.M)) < 1e-9:
            raise ValueError("space was squished flat (det = 0), so it cannot be undone: build a new Space3D")
        self.transform_space(np.linalg.inv(self.M), run_time=run_time)

"""
Norms — visual walkthrough of Goodfellow et al., Deep Learning, §2.5

Run from your manimations folder:
    uv run manim -pql norms.py LpNorms           # L1, L2, L-inf on one vector, then general Lp
    uv run manim -pql norms.py UnitBalls         # shape of "all vectors with norm 1" as p changes
    uv run manim -pql norms.py NormRules         # the 3 rules: zero, triangle inequality, scaling
    uv run manim -pql norms.py SquaredVsL1       # why squared L2 is "flat near 0" and L1 isn't
    uv run manim -pql norms.py CountingIsNotNorm # L1, max norm, and why "L0" is not a norm
    uv run manim -pql norms.py Frobenius         # matrix norm = L2 of all entries
    uv run manim -pql norms.py DotProductAngle   # x·y = ||x|| ||y|| cos(theta)

Render everything:  uv run manim -pql -a norms.py
No LaTeX needed (captions use plain Text).
"""
from manim import *
import numpy as np

FS = 24


def box(text, corner=UL, fs=FS):
    t = Text(text, font_size=fs, line_spacing=0.8)
    t.add_background_rectangle(opacity=0.88, buff=0.18)
    return t.to_corner(corner).set_z_index(10)


def lp(v, p):
    v = np.abs(np.asarray(v, dtype=float))
    if p == np.inf:
        return v.max()
    return (v ** p).sum() ** (1 / p)


class Base(Scene):
    note = None

    def say(self, text, corner=UL, wait=2.5):
        new = box(text, corner)
        if self.note is None:
            self.play(FadeIn(new))
        else:
            self.play(FadeOut(self.note), FadeIn(new))
        self.note = new
        self.wait(wait)


# ---------------------------------------------------------------------------
# 1. L1, L2, L-infinity of the same vector
# ---------------------------------------------------------------------------
class LpNorms(Base):
    def construct(self):
        plane = NumberPlane(x_range=[-1, 5, 1], y_range=[-1, 5, 1],
                            x_length=7.2, y_length=7.2,
                            background_line_style={"stroke_opacity": 0.35}).to_edge(RIGHT, buff=0.4)
        self.add(plane)
        O, P, C = plane.c2p(0, 0), plane.c2p(3, 4), plane.c2p(3, 0)

        x_vec = Arrow(O, P, buff=0, color=YELLOW)
        x_lbl = Text("x = (3, 4)", font_size=26, color=YELLOW).next_to(P, UP, buff=0.1)
        self.say("A norm = the 'size' of a vector.\n"
                 "Example: x = (3, 4).\n"
                 "Different norms measure size\n"
                 "in different ways.")
        self.play(GrowArrow(x_vec), FadeIn(x_lbl))

        # L2
        self.say("L2 (Euclidean) norm, p = 2:\n"
                 "straight-line distance to x.\n"
                 "sqrt(3² + 4²) = sqrt(25) = 5\n"
                 "Written ||x||₂ or just ||x||.")
        legs = VGroup(DashedLine(O, C), DashedLine(C, P)).set_color(GREY_B)
        l2 = Line(O, P, color=WHITE, stroke_width=9)
        self.play(Create(legs), Create(l2))
        self.wait()

        # L1
        self.say("L1 norm, p = 1: add the absolute\n"
                 "values. Like walking a city grid\n"
                 "('taxicab' distance):\n"
                 "|3| + |4| = 7")
        l1 = VMobject(color=ORANGE, stroke_width=9).set_points_as_corners([O, C, P])
        self.play(FadeOut(l2), Create(l1), run_time=2)
        self.wait()

        # L-inf
        self.say("L∞ (max) norm: only the BIGGEST\n"
                 "coordinate counts.\n"
                 "max(|3|, |4|) = 4")
        linf = Line(C, P, color=RED, stroke_width=11)
        self.play(FadeOut(l1), Create(linf))
        self.wait()
        self.play(FadeOut(linf))

        # General p
        self.say("General Lp (eq. 2.30):\n"
                 "||x||p = (|x1|^p + |x2|^p)^(1/p)\n"
                 "Watch the value as p grows:", wait=1)
        p = ValueTracker(1)
        readout = always_redraw(lambda: box(
            f"p = {p.get_value():5.2f}\n||x||p = {lp([3, 4], p.get_value()):.3f}", DL, fs=30))
        self.add(readout)
        self.play(p.animate.set_value(2), run_time=3)
        self.wait()
        self.play(p.animate.set_value(20), run_time=5)
        self.say("p = 1 → 7,  p = 2 → 5,  p → ∞ → 4\n"
                 "Bigger p = the largest coordinate\n"
                 "dominates more and more.", wait=4)


# ---------------------------------------------------------------------------
# 2. Unit balls: all vectors with ||x||p = 1
# ---------------------------------------------------------------------------
class UnitBalls(Base):
    def construct(self):
        plane = NumberPlane(x_range=[-1.5, 1.5, 0.5], y_range=[-1.5, 1.5, 0.5],
                            x_length=7.5, y_length=7.5,
                            background_line_style={"stroke_opacity": 0.35}).to_edge(RIGHT, buff=0.5)
        self.add(plane)

        def ball(p, color, width=5):
            e = 2 / p
            return ParametricFunction(
                lambda t: plane.c2p(np.sign(np.cos(t)) * abs(np.cos(t)) ** e,
                                    np.sign(np.sin(t)) * abs(np.sin(t)) ** e),
                t_range=[0, TAU, TAU / 800], color=color, stroke_width=width,
                use_smoothing=False)

        self.say("Draw EVERY vector whose norm is\n"
                 "exactly 1. That shape is called\n"
                 "the 'unit ball' of the norm.")
        c2 = ball(2, WHITE)
        self.say("L2: all points at distance 1\n"
                 "→ a circle.", wait=1)
        self.play(Create(c2), run_time=2)
        self.wait()
        c1 = ball(1, ORANGE)
        self.say("L1: |x1| + |x2| = 1\n"
                 "→ a diamond. Its corners sit\n"
                 "ON the axes, where one coordinate\n"
                 "is exactly 0.", wait=1)
        self.play(Create(c1), run_time=2)
        self.wait(2)
        cinf = Square(side_length=plane.c2p(2, 0)[0] - plane.c2p(0, 0)[0],
                      color=RED, stroke_width=5).move_to(plane.c2p(0, 0))
        self.say("L∞: max(|x1|, |x2|) = 1\n→ a square.", wait=1)
        self.play(Create(cinf), run_time=2)
        self.wait()

        self.say("Now one shape morphing as p grows.\n"
                 "p = 1 diamond → p = 2 circle\n"
                 "→ p = ∞ square.", wait=1)
        p = ValueTracker(1)
        moving = always_redraw(lambda: ball(p.get_value(), YELLOW, 7))
        readout = always_redraw(lambda: box(f"p = {p.get_value():.2f}", DL, fs=34))
        self.play(FadeOut(c1), FadeOut(c2), FadeOut(cinf))
        self.add(moving, readout)
        self.play(p.animate.set_value(2), run_time=3)
        self.play(p.animate.set_value(30), run_time=5, rate_func=rate_functions.ease_in_quad)
        self.play(p.animate.set_value(1), run_time=4)
        self.say("Why this matters in ML: L1's pointy\n"
                 "corners on the axes are why L1\n"
                 "regularization pushes weights to\n"
                 "EXACTLY zero (sparse models).", wait=4)


# ---------------------------------------------------------------------------
# 3. The three rules every norm must obey
# ---------------------------------------------------------------------------
class NormRules(Base):
    def construct(self):
        plane = NumberPlane(background_line_style={"stroke_opacity": 0.35}).shift(1.5 * RIGHT + 1.5 * DOWN)
        self.add(plane)
        O = plane.c2p(0, 0)
        x = np.array([3.0, 1.0])

        self.say("A function f is a norm only if\n"
                 "it obeys 3 rules:\n"
                 "1) f(x) = 0  →  x = 0\n"
                 "2) f(x + y) ≤ f(x) + f(y)\n"
                 "3) f(αx) = |α| f(x)", wait=3.5)
        self.say("Rule 1: only the zero vector\n"
                 "has size 0. Any other vector,\n"
                 "however tiny, has size > 0.")

        # Rule 2: triangle inequality with a rotating y
        self.say("Rule 2, triangle inequality:\n"
                 "going x then y (yellow + teal)\n"
                 "is never shorter than going\n"
                 "straight to x + y (white).", wait=1)
        r = np.sqrt(5)
        th = ValueTracker(np.arctan2(2, 1) + 0.9)

        def y():
            return r * np.array([np.cos(th.get_value()), np.sin(th.get_value())])

        xa = Arrow(O, plane.c2p(*x), buff=0, color=YELLOW)
        ya = always_redraw(lambda: Arrow(plane.c2p(*x), plane.c2p(*(x + y())), buff=0, color=TEAL))
        sa = always_redraw(lambda: Arrow(O, plane.c2p(*(x + y())), buff=0, color=WHITE, stroke_width=7))
        read = always_redraw(lambda: box(
            f"||x + y||       = {np.linalg.norm(x + y()):.2f}\n"
            f"||x|| + ||y|| = {np.linalg.norm(x) + r:.2f}", DL, fs=28))
        self.add(xa, ya, sa, read)
        self.wait()
        self.play(th.animate.set_value(np.arctan2(1, 3)), run_time=5)
        self.say("When y points the same way as x,\n"
                 "the detour IS the straight path:\n"
                 "equality. Otherwise always ≤.", wait=3)
        self.play(FadeOut(ya), FadeOut(sa), FadeOut(read), FadeOut(xa))

        # Rule 3: scaling
        self.say("Rule 3: scaling x by α scales its\n"
                 "size by |α|. Double the vector,\n"
                 "double the norm.", wait=1)
        a = ValueTracker(1)
        va = always_redraw(lambda: Arrow(O, plane.c2p(*(a.get_value() * x)), buff=0, color=YELLOW))
        read2 = always_redraw(lambda: box(
            f"α = {a.get_value():.2f}\n"
            f"||αx|| = {abs(a.get_value()) * np.linalg.norm(x):.2f} = |α|·{np.linalg.norm(x):.2f}", DL, fs=28))
        self.add(va, read2)
        self.play(a.animate.set_value(1.8), run_time=2.5)
        self.play(a.animate.set_value(0.4), run_time=2.5)
        self.say("Negative α flips the direction\n"
                 "but the length uses |α|:\n"
                 "size can never be negative.", wait=1)
        self.play(a.animate.set_value(-1.3), run_time=2.5)
        self.wait(3)


# ---------------------------------------------------------------------------
# 4. Squared L2 vs L1 near zero
# ---------------------------------------------------------------------------
class SquaredVsL1(Base):
    def construct(self):
        ax = Axes(x_range=[-2, 2, 0.5], y_range=[0, 2.2, 0.5], x_length=7.5, y_length=5.2,
                  tips=False).to_edge(RIGHT, buff=0.4).shift(0.3 * DOWN)
        sq = ax.plot(lambda t: t ** 2, x_range=[-1.48, 1.48], color=BLUE, stroke_width=5)
        l1 = ax.plot(lambda t: abs(t), x_range=[-2, 2], color=ORANGE, stroke_width=5)
        sq_lbl = Text("x²  (squared L2)", font_size=24, color=BLUE).next_to(ax.c2p(1.45, 2.1), LEFT)
        l1_lbl = Text("|x|  (L1)", font_size=24, color=ORANGE).next_to(ax.c2p(-1.9, 1.9), RIGHT)

        self.say("Look at ONE coordinate x of a\n"
                 "vector, and how much it adds\n"
                 "to each norm.")
        self.play(Create(ax))
        self.play(Create(sq), FadeIn(sq_lbl))
        self.play(Create(l1), FadeIn(l1_lbl))
        self.wait()

        t = ValueTracker(1.4)
        d_sq = always_redraw(lambda: Dot(ax.c2p(t.get_value(), t.get_value() ** 2), color=BLUE, radius=0.09))
        d_l1 = always_redraw(lambda: Dot(ax.c2p(t.get_value(), abs(t.get_value())), color=ORANGE, radius=0.09))
        read = always_redraw(lambda: box(
            f"x   = {t.get_value():.3f}\n"
            f"x²  = {t.get_value() ** 2:.4f}\n"
            f"|x| = {abs(t.get_value()):.3f}", DL, fs=28))
        self.add(d_sq, d_l1, read)
        self.say("Slide x toward 0 and compare.", wait=0.5)
        self.play(t.animate.set_value(0.1), run_time=5)
        self.say("At x = 0.1:  x² = 0.01,  |x| = 0.1\n"
                 "Squared L2 is almost FLAT near 0:\n"
                 "small-but-nonzero entries barely\n"
                 "register, they look like zero.", wait=4)
        self.say("L1 grows at the same rate\n"
                 "everywhere: move ε away from 0,\n"
                 "L1 goes up by exactly ε.\n"
                 "So L1 tells 0 apart from 'small'.", wait=4)
        self.say("Why people still love squared L2:\n"
                 "xᵀx = x1² + x2² + ...\n"
                 "d/dxi (xᵀx) = 2·xi  ← only its own entry\n"
                 "d/dxi ||x|| = xi / ||x||  ← needs ALL of x\n"
                 "Simpler gradients = cheaper training.", wait=5)


# ---------------------------------------------------------------------------
# 5. L1, max norm, and why counting nonzeros ("L0") is NOT a norm
# ---------------------------------------------------------------------------
class CountingIsNotNorm(Base):
    V = np.array([2, 0, -3, 0, 1])

    def construct(self):
        base_y, unit, x0, gap = 0.6, 0.45, 0.8, 1.2
        a = ValueTracker(1)
        baseline = Line([x0 - 0.8, base_y, 0], [x0 + gap * 4 + 0.8, base_y, 0], color=GREY_B)

        def bars():
            g = VGroup()
            for i, val in enumerate(self.V * a.get_value()):
                h = abs(val) * unit
                cx = x0 + gap * i
                if h > 1e-6:
                    r = Rectangle(width=0.7, height=h, fill_opacity=0.8, stroke_width=2,
                                  color=GREEN if val > 0 else RED)
                    r.move_to([cx, base_y + np.sign(val) * h / 2, 0])
                    g.add(r)
                lbl = Text(f"{val:.1f}".rstrip("0").rstrip("."), font_size=22).move_to(
                    [cx, base_y + (-0.3 if val >= 0 else 0.3), 0])
                g.add(lbl)
            return g

        def stats():
            v = self.V * a.get_value()
            return box(f"α = {a.get_value():.1f}\n"
                       f"L1   = sum |xi|        = {np.abs(v).sum():.1f}\n"
                       f"L∞  = max |xi|        = {np.abs(v).max():.1f}\n"
                       f"count of nonzeros = {np.count_nonzero(v)}", DL, fs=26)

        b = always_redraw(bars)
        s = always_redraw(stats)
        self.say("x = (2, 0, -3, 0, 1), drawn as bars\n"
                 "(green +, red -).")
        self.play(Create(baseline), FadeIn(b))
        self.wait()
        self.say("L1 = total bar length: 2+0+3+0+1 = 6\n"
                 "L∞ = tallest bar: 3\n"
                 "nonzero count: 3 bars", wait=1)
        self.play(FadeIn(s))
        self.wait(2)
        self.say("Now scale x by α = 2.\n"
                 "Rule 3 says every norm must\n"
                 "double too.", wait=1)
        self.play(a.animate.set_value(2), run_time=3)
        self.wait()
        self.say("L1: 6 → 12   doubled ✓\n"
                 "L∞: 3 → 6    doubled ✓\n"
                 "count: 3 → 3   NOT doubled ✗\n"
                 "So the 'L0 norm' is not a real norm.", wait=4)
        self.say("In practice ML uses L1 as a\n"
                 "stand-in for 'how many nonzeros',\n"
                 "because it behaves like a norm\n"
                 "AND rewards exact zeros.", wait=4)


# ---------------------------------------------------------------------------
# 6. Frobenius norm of a matrix
# ---------------------------------------------------------------------------
class Frobenius(Base):
    def construct(self):
        M = [[1, -2], [3, 0], [2, 1]]
        cells = VGroup(*[Text(str(v), font_size=44) for row in M for v in row])
        for k, c in enumerate(cells):
            i, j = divmod(k, 2)
            c.move_to([1.2 * j + 1.5, 1.0 - 1.0 * i, 0])
        brackets = VGroup(Text("[", font_size=150), Text("]", font_size=150))
        brackets[0].next_to(cells, LEFT, buff=0.25)
        brackets[1].next_to(cells, RIGHT, buff=0.25)
        name = Text("A =", font_size=44).next_to(brackets, LEFT)

        self.say("How big is a MATRIX?\n"
                 "Frobenius norm (eq. 2.33).")
        self.play(FadeIn(name), FadeIn(brackets), FadeIn(cells))
        self.wait()
        self.say("Trick: forget the rows/columns.\n"
                 "Line ALL entries up as one long\n"
                 "vector...", wait=1)
        row = cells.copy()
        row.arrange(RIGHT, buff=0.6).move_to(2 * DOWN + 1.0 * RIGHT)
        self.play(TransformFromCopy(cells, row), run_time=2)
        self.wait()
        self.say("...then take its ordinary L2 norm:\n"
                 "1² + 2² + 3² + 0² + 2² + 1² = 19\n"
                 "||A||F = sqrt(19) ≈ 4.36", wait=3)
        squares = VGroup(*[Text(f"{int(c.text.replace('−','-'))**2}", font_size=36, color=YELLOW)
                           .next_to(c, DOWN, buff=0.35) for c in row])
        self.play(LaggedStart(*[FadeIn(sq, shift=0.2 * DOWN) for sq in squares], lag_ratio=0.15))
        self.say("So Frobenius = 'L2 norm for\n"
                 "matrices'. In deep learning it\n"
                 "measures how big a weight\n"
                 "matrix is (weight decay).", wait=4)


# ---------------------------------------------------------------------------
# 7. Dot product = ||x|| ||y|| cos(theta)
# ---------------------------------------------------------------------------
class DotProductAngle(Base):
    def construct(self):
        plane = NumberPlane(background_line_style={"stroke_opacity": 0.35}).shift(1.5 * RIGHT + 1.2 * DOWN)
        self.add(plane)
        O = plane.c2p(0, 0)
        x = np.array([3.0, 0.0])
        r = 2.5
        th = ValueTracker(35 * DEGREES)

        def y():
            return r * np.array([np.cos(th.get_value()), np.sin(th.get_value())])

        xa = Arrow(O, plane.c2p(*x), buff=0, color=YELLOW)
        ya = always_redraw(lambda: Arrow(O, plane.c2p(*y()), buff=0, color=TEAL))
        arc = always_redraw(lambda: Arc(radius=0.6, start_angle=0, angle=th.get_value(),
                                        arc_center=O, color=WHITE))
        drop = always_redraw(lambda: DashedLine(plane.c2p(*y()), plane.c2p(y()[0], 0), color=GREY_B))
        shadow = always_redraw(lambda: Line(O, plane.c2p(y()[0], 0),
                                            color=GREEN if y()[0] >= 0 else RED, stroke_width=10))

        def read():
            yy = y()
            return box(f"θ = {np.degrees(th.get_value()):6.1f}°\n"
                       f"x·y = x1y1 + x2y2           = {x @ yy:6.2f}\n"
                       f"||x|| ||y|| cos θ = 3·2.5·{np.cos(th.get_value()):5.2f} = "
                       f"{3 * r * np.cos(th.get_value()):6.2f}", DL, fs=24)

        self.say("Eq. 2.34: x·y = ||x||₂ ||y||₂ cos θ\n"
                 "The dot product mixes LENGTHS\n"
                 "and the ANGLE between vectors.")
        self.play(GrowArrow(xa), Create(ya), Create(arc))
        rd = always_redraw(read)
        self.add(rd)
        self.wait()
        self.say("Green bar = 'shadow' of y on x\n"
                 "= ||y|| cos θ.\n"
                 "Dot product = ||x|| × shadow.", wait=1)
        self.play(Create(drop), Create(shadow))
        self.wait(2)
        self.say("Same direction (θ = 0):\n"
                 "cos = 1, dot product is maximal.", wait=0.5)
        self.play(th.animate.set_value(0.0001), run_time=2.5)
        self.wait(1.5)
        self.say("θ = 90°: cos = 0, no shadow,\n"
                 "dot product = 0.\n"
                 "The vectors are ORTHOGONAL.", wait=0.5)
        self.play(th.animate.set_value(90 * DEGREES), run_time=3)
        self.wait(1.5)
        self.say("θ > 90°: shadow points backwards,\n"
                 "dot product is NEGATIVE.", wait=0.5)
        self.play(th.animate.set_value(150 * DEGREES), run_time=3)
        self.wait(1.5)
        self.say("ML use: cosine similarity =\n"
                 "x·y / (||x|| ||y||) = cos θ.\n"
                 "How 'aligned' two embeddings are,\n"
                 "ignoring their lengths.", wait=1)
        self.play(th.animate.set_value(20 * DEGREES), run_time=3)
        self.wait(3)

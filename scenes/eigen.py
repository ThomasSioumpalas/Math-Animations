"""
Eigendecomposition — Goodfellow et al., Deep Learning, §2.7
Uses the matrix from the TikTok clip: A = [[1.5, 1], [1, 1.5]]

Put this file in your scenes\ folder and run from the main manimations folder:
    uv run manim -pql scenes\eigen.py E1_WhatIsAnEigenvector   # most vectors get knocked off their line, eigenvectors don't
    uv run manim -pql scenes\eigen.py E2_AnyLengthWorks        # s·v is also an eigenvector -> use unit length
    uv run manim -pql scenes\eigen.py E3_CircleToEllipse       # the book's Figure 2.3
    uv run manim -pql scenes\eigen.py E4_Decomposition         # A = V diag(λ) V⁻¹ as 3 moves
    uv run manim -pql scenes\eigen.py E5_WhenItFails           # rotation (complex) and shear (too few eigenvectors)

All scenes:  uv run manim -pql -a scenes\eigen.py
No LaTeX needed.
"""
from manim import *
import numpy as np

FS = 24
A = np.array([[1.5, 1.0], [1.0, 1.5]])
LAM1, LAM2 = 2.5, 0.5
V1 = np.array([1, 1]) / np.sqrt(2)     # eigenvector for λ1 = 2.5
V2 = np.array([-1, 1]) / np.sqrt(2)    # eigenvector for λ2 = 0.5
V = np.column_stack([V1, V2])          # rotation by 45°, so V⁻¹ = Vᵀ


def box(text, corner=UL, fs=FS):
    t = Text(text, font_size=fs, line_spacing=0.8)
    t.add_background_rectangle(opacity=0.88, buff=0.18)
    return t.to_corner(corner).set_z_index(10)


def span_line(direction, color=GREY_B):
    d = np.array([*direction, 0], dtype=float)
    d = d / np.linalg.norm(d)
    return DashedLine(-12 * d, 12 * d, color=color, stroke_width=2, dash_length=0.15)


class GridBase(Scene):
    note = None

    def say(self, text, corner=UL, wait=3):
        new = box(text, corner)
        if self.note is None:
            self.play(FadeIn(new))
        else:
            self.play(FadeOut(self.note), FadeIn(new))
        self.note = new
        self.wait(wait)

    def setup_grid(self, basis=False):
        self.add(NumberPlane(background_line_style={"stroke_opacity": 0.25},
                             axis_config={"stroke_opacity": 0.4}))
        self.grid = NumberPlane(x_range=[-16, 16], y_range=[-16, 16],
                                background_line_style={"stroke_color": BLUE, "stroke_width": 2,
                                                       "stroke_opacity": 0.7},
                                axis_config={"stroke_opacity": 0.6})
        self.add(self.grid)
        self.movers = [self.grid]
        if basis:
            self.i_hat = Vector([1, 0], color=GREEN)
            self.j_hat = Vector([0, 1], color=RED)
            self.add(self.i_hat, self.j_hat)
            self.movers += [self.i_hat, self.j_hat]

    def apply(self, matrix, extra=(), run_time=2.5):
        m = np.array(matrix, dtype=float)
        anims = []
        for mob in self.movers + list(extra):
            if isinstance(mob, Arrow):
                anims.append(Transform(mob, Vector(m @ mob.get_end()[:2], color=mob.get_color())))
            elif isinstance(mob, Dot):
                p = mob.get_center()
                anims.append(mob.animate.move_to([*(m @ p[:2]), 0]))
            else:
                anims.append(ApplyMatrix(m, mob))
        self.play(*anims, run_time=run_time)
        self.wait(0.5)


# ---------------------------------------------------------------------------
# E1. What an eigenvector is
# ---------------------------------------------------------------------------
class E1_WhatIsAnEigenvector(GridBase):
    def construct(self):
        self.say("Book's idea: 12 = 2 × 2 × 3 tells you things about 12\n"
                 "no matter how you write it (decimal, binary...).\n"
                 "Eigendecomposition does the same for a MATRIX:\n"
                 "breaks it into parts that reveal what it really does.", wait=5)
        self.setup_grid()
        self.say("The TikTok matrix: A = [[1.5, 1], [1, 1.5]].\n"
                 "Each dashed line = the line a vector sits on\n"
                 "(its 'span').", wait=1)
        a = Vector([2, 0.5], color=YELLOW)
        b = Vector([-1, 2], color=TEAL)
        la, lb = span_line([2, 0.5], YELLOW), span_line([-1, 2], TEAL)
        self.play(Create(la), Create(lb), GrowArrow(a), GrowArrow(b))
        self.wait()
        self.say("Apply A. Watch the arrows vs. their own lines.", wait=0.5)
        self.apply(A, extra=[a, b])
        self.say("Both got KNOCKED OFF their lines:\n"
                 "A rotated them, not just stretched them.\n"
                 "That's what happens to almost every vector.", wait=3.5)
        self.apply(np.linalg.inv(A), extra=[a, b], run_time=1.5)
        self.play(FadeOut(a), FadeOut(b), FadeOut(la), FadeOut(lb))

        e1 = Vector(V1 * 1.2, color=ORANGE)
        e2 = Vector(V2 * 1.2, color=PINK)
        l1, l2 = span_line(V1, ORANGE), span_line(V2, PINK)
        self.say("But two special directions exist.\n"
                 "orange along (1, 1),  pink along (-1, 1).", wait=1)
        self.play(Create(l1), Create(l2), GrowArrow(e1), GrowArrow(e2))
        self.wait()
        self.apply(A, extra=[e1, e2])
        self.say("They STAYED on their lines! Only their length changed:\n"
                 "orange × 2.5  (stretched)\n"
                 "pink   × 0.5  (shrunk)\n"
                 "These are the EIGENVECTORS of A.", wait=4)
        self.say("Definition (eq. 2.39):  A v = λ v\n"
                 "v = eigenvector: A only changes its SCALE\n"
                 "λ = eigenvalue: by HOW MUCH (here 2.5 and 0.5)\n"
                 "Check: A·(1,1) = (1.5+1, 1+1.5) = (2.5, 2.5) = 2.5·(1,1)", wait=6)


# ---------------------------------------------------------------------------
# E2. Any rescaled eigenvector is still an eigenvector
# ---------------------------------------------------------------------------
class E2_AnyLengthWorks(GridBase):
    def construct(self):
        self.setup_grid()
        self.add(span_line(V1, ORANGE))
        vs = [Vector([0.6, 0.6], color=ORANGE),
              Vector([-1, -1], color=YELLOW),
              Vector([1.2, 1.2], color=GOLD)]
        vs[2].shift(0)  # same line, different lengths
        self.say("Any vector on the orange line works:\n"
                 "(0.6, 0.6),  (-1, -1),  (1.2, 1.2)...\n"
                 "v times any number s ≠ 0.", wait=1)
        self.play(*[GrowArrow(v) for v in vs])
        self.wait()
        self.apply(A, extra=vs)
        self.say("ALL of them stay on the line and ALL get × 2.5.\n"
                 "So s·v is an eigenvector with the SAME λ.\n"
                 "(even the backwards one: (-1,-1) -> (-2.5,-2.5))", wait=4)
        self.apply(np.linalg.inv(A), extra=vs, run_time=1.5)
        self.play(*[FadeOut(v) for v in vs])
        u = Vector(V1, color=ORANGE)
        circ = Circle(radius=1, color=WHITE, stroke_width=2)
        self.play(Create(circ), GrowArrow(u))
        self.say("Infinitely many choices, so we agree on one:\n"
                 "the UNIT eigenvector (length 1, on the circle).\n"
                 "v = (1, 1)/sqrt(2) ≈ (0.71, 0.71)", wait=4)


# ---------------------------------------------------------------------------
# E3. Figure 2.3: unit circle -> ellipse
# ---------------------------------------------------------------------------
class E3_CircleToEllipse(GridBase):
    def construct(self):
        k = 1.2  # zoom: 1 unit = 1.2 screen units
        self.movers = []
        self.add(NumberPlane(x_range=[-6, 6], y_range=[-4, 4], x_length=12 * k, y_length=8 * k,
                             background_line_style={"stroke_opacity": 0.3}))
        circle = Circle(radius=k, color=WHITE, stroke_width=4)
        dots = VGroup(*[Dot([k * np.cos(t), k * np.sin(t), 0], radius=0.05, color=BLUE_B)
                        for t in np.linspace(0, TAU, 24, endpoint=False)])
        e1 = Vector(k * V1, color=ORANGE)
        e2 = Vector(k * V2, color=PINK)

        self.say("Book Figure 2.3. Left side: ALL unit vectors u\n"
                 "= every point on the unit circle.", wait=1)
        self.play(Create(circle), FadeIn(dots))
        self.play(GrowArrow(e1), GrowArrow(e2))
        self.say("orange v(1), pink v(2): the two unit eigenvectors.\n"
                 "They are at 90° (orthonormal) because\n"
                 "A is symmetric.", wait=3)
        self.say("Right side: apply A to EVERY point of the circle\n"
                 "(plot all the A·u).", wait=1)
        self.apply(A, extra=[circle, *dots, e1, e2], run_time=3.5)
        l1 = Text("× 2.5 (λ1)", font_size=26, color=ORANGE).next_to(e1.get_end(), DR, buff=0.1)
        l2 = Text("× 0.5 (λ2)", font_size=26, color=PINK).next_to(e2.get_end(), LEFT, buff=0.15)
        self.play(FadeIn(l1), FadeIn(l2))
        self.say("The circle became an ELLIPSE.\n"
                 "Its longest axis = orange direction, stretched by λ1 = 2.5\n"
                 "Its shortest axis = pink direction, squeezed by λ2 = 0.5\n"
                 "Eigenvectors = the axes, eigenvalues = how much.", DR, wait=5)
        self.say("So without reading the 4 numbers of A one by one,\n"
                 "eigen-pairs tell you WHAT A DOES:\n"
                 "'stretch 2.5× along (1,1), squash 0.5× along (-1,1)'.", DR, wait=5)


# ---------------------------------------------------------------------------
# E4. A = V diag(λ) V⁻¹  as three simple moves
# ---------------------------------------------------------------------------
class E4_Decomposition(GridBase):
    def construct(self):
        self.setup_grid()
        circle = Circle(radius=1, color=WHITE, stroke_width=4)
        e1, e2 = Vector(V1, color=ORANGE), Vector(V2, color=PINK)
        self.add(circle, e1, e2)
        self.say("Eq. 2.40:  A = V · diag(λ) · V⁻¹\n"
                 "V = eigenvectors as COLUMNS = [v(1) v(2)]\n"
                 "diag(λ) = diag(2.5, 0.5)\n"
                 "First, A in one go:", wait=2)
        self.apply(A, extra=[circle, e1, e2])
        self.wait()
        self.apply(np.linalg.inv(A), extra=[circle, e1, e2], run_time=1.5)

        self.say("Now the SAME result in 3 simple steps.\n"
                 "(read right to left: V⁻¹ first)\n"
                 "Step 1, V⁻¹: turn the eigenvectors onto the axes.\n"
                 "Here V is a rotation, so V⁻¹ = Vᵀ (last chapter!).", wait=1.5)
        self.apply(V.T, extra=[circle, e1, e2])
        self.say("orange is now on the x-axis, pink on the y-axis.\n"
                 "Step 2, diag(λ): stretch along the AXES only:\n"
                 "x × 2.5,  y × 0.5  (cheap diagonal matrix!).", wait=1.5)
        self.apply(np.diag([LAM1, LAM2]), extra=[circle, e1, e2])
        self.say("Step 3, V: turn everything back.", wait=0.5)
        self.apply(V, extra=[circle, e1, e2])
        self.say("Identical to applying A directly!\n"
                 "Every A with enough eigenvectors =\n"
                 "rotate in, stretch along axes, rotate back.", wait=4)
        self.say("Why ML cares: XᵀX (least squares) and covariance\n"
                 "matrices are symmetric, so they ALWAYS split\n"
                 "like this with orthonormal V. PCA uses it to find\n"
                 "the main directions in data.", wait=5)


# ---------------------------------------------------------------------------
# E5. When there is no (real) eigendecomposition
# ---------------------------------------------------------------------------
class E5_WhenItFails(GridBase):
    def construct(self):
        self.setup_grid()
        dirs = [([1, 0], YELLOW), ([1, 1], ORANGE), ([0, 1], TEAL), ([-1, 1], PINK)]
        lines = VGroup(*[span_line(d, c) for d, c in dirs])
        vecs = [Vector(np.array(d) * 1.3 / np.linalg.norm(d), color=c) for d, c in dirs]
        self.say("Not every matrix can be decomposed.\n"
                 "Case 1: rotation by 90°, R = [[0, -1], [1, 0]].", wait=1)
        self.play(Create(lines), *[GrowArrow(v) for v in vecs])
        self.apply([[0, -1], [1, 0]], extra=vecs)
        self.say("EVERY vector left its line: no direction is\n"
                 "only scaled. No real eigenvectors.\n"
                 "(the eigenvalues exist but are complex: ±i)", wait=4)
        self.apply([[0, 1], [-1, 0]], extra=vecs, run_time=1.5)

        self.say("Case 2: shear S = [[1, 1], [0, 1]].", wait=1)
        self.apply([[1, 1], [0, 1]], extra=vecs)
        self.say("Only the yellow x-axis stays on its line (λ = 1).\n"
                 "ONE eigen-direction in 2D is not enough: V needs\n"
                 "2 independent columns to be invertible.\n"
                 "-> no eigendecomposition.", wait=5)
        self.say("Good news (book): ML mostly needs REAL SYMMETRIC\n"
                 "matrices, and those always decompose, with\n"
                 "real λ and orthonormal eigenvectors.", wait=4)

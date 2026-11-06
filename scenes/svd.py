"""
SVD and the Moore-Penrose pseudoinverse — Goodfellow et al., Deep Learning, §2.8–2.9

Put this file in your scenes\ folder and run from the main manimations folder:
    uv run manim -pql scenes\svd.py SVD1_ShearInThreeMoves     # A = U D Vᵀ on the shear that eigen couldn't handle
    uv run manim -pql scenes\svd.py SVD2_NonSquare             # 3×2 matrix: rotate, stretch + add a dimension, rotate
    uv run manim -pql scenes\svd.py PINV1_BuildDplus           # D⁺: flip nonzero entries, then transpose
    uv run manim -pql scenes\svd.py PINV2_TallLeastSquares     # tall A: A⁺y = closest point (3 students)
    uv run manim -pql scenes\svd.py PINV3_WideMinNorm          # wide A: A⁺y = smallest-norm solution (1 student)

All scenes:  uv run manim -pql -a scenes\svd.py
No LaTeX needed.
"""
from manim import *
import numpy as np

FS = 24


def box(text, corner=UL, fs=FS):
    t = Text(text, font_size=fs, line_spacing=0.8)
    t.add_background_rectangle(opacity=0.88, buff=0.18)
    return t.to_corner(corner).set_z_index(10)


def svd_rot(A):
    """SVD with U and V chosen as proper rotations when possible (nicer to watch)."""
    U, S, Vt = np.linalg.svd(A)
    if A.shape[0] == A.shape[1] and np.linalg.det(U) < 0:
        U[:, -1] *= -1
        Vt[-1, :] *= -1
    return U, S, Vt


def bracketed(entries, ncols, fs=30, colors=None):
    cells = VGroup(*[Text(str(e), font_size=fs, color=(colors[i] if colors else WHITE))
                     for i, e in enumerate(entries)])
    cells.arrange_in_grid(cols=ncols, buff=(0.55, 0.22))
    h = cells.height + 0.25

    def bracket(side):
        d = 0.15 * side
        return VMobject(stroke_width=3).set_points_as_corners(
            [UP * h / 2 + RIGHT * d, UP * h / 2, DOWN * h / 2, DOWN * h / 2 + RIGHT * d])

    return VGroup(bracket(1).next_to(cells, LEFT, buff=0.12), cells,
                  bracket(-1).next_to(cells, RIGHT, buff=0.12))


class Narrated:
    note = None

    def say(self, text, corner=UL, wait=3, fs=FS):
        new = box(text, corner, fs)
        if isinstance(self, ThreeDScene):
            self.add_fixed_in_frame_mobjects(new)
        if self.note is None:
            self.play(FadeIn(new))
        else:
            self.play(FadeOut(self.note), FadeIn(new))
        self.note = new
        self.wait(wait)


class GridBase(Narrated, Scene):
    def setup_grid(self):
        self.add(NumberPlane(background_line_style={"stroke_opacity": 0.25},
                             axis_config={"stroke_opacity": 0.4}))
        self.grid = NumberPlane(x_range=[-16, 16], y_range=[-16, 16],
                                background_line_style={"stroke_color": BLUE, "stroke_width": 2,
                                                       "stroke_opacity": 0.7},
                                axis_config={"stroke_opacity": 0.6})
        self.add(self.grid)
        self.movers = [self.grid]

    def apply(self, matrix, extra=(), run_time=2.5):
        m = np.array(matrix, dtype=float)
        anims = []
        for mob in self.movers + list(extra):
            if isinstance(mob, Arrow):
                anims.append(Transform(mob, Vector(m @ mob.get_end()[:2], color=mob.get_color())))
            else:
                anims.append(ApplyMatrix(m, mob))
        self.play(*anims, run_time=run_time)
        self.wait(0.5)


# ---------------------------------------------------------------------------
# SVD1. A = U D Vᵀ  on the shear (where eigendecomposition failed)
# ---------------------------------------------------------------------------
class SVD1_ShearInThreeMoves(GridBase):
    def construct(self):
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        U, S, Vt = svd_rot(A)
        self.setup_grid()
        circle = Circle(radius=1, color=WHITE, stroke_width=4)
        v1 = Vector(Vt[0], color=ORANGE)   # right-singular vectors = rows of Vᵀ
        v2 = Vector(Vt[1], color=PINK)
        objs = [circle, v1, v2]

        self.say("Remember the SHEAR A = [[1, 1], [0, 1]]?\n"
                 "Eigendecomposition FAILED on it: only one\n"
                 "eigen-direction. SVD works on EVERY matrix.", wait=4)
        self.say("Eq. 2.43:  A = U · D · Vᵀ\n"
                 "U, V = orthogonal (pure rotations)\n"
                 "D = diagonal (stretch along axes)\n"
                 "First, A in one go on the unit circle:", wait=2)
        self.add(*objs)
        self.apply(A, extra=objs)
        self.wait()
        self.apply(np.linalg.inv(A), extra=objs, run_time=1.5)

        self.say("Orange & pink = the RIGHT-singular vectors v\n"
                 "(columns of V). They're orthonormal: 90°, length 1.\n"
                 "Step 1, Vᵀ: rotate them onto the axes.", wait=1)
        self.apply(Vt, extra=objs)
        self.say(f"Step 2, D = diag({S[0]:.3f}, {S[1]:.3f}): stretch the axes.\n"
                 "These numbers are the SINGULAR VALUES σ.\n"
                 "x × 1.618,  y × 0.618", wait=1)
        self.apply(np.diag(S), extra=objs)
        self.say("Step 3, U: rotate into the final directions.", wait=0.5)
        self.apply(U, extra=objs)
        self.say("Same ellipse as A in one go!\n"
                 "orange -> σ1·u1 = long axis,  pink -> σ2·u2 = short axis\n"
                 "u = LEFT-singular vectors (columns of U).", wait=4)
        self.say("The big difference from eigendecomposition:\n"
                 "the INPUT directions (v) and OUTPUT directions (u)\n"
                 "are allowed to be DIFFERENT. Eigen needed them\n"
                 "to be the same, that's why it failed here.", wait=5)
        self.say("Link to eigen (book): σ² = eigenvalues of AᵀA.\n"
                 "AᵀA = [[1, 1], [1, 2]] -> eigenvalues 2.618 and 0.382\n"
                 "sqrt -> 1.618 and 0.618 = the singular values.\n"
                 "v = eigenvectors of AᵀA,  u = eigenvectors of AAᵀ.", wait=6)


# ---------------------------------------------------------------------------
# SVD2. A non-square (3×2) matrix: rotate, stretch + add a dimension, rotate in 3-D
# ---------------------------------------------------------------------------
class SVD2_NonSquare(Narrated, ThreeDScene):
    def construct(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        U, S, Vt = np.linalg.svd(A)          # U 3×3, S (2,), Vt 2×2
        D = np.zeros((3, 2)); D[0, 0], D[1, 1] = S
        k = 1.25

        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-2.5, 2.5],
                          x_length=6, y_length=6, z_length=5)
        self.set_camera_orientation(phi=65 * DEGREES, theta=-55 * DEGREES)
        self.add(axes)

        stages = [
            lambda p: np.array([p[0], p[1], 0.0]),
            lambda p: np.array([*(Vt @ p), 0.0]),
            lambda p: D @ (Vt @ p),
            lambda p: U @ D @ (Vt @ p),
        ]

        def curve(f):
            return ParametricFunction(
                lambda t: axes.c2p(*(k * f(np.array([np.cos(t), np.sin(t)])))),
                t_range=[0, TAU, TAU / 120], color=WHITE, stroke_width=4)

        def arrow(f, v, color):
            return Arrow3D(axes.c2p(0, 0, 0), axes.c2p(*(k * f(v))), color=color)

        c = curve(stages[0])
        a1 = arrow(stages[0], Vt[0], ORANGE)
        a2 = arrow(stages[0], Vt[1], PINK)

        self.say("Now a NON-square A (3×2), same as the ThreeByTwo\n"
                 "scene: columns (1, 0, 1) and (0, 1, 1).\n"
                 "2 numbers in, 3 numbers out. Eigen: impossible.\n"
                 "SVD: U is 3×3, D is 3×2, V is 2×2.", wait=4)
        self.play(Create(c), Create(a1), Create(a2))
        self.say("Input = the flat unit circle (2-D).\n"
                 "Step 1, Vᵀ (2×2): rotate the right-singular\n"
                 "vectors onto the axes. Still flat.", wait=1)
        self.play(Transform(c, curve(stages[1])), Transform(a1, arrow(stages[1], Vt[0], ORANGE)),
                  Transform(a2, arrow(stages[1], Vt[1], PINK)), run_time=2.5)
        self.say(f"Step 2, D (3×2 rectangular diagonal!):\n"
                 f"stretch by σ = {S[0]:.2f} and {S[1]:.2f}, then ADD a zero\n"
                 f"3rd coordinate. Now it lives in 3-D, still flat.", wait=1)
        self.play(Transform(c, curve(stages[2])), Transform(a1, arrow(stages[2], Vt[0], ORANGE)),
                  Transform(a2, arrow(stages[2], Vt[1], PINK)), run_time=2.5)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.say("Step 3, U (3×3 rotation): tilt it into its final\n"
                 "position in 3-D space.", wait=1)
        self.play(Transform(c, curve(stages[3])), Transform(a1, arrow(stages[3], Vt[0], ORANGE)),
                  Transform(a2, arrow(stages[3], Vt[1], PINK)), run_time=3)
        u1, u2 = U[:, 0], U[:, 1]
        plane = Surface(lambda s_, t_: axes.c2p(*(2.4 * (s_ * u1 + t_ * u2))),
                        u_range=[-1, 1], v_range=[-1, 1], resolution=(8, 8),
                        fill_opacity=0.25, checkerboard_colors=[BLUE_D, BLUE_E])
        self.play(Create(plane))
        self.say("The ellipse lies on a tilted PLANE: the column space\n"
                 "of A. Every Ax lands somewhere on it.\n"
                 "σ1 = sqrt(3) ≈ 1.73, σ2 = 1 (sqrt of eig of AᵀA = 3, 1)", wait=6)
        self.stop_ambient_camera_rotation()


# ---------------------------------------------------------------------------
# PINV1. Building D⁺ and A⁺ = V D⁺ Uᵀ
# ---------------------------------------------------------------------------
class PINV1_BuildDplus(Narrated, Scene):
    def construct(self):
        self.say("Problem (§2.9): non-square A has no inverse.\n"
                 "The PSEUDOINVERSE A⁺ is the next best thing.\n"
                 "Recipe (eq. 2.47):  A⁺ = V · D⁺ · Uᵀ", wait=4)

        D = bracketed(["1.73", "0", "0", "1", "0", "0"], 2, fs=34,
                      colors=[YELLOW, GREY, GREY, YELLOW, GREY, GREY])
        Dl = Text("D =", font_size=34).next_to(D, LEFT)
        g = VGroup(Dl, D).move_to(3 * LEFT + 0.4 * DOWN)
        self.play(FadeIn(g))
        self.say("Start with D from the 3×2 example (3×2).\n"
                 "Step 1: take 1/σ of every NONZERO entry.\n"
                 "Zeros stay zero (no 1/0 ever happens).", wait=1)
        D1 = bracketed(["0.58", "0", "0", "1", "0", "0"], 2, fs=34,
                       colors=[YELLOW, GREY, GREY, YELLOW, GREY, GREY]).move_to(D)
        self.play(Transform(D, D1), run_time=1.5)
        self.wait(1.5)
        self.say("Step 2: TRANSPOSE it (rows <-> columns).\n"
                 "3×2 becomes 2×3.", wait=1)
        Dp = bracketed(["0.58", "0", "0", "0", "1", "0"], 3, fs=34,
                       colors=[YELLOW, GREY, GREY, GREY, YELLOW, GREY])
        Dpl = Text("D⁺ =", font_size=34).next_to(Dp, LEFT)
        g2 = VGroup(Dpl, Dp).move_to(3 * RIGHT + 0.4 * DOWN)
        arrow = Arrow(g.get_right(), g2.get_left(), buff=0.3)
        self.play(GrowArrow(arrow), TransformFromCopy(D, Dp), FadeIn(Dpl), run_time=2)
        self.say("D⁺ undoes D as much as possible:\n"
                 "D⁺D = [[1, 0], [0, 1]] = I\n"
                 "(shrink back by 1/σ, throw away the added 3rd coord.)", wait=4)
        self.play(FadeOut(g), FadeOut(g2), FadeOut(arrow))

        steps = VGroup(
            Text("A  = U · D · Vᵀ      (rotate, stretch, rotate)", font_size=30),
            Text("A⁺ = V · D⁺ · Uᵀ     (the same moves BACKWARDS)", font_size=30, color=YELLOW),
            Text("Uᵀ: undo the last rotation   (orthogonal: inverse = transpose)", font_size=24),
            Text("D⁺: undo the stretch by 1/σ, drop extra dimensions", font_size=24),
            Text("V : undo the first rotation", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(0.6 * DOWN)
        for s in steps:
            self.play(FadeIn(s, shift=0.2 * UP))
            self.wait(0.8)
        self.say("Every piece is CHEAP to invert: rotations by\n"
                 "transpose, the diagonal by 1/σ. That's why real\n"
                 "libraries (np.linalg.pinv) compute A⁺ this way.", wait=4)
        self.say("If A is square and invertible: A⁺ = A⁻¹ exactly.\n"
                 "Otherwise it gives a 'best' answer:\n"
                 "tall A -> closest fit,  wide A -> smallest solution.\n"
                 "Next two scenes.", wait=5)
        self.say("Eq. 2.46: A⁺ = lim (AᵀA + αI)⁻¹ Aᵀ  as α -> 0\n"
                 "With a small α > 0 kept, that's exactly RIDGE\n"
                 "regression (L2-regularized least squares).", wait=5)


# ---------------------------------------------------------------------------
# PINV2. Tall A: A⁺y = least-squares (the 3 students)
# ---------------------------------------------------------------------------
class PINV2_TallLeastSquares(Narrated, ThreeDScene):
    def construct(self):
        X = np.array([[2.0, 6.0], [4.0, 7.0], [5.0, 5.0]])
        y = np.array([52.0, 63.0, 64.0])
        w = np.linalg.pinv(X) @ y
        p = X @ w
        k = 1 / 12   # display scale

        axes = ThreeDAxes(x_range=[0, 6], y_range=[0, 6], z_range=[0, 6],
                          x_length=5.5, y_length=5.5, z_length=5.5)
        self.set_camera_orientation(phi=65 * DEGREES, theta=-35 * DEGREES, zoom=0.8,
                                    frame_center=axes.c2p(*(k * p * 0.55)))
        self.add(axes)
        O = axes.c2p(0, 0, 0)

        q, _ = np.linalg.qr(X)
        e1, e2 = q[:, 0], q[:, 1]
        plane = Surface(lambda s_, t_: axes.c2p(*(k * p + 1.8 * (s_ * e1 + t_ * e2))),
                        u_range=[-1, 1], v_range=[-1, 1], resolution=(8, 8),
                        fill_opacity=0.3, checkerboard_colors=[BLUE_D, BLUE_E])

        self.say("TALL A: our 3 students, 2 weights (S4 scene).\n"
                 "X = [[2,6],[4,7],[5,5]],  y = (52, 63, 64)\n"
                 "No exact w exists: y is off the column space.", wait=1)
        self.play(Create(plane))
        ya = Arrow3D(O, axes.c2p(*(k * y)), color=GREEN)
        self.play(Create(ya))
        self.say("Blue plane = every possible prediction Xw.\n"
                 "Green = the real grades y, sticking OUT of it.", wait=2)
        self.begin_ambient_camera_rotation(rate=0.15)
        pa = Arrow3D(O, axes.c2p(*(k * p)), color=YELLOW)
        err = DashedLine(axes.c2p(*(k * p)), axes.c2p(*(k * y)), color=RED)
        self.say("A⁺y picks the point of the plane CLOSEST to y\n"
                 "(yellow). The red error is perpendicular to the\n"
                 "plane: the shortest possible gap.", wait=1)
        self.play(Create(pa), Create(err))
        self.wait(3)
        self.say(f"w = A⁺y = ({w[0]:.2f}, {w[1]:.2f})\n"
                 f"predictions ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) vs (52, 63, 64)\n"
                 f"||Xw - y||² = {np.sum((p - y) ** 2):.1f}: as small as it can get.\n"
                 "SAME w as least squares! A⁺ = (XᵀX)⁻¹Xᵀ here.", wait=6)
        self.say("ML: this is linear regression training in one\n"
                 "formula: w = pinv(X) @ y   (numpy).", wait=4)
        self.stop_ambient_camera_rotation()


# ---------------------------------------------------------------------------
# PINV3. Wide A: A⁺y = the smallest-norm solution (1 student, 2 weights)
# ---------------------------------------------------------------------------
class PINV3_WideMinNorm(Narrated, Scene):
    def construct(self):
        a = np.array([2.0, 6.0])   # one student: study 2, sleep 6
        b = 52.0
        w_star = a * b / (a @ a)   # = A⁺ b = (2.6, 7.8)

        ax = Axes(x_range=[-2, 16, 2], y_range=[-1, 11, 1], x_length=7.4, y_length=5.0,
                  tips=False).to_edge(RIGHT, buff=0.4).shift(0.9 * DOWN)
        xl = Text("w1 (per study hour)", font_size=20).next_to(ax.x_axis, DOWN, buff=0.2)
        yl = Text("w2 (per sleep hour)", font_size=20).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.2)
        self.add(ax, xl, yl)
        O = ax.c2p(0, 0)

        self.say("WIDE A: only ONE student (Maria), 2 weights.\n"
                 "2·w1 + 6·w2 = 52   ->  A = [2, 6] is 1×2\n"
                 "1 equation, 2 unknowns: infinitely many answers.", wait=3)
        line = ax.plot(lambda t: (b - a[0] * t) / a[1], x_range=[-2, 16], color=GREEN, stroke_width=5)
        self.play(Create(line))
        self.say("Every point on the green line is a PERFECT fit\n"
                 "for Maria. Which w should we pick?", wait=1)
        t = ValueTracker(14.0)

        def wpt():
            return np.array([t.get_value(), (b - a[0] * t.get_value()) / a[1]])

        arr = always_redraw(lambda: Arrow(O, ax.c2p(*wpt()), buff=0, color=YELLOW))
        dot = always_redraw(lambda: Dot(ax.c2p(*wpt()), color=YELLOW))
        rd = always_redraw(lambda: box(
            f"w = ({wpt()[0]:5.2f}, {wpt()[1]:5.2f})\n"
            f"check: 2·w1 + 6·w2 = {a @ wpt():.0f}\n"
            f"||w|| = {np.linalg.norm(wpt()):.2f}", DL, fs=26))
        self.add(arr, dot, rd)
        self.play(t.animate.set_value(-1.5), run_time=4)
        self.play(t.animate.set_value(10), run_time=3)
        self.say("All give 52, but the LENGTH ||w|| changes.\n"
                 "The pseudoinverse picks the SHORTEST one:\n"
                 "the point of the line closest to the origin.", wait=1)
        self.play(t.animate.set_value(w_star[0]), run_time=3)
        perp = DashedLine(O, ax.c2p(*w_star), color=WHITE)
        self.add(perp)
        self.say(f"w = A⁺·52 = ({w_star[0]:.1f}, {w_star[1]:.1f}),  ||w|| = {np.linalg.norm(w_star):.2f}\n"
                 "Minimal Euclidean norm among ALL solutions.\n"
                 "It points straight at the line (90°).", wait=5)
        self.say("ML: when you have more features than examples,\n"
                 "smallest weights = simplest model, least likely\n"
                 "to rely on any single feature too much. Same idea\n"
                 "as L2 regularization / weight decay.", wait=5)

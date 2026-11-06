"""
Column space & linear dependence — visualizing Goodfellow et al., Deep Learning, §2.4

Render (from the folder containing this file):
    manim -pql column_space.py Combination     # Ax is a mix of A's columns
    manim -pql column_space.py FullRank        # columns span all of R^2 -> every b solvable
    manim -pql column_space.py Singular        # identical columns -> plane squished to a line
    manim -pql column_space.py ThreeByTwo      # 3x2 matrix: column space is a plane in R^3

-p = open the video when done, -ql = low quality (fast). Use -qh for HD.
Uses Text (not MathTex) so it works without a LaTeX install.
"""
from manim import *
import numpy as np

FONT_SIZE = 30


def caption(text, scene_corner=UL):
    t = Text(text, font_size=FONT_SIZE, line_spacing=0.8)
    t.add_background_rectangle(opacity=0.85, buff=0.15)
    return t.to_corner(scene_corner)


# ---------------------------------------------------------------------------
# 1. Ax = x1 * (column 1) + x2 * (column 2)
# ---------------------------------------------------------------------------
class Combination(Scene):
    def construct(self):
        self.add(NumberPlane(background_line_style={"stroke_opacity": 0.4}))

        c1 = np.array([2, 1, 0])
        c2 = np.array([-1, 1, 0])
        x1 = ValueTracker(1.0)
        x2 = ValueTracker(0.5)

        def tip():
            return x1.get_value() * c1 + x2.get_value() * c2

        part1 = always_redraw(lambda: Arrow(ORIGIN, x1.get_value() * c1, buff=0, color=YELLOW))
        part2 = always_redraw(lambda: Arrow(x1.get_value() * c1, tip(), buff=0, color=TEAL))
        result = always_redraw(lambda: Arrow(ORIGIN, tip(), buff=0, color=WHITE, stroke_width=7))
        readout = always_redraw(
            lambda: caption(
                f"A = [[2, -1], [1, 1]]\n"
                f"x = ({x1.get_value():+.1f}, {x2.get_value():+.1f})\n"
                f"Ax = x1*col1 + x2*col2",
            )
        )
        trace = TracedPath(lambda: tip(), stroke_color=WHITE, stroke_opacity=0.5)

        self.add(trace, part1, part2, result, readout)
        self.wait()
        # Sweep x around: the tip of Ax can reach anywhere in the plane
        self.play(x1.animate.set_value(-1.3), x2.animate.set_value(1.8), run_time=3)
        self.play(x1.animate.set_value(0.7), x2.animate.set_value(-2.1), run_time=3)
        self.play(x1.animate.set_value(1.6), x2.animate.set_value(1.2), run_time=3)
        self.wait()


class MatrixScene(Scene):
    """Faint fixed grid + a blue grid and basis vectors that the matrix moves."""

    def setup_grid(self):
        self.add(NumberPlane(background_line_style={"stroke_opacity": 0.25},
                             axis_config={"stroke_opacity": 0.4}))
        self.grid = NumberPlane(x_range=[-12, 12], y_range=[-12, 12],
                                background_line_style={"stroke_color": BLUE, "stroke_width": 2})
        self.i_hat = Vector([1, 0], color=GREEN)
        self.j_hat = Vector([0, 1], color=RED)
        self.add(self.grid, self.i_hat, self.j_hat)

    def apply(self, matrix):
        m = np.array(matrix)
        self.play(*[ApplyMatrix(m, mob) for mob in (self.grid, self.i_hat, self.j_hat)],
                  run_time=3)


# ---------------------------------------------------------------------------
# 2. Full-rank 2x2: column space = all of R^2, so Ax = b always has a solution
# ---------------------------------------------------------------------------
class FullRank(MatrixScene):
    def construct(self):
        self.setup_grid()
        A = [[2, -1], [1, 1]]
        self.add(caption("A = [[2, -1], [1, 1]]\ncolumns are independent\n"
                         "green -> col 1, red -> col 2").set_z_index(10))
        self.wait()
        self.apply(A)
        self.wait()

        # Any b is reachable; show one and its solution x = A^-1 b
        b = np.array([1, 3])
        x = np.linalg.solve(A, b)
        b_vec = Vector(b, color=YELLOW)
        msg = caption(
            f"column space = all of R^2\n"
            f"b = (1, 3) -> x = ({x[0]:.2f}, {x[1]:.2f})\n"
            f"A is invertible",
            DR,
        ).set_z_index(10)
        self.play(GrowArrow(b_vec), FadeIn(msg))
        self.wait(2)


# ---------------------------------------------------------------------------
# 3. Singular 2x2 (both columns identical): column space collapses to a line
# ---------------------------------------------------------------------------
class Singular(MatrixScene):
    def construct(self):
        self.setup_grid()
        self.add(caption("A = [[1, 1], [1, 1]]\nboth columns identical").set_z_index(10))
        self.wait()
        self.apply([[1, 1], [1, 1]])
        self.wait()

        line = Line(4 * DL, 4 * UR, color=YELLOW, stroke_width=6)
        b_on = Vector([2, 2], color=GREEN)
        b_off = Vector([-1, 2], color=RED)
        cross = Cross(stroke_width=6).scale(0.2).move_to(b_off.get_end())

        msg = caption(
            "column space = a single line\n"
            "green b: on the line -> solvable\n"
            "red b: off the line -> NO x works\n"
            "det(A) = 0 -> singular, no inverse",
            DR,
        ).set_z_index(10)
        self.play(Create(line), FadeIn(msg))
        self.play(GrowArrow(b_on))
        self.play(GrowArrow(b_off), Create(cross))
        self.wait(2)


# ---------------------------------------------------------------------------
# 4. 3x2 matrix: b lives in R^3 but x is only 2-D -> Ax traces out a plane
# ---------------------------------------------------------------------------
class ThreeByTwo(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3],
                          x_length=6, y_length=6, z_length=6)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-40 * DEGREES)
        self.add(axes)

        c1 = np.array([1, 0, 1])
        c2 = np.array([0, 1, 1])
        o = axes.c2p(0, 0, 0)

        col1 = Arrow3D(o, axes.c2p(*c1), color=YELLOW)
        col2 = Arrow3D(o, axes.c2p(*c2), color=TEAL)
        plane = Surface(
            lambda u, v: axes.c2p(*(u * c1 + v * c2)),
            u_range=[-1.8, 1.8], v_range=[-1.8, 1.8], resolution=(10, 10),
            fill_opacity=0.35, checkerboard_colors=[BLUE_D, BLUE_E],
        )

        title = caption("A is 3x2: columns (1,0,1) and (0,1,1)")
        self.add_fixed_in_frame_mobjects(title)
        self.play(Create(col1), Create(col2))
        self.wait()

        self.begin_ambient_camera_rotation(rate=0.15)
        span_msg = caption("their span (column space) is only a 2-D plane", DL)
        self.add_fixed_in_frame_mobjects(span_msg)
        self.play(Create(plane), run_time=2)
        self.wait()

        b_in = Arrow3D(o, axes.c2p(*(1.2 * c1 - 0.8 * c2)), color=GREEN)  # on the plane
        b_out = Arrow3D(o, axes.c2p(1, 1, -1.5), color=RED)               # off the plane
        self.play(Create(b_in))
        self.play(Create(b_out))
        end_msg = caption("green b: on plane -> solvable\nred b: off plane -> no solution", DL)
        self.play(FadeOut(span_msg))
        self.add_fixed_in_frame_mobjects(end_msg)
        self.wait(5)
        self.stop_ambient_camera_rotation()

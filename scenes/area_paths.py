"""
WHEN DOES THE AREA CHANGE?  Path 1 (normal equations) and Path 2 (pseudoinverse / SVD)
Same 6 students as the paths/ scripts. One question only: which step changes area, by how much, and why.

Put in scenes\ and run from the main manimations folder:
    uv run manim -pql scenes\area_paths.py AreaPath1_NormalEquations
    uv run manim -pql scenes\area_paths.py AreaPath2_SVD
No LaTeX needed.
"""
from manim import *
import numpy as np

FS = 24

# ---- numbers from the student data (see paths/p1 and p2) -------------------
RHO = 1.974 / 6                       # correlation of z_study and z_sleep = 0.329
R = np.array([[1, RHO], [RHO, 1]])    # XᵀX / 6 (feature block): same w, smaller numbers
C_VEC = np.array([69.545, 30.290]) / 6  # Xᵀy / 6 (feature part) = (11.59, 5.05)
W = np.linalg.solve(R, C_VEC)           # = (11.135, 1.385)
SIG1, SIG3 = 2.8238, 2.0065             # singular values of the feature part of X
r2 = 1 / np.sqrt(2)
V2 = np.array([[r2, -r2], [r2, r2]])    # columns v1 = study+sleep, v3 = study−sleep (rotation 45°)


def box(text, corner=UL, fs=FS):
    t = Text(text, font_size=fs, line_spacing=0.8)
    t.add_background_rectangle(opacity=0.9, buff=0.18)
    return t.to_corner(corner).set_z_index(10)


class Narrated:
    note = None
    area_note = None

    def _fix(self, m):
        if isinstance(self, ThreeDScene):
            self.add_fixed_in_frame_mobjects(m)

    def say(self, text, corner=UL, wait=3):
        new = box(text, corner)
        self._fix(new)
        self.play(*([FadeOut(self.note)] if self.note else []), FadeIn(new))
        self.note = new
        self.wait(wait)

    def area(self, text, color=YELLOW):
        new = box(text, DR, fs=30)
        new[1:].set_color(color)
        self._fix(new)
        self.play(*([FadeOut(self.area_note)] if self.area_note else []), FadeIn(new), run_time=0.6)
        self.area_note = new


# =============================================================================
# PATH 1: the normal equations live in WEIGHT SPACE
# =============================================================================
class AreaPath1_NormalEquations(Narrated, Scene):
    def construct(self):
        k = 0.45                                  # 1 weight unit = 0.45 screen units
        O = np.array([-3.4, -1.7, 0])
        to_s = lambda v: O + k * np.array([v[0], v[1], 0])

        bg = NumberPlane(x_range=[-6, 18, 1], y_range=[-5, 12, 1], x_length=24 * k, y_length=17 * k,
                         background_line_style={"stroke_opacity": 0.25}, axis_config={"stroke_opacity": 0.5})
        bg.shift(O - bg.c2p(0, 0))
        self.add(bg)
        xl = Text("w_study", font_size=20, color=GREY_B).move_to(to_s([16.5, -0.7]))
        yl = Text("w_sleep", font_size=20, color=GREY_B).move_to(to_s([-1.4, 9.5]))
        self.add(xl, yl)

        self.say("There are TWO different spaces in this problem:\n"
                 "• STUDENT space: 6 numbers, one per student (y, Xw)\n"
                 "• WEIGHT space: (w_study, w_sleep)\n"
                 "Path 1 works entirely in WEIGHT space (shown here).", wait=4.5)

        # ---- the unit square and a live correlation ------------------------
        rho = ValueTracker(0.0)

        def M():
            return np.array([[1, rho.get_value()], [rho.get_value(), 1]])

        def shape():
            m = M()
            pts = [to_s(m @ p) for p in ([0, 0], [1, 0], [1, 1], [0, 1])]
            g = VGroup(Polygon(*pts, color=YELLOW, fill_opacity=0.5, stroke_width=3))
            for i in range(-1, 4):   # a few transformed grid lines around it
                g.add(Line(to_s(m @ [i, -1]), to_s(m @ [i, 3]), color=BLUE, stroke_width=1.5, stroke_opacity=0.7))
                g.add(Line(to_s(m @ [-1, i]), to_s(m @ [3, i]), color=BLUE, stroke_width=1.5, stroke_opacity=0.7))
            g.add(Arrow(to_s([0, 0]), to_s(m @ [1, 0]), buff=0, color=GREEN, stroke_width=5))
            g.add(Arrow(to_s([0, 0]), to_s(m @ [0, 1]), buff=0, color=RED, stroke_width=5))
            return g

        sq = always_redraw(shape)
        self.add(sq)
        self.say("Yellow = the unit square: area 1.\n"
                 "We'll watch what the matrices of path 1 do to it.", wait=2)
        self.area("area = 1")

        self.say("Path 1 builds XᵀX. Divide by m = 6 (w doesn't change):\n"
                 "R = XᵀX / 6 = [[1, ρ], [ρ, 1]]\n"
                 "ρ = correlation between study and sleep.\n"
                 "If they were unrelated (ρ = 0), R = I: nothing moves.", wait=4)
        area_live = always_redraw(lambda: box(
            f"ρ = {rho.get_value():.3f}\narea = det R = 1·1 − ρ·ρ = {1 - rho.get_value() ** 2:.3f}", DR, fs=30))
        self.play(FadeOut(self.area_note), FadeIn(area_live))
        self.area_note = None

        self.say("Our students: ρ = 0.329 (they overlap a little).\n"
                 "The square leans into a parallelogram.", wait=0.5)
        self.play(rho.animate.set_value(RHO), run_time=3)
        self.say("AREA CHANGES HERE: × det R = 1 − 0.329² = 0.892\n"
                 "Smaller than 1 because the two features overlap:\n"
                 "they carry less independent information.", wait=4)

        self.say("More overlap -> thinner parallelogram -> smaller area.", wait=0.5)
        self.play(rho.animate.set_value(0.8), run_time=2.5)
        self.wait()
        self.say("ρ = 1: study and sleep are the SAME feature.\n"
                 "The square collapses onto a line: area = det = 0.\n"
                 "No inverse -> no unique w (path 1 breaks).", wait=0.5)
        self.play(rho.animate.set_value(1.0), run_time=2.5)
        self.wait(3)
        self.play(rho.animate.set_value(RHO), run_time=2.5)
        self.wait()

        # ---- freeze, then use the inverse ---------------------------------
        frozen = shape()
        self.remove(sq, area_live)
        self.add(frozen)
        self.area(f"area = {1 - RHO ** 2:.3f}")
        self.say("Now the inverse R⁻¹ (what path 1 uses to get w).\n"
                 "It undoes R, so it must multiply area by\n"
                 f"1 / det R = 1 / {1 - RHO ** 2:.3f} = {1 / (1 - RHO ** 2):.3f}", wait=1)
        Rinv = np.linalg.inv(R)
        self.play(ApplyMatrix(Rinv, frozen, about_point=O), run_time=3)
        self.area("area = 0.892 × 1.121 = 1")
        self.say("Back to the unit square: R⁻¹R = I, area 1 again.\n"
                 "Every matrix multiplies ALL areas by its det.", wait=3)

        # ---- where w comes from -------------------------------------------
        self.play(FadeOut(frozen))
        grid = NumberPlane(x_range=[-6, 18, 1], y_range=[-5, 12, 1], x_length=24 * k, y_length=17 * k,
                           background_line_style={"stroke_color": BLUE, "stroke_width": 1.5, "stroke_opacity": 0.6},
                           axis_config={"stroke_opacity": 0})
        grid.shift(O - grid.c2p(0, 0))
        unit = Polygon(*[to_s(p) for p in ([0, 0], [1, 0], [1, 1], [0, 1])], color=YELLOW, fill_opacity=0.5)
        c_arrow = Arrow(to_s([0, 0]), to_s(C_VEC), buff=0, color=ORANGE, stroke_width=6)
        c_lbl = Text("Xᵀy/6 = (11.59, 5.05)", font_size=22, color=ORANGE).next_to(c_arrow.get_end(), UR, buff=0.1)
        self.play(FadeIn(grid), FadeIn(unit), GrowArrow(c_arrow), FadeIn(c_lbl))
        self.area("area = 1")
        self.say("Orange = Xᵀy / 6: how strongly each feature\n"
                 "moves with the grade. Path 1: w = R⁻¹ (Xᵀy/6).\n"
                 "Apply R⁻¹ to the WHOLE weight space:", wait=1)
        self.play(FadeOut(c_lbl))
        self.play(ApplyMatrix(Rinv, grid, about_point=O), ApplyMatrix(Rinv, unit, about_point=O),
                  Transform(c_arrow, Arrow(to_s([0, 0]), to_s(W), buff=0, color=ORANGE, stroke_width=6)),
                  run_time=3.5)
        dot = Dot(to_s(W), color=RED, radius=0.09)
        w_lbl = Text(f"w = ({W[0]:.3f}, {W[1]:.3f})", font_size=22, color=RED).next_to(dot, DOWN, buff=0.15)
        self.play(FadeIn(dot), FadeIn(w_lbl))
        self.area(f"area × {1 / (1 - RHO ** 2):.3f}")
        self.say("Every area grew by 1.121 and the orange arrow landed on w.\n"
                 "The inverse 'un-leans' the overlap: sleep's share of\n"
                 "the grade link is handed back to study.", wait=4.5)

        self.say("Where your 32.10 and 192.62 come from (no division by 6):\n"
                 "feature block of XᵀX = 6·R  -> area × 6·6·0.892 = 32.10\n"
                 "full XᵀX adds the bias axis (× 6) -> VOLUME\n"
                 "× 6·6·6·0.892 = 192.62 = det(XᵀX)", wait=6)
        self.say("Path 1 summary:\n"
                 "• area changes ONLY when a matrix is applied\n"
                 "• factor = det (XᵀX squeezes because features overlap)\n"
                 "• the inverse multiplies by 1/det\n"
                 "• det = 0 (duplicate features) -> no way back", wait=6)


# =============================================================================
# PATH 2: SVD shows WHERE inside a matrix the area changes
# =============================================================================
class AreaPath2_SVD(Narrated, ThreeDScene):
    def construct(self):
        k = 0.62
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        axes = ThreeDAxes(x_range=[-4, 4], y_range=[-4, 4], z_range=[-3, 3],
                          x_length=8 * k * 1.6, y_length=8 * k * 1.6, z_length=6 * k * 1.6,
                          axis_config={"stroke_opacity": 0.5})
        floor = NumberPlane(x_range=[-6, 6], y_range=[-4, 4], background_line_style={"stroke_opacity": 0.2},
                            axis_config={"stroke_opacity": 0})
        self.add(floor, axes)

        tilt = 35 * DEGREES                      # U: place the flat picture into a tilted plane
        T = np.array([[1, 0], [0, np.cos(tilt)], [0, np.sin(tilt)]])
        E = np.array([[1, 0], [0, 1], [0, 0]])
        D = np.diag([SIG1, SIG3])
        stages = {
            "start": E,
            "Vt":    E @ V2.T,
            "D":     E @ np.diag([SIG1, SIG3]) @ V2.T,
            "U":     T @ D @ V2.T,
        }

        def make(Mx, sig=None):
            pts = [k * (Mx @ np.array(p, float)) for p in ([0, 0], [1, 0], [1, 1], [0, 1])]
            g = VGroup(Polygon(*pts, color=YELLOW, fill_opacity=0.5, stroke_width=3))
            g.add(Arrow(ORIGIN, k * (Mx @ V2[:, 0]), buff=0, color=PURPLE, stroke_width=6))
            g.add(Arrow(ORIGIN, k * (Mx @ V2[:, 1]), buff=0, color=TEAL, stroke_width=6))
            return g

        shp = make(stages["start"])
        self.say("Path 2 splits X into 3 moves:  X = U · D · Vᵀ\n"
                 "Question: which of the three changes area?\n"
                 "Yellow = unit square in WEIGHT space, area 1.\n"
                 "Purple v1 = study+sleep, teal v3 = study−sleep.", wait=1)
        self.play(FadeIn(shp))
        self.area("area = 1")
        self.wait(3)

        self.say("Move 1, Vᵀ: a ROTATION (turn v1, v3 onto the axes).\n"
                 "Rotations never stretch: det = 1.", wait=0.5)
        self.play(Transform(shp, make(stages["Vt"])), run_time=2.5)
        self.area("area = 1   (unchanged)", GREEN)
        self.wait(2)

        self.say(f"Move 2, D: stretch the axes by σ1 = {SIG1:.3f}, σ3 = {SIG3:.3f}.\n"
                 "THIS is the only move that changes area.", wait=0.5)
        self.play(Transform(shp, make(stages["D"])), run_time=2.5)
        self.area(f"area = σ1·σ3 = {SIG1:.3f}·{SIG3:.3f} = {SIG1 * SIG3:.3f}", ORANGE)
        self.say(f"{SIG1 * SIG3:.3f} = √32.10  (the 32.10 from path 1!)\n"
                 "σ² are the eigenvalues of XᵀX, so σ1²·σ3² = det = 32.10.\n"
                 "XᵀX applies X twice -> squares the area.", wait=5)

        self.say("Move 3, U: place the shape inside STUDENT space\n"
                 "(really 6-D, drawn as 3-D). Another rotation:\n"
                 "the shape tilts, the area stays.", wait=0.5)
        self.move_camera(phi=62 * DEGREES, theta=-55 * DEGREES, run_time=2.5)
        self.play(Transform(shp, make(stages["U"])), run_time=2.5)
        plane = Surface(lambda a, b: k * (T @ np.array([a, b])), u_range=[-4.5, 4.5], v_range=[-3.5, 3.5],
                        resolution=(8, 8), fill_opacity=0.18, checkerboard_colors=[BLUE_D, BLUE_E])
        self.play(Create(plane))
        self.area(f"area = {SIG1 * SIG3:.3f}   (unchanged)", GREEN)
        self.say("Blue plane = every prediction X·w can ever reach\n"
                 "(the column space). The unit square of weights\n"
                 f"became a shape of area {SIG1 * SIG3:.2f} on that plane.", wait=4)

        # ---- backwards: the pseudoinverse --------------------------------------
        p = k * (T @ np.array([1.9, 1.2]))
        n_ = np.cross(T[:, 0], T[:, 1])
        ytip = p + 1.6 * n_
        y_arrow = Arrow3D(ORIGIN, ytip, color=GREEN)
        p_arrow = Arrow3D(ORIGIN, p, color=YELLOW_A)
        drop = DashedLine(ytip, p, color=RED)
        self.say("Now path 2 runs BACKWARDS on the grades y (green, schematic).\n"
                 "y sticks OUT of the plane: no w reaches it exactly.", wait=1)
        self.play(Create(y_arrow))
        self.wait(2)
        self.say("Uᵀ: keep only the part of y lying IN the plane.\n"
                 "Uᵀy = (25.00, 13.83). The part sticking out (red) is the\n"
                 "error: it is thrown away, it can never be undone.", wait=0.5)
        self.play(Create(drop), Create(p_arrow), run_time=2)
        self.wait(3)
        self.play(FadeOut(y_arrow), FadeOut(drop), FadeOut(p_arrow), FadeOut(plane))
        self.play(Transform(shp, make(stages["D"])), run_time=2)
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=2)
        self.area(f"area = {SIG1 * SIG3:.3f}   (Uᵀ = rotation back)", GREEN)

        self.say("D⁺: divide by σ instead of multiplying.\n"
                 f"(25.00, 13.83) ÷ σ -> (8.85, 6.89)\n"
                 f"area × 1/(σ1σ3) = × {1 / (SIG1 * SIG3):.3f}", wait=0.5)
        self.play(Transform(shp, make(stages["Vt"])), run_time=2.5)
        self.area(f"area = {SIG1 * SIG3:.3f} × {1 / (SIG1 * SIG3):.3f} = 1", ORANGE)
        self.wait(2)
        self.say("V: rotate back. 8.85·v1 + 6.89·v3 = (11.135, 1.385) = w\n"
                 "Area unchanged. Same w as path 1.", wait=0.5)
        self.play(Transform(shp, make(stages["start"])), run_time=2.5)
        self.area("area = 1   (unchanged)", GREEN)
        self.wait(3)

        # ---- what if a σ were 0? -------------------------------------------------
        self.say("Why path 2 never crashes: suppose σ3 = 0\n"
                 "(study and sleep identical).", wait=1)
        self.play(Transform(shp, make(E @ np.diag([SIG1, 0.0]) @ V2.T)), run_time=2.5)
        self.area("area = σ1 · 0 = 0", RED)
        self.say("Path 1 would need 1/0 here and fail.\n"
                 "D⁺ just writes 0 instead of 1/0: that direction is\n"
                 "skipped (w gets the smallest possible size there).", wait=4.5)

        self.say("Path 2 summary:\n"
                 "• Vᵀ, U, V (rotations): area NEVER changes\n"
                 "• D: area × σ1·σ2 · · ·   D⁺: area × 1/(σ1·σ2 · · ·)\n"
                 "• Uᵀ (projection): drops the part of y off the plane = error\n"
                 "• det(XᵀX) = (σ1·σ2 · · ·)²: path 1 sees the square of it", wait=7)

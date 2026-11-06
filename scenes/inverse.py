"""
How the inverse matrix A^-1 works — step-by-step visual explanations
(companion to Goodfellow et al., Deep Learning, §2.3–2.4)

Run from your manimations folder:
    uv run manim -pql inverse.py UndoA            # A^-1 undoes A  (A^-1 A = I and A A^-1 = I)
    uv run manim -pql inverse.py SolveAxB         # solving Ax = b with x = A^-1 b
    uv run manim -pql inverse.py InverseColumns   # what the columns of A^-1 mean + 2x2 formula
    uv run manim -pql inverse.py AreaAndDet       # determinant, 1/det, and why singular = no inverse

Render them all at once:
    uv run manim -pql -a inverse.py

Things to try: change A / A_INV at the top (keep them true inverses —
check with numpy: np.linalg.inv(A)), or change the vector X in SolveAxB.
"""
from manim import *
import numpy as np

# The main example matrix and its inverse (det = 2*1 - 1*1 = 1)
A = np.array([[2, 1], [1, 1]])
A_INV = np.linalg.inv(A)          # = [[1, -1], [-1, 2]]

FS = 26  # caption font size


def box(text, corner=UL):
    t = Text(text, font_size=FS, line_spacing=0.8)
    t.add_background_rectangle(opacity=0.88, buff=0.18)
    return t.to_corner(corner).set_z_index(10)


class Base(Scene):
    """Faint fixed reference grid + a blue grid that matrices move."""

    def setup_grid(self, basis=True):
        self.add(NumberPlane(background_line_style={"stroke_opacity": 0.25},
                             axis_config={"stroke_opacity": 0.4}))
        self.grid = NumberPlane(x_range=[-14, 14], y_range=[-14, 14],
                                background_line_style={"stroke_color": BLUE, "stroke_width": 2},
                                axis_config={"stroke_opacity": 0.6})
        self.add(self.grid)
        self.movers = [self.grid]
        if basis:
            self.i_hat = Vector([1, 0], color=GREEN)
            self.j_hat = Vector([0, 1], color=RED)
            self.add(self.i_hat, self.j_hat)
            self.movers += [self.i_hat, self.j_hat]
        self.note = None

    def say(self, text, corner=UL, wait=2.5):
        new = box(text, corner)
        if self.note is None:
            self.play(FadeIn(new))
        else:
            self.play(FadeOut(self.note), FadeIn(new))
        self.note = new
        self.wait(wait)

    def apply(self, matrix, extra=(), run_time=2.5):
        m = np.array(matrix, dtype=float)
        anims = []
        for mob in self.movers + list(extra):
            if isinstance(mob, Arrow):
                # move arrows by their tip so the arrowhead isn't distorted
                new_tip = m @ mob.get_end()[:2]
                anims.append(Transform(mob, Vector(new_tip, color=mob.get_color())))
            else:
                anims.append(ApplyMatrix(m, mob))
        self.play(*anims, run_time=run_time)
        self.wait(0.5)


# ---------------------------------------------------------------------------
# 1. A^-1 undoes A
# ---------------------------------------------------------------------------
class UndoA(Base):
    def construct(self):
        self.setup_grid()
        self.say("Step 1: the plane before any matrix.\n"
                 "Green = i-hat (1, 0), red = j-hat (0, 1).\n"
                 "Every point x on the grid is an input.")
        self.say("Step 2: apply A = [[2, 1], [1, 1]].\n"
                 "Watch where green and red land:\n"
                 "they become the COLUMNS of A.")
        self.apply(A)
        self.say("green -> (2, 1) = column 1 of A\n"
                 "red   -> (1, 1) = column 2 of A\n"
                 "Grid is stretched + sheared, but it still\n"
                 "covers the whole plane (columns independent).")
        self.say("Step 3: apply A⁻¹ = [[1, -1], [-1, 2]].\n"
                 "If it is really the inverse, it must\n"
                 "put every point back where it started.")
        self.apply(A_INV)
        self.say("Back to the original grid!\n"
                 "A⁻¹(Ax) = x for EVERY x, i.e.  A⁻¹A = I\n"
                 "(I = identity matrix = 'do nothing').")
        self.say("Book eq. 2.29: the other order works too.\n"
                 "Apply A⁻¹ FIRST, then A ...")
        self.apply(A_INV)
        self.say("A⁻¹ alone moves the grid somewhere else.\n"
                 "Now apply A on top of it ...", wait=1.5)
        self.apply(A)
        self.say("... and we are back again:  A A⁻¹ = I.\n"
                 "For square matrices the left inverse and\n"
                 "the right inverse are the same matrix.", wait=4)


# ---------------------------------------------------------------------------
# 2. Solving Ax = b with x = A^-1 b
# ---------------------------------------------------------------------------
class SolveAxB(Base):
    X = np.array([1, 2])

    def construct(self):
        self.setup_grid(basis=False)
        b = A @ self.X
        x_vec = Vector(self.X, color=YELLOW)
        x_lbl = Text("x", font_size=30, color=YELLOW).next_to(x_vec.get_end(), RIGHT, buff=0.15)

        self.say("The problem: Ax = b.\n"
                 "We know A and b, we want the input x.\n"
                 "First, let's go FORWARD with x = (1, 2).")
        self.play(GrowArrow(x_vec), FadeIn(x_lbl))
        self.wait()
        self.say("Ax = 1·column1 + 2·column2\n"
                 "   = 1·(2, 1) + 2·(1, 1) = (4, 3) = b\n"
                 "Watch x get carried along by A:")
        self.play(FadeOut(x_lbl))
        self.apply(A, extra=[x_vec])

        b_dot = Dot([*b, 0], color=ORANGE, radius=0.1).set_z_index(5)
        b_lbl = Text(f"b = ({b[0]}, {b[1]})", font_size=28, color=ORANGE)
        b_lbl.next_to(b_dot, RIGHT, buff=0.15).add_background_rectangle(opacity=0.8)
        self.play(FadeIn(b_dot), FadeIn(b_lbl))
        self.say("x landed on b = (4, 3).\n"
                 "Now the real question: if someone only\n"
                 "gives us b, which x was it?")
        self.say("Answer: apply A⁻¹ to the whole plane.\n"
                 "Since A⁻¹ undoes A, whatever sits at b\n"
                 "travels back to its original x.")
        self.apply(A_INV, extra=[x_vec])
        x_lbl2 = Text("x = (1, 2)", font_size=28, color=YELLOW)
        x_lbl2.next_to(x_vec.get_end(), LEFT, buff=0.15).add_background_rectangle(opacity=0.8)
        self.play(FadeIn(x_lbl2))
        self.say("x = A⁻¹ b = [[1, -1], [-1, 2]] · (4, 3)\n"
                 "  = (1·4 - 1·3,  -1·4 + 2·3) = (1, 2)   (correct)\n"
                 "This is the book's 'matrix inversion' method.", wait=4)


# ---------------------------------------------------------------------------
# 3. What the columns of A^-1 mean
# ---------------------------------------------------------------------------
class InverseColumns(Base):
    def construct(self):
        self.setup_grid(basis=False)
        # Faint targets: i-hat and j-hat stay fixed on screen
        i_t = Vector([1, 0], color=GREEN, stroke_opacity=0.6)
        j_t = Vector([0, 1], color=RED, stroke_opacity=0.6)
        self.add(i_t, j_t)

        u = Vector(A_INV[:, 0], color=ORANGE)   # column 1 of A^-1 = (1, -1)
        w = Vector(A_INV[:, 1], color=PURPLE)   # column 2 of A^-1 = (-1, 2)

        self.say("Remember: columns of A = where A sends\n"
                 "i-hat (green) and j-hat (red).\n"
                 "So what are the columns of A⁻¹?", DL)
        self.say("A⁻¹ goes the other way. Its columns are\n"
                 "the vectors that A sends TO i-hat and j-hat.", DL)
        self.play(GrowArrow(u), GrowArrow(w))
        self.say("orange (1, -1) = column 1 of A⁻¹\n"
                 "purple (-1, 2) = column 2 of A⁻¹\n"
                 "Prediction: A moves them onto green and red.", DL)
        self.apply(A, extra=[u, w])
        self.say("A·(1, -1) = (2-1, 1-1) = (1, 0) = i-hat  (correct)\n"
                 "A·(-1, 2) = (-2+2, -1+2) = (0, 1) = j-hat  (correct)\n"
                 "So A·A⁻¹ = [i-hat  j-hat] = I.", DL, wait=3.5)
        self.say("2x2 shortcut:  A = [[a, b], [c, d]]\n"
                 "A⁻¹ = (1/det) · [[d, -b], [-c, a]],  det = ad - bc\n"
                 "Here det = 2·1 - 1·1 = 1 -> A⁻¹ = [[1, -1], [-1, 2]]\n"
                 "If det = 0 you'd divide by zero -> no inverse.", DL, wait=5)


# ---------------------------------------------------------------------------
# 4. Determinant, 1/det, and why singular means no inverse
# ---------------------------------------------------------------------------
class AreaAndDet(Base):
    def construct(self):
        self.setup_grid()
        B = np.array([[3, 1], [1, 1]])        # det = 2
        B_INV = np.linalg.inv(B)              # = 1/2 * [[1, -1], [-1, 3]]
        sq = Polygon([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     color=YELLOW, fill_opacity=0.45, stroke_width=3)

        self.say("Another view: the DETERMINANT.\n"
                 "Yellow = unit square, area 1.")
        self.play(FadeIn(sq))
        self.say("Apply A = [[3, 1], [1, 1]].\n"
                 "det = 3·1 - 1·1 = 2, so areas should double.")
        self.apply(B, extra=[sq])
        self.say("The square became a parallelogram of area 2.\n"
                 "EVERY shape in the plane got 2x bigger.")
        self.say("A⁻¹ must shrink areas by 1/det = 1/2.\n"
                 "A⁻¹ = (1/2)·[[1, -1], [-1, 3]]")
        self.apply(B_INV, extra=[sq])
        self.say("Area 2 -> area 1: back to the unit square.\n"
                 "det(A⁻¹) = 1 / det(A).")
        self.say("Now a SINGULAR matrix: [[1, 1], [1, 1]].\n"
                 "det = 1·1 - 1·1 = 0 (columns identical).")
        self.apply([[1, 1], [1, 1]], extra=[sq])
        self.say("Area 0: the whole plane is flattened onto a line.\n"
                 "To undo it, A⁻¹ would need to scale by 1/0.\n"
                 "Impossible -> singular matrices have NO inverse.", wait=3)
        self.say("Intuition: many different x landed on the\n"
                 "same point, so there's no way to know which\n"
                 "one to send back. Information was lost.", wait=4)

"""
Special matrices and vectors — Goodfellow et al., Deep Learning, §2.6
diagonal, rectangular diagonal, symmetric, unit vectors, orthogonal/orthonormal, orthogonal matrices

Put this file in your scenes\ folder and run from the main manimations folder:
    uv run manim -pql scenes\special_matrices.py DiagonalScaling       # diag(v) x = v ⊙ x
    uv run manim -pql scenes\special_matrices.py DiagonalInverse       # diag(v)^-1 = diag(1/v) ... unless a 0
    uv run manim -pql scenes\special_matrices.py RectangularDiagonal   # tall: adds zeros, wide: drops entries
    uv run manim -pql scenes\special_matrices.py Symmetric             # A = Aᵀ, distance matrix example
    uv run manim -pql scenes\special_matrices.py UnitAndOrthogonal     # unit vectors, orthogonal, orthonormal
    uv run manim -pql scenes\special_matrices.py OrthogonalMatrix      # rotation: Q⁻¹ = Qᵀ, and the trap

All scenes:  uv run manim -pql -a scenes\special_matrices.py
No LaTeX needed.
"""
from manim import *
import numpy as np

FS = 24


def box(text, corner=UL, fs=FS):
    t = Text(text, font_size=fs, line_spacing=0.8)
    t.add_background_rectangle(opacity=0.88, buff=0.18)
    return t.to_corner(corner).set_z_index(10)


def bracketed(entries, ncols, fs=30, colors=None):
    cells = VGroup(*[Text(str(e), font_size=fs, color=(colors[i] if colors else WHITE))
                     for i, e in enumerate(entries)])
    cells.arrange_in_grid(cols=ncols, buff=(0.5, 0.22))
    h = cells.height + 0.25

    def bracket(side):
        d = 0.15 * side
        return VMobject(stroke_width=3).set_points_as_corners(
            [UP * h / 2 + RIGHT * d, UP * h / 2, DOWN * h / 2, DOWN * h / 2 + RIGHT * d])

    return VGroup(bracket(1).next_to(cells, LEFT, buff=0.12), cells,
                  bracket(-1).next_to(cells, RIGHT, buff=0.12))


class Base(Scene):
    note = None

    def say(self, text, corner=UL, wait=3):
        new = box(text, corner)
        if self.note is None:
            self.play(FadeIn(new))
        else:
            self.play(FadeOut(self.note), FadeIn(new))
        self.note = new
        self.wait(wait)


class GridBase(Base):
    """Faint fixed grid + a blue grid, i-hat, j-hat that matrices move."""

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

    def apply(self, matrix, extra=(), run_time=2.5):
        m = np.array(matrix, dtype=float)
        anims = []
        for mob in self.movers + list(extra):
            if isinstance(mob, Arrow):
                new_tip = m @ mob.get_end()[:2]
                anims.append(Transform(mob, Vector(new_tip, color=mob.get_color())))
            else:
                anims.append(ApplyMatrix(m, mob))
        self.play(*anims, run_time=run_time)
        self.wait(0.5)


# ---------------------------------------------------------------------------
# 1. Diagonal matrix = scale each axis on its own
# ---------------------------------------------------------------------------
class DiagonalScaling(GridBase):
    def construct(self):
        self.setup_grid()
        self.say("A DIAGONAL matrix has numbers only on the\n"
                 "diagonal, zeros everywhere else:\n"
                 "D = diag(v) = [[2, 0], [0, 0.5]],  v = (2, 0.5)\n"
                 "(the identity I = diag(1, 1) is one too)", wait=4)
        x = Vector([1.5, 2], color=YELLOW)
        self.play(GrowArrow(x))
        self.say("Take x = (1.5, 2) and apply D.\n"
                 "Watch: the grid is only STRETCHED along the\n"
                 "axes. Nothing is rotated or sheared.", wait=1)
        self.apply([[2, 0], [0, 0.5]], extra=[x])
        self.say("green (x-axis) × 2,  red (y-axis) × 0.5\n"
                 "x = (1.5, 2)  ->  (2·1.5, 0.5·2) = (3, 1)\n"
                 "Each entry is just multiplied by its own v_i:\n"
                 "diag(v) x = v ⊙ x  (element-wise product)", wait=4.5)
        self.say("Why the book cares: SPEED.\n"
                 "Full n×n matrix × vector: n² multiplications.\n"
                 "Diagonal: only n (skip all the zeros).\n"
                 "n = 1000 -> 1,000 instead of 1,000,000.", wait=4.5)
        self.say("ML example: rescaling features.\n"
                 "Student = (study hours, sleep hours).\n"
                 "diag(1/10, 1/8) puts both on a 0-1 scale\n"
                 "without mixing them together.", wait=4.5)


# ---------------------------------------------------------------------------
# 2. Inverse of a diagonal matrix
# ---------------------------------------------------------------------------
class DiagonalInverse(GridBase):
    def construct(self):
        self.setup_grid()
        self.say("Inverting a diagonal matrix is just as easy.\n"
                 "D = diag(2, 0.5) stretches x by 2, y by 0.5.", wait=1)
        self.apply([[2, 0], [0, 0.5]])
        self.say("To undo it: shrink x by 1/2, stretch y by 1/0.5 = 2.\n"
                 "D⁻¹ = diag(1/2, 1/0.5) = diag(0.5, 2)\n"
                 "No formula, no determinant: flip each number.", wait=1)
        self.apply([[0.5, 0], [0, 2]])
        self.say("Back to the original grid: D⁻¹ D = I.\n"
                 "Rule: diag(v)⁻¹ = diag(1/v1, ..., 1/vn)", wait=3)
        self.say("Catch: what if one diagonal entry is 0?\n"
                 "D = diag(2, 0)", wait=1)
        self.apply([[2, 0], [0, 0]])
        self.say("The whole plane is flattened onto the x-axis.\n"
                 "To undo it we'd need 1/0. Impossible.\n"
                 "So: diag(v) is invertible ONLY if every\n"
                 "v_i ≠ 0.", wait=4.5)


# ---------------------------------------------------------------------------
# 3. Rectangular (non-square) diagonal matrices
# ---------------------------------------------------------------------------
class RectangularDiagonal(Base):
    def construct(self):
        self.say("Diagonal matrices don't have to be square.\n"
                 "Still only the 'diagonal' positions (i = j)\n"
                 "can be nonzero.", wait=3)

        # Tall: 3x2
        D = bracketed([2, 0, 0, 3, 0, 0], 2, fs=34, colors=[YELLOW, GREY, GREY, YELLOW, GREY, GREY])
        x = bracketed(["a", "b"], 1, fs=34, colors=[TEAL, TEAL])
        r = bracketed(["2a", "3b", "0"], 1, fs=34, colors=[TEAL, TEAL, RED])
        eq = VGroup(Text("TALL  (3×2):", font_size=28), D, x, Text("=", font_size=40), r)
        eq.arrange(RIGHT, buff=0.35).move_to(0.7 * UP + 0.5 * RIGHT)
        self.play(FadeIn(eq[:4]))
        self.wait()
        self.play(FadeIn(eq[4]))
        self.say("TALL (more rows than columns):\n"
                 "scale each entry (a->2a, b->3b), then\n"
                 "ADD ZEROS at the end. 2 numbers in, 3 out.", wait=4)

        # Wide: 2x3
        D2 = bracketed([2, 0, 0, 0, 3, 0], 3, fs=34, colors=[YELLOW, GREY, GREY, GREY, YELLOW, GREY])
        x2 = bracketed(["a", "b", "c"], 1, fs=34, colors=[TEAL, TEAL, RED])
        r2 = bracketed(["2a", "3b"], 1, fs=34, colors=[TEAL, TEAL])
        eq2 = VGroup(Text("WIDE  (2×3):", font_size=28), D2, x2, Text("=", font_size=40), r2)
        eq2.arrange(RIGHT, buff=0.35).move_to(2.0 * DOWN + 0.5 * RIGHT)
        self.play(FadeIn(eq2[:4]))
        self.wait()
        cross = Cross(eq2[2][1][2], stroke_width=5)
        self.play(FadeIn(eq2[4]), Create(cross))
        self.say("WIDE (more columns than rows):\n"
                 "scale each entry, and the LAST entry c\n"
                 "is DISCARDED. 3 numbers in, 2 out.", wait=4)
        self.say("Neither has an inverse: the tall one can't\n"
                 "reach outputs with a nonzero 3rd entry, the\n"
                 "wide one forgets c forever.\n"
                 "But multiplying by them is still very cheap.", wait=5)


# ---------------------------------------------------------------------------
# 4. Symmetric matrices: A = Aᵀ
# ---------------------------------------------------------------------------
class Symmetric(Base):
    def construct(self):
        cities = ["Athens", "Thessaloniki", "Patras"]
        dist = [[0, 300, 180], [300, 0, 340], [180, 340, 0]]   # rough straight-line km
        cells = VGroup(*[Text(str(dist[i][j]), font_size=34) for i in range(3) for j in range(3)])
        cells.arrange_in_grid(rows=3, cols=3, buff=(0.9, 0.45)).move_to(1.2 * RIGHT + 0.6 * DOWN)
        rlab = VGroup(*[Text(c, font_size=24, color=GREY_B).next_to(cells[3 * i], LEFT, buff=0.6)
                        for i, c in enumerate(cities)])
        clab = VGroup(*[Text(c, font_size=22, color=GREY_B).next_to(cells[j], UP, buff=0.45)
                        for j, c in enumerate(cities)])
        cell = lambda i, j: cells[3 * i + j]

        self.say("A matrix of DISTANCES (≈ km, straight line).\n"
                 "Entry (i, j) = distance from city i to city j.", wait=1)
        self.play(FadeIn(cells), FadeIn(rlab), FadeIn(clab))
        self.wait(2)

        pairs = [((0, 1), (1, 0), YELLOW), ((0, 2), (2, 0), TEAL), ((1, 2), (2, 1), ORANGE)]
        self.say("Athens -> Thessaloniki = Thessaloniki -> Athens.\n"
                 "So entry (i, j) always equals entry (j, i).", wait=0.5)
        for a, b, col in pairs:
            self.play(Create(SurroundingRectangle(cell(*a), color=col)),
                      Create(SurroundingRectangle(cell(*b), color=col)), run_time=0.8)
        diag = DashedLine(cell(0, 0).get_center() + 0.4 * UL, cell(2, 2).get_center() + 0.4 * DR,
                          color=WHITE)
        self.play(Create(diag))
        self.say("The matrix is a MIRROR image across its diagonal.\n"
                 "Transpose = flip across that diagonal (rows <-> cols).\n"
                 "Watch: flipping changes nothing.", wait=1)
        moves = [cells[3 * i + j].animate.move_to(cell(j, i).get_center())
                 for i in range(3) for j in range(3)]
        self.play(*moves, run_time=2.5)
        self.say("Same matrix after the flip: A = Aᵀ.\n"
                 "That's the definition of SYMMETRIC (eq. 2.35).\n"
                 "Happens whenever entry = f(i, j) and order\n"
                 "doesn't matter: distance, similarity, ...", wait=4.5)
        self.say("ML example you already met: XᵀX from least\n"
                 "squares is ALWAYS symmetric. Entry (i, j) =\n"
                 "column i · column j, and a dot product doesn't\n"
                 "care about order.", wait=5)


# ---------------------------------------------------------------------------
# 5. Unit vectors, orthogonal, orthonormal
# ---------------------------------------------------------------------------
class UnitAndOrthogonal(Base):
    def construct(self):
        plane = NumberPlane(x_range=[-4, 4], y_range=[-3, 3], x_length=8 * 1.15, y_length=6 * 1.15,
                            background_line_style={"stroke_opacity": 0.35}).to_edge(RIGHT, buff=0.2)
        self.add(plane)
        O = plane.c2p(0, 0)
        circle = Circle(radius=plane.c2p(1, 0)[0] - O[0], color=WHITE, stroke_width=3).move_to(O)

        self.say("UNIT vector: length (L2 norm) exactly 1.\n"
                 "All of them lie on this circle (eq. 2.36).", wait=1)
        self.play(Create(circle))
        self.wait()

        v = np.array([3.0, 2.0])
        va = Arrow(O, plane.c2p(*v), buff=0, color=YELLOW)
        self.play(GrowArrow(va))
        self.say("x = (3, 2) has length sqrt(13) ≈ 3.61.\n"
                 "Divide by its length -> x / ||x||\n"
                 "same DIRECTION, length 1 ('normalizing').", wait=1)
        u = v / np.linalg.norm(v)
        self.play(Transform(va, Arrow(O, plane.c2p(*u), buff=0, color=YELLOW)), run_time=2)
        self.say("Now x/||x|| ≈ (0.83, 0.55): it sits on the circle.\n"
                 "ML: normalized embeddings, so only direction counts\n"
                 "(that's what cosine similarity compares).", wait=4)

        w = np.array([-2.0, 3.0])
        wa = Arrow(O, plane.c2p(*w), buff=0, color=TEAL)
        sq = Square(side_length=0.3, color=WHITE).move_to(O, aligned_edge=DL).rotate(
            np.arctan2(u[1], u[0]), about_point=O)
        self.say("ORTHOGONAL: x · y = 0.\n"
                 "y = (-2, 3):  3·(-2) + 2·3 = 0  -> 90° apart.", wait=1)
        self.play(GrowArrow(wa), Create(sq))
        self.wait(2)
        self.say("ORTHONORMAL = orthogonal AND both unit length.\n"
                 "Shrink y onto the circle too:", wait=1)
        self.play(Transform(wa, Arrow(O, plane.c2p(*(w / np.linalg.norm(w))), buff=0, color=TEAL)),
                  run_time=2)
        self.say("Two perpendicular arrows on the circle:\n"
                 "an orthonormal pair, like i-hat and j-hat\n"
                 "but rotated.", wait=3)
        self.say("In 2D you can't add a 3rd arrow at 90° to both.\n"
                 "In Rⁿ at most n vectors can be mutually\n"
                 "orthogonal (with nonzero length).", wait=4)


# ---------------------------------------------------------------------------
# 6. Orthogonal matrices: Q⁻¹ = Qᵀ (and the trap)
# ---------------------------------------------------------------------------
class OrthogonalMatrix(GridBase):
    def construct(self):
        self.setup_grid()
        th = 30 * DEGREES
        c, s = np.cos(th), np.sin(th)
        Q = [[c, -s], [s, c]]
        QT = [[c, s], [-s, c]]
        sq = Polygon([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     color=YELLOW, fill_opacity=0.35, stroke_width=3)
        self.add(sq)

        self.say("An ORTHOGONAL matrix: its columns are an\n"
                 "ORTHONORMAL set. Example: rotation by 30°\n"
                 "Q = [[0.87, -0.5], [0.5, 0.87]]", wait=3.5)
        self.apply(Q, extra=[sq])
        self.say("Columns of Q = where green and red land:\n"
                 "(0.87, 0.5) and (-0.5, 0.87).\n"
                 "Both length 1, at 90°: orthonormal.\n"
                 "The square only TURNED: no stretch, no squish.", wait=4.5)
        self.say("Inverse = turn back by 30°. That's just the\n"
                 "TRANSPOSE: Qᵀ = [[0.87, 0.5], [-0.5, 0.87]]\n"
                 "(rows and columns swapped).", wait=1)
        self.apply(QT, extra=[sq])
        self.say("Back to the start: QᵀQ = I, so Q⁻¹ = Qᵀ\n"
                 "(eq. 2.37-2.38). No elimination, no det:\n"
                 "inverting = swapping indexes. Super cheap.", wait=4.5)

        self.say("The trap the book warns about: columns that are\n"
                 "orthogonal but NOT unit length.\n"
                 "A = [[2, 0], [0, 1]]: columns (2,0) and (0,1),\n"
                 "at 90°, but the first has length 2.", wait=4)
        self.apply([[2, 0], [0, 1]], extra=[sq])
        self.say("Try 'undo with the transpose': Aᵀ = A here.", wait=1)
        self.apply([[2, 0], [0, 1]], extra=[sq])
        self.say("NOT back! The square is now 4× wide.\n"
                 "AᵀA = diag(4, 1) ≠ I.\n"
                 "Q⁻¹ = Qᵀ only works when columns are\n"
                 "orthogonal AND length 1 (orthonormal).", wait=5)

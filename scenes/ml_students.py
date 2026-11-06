"""
WHY machine learning needs vectors, matrices, Ax = b, inverses and norms
A real problem, step by step: predicting students' final grades.

Run from your manimations folder (watch them in this order):
    uv run manim -pql ml_students.py S1_StudentAsVector     # a student -> a vector (features)
    uv run manim -pql ml_students.py S2_ClassAsMatrix       # a class   -> a matrix X + grades y = training set
    uv run manim -pql ml_students.py S3_PredictionIsXw      # the model: dot product, then X w = y  (it's Ax = b!)
    uv run manim -pql ml_students.py S4_WhyInverseFails     # solve with X^-1 ... until real data breaks it
    uv run manim -pql ml_students.py S5_TrainingIsANorm     # training = make ||error|| as small as possible

Or all at once:  uv run manim -pql -a ml_students.py
No LaTeX needed.
"""
from manim import *
import numpy as np

FS = 24

# ---- the data: 6 students -------------------------------------------------
NAMES = ["Maria", "Nikos", "Eleni", "Giorgos", "Anna", "Kostas"]
STUDY = np.array([2, 4, 5, 7, 8, 10])      # hours studied per week
SLEEP = np.array([6, 7, 5, 8, 6, 7])       # hours slept per night
GRADE = np.array([52, 63, 64, 76, 77, 88])  # final grade /100  (what we want to predict)


def box(text, corner=UL, fs=FS):
    t = Text(text, font_size=fs, line_spacing=0.8)
    t.add_background_rectangle(opacity=0.88, buff=0.18)
    return t.to_corner(corner).set_z_index(10)


def bracketed(entries, ncols, fs=30, color=WHITE):
    """A matrix / vector drawn with Text entries and square brackets."""
    cells = VGroup(*[Text(str(e), font_size=fs, color=color) for e in entries])
    cells.arrange_in_grid(cols=ncols, buff=(0.45, 0.2))
    h = cells.height + 0.25

    def bracket(side):
        d = 0.15 * side
        b = VMobject(stroke_width=3).set_points_as_corners(
            [UP * h / 2 + RIGHT * d, UP * h / 2, DOWN * h / 2, DOWN * h / 2 + RIGHT * d])
        return b

    left = bracket(1).next_to(cells, LEFT, buff=0.12)
    right = bracket(-1).next_to(cells, RIGHT, buff=0.12)
    return VGroup(left, cells, right)


def tick_labels(ax, xs, ys, fs=18):
    g = VGroup()
    for v in xs:
        g.add(Text(str(v), font_size=fs).next_to(ax.c2p(v, ax.y_range[0]), DOWN, buff=0.12))
    for v in ys:
        g.add(Text(str(v), font_size=fs).next_to(ax.c2p(ax.x_range[0], v), LEFT, buff=0.12))
    return g


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


# ---------------------------------------------------------------------------
# S1. A student becomes a vector
# ---------------------------------------------------------------------------
class S1_StudentAsVector(Base):
    def construct(self):
        self.say("THE PROBLEM: a school wants to predict\n"
                 "each student's final grade BEFORE the exam,\n"
                 "so it can help the ones at risk.", wait=3.5)

        card = VGroup(
            Text("Maria", font_size=36, color=YELLOW),
            Text("hours studied per week:  2", font_size=28),
            Text("hours of sleep per night:  6", font_size=28),
            Text("likes cats, plays guitar, ...", font_size=28, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        card.add_to_back(SurroundingRectangle(card, color=GREY_B, buff=0.3, corner_radius=0.1))
        card.move_to(3.2 * LEFT + 0.3 * DOWN)
        self.play(FadeIn(card))
        self.say("Step 1: describe a student with numbers.\n"
                 "We choose FEATURES: facts we think affect\n"
                 "the grade. Cats and guitar? Ignored.", wait=3.5)
        self.play(card[4].animate.set_opacity(0.2))

        vec = bracketed([2, 6], 1, fs=40, color=YELLOW).move_to(4.3 * RIGHT + 0.3 * DOWN)
        vname = Text("x_Maria =", font_size=32, color=YELLOW).next_to(vec, LEFT)
        flabels = VGroup(Text("← study", font_size=24), Text("← sleep", font_size=24))
        flabels[0].next_to(vec[1][0], RIGHT, buff=0.6)
        flabels[1].next_to(vec[1][1], RIGHT, buff=0.6)
        self.play(TransformFromCopy(card[2][-1], vec[1][0]), TransformFromCopy(card[3][-1], vec[1][1]),
                  FadeIn(vec[0]), FadeIn(vec[2]), FadeIn(vname), FadeIn(flabels))
        self.say("The computer can't read 'Maria'. It gets an\n"
                 "ordered list of numbers: [study, sleep].\n"
                 "That list is a VECTOR. Each slot = 1 feature.\n"
                 "The ORDER is fixed for every student.", wait=4)
        self.play(FadeOut(card), VGroup(vec, vname, flabels).animate.scale(0.7).to_corner(UR))

        # A vector is a point in "student space"
        ax = Axes(x_range=[0, 11, 1], y_range=[0, 10, 1], x_length=6.5, y_length=5,
                  tips=False).move_to(1.6 * RIGHT + 0.6 * DOWN)
        xt = Text("hours studied / week", font_size=20).next_to(ax.x_axis, DOWN, buff=0.45)
        yt = Text("sleep / night", font_size=20).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45)
        self.play(Create(ax), FadeIn(tick_labels(ax, [2, 4, 6, 8, 10], [2, 4, 6, 8])), FadeIn(xt), FadeIn(yt))
        m = Arrow(ax.c2p(0, 0), ax.c2p(2, 6), buff=0, color=YELLOW)
        md = Dot(ax.c2p(2, 6), color=YELLOW)
        ml = Text("Maria (2, 6)", font_size=22, color=YELLOW).next_to(md, UL, buff=0.1)
        self.play(GrowArrow(m), FadeIn(md), FadeIn(ml))
        self.say("A vector is also a POINT. Maria now has\n"
                 "a location in 'student space':\n"
                 "2 steps right (study), 6 steps up (sleep).", wait=3.5)

        n = Dot(ax.c2p(4, 7), color=TEAL)
        nl = Text("Nikos (4, 7)", font_size=22, color=TEAL).next_to(n, UR, buff=0.1)
        k = Dot(ax.c2p(10, 7), color=ORANGE)
        kl = Text("Kostas (10, 7)", font_size=22, color=ORANGE).next_to(k, UP, buff=0.1)
        self.play(FadeIn(n), FadeIn(nl), FadeIn(k), FadeIn(kl))
        d1 = DashedLine(md.get_center(), n.get_center(), color=TEAL)
        d2 = DashedLine(md.get_center(), k.get_center(), color=ORANGE)
        self.play(Create(d1), Create(d2))
        self.say("Similar students = nearby points.\n"
                 "Distance = NORM of the difference:\n"
                 "||Maria - Nikos||  = ||(-2, -1)|| ≈ 2.2\n"
                 "||Maria - Kostas|| = ||(-8, -1)|| ≈ 8.1\n"
                 "So Maria is 'like' Nikos, not Kostas.", wait=5)


# ---------------------------------------------------------------------------
# S2. A class becomes a matrix X (+ the answers y) = the training set
# ---------------------------------------------------------------------------
class S2_ClassAsMatrix(Base):
    def construct(self):
        header = ["student", "study", "sleep", "grade"]
        rows = [[NAMES[i], STUDY[i], SLEEP[i], GRADE[i]] for i in range(6)]
        cells = VGroup(*[Text(str(v), font_size=28, weight=BOLD if r == 0 else NORMAL)
                         for r, row in enumerate([header] + rows) for v in row])
        cells.arrange_in_grid(rows=7, cols=4, buff=(0.8, 0.28)).move_to(0.6 * DOWN)
        col = lambda j: VGroup(*[cells[r * 4 + j] for r in range(7)])
        self.say("Step 2: last year's class. For each student\n"
                 "we know the features AND the real grade.")
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.02))
        self.wait()

        self.say("Every row is one student = one vector.\n"
                 "Stack the vectors as ROWS -> a MATRIX.", wait=1)
        rbox = SurroundingRectangle(VGroup(*[cells[4 + j] for j in range(4)]), color=YELLOW)
        self.play(Create(rbox))
        for r in range(2, 7):
            self.play(rbox.animate.become(SurroundingRectangle(
                VGroup(*[cells[r * 4 + j] for j in range(4)]), color=YELLOW)), run_time=0.4)
        self.play(FadeOut(rbox))

        fbox = SurroundingRectangle(VGroup(col(1), col(2)), color=BLUE)
        gbox = SurroundingRectangle(col(3), color=GREEN)
        self.play(Create(fbox), Create(gbox))
        self.say("BLUE = features -> matrix X (6 rows x 2 cols)\n"
                 "  rows = students (examples)\n"
                 "  columns = features\n"
                 "GREEN = the answers -> vector y (labels)", wait=4)

        X = bracketed([v for i in range(6) for v in (STUDY[i], SLEEP[i])], 2, fs=28, color=BLUE)
        y = bracketed(list(GRADE), 1, fs=28, color=GREEN)
        Xl = Text("X =", font_size=32, color=BLUE)
        yl = Text("y =", font_size=32, color=GREEN)
        grp = VGroup(Xl, X, yl, y).arrange(RIGHT, buff=0.35).move_to(1.2 * RIGHT + 0.6 * DOWN)
        yl.shift(0.5 * RIGHT); y.shift(0.5 * RIGHT)
        self.play(FadeOut(col(0)), FadeOut(fbox), FadeOut(gbox),
                  *[FadeOut(cells[j]) for j in range(4)],
                  ReplacementTransform(VGroup(col(1)[1:], col(2)[1:]), X[1]),
                  ReplacementTransform(col(3)[1:], y[1]),
                  FadeIn(X[0]), FadeIn(X[2]), FadeIn(y[0]), FadeIn(y[2]), FadeIn(Xl), FadeIn(yl),
                  run_time=2)
        names = VGroup(*[Text(n, font_size=20, color=GREY_B)
                         .move_to([Xl.get_left()[0] - 0.9, X[1][2 * i].get_y(), 0])
                         for i, n in enumerate(NAMES)])
        self.play(FadeIn(names))
        self.say("X and y together = the TRAINING SET.\n"
                 "That is the 'making a training set' part:\n"
                 "turning real things into rows of numbers.", wait=4)
        self.say("The GOAL of machine learning:\n"
                 "find a rule that turns each row of X\n"
                 "into (almost) its number in y,\n"
                 "so we can use it on NEW students.", wait=4)


# ---------------------------------------------------------------------------
# S3. The model: dot product for one student, X w for the whole class
# ---------------------------------------------------------------------------
class S3_PredictionIsXw(Base):
    def construct(self):
        w = np.array([5, 7])
        self.say("Step 3: the model. Guess how much each\n"
                 "feature is 'worth' in grade points.\n"
                 "These are the WEIGHTS, also a vector:\n"
                 "w = (5 per study hour, 7 per sleep hour)", wait=4)

        xv = bracketed([2, 6], 1, fs=38, color=YELLOW)
        wv = bracketed([5, 7], 1, fs=38, color=PURPLE)
        dot = Text("·", font_size=60)
        lhs = VGroup(Text("Maria", font_size=28, color=YELLOW), xv, dot, wv,
                     Text("weights", font_size=28, color=PURPLE)).arrange(RIGHT, buff=0.3)
        lhs.move_to(3.9 * LEFT + 0.3 * DOWN)
        self.play(FadeIn(lhs))
        lines = VGroup(
            Text("study:  2 × 5 = 10 points", font_size=28),
            Text("sleep:  6 × 7 = 42 points", font_size=28),
            Text("predicted grade = 10 + 42 = 52", font_size=30, color=YELLOW),
            Text("real grade = 52   ✓", font_size=30, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(3.0 * RIGHT + 0.3 * DOWN)
        for ln in lines:
            self.play(FadeIn(ln, shift=0.2 * RIGHT))
            self.wait(0.6)
        self.say("Multiply feature by weight, add them up:\n"
                 "that's the DOT PRODUCT  w · x.\n"
                 "Prediction for one student = one dot product.", wait=4)
        self.play(FadeOut(lhs), FadeOut(lines))

        # Whole class at once
        X = bracketed([v for i in range(6) for v in (STUDY[i], SLEEP[i])], 2, fs=26, color=BLUE)
        W = bracketed(list(w), 1, fs=26, color=PURPLE)
        pred = STUDY * w[0] + SLEEP * w[1]
        P = bracketed(list(pred), 1, fs=26, color=YELLOW)
        Y = bracketed(list(GRADE), 1, fs=26, color=GREEN)
        eq = VGroup(X, W, Text("=", font_size=36), P).arrange(RIGHT, buff=0.3)
        labels = VGroup(Text("X (data)", font_size=22, color=BLUE),
                        Text("w", font_size=22, color=PURPLE),
                        Text("prediction", font_size=22, color=YELLOW))
        grp = VGroup(eq, Y).arrange(RIGHT, buff=1.2).move_to(1.5 * RIGHT + 1.4 * DOWN)
        for lbl, m in zip(labels, [X, W, P]):
            lbl.next_to(m, UP, buff=0.2)
        ylbl = Text("real y", font_size=22, color=GREEN).next_to(Y, UP, buff=0.2)
        self.say("Do it for ALL students at once:\n"
                 "each row of X 'dot' w.\n"
                 "That is exactly a MATRIX × VECTOR.", wait=1)
        self.play(FadeIn(X), FadeIn(W), FadeIn(labels[:2]))
        row_box = SurroundingRectangle(X[1][0:2], color=YELLOW)
        self.play(Create(row_box), FadeIn(eq[2]), FadeIn(P[0]), FadeIn(P[2]), FadeIn(labels[2]))
        for i in range(6):
            self.play(row_box.animate.move_to(VGroup(X[1][2 * i], X[1][2 * i + 1])),
                      FadeIn(P[1][i]), run_time=0.5)
        self.play(FadeOut(row_box), FadeIn(Y), FadeIn(ylbl))
        self.wait()
        self.say("Compare with the real grades:\n"
                 "Maria is right, but Giorgos is off by 15!\n"
                 "Our guessed w is not good enough.", wait=4)
        self.say("Look at the shape:  X w = y\n"
                 "That's the book's  A x = b !\n"
                 "  A = X  (data, known)\n"
                 "  b = y  (grades, known)\n"
                 "  x = w  (weights, UNKNOWN)\n"
                 "LEARNING = solving for w.", wait=6)


# ---------------------------------------------------------------------------
# S4. Solving with the inverse... and why real data breaks it
# ---------------------------------------------------------------------------
class S4_WhyInverseFails(Base):
    def construct(self):
        self.say("Step 4: can we just solve X w = y\n"
                 "with the inverse,  w = X⁻¹ y ?\n"
                 "Try a tiny class: only Maria and Nikos.", wait=3.5)
        X2 = bracketed([2, 6, 4, 7], 2, fs=34, color=BLUE)
        W = bracketed(["w1", "w2"], 1, fs=34, color=PURPLE)
        Y2 = bracketed([52, 63], 1, fs=34, color=GREEN)
        eq = VGroup(X2, W, Text("=", font_size=40), Y2).arrange(RIGHT, buff=0.3).move_to(UP * 0.2 + RIGHT)
        self.play(FadeIn(eq))
        steps = VGroup(
            Text("2 students, 2 weights -> X is SQUARE (2×2)", font_size=26),
            Text("det = 2·7 - 6·4 = -10  ≠ 0  -> invertible", font_size=26),
            Text("w = X⁻¹ y = (1.4, 8.2)", font_size=28, color=PURPLE),
            Text("check Maria: 1.4·2 + 8.2·6 = 52 ✓   Nikos: 1.4·4 + 8.2·7 = 63 ✓", font_size=24, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(eq, DOWN, buff=0.5)
        for s in steps:
            self.play(FadeIn(s, shift=0.2 * UP))
            self.wait(1)
        self.say("Perfect fit! This is the inverse from\n"
                 "the book, doing real work.", wait=3)
        self.play(FadeOut(steps), FadeOut(eq))

        X3 = bracketed([2, 6, 4, 7, 5, 5], 2, fs=34, color=BLUE)
        Y3 = bracketed([52, 63, 64], 1, fs=34, color=GREEN)
        W3 = bracketed(["w1", "w2"], 1, fs=34, color=PURPLE)
        eq3 = VGroup(X3, W3, Text("=", font_size=40), Y3).arrange(RIGHT, buff=0.3).move_to(UP * 0.2 + RIGHT)
        self.say("Now add a 3rd student: Eleni (5, 5) -> 64.", wait=1)
        self.play(FadeIn(eq3))
        steps2 = VGroup(
            Text("3 equations, only 2 unknowns. X is 3×2: NOT square -> no X⁻¹", font_size=24),
            Text("old w = (1.4, 8.2) predicts Eleni: 1.4·5 + 8.2·5 = 48   (real: 64)  ✗", font_size=24, color=RED),
            Text("no w hits all three exactly", font_size=26, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(eq3, DOWN, buff=0.5)
        for s in steps2:
            self.play(FadeIn(s, shift=0.2 * UP))
            self.wait(1.2)
        self.say("Remember the ThreeByTwo scene: 2 columns in 3D\n"
                 "span only a PLANE. y = (52, 63, 64) is off that\n"
                 "plane -> 'not in the column space' -> no solution.", wait=5)
        self.say("Real data is ALWAYS like this: many students,\n"
                 "few features, and noise (bad days, luck...).\n"
                 "So we stop asking for EXACT and ask for\n"
                 "'as close as possible'. How close? A NORM.", wait=5)


# ---------------------------------------------------------------------------
# S5. Training = making the error vector's norm as small as possible
# ---------------------------------------------------------------------------
class S5_TrainingIsANorm(Base):
    def construct(self):
        ax = Axes(x_range=[0, 11, 1], y_range=[40, 100, 10], x_length=7.2, y_length=4.6,
                  tips=False).to_edge(RIGHT, buff=0.5).shift(0.6 * DOWN)
        xt = Text("hours studied / week", font_size=20).next_to(ax.x_axis, DOWN, buff=0.45)
        yt = Text("grade", font_size=20).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.5)
        self.add(ax, xt, yt, tick_labels(ax, [2, 4, 6, 8, 10], [50, 60, 70, 80, 90]))

        dots = VGroup(*[Dot(ax.c2p(h, g), color=GREEN, radius=0.09) for h, g in zip(STUDY, GRADE)])
        names = VGroup(*[Text(n, font_size=16, color=GREY_B).next_to(d, UL, buff=0.05)
                         for n, d in zip(NAMES, dots)])
        self.say("Step 5: training. To SEE it, keep one feature\n"
                 "(study hours). Each green dot = one student\n"
                 "= one row of X with its grade from y.", wait=1)
        self.play(FadeIn(dots), FadeIn(names))
        self.wait(2)

        w, b = ValueTracker(1.0), ValueTracker(75.0)
        best_w, best_b = np.polyfit(STUDY, GRADE, 1)

        def pred():
            return w.get_value() * STUDY + b.get_value()

        line = always_redraw(lambda: ax.plot(lambda h: w.get_value() * h + b.get_value(),
                                             x_range=[0, 11], color=YELLOW, stroke_width=5))
        resid = always_redraw(lambda: VGroup(*[
            Line(ax.c2p(h, g), ax.c2p(h, p), color=RED, stroke_width=4)
            for h, g, p in zip(STUDY, GRADE, pred())]))

        def readout():
            e = pred() - GRADE
            return box(f"model: grade = w·hours + b\n"
                       f"w = {w.get_value():.2f}   b = {b.get_value():.1f}\n"
                       f"errors e = ({', '.join(f'{int(round(v)):+d}' for v in e)})\n"
                       f"||e||² = {np.sum(e ** 2):8.1f}", DL, fs=22)

        self.say("Model = the yellow line: grade = w·hours + b\n"
                 "(b = starting grade; in X it's a column of 1s).\n"
                 "Red bars = errors: predicted - real.", wait=1)
        self.play(Create(line))
        self.play(Create(resid), FadeIn(rd := always_redraw(readout)))
        self.wait(2)
        self.say("All the red bars together form the ERROR\n"
                 "VECTOR e. One number for 'how wrong':\n"
                 "its squared L2 norm ||e||² = sum of e_i².\n"
                 "(the 'loss')", wait=4)

        self.say("TRAINING = change w and b until ||e||²\n"
                 "is as SMALL as possible. Watch the\n"
                 "red bars shrink and the loss drop:", wait=1)
        self.play(w.animate.set_value(2.5), b.animate.set_value(60), run_time=3)
        self.play(w.animate.set_value(6), b.animate.set_value(38), run_time=3)
        self.play(w.animate.set_value(best_w), b.animate.set_value(best_b), run_time=3)
        self.wait()
        self.say(f"Best line: w ≈ {best_w:.1f}, b ≈ {best_b:.0f}.\n"
                 f"Meaning: every extra study hour/week\n"
                 f"≈ +{best_w:.1f} grade points. No other line has\n"
                 f"a smaller ||e||². That line = the trained model.", wait=5)
        self.say("Real libraries (scikit-learn, PyTorch) do\n"
                 "exactly this, with many features and\n"
                 "millions of rows, by following the\n"
                 "derivative of ||e||² downhill.", wait=4)

        # Predict a new student
        h_new = 6
        g_new = best_w * h_new + best_b
        nd = Dot(ax.c2p(h_new, g_new), color=PINK, radius=0.12)
        guide = VGroup(DashedLine(ax.c2p(h_new, 40), ax.c2p(h_new, g_new), color=PINK),
                       DashedLine(ax.c2p(0, g_new), ax.c2p(h_new, g_new), color=PINK))
        nl = Text(f"Sofia: {g_new:.0f}?", font_size=22, color=PINK).next_to(nd, UL, buff=0.1)
        self.say("THE PAYOFF: Sofia is new. She studies\n"
                 "6 h/week. Her vector goes in, the\n"
                 f"model predicts ≈ {g_new:.0f}/100 before any exam.", wait=1)
        self.play(Create(guide), FadeIn(nd), FadeIn(nl))
        self.wait(3)
        self.say("Summary:\n"
                 "vector = one thing (a student) as numbers\n"
                 "matrix X = all the things = training set\n"
                 "X w = predictions (Ax = b, w unknown)\n"
                 "inverse = exact solve, only in tiny cases\n"
                 "norm ||Xw - y|| = how wrong; training shrinks it", wait=7)

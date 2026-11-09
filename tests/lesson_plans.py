"""Hand-written lesson plans that exercise every view and action of studio_lesson.
Used by tests/test_lessons.py (fast checks) and tools/render_plan.py (real renders)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import prompts  # noqa: E402

det = json.loads(re.search(r"```json\s*(\{.*?\})\s*```", prompts.LESSON_SYSTEM, re.S).group(1))

eigen = {"kind": "lesson", "title": "Eigenvectors of A",
         "question": "Which directions does A = [[2, 1], [1, 2]] only stretch?",
         "answer": "The diagonal directions: [1, 1] is stretched by 3 and [1, -1] by 1.",
         "steps": [
             {"title": "Most vectors get turned", "view": {"type": "plane"}, "beats": [
                 {"say": "Here is A.", "do": "show_matrix", "matrix": [[2, 1], [1, 2]], "name": "A"},
                 {"say": "Take the vector [1, 0] and the line through it.", "do": "off_span", "coords": [1, 0]},
                 {"say": "These dashed lines are the special directions.", "do": "eigenvectors", "matrix": "A"},
                 {"say": "Apply A and watch.", "do": "transform", "matrix": "A"},
                 {"say": "The pink vector left its line; the others stayed on theirs.", "do": "eigen_stretch"}]},
             {"title": "Checking with numbers", "view": {"type": "same"}, "beats": [
                 {"say": "A times [1, 1] is 3 times [1, 1].", "do": "calc", "what": "eigen", "matrix": "A", "vector": [1, 1]},
                 {"say": "So λ = 3 and λ = 1.", "do": "calc", "what": "eigen", "matrix": "A"}]},
             {"title": "Vectors as mixes", "view": {"type": "plane"}, "beats": [
                 {"say": "v = [2, 1].", "do": "vector", "coords": [2, 1], "name": "v"},
                 {"say": "It is 2 î plus 1 ĵ.", "do": "combination", "vector": "v"},
                 {"say": "Apply A.", "do": "transform", "matrix": [[2, 1], [1, 2]], "landings": True},
                 {"say": "A·v computed.", "do": "calc", "what": "matvec", "matrix": [[2, 1], [1, 2]], "vector": "v"},
                 {"say": "Undo it.", "do": "reset"}]}]}

matrices = {"kind": "lesson", "title": "Matrix products",
            "question": "How do you multiply A = [[1, 2], [3, 4]] by B = [[0, 1], [1, 0]]?",
            "answer": "Each entry is a row of A dotted with a column of B: A·B = [[2, 1], [4, 3]].",
            "steps": [
                {"title": "The two matrices", "view": {"type": "matrices"}, "beats": [
                    {"say": "This is A.", "do": "matrix", "name": "A", "values": [[1, 2], [3, 4]], "at": "left"},
                    {"say": "This is B.", "do": "matrix", "name": "B", "values": [[0, 1], [1, 0]], "at": "right"},
                    {"say": "Look at row 1 of A.", "do": "highlight", "name": "A", "row": 0},
                    {"say": "One entry: row 1 of A with column 1 of B.", "do": "dot", "a": "A", "row": 0, "b": "B", "col": 0}]},
                {"title": "All entries", "view": {"type": "same"}, "beats": [
                    {"say": "Every entry the same way.", "do": "multiply", "a": "A", "b": "B", "name": "C"},
                    {"say": "A new matrix next to it.", "do": "matrix", "name": "X", "values": [[2, 4], [6, 8], [1, 3]], "at": "left"},
                    {"say": "Flip it.", "do": "transpose", "name": "X"}]},
                {"title": "Centering", "view": {"type": "matrices"}, "beats": [
                    {"say": "Data.", "do": "matrix", "name": "D", "values": [[1, 10], [3, 20], [5, 30]], "at": "main"},
                    {"say": "Center column 1.", "do": "column_op", "name": "D", "col": 0, "op": "center"},
                    {"say": "The mean.", "do": "calc", "what": "mean", "values": [1, 3, 5]},
                    {"say": "The variance.", "do": "calc", "what": "variance", "values": [1, 3, 5], "sample": True}]}]}

prob = {"kind": "lesson", "title": "Sigmoid and a die",
        "question": "What does the sigmoid look like, and what is the expected value of a die?",
        "answer": "The sigmoid squashes any number into (0, 1); a fair die averages 3.5.",
        "steps": [
            {"title": "The sigmoid", "view": {"type": "plot", "x_range": [-6, 6], "x_label": "x", "y_label": "σ(x)"}, "beats": [
                {"say": "σ(x) = 1/(1+e^(−x)).", "do": "curve", "f": "1/(1+exp(-x))", "label": "σ(x)"},
                {"say": "At x = 0 it is exactly one half.", "do": "point", "x": 0, "y": 0.5, "label": "(0, 0.5)"},
                {"say": "The shaded part.", "do": "area", "curve": "σ(x)", "from": -2, "to": 2},
                {"say": "A threshold.", "do": "vline", "x": 2, "label": "x = 2"}]},
            {"title": "A fair die", "view": {"type": "plot", "x_label": "face", "y_label": "P"}, "beats": [
                {"say": "Each face has probability 1/6.", "do": "bars", "labels": ["1", "2", "3", "4", "5", "6"],
                 "values": [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]},
                {"say": "The expected value.", "do": "calc", "what": "expectation", "values": [1, 2, 3, 4, 5, 6],
                 "probs": [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]}]},
            {"title": "Fitting a line", "view": {"type": "plot", "x_label": "hours", "y_label": "score"}, "beats": [
                {"say": "Five students.", "do": "points", "name": "s", "data": [[1, 52], [2, 55], [3, 61], [4, 64], [5, 70]]},
                {"say": "The best line.", "do": "fit_line", "name": "s"},
                {"say": "Center them.", "do": "move_points", "name": "s", "op": "center"},
                {"say": "A segment.", "do": "segment", "from": [0, 0], "to": [2, 8], "label": "Δ"}]}]}

space = {"kind": "lesson", "title": "Volume scaling in 3-D",
         "question": "What does det = 6 mean for M = [[2, 0, 0], [0, 1, 0], [0, 0, 3]]?",
         "answer": "M multiplies every volume by 6.",
         "steps": [
             {"title": "The unit cube", "view": {"type": "space"}, "beats": [
                 {"say": "This is M.", "do": "show_matrix", "matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 3]], "name": "M"},
                 {"say": "The unit cube.", "do": "unit_cube"},
                 {"say": "Its volume.", "do": "determinant"},
                 {"say": "Apply M.", "do": "transform", "matrix": "M"},
                 {"say": "Look around.", "do": "orbit"},
                 {"say": "The number.", "do": "calc", "what": "det", "matrix": "M"}]},
             {"title": "A formula", "view": {"type": "text"}, "beats": [
                 {"say": "In words.", "do": "note", "text": "volume after = det(M) · volume before"}]}]}

PLANS = {"det": det, "eigen": eigen, "matrices": matrices, "prob": prob, "space": space}

corr = {"kind": "lesson", "title": "Colors of a correlation matrix",
        "question": "For X = [1, 2, 3, 4], Y = [2, 1, 4, 3], Z = [5, 6, 5, 6]: what do the colors of the correlation matrix show, and what does it do to space?",
        "answer": "R is red where variables rise together and blue where they move oppositely; det(R) = {det(R)}, so R flattens space: Z is a combination of X and Y.",
        "steps": [
            {"title": "The data", "view": {"type": "matrices"}, "beats": [
                {"say": "Four samples of three variables X, Y and Z, one column each.", "do": "data", "columns": {"X": [1, 2, 3, 4], "Y": [2, 1, 4, 3], "Z": [5, 6, 5, 6]}},
                {"say": "The average of X is {mean(X)}.", "do": "calc", "what": "mean", "values": "X"},
                {"say": "Its spread: the standard deviation of X is {std(X)}.", "do": "calc", "what": "std", "values": "X"}]},
            {"title": "X and Y together", "view": {"type": "plot"}, "beats": [
                {"say": "Each dot is one sample: X across, Y up.", "do": "points", "name": "xy", "x": "X", "y": "Y"},
                {"say": "The best line rises, so X and Y tend to grow together.", "do": "fit_line", "name": "xy"},
                {"say": "Correlation measures that in one number between −1 and 1.", "do": "calc", "what": "correlation", "a": "X", "b": "Y"}]},
            {"title": "All pairs at once", "view": {"type": "matrices"}, "beats": [
                {"say": "R holds the correlation of every pair; the diagonal is each variable with itself.", "do": "matrix", "name": "R", "compute": "correlation", "of": "data", "at": "main"},
                {"say": "Color it: red for together, blue for opposite, white for unrelated.", "do": "heatmap", "name": "R"},
                {"say": "X and Y are {R[0][1]}: light red. Y and Z are {R[1][2]}: light blue."}]},
            {"title": "R as a transformation", "view": {"type": "space"}, "beats": [
                {"say": "Now use R to move 3-D space.", "do": "show_matrix", "matrix": "R", "name": "R"},
                {"say": "The unit cube has volume 1.", "do": "unit_cube"},
                {"say": "The readout tracks its volume.", "do": "determinant"},
                {"say": "Apply R: the cube is squashed flat, its volume becomes {det(R)}.", "do": "transform", "matrix": "R"},
                {"say": "Flat means one variable adds nothing new: Z = 5.5 + 0.5·(X − Y)."}]}]}
PLANS["corr"] = corr

ml = {"kind": "lesson", "title": "Machine learning gallery", "question": "What does each ML picture show?",
      "answer": "Every picture computes its numbers from the data shown.",
      "steps": [
          {"title": "Classification", "view": {"type": "plot", "x_label": "x₁", "y_label": "x₂"}, "beats": [
              {"say": "Two classes of points.", "do": "points", "name": "c",
               "data": [[1, 1], [1.5, 2], [2, 1.2], [2.5, 2.2], [1.2, 2.8], [4, 4], [4.5, 3.2], [5, 4.5], [3.8, 5], [5.2, 3.6]],
               "labels": ["cat", "cat", "cat", "cat", "cat", "dog", "dog", "dog", "dog", "dog"]},
              {"say": "Logistic regression draws the line between them.", "do": "boundary", "name": "c"},
              {"say": "A new point is labelled by its 3 nearest neighbours.", "do": "knn", "name": "c", "query": [3.2, 3], "k": 3}]},
          {"title": "Regression and overfitting", "view": {"type": "plot", "x_label": "x", "y_label": "y"}, "beats": [
              {"say": "Noisy data.", "do": "points", "name": "r", "data": [[0, 1.2], [1, 1.9], [2, 3.4], [3, 3.1], [4, 4.8], [5, 5.2], [6, 6.9], [7, 6.6]]},
              {"say": "Residuals.", "do": "residuals", "name": "r"},
              {"say": "A degree-3 curve.", "do": "fit_curve", "name": "r", "degree": 3},
              {"say": "A degree-7 curve passes through every point: overfitting.", "do": "fit_curve", "name": "r", "degree": 7}]},
          {"title": "PCA", "view": {"type": "plot"}, "beats": [
              {"say": "A tilted cloud.", "do": "points", "name": "p", "data": [[1, 1.2], [2, 1.9], [3, 3.3], [4, 3.8], [5, 5.4], [2.5, 2.1], [3.5, 3.9], [1.5, 1.1]]},
              {"say": "The principal axes.", "do": "pca", "name": "p"},
              {"say": "Project onto PC1.", "do": "project", "name": "p"}]},
          {"title": "k-means", "view": {"type": "plot"}, "beats": [
              {"say": "Unlabelled points.", "do": "points", "name": "k", "data": [[1, 1], [1.3, 1.6], [0.8, 1.4], [5, 5], [5.4, 4.6], [4.8, 5.5], [1, 5], [1.4, 5.4], [0.7, 4.6]]},
              {"say": "k-means with 3 centers.", "do": "kmeans", "name": "k", "k": 3, "iterations": 3}]},
          {"title": "Distributions", "view": {"type": "plot", "x_label": "height", "y_label": "density"}, "beats": [
              {"say": "A histogram of samples.", "do": "histogram", "values": [160, 165, 170, 172, 168, 175, 180, 171, 169, 174, 166, 178, 173, 170, 167], "bins": 6, "density": True},
              {"say": "A normal curve on top.", "do": "curve", "f": "exp(-(x-170.5)^2/(2*25))/sqrt(2*pi*25)", "label": "normal"}]},
          {"title": "Training curves", "view": {"type": "plot", "x_label": "epoch", "y_label": "loss"}, "beats": [
              {"say": "Training and validation loss.", "do": "series", "ys": {"training": [2.0, 1.4, 1.0, 0.7, 0.5, 0.35, 0.25, 0.18], "validation": [2.1, 1.5, 1.15, 0.95, 0.9, 0.93, 1.0, 1.1]}, "mark_min": "validation"}]},
          {"title": "Gradient descent", "view": {"type": "plot", "x_range": [-3, 3]}, "beats": [
              {"say": "Walk downhill on f(x) = x².", "do": "descent", "f": "x^2", "start": 2.6, "lr": 0.2, "steps": 6},
              {"say": "Too big a step overshoots.", "do": "descent", "f": "x^2", "start": -2.5, "lr": 0.95, "steps": 6, "color": "pink"}]},
          {"title": "A loss landscape", "view": {"type": "contour", "f": "x^2 + 4*y^2", "x_range": [-3, 3], "y_range": [-2, 2], "x_label": "w₁", "y_label": "w₂"}, "beats": [
              {"say": "Descent on the bowl.", "do": "descent2d", "start": [-2.6, 1.6], "lr": 0.1, "steps": 12},
              {"say": "A bigger learning rate zig-zags.", "do": "descent2d", "start": [2.6, 1.5], "lr": 0.23, "steps": 10, "color": "pink"}]},
          {"title": "A neural network", "view": {"type": "network", "layers": [3, 4, 2], "names": ["input", "hidden", "output"]}, "beats": [
              {"say": "Inputs flow forward.", "do": "forward", "values": [0.2, 0.9, 0.5], "outputs": [0.8, 0.2]},
              {"say": "The hidden layer.", "do": "focus", "layer": 1},
              {"say": "Its activations.", "do": "activate", "layer": 1, "values": [0.1, 0.7, 0.4, 0.95]}]}]}
PLANS["ml"] = ml

net = {"kind": "lesson", "title": "A tiny network", "question": "How does a neural network pass information forward?",
       "answer": "Each layer mixes the previous layer's values with weights and passes them on.",
       "steps": [ml["steps"][-1], {"title": "Sigmoid activation", "view": {"type": "plot", "x_range": [-6, 6]}, "beats": [
           {"say": "Each neuron squashes its input with a sigmoid.", "do": "curve", "f": "1/(1+exp(-x))", "label": "σ(x)"}]}]}
PLANS["net"] = net

decomp = {"kind": "lesson", "title": "SVD and friends", "question": "What does the SVD of A = [[3, 0], [4, 5]] do?",
          "answer": "A = UΣVᵀ: rotate, stretch by {sv(A)}, rotate.",
          "steps": [
              {"title": "A turns the circle into an ellipse", "view": {"type": "plane"}, "beats": [
                  {"say": "This is A.", "do": "show_matrix", "matrix": [[3, 0], [4, 5]], "name": "A"},
                  {"say": "The unit circle: all vectors of length 1.", "do": "unit_circle"},
                  {"say": "Apply A: the circle becomes an ellipse.", "do": "transform", "matrix": "A"},
                  {"say": "Its half-axes are the singular values.", "do": "calc", "what": "svd", "matrix": "A"}]},
              {"title": "Three simple moves", "view": {"type": "plane"}, "beats": [
                  {"say": "The same circle again.", "do": "unit_circle"},
                  {"say": "Rotate with Vᵀ, stretch with Σ, rotate with U; then A in one move.", "do": "decompose", "matrix": "A", "kind": "svd"},
                  {"say": "Σ holds {S[0][0]} and {S[1][1]}."}]},
              {"title": "Eigendecomposition of a symmetric matrix", "view": {"type": "plane"}, "beats": [
                  {"say": "B is symmetric.", "do": "show_matrix", "matrix": [[2, 1], [1, 2]], "name": "B"},
                  {"say": "Qᵀ, Λ, Q.", "do": "decompose", "matrix": "B", "kind": "eigen"}]},
              {"title": "Projection", "view": {"type": "plane"}, "beats": [
                  {"say": "Project b onto the line through a.", "do": "projection", "vector": [1, 3], "onto": [2, 1]},
                  {"say": "The formula.", "do": "calc", "what": "projection", "vector": [1, 3], "onto": [2, 1]}]},
              {"title": "Unit balls", "view": {"type": "plot", "equal": True, "x_label": "x₁", "y_label": "x₂"}, "beats": [
                  {"say": "L2: a circle.", "do": "norm_ball", "p": 2, "label": "L2"},
                  {"say": "L1: a diamond.", "do": "norm_ball", "p": 1, "label": "L1", "color": "pink"},
                  {"say": "L∞: a square.", "do": "norm_ball", "p": "inf", "label": "L∞", "color": "teal"},
                  {"say": "A point.", "do": "label", "at": [0.6, 0.8], "text": "x"}]},
              {"title": "Covariance shapes", "view": {"type": "plot", "equal": True}, "beats": [
                  {"say": "Round.", "do": "ellipse", "cov": [[1, 0], [0, 1]], "label": "isotropic"},
                  {"say": "Tilted.", "do": "ellipse", "cov": [[2, 1.5], [1.5, 2]], "label": "correlated", "color": "pink", "fill": True},
                  {"say": "An arrow along the long axis.", "do": "arrow", "from": [0, 0], "to": [2.6, 2.6]}]},
              {"title": "Weight decay", "view": {"type": "plot"}, "beats": [
                  {"say": "Data.", "do": "points", "name": "d", "data": [[0, 1.1], [1, 2.3], [2, 3.8], [3, 4.0], [4, 5.9], [5, 6.2], [6, 8.1], [7, 8.0], [8, 9.9]]},
                  {"say": "No penalty.", "do": "fit_curve", "name": "d", "degree": 8, "ridge": 0},
                  {"say": "Small penalty.", "do": "fit_curve", "name": "d", "degree": 8, "ridge": 0.01},
                  {"say": "Huge penalty.", "do": "fit_curve", "name": "d", "degree": 8, "ridge": 100}]}]}
PLANS["decomp"] = decomp

bayes = {"kind": "lesson", "title": "Bayes' rule", "question": "How likely is the disease after a positive test?",
         "answer": "About 15%: most positives come from the much larger healthy group.",
         "steps": [
             {"title": "Counting people", "view": {"type": "text"}, "beats": [
                 {"say": "Of 10,000 people, 1% are sick.", "do": "bayes_tree", "prior": 0.01, "likelihood": 0.9, "false_positive": 0.05, "show": "prior"},
                 {"say": "The test finds 90% of the sick, and wrongly flags 5% of the healthy.", "do": "bayes_tree", "prior": 0.01, "likelihood": 0.9, "false_positive": 0.05, "show": "test"},
                 {"say": "Among all positives, only a small share is sick.", "do": "bayes_tree", "prior": 0.01, "likelihood": 0.9, "false_positive": 0.05, "show": "posterior"}]},
             {"title": "The formula", "view": {"type": "text"}, "beats": [
                 {"say": "The same numbers in Bayes' rule.", "do": "calc", "what": "bayes", "prior": 0.01, "likelihood": 0.9, "false_positive": 0.05}]},
             {"title": "Moves with captions", "view": {"type": "plane"}, "beats": [
                 {"say": "A symmetric matrix.", "do": "show_matrix", "matrix": [[2, 1], [1, 2]], "name": "B"},
                 {"say": "Three moves.", "do": "decompose", "matrix": "B", "kind": "eigen",
                  "says": ["Qᵀ turns the eigenvectors onto the axes.", "Λ stretches the axes by {L[0][0]} and {L[1][1]}.", "Q turns them back.", "B in one move lands in the same place."]}]}]}
PLANS["bayes"] = bayes

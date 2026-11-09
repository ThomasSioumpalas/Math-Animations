"""Machine-learning visuals for lesson plans (mixed into studio_lesson._Lesson).

plot view:    points with class labels, boundary (logistic regression / given w, b), knn, residuals,
              fit_curve (polynomial degree d), pca, project, kmeans, histogram, series (loss curves),
              descent (gradient descent on a 1-D curve)
contour view: a loss surface f(x, y) as contour lines, descent2d (gradient descent path on it)
network view: a neural network diagram; forward (signal flows through it), activate (values in a
              layer), focus (point at a layer / node)
Everything is computed here with numpy from the plan's data: the model never supplies results.
"""
from __future__ import annotations

import ast

import numpy as np
from manim import (
    BLACK, DOWN, GREY_B, LEFT, RIGHT, UL, UR, WHITE, YELLOW, AnimationGroup, Circle, Create, DashedLine, Dot,
    Ellipse, FadeIn, FadeOut, GrowFromEdge, Indicate, Line, ManimColor, Polygon, Rectangle, ShowPassingFlash, Text,
    VGroup, VMobject, interpolate_color,
)
from studio_kit import DataPlot, boxed, fit_to, fmt
from studio_space import _segments

CLASS_COLORS = [ManimColor(c) for c in ("#58C4DD", "#FC6255", "#83C167", "#FFD43B", "#C77DFF", "#FF9F43")]
LOW, HIGH = ManimColor("#3B6FD8"), ManimColor("#FFD43B")


def formula_xy(src: str):
    """A safe f(x, y) from text like 'x^2 + 3*y^2' (the tree is walked by hand, no eval)."""
    from studio_lesson import _CONSTS, _FUNCS
    tree = ast.parse(str(src).replace("^", "**").replace("·", "*").replace("−", "-"), mode="eval")
    ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}

    def ev(n, x, y):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id in ("x", "w1", "a"):
                return x
            if n.id in ("y", "w2", "b"):
                return y
            if n.id in _CONSTS:
                return _CONSTS[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left, x, y), ev(n.right, x, y))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand, x, y)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*(ev(a, x, y) for a in n.args))
        raise ValueError(f"unsupported formula part: {ast.dump(n)[:60]}")

    def f(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        with np.errstate(all="ignore"):
            return np.asarray(ev(tree.body, x, y), float) * np.ones(np.broadcast(x, y).shape)
    f(np.array([0.5]), np.array([0.5]))
    return f


def _marching(F, xs, ys, level):
    """Contour line segments of the grid F (rows = ys, cols = xs) at one level (marching squares)."""
    segs = []
    above = F >= level
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            corners = (above[i, j], above[i, j + 1], above[i + 1, j + 1], above[i + 1, j])
            if all(corners) or not any(corners):
                continue
            v = (F[i, j], F[i, j + 1], F[i + 1, j + 1], F[i + 1, j])
            p = ((xs[j], ys[i]), (xs[j + 1], ys[i]), (xs[j + 1], ys[i + 1]), (xs[j], ys[i + 1]))
            cross = []
            for k in range(4):
                a, b = v[k], v[(k + 1) % 4]
                if (a >= level) != (b >= level) and np.isfinite(a) and np.isfinite(b):
                    t = (level - a) / (b - a)
                    cross.append((p[k][0] + t * (p[(k + 1) % 4][0] - p[k][0]), p[k][1] + t * (p[(k + 1) % 4][1] - p[k][1])))
            if len(cross) == 2:
                segs.append(cross)
            elif len(cross) == 4:
                segs += [cross[:2], cross[2:]]
    return segs


def _clip(poly, w, c, sign):
    """The part of a convex polygon where sign·(w·p + c) >= 0 (Sutherland-Hodgman, one plane)."""
    out = []
    for i in range(len(poly)):
        P, Q = np.asarray(poly[i]), np.asarray(poly[(i + 1) % len(poly)])
        fp, fq = sign * (w @ P + c), sign * (w @ Q + c)
        if fp >= 0:
            out.append(P)
        if (fp >= 0) != (fq >= 0):
            out.append(P + fp / (fp - fq) * (Q - P))
    return out


class MLActions:
    """_do_* actions for ML pictures; relies on _Lesson for self.s, self.plot, self._xy, self._adopt,
    self._series, self._color, self._need, self._arr, self.series, self.fits."""

    # ---------------------------------------------------------------- helpers
    def _ml_reset(self) -> None:
        self.labels, self.pcas, self.status, self.net, self.surface = {}, {}, None, None, None

    def _ranges(self):
        return tuple(self.plot.axes.x_range[:2]), tuple(self.plot.axes.y_range[:2])

    def _status(self, text: str, color=WHITE):
        """One status line under the plot (replaces the previous one and any note there)."""
        from studio_kit import ZONES
        lab = boxed(text, color=color, font_size=26, opacity=0.75)
        if lab.width > 12.6:
            lab.scale_to_fit_width(12.6)
        lab.move_to(ZONES["work"][0])
        anims = [FadeIn(lab)]
        if self.note is not None and self.note in self.s.mobjects:
            anims.append(FadeOut(self.note))
            self.note = None
        prev, self.status = self.status, None
        if prev is not None:   # the previous status goes, wherever it is (in the plot or on its own)
            if prev in self.plot:
                self.plot.remove(prev)
                self.s.add(prev)
            anims.append(FadeOut(prev))
        self.status = lab
        return anims

    def _legend_box(self, items):
        rows = VGroup(*(VGroup(Dot(radius=0.07, color=c), Text(str(t), font_size=20)).arrange(RIGHT, buff=0.1)
                        for t, c in items)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rows.add_background_rectangle(color=BLACK, opacity=0.7, buff=0.1)
        rows.next_to(self.plot.axes.get_corner(UR), DOWN + LEFT, buff=0.12)
        return rows

    def _ml_extent(self, b: dict, do: str):
        """Data extents of the ML actions, so the plot's ranges fit them."""
        xs, ys = [], []
        if do == "histogram":
            v = self._arr(b.get("values")).ravel()
            counts, edges = np.histogram(v, bins=self._bins(b, v), density=bool(b.get("density")))
            xs += [edges[0], edges[-1]]
            ys += [0.0, float(counts.max())]
        elif do == "series":
            for y in self._series_ys(b).values():
                ys += list(y)
                xs += list(self._series_x(b, len(y)))
        elif do == "knn" and b.get("query") is not None:
            q = np.asarray(b["query"], float)
            xs.append(q[0]); ys.append(q[1])
        return xs, ys

    # ---------------------------------------------------------------- classes, boundary, knn
    def _class_dots(self, key, data, labels):
        names = [str(v) for v in labels][: len(data)]
        uniq = list(dict.fromkeys(names))
        color = {u: CLASS_COLORS[k % len(CLASS_COLORS)] for k, u in enumerate(uniq)}
        dots = self.plot.dots(data[: len(names), :2], color=WHITE)
        for d, n in zip(dots, names):
            d.set_color(color[n])
        self.labels[key] = np.array(names)
        return dots, self._legend_box([(u, color[u]) for u in uniq])

    def _do_boundary(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        labels = self.labels.get(key)
        if b.get("w") is not None:
            w, c = np.asarray(b["w"], float)[:2], float(b.get("b") or 0)
        else:
            if labels is None or len(set(labels)) != 2:
                raise ValueError("boundary needs points with 2 classes (labels) or w and b")
            uniq = list(dict.fromkeys(labels))
            y = (labels == uniq[1]).astype(float)
            mu, sd = data.mean(0), np.where(data.std(0) > 0, data.std(0), 1)
            Z = (data - mu) / sd
            wz, cz = np.zeros(2), 0.0
            for _ in range(3000):   # logistic regression by gradient descent (tiny data: instant)
                p = 1 / (1 + np.exp(-(Z @ wz + cz)))
                wz -= 0.5 * (Z.T @ (p - y) / len(y) + 1e-3 * wz)
                cz -= 0.5 * float(np.mean(p - y))
            w, c = wz / sd, cz - float((wz * mu / sd).sum())
        (xlo, xhi), (ylo, yhi) = self._ranges()
        box = [(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)]
        regions, colors = VGroup(), [CLASS_COLORS[0], CLASS_COLORS[1]]
        for sign, col in ((-1, colors[0]), (1, colors[1])):
            poly = _clip(box, w, c, sign)
            if len(poly) >= 3:
                regions.add(Polygon(*(self._xy(*p) for p in poly), fill_color=col, fill_opacity=0.14, stroke_width=0))
        ends = [p for p in _clip(box, w, c, 1) if abs(w @ p + c) < 1e-6 * (1 + np.abs(w).sum())]
        anims = [FadeIn(regions)]
        if len(ends) >= 2:
            anims.append(Create(Line(self._xy(*ends[0]), self._xy(*ends[-1]), color=WHITE, stroke_width=4)))
        if labels is not None and len(set(labels)) == 2:
            uniq = list(dict.fromkeys(labels))
            acc = float(np.mean((data @ w + c > 0) == (labels == uniq[1])))
            anims += self._status(f"the line separates the classes: {acc * 100:.0f}% of the points on the correct side")
        self.s.play(*anims, run_time=1.4)
        self._adopt(*(a.mobject for a in anims))
        for r in regions:   # behind the dots
            self.plot.remove(r)
            self.plot.add_to_back(r)

    def _do_knn(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        q = np.asarray(b.get("query"), float)[:2]
        k = int(np.clip(int(b.get("k") or 3), 1, len(data)))
        d = np.linalg.norm(data - q, axis=1)
        idx = np.argsort(d)[:k]
        r = float(d[idx[-1]])
        ux = abs(self._xy(1, 0)[0] - self._xy(0, 0)[0])
        uy = abs(self._xy(0, 1)[1] - self._xy(0, 0)[1])
        qdot = Dot(self._xy(*q), radius=0.11, color=WHITE).set_stroke(BLACK, 2)
        ring = Ellipse(width=2 * r * ux, height=2 * r * uy, color=WHITE, stroke_width=2).move_to(self._xy(*q))
        lines = VGroup(*(Line(self._xy(*q), self._xy(*data[i]), color=WHITE, stroke_width=2, stroke_opacity=0.7)
                         for i in idx))
        anims = [FadeIn(qdot, scale=1.5), Create(ring), Create(lines)]
        labels = self.labels.get(key)
        if labels is not None:
            votes = {}
            for i in idx:
                votes[labels[i]] = votes.get(labels[i], 0) + 1
            winner = max(votes, key=votes.get)
            text = ", ".join(f"{n}× {lab}" for lab, n in votes.items())
            anims += self._status(f"{k} nearest: {text} → predict {winner}")
        self.s.play(*anims, run_time=1.4)
        self._adopt(*(a.mobject for a in anims))

    # ---------------------------------------------------------------- regression: residuals, polynomial fits
    def _do_residuals(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        if key not in self.fits:
            self._do_fit_line(b)
        x, y = data[:, 0], data[:, 1]
        slope, icept = np.polyfit(x, y, 1)
        fit = slope * x + icept
        lines = VGroup(*(DashedLine(self._xy(xi, yi), self._xy(xi, fi), dash_length=0.06, color="#FC6255",
                                    stroke_width=3) for xi, yi, fi in zip(x, y, fit)))
        sse = float(((y - fit) ** 2).sum())
        anims = [Create(lines)] + self._status(f"sum of squared errors = {fmt(sse)}", "#FC6255")
        self.s.play(*anims, run_time=1.3)
        self._adopt(*(a.mobject for a in anims))

    def _do_fit_curve(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        x, y = data[:, 0], data[:, 1]
        deg = int(np.clip(int(b.get("degree") or 2), 0, min(12, len(x) - 1)))
        with np.errstate(all="ignore"):
            poly = np.poly1d(np.polyfit(x, y, deg))
        sse = float(((poly(x) - y) ** 2).sum())
        (xlo, xhi), (ylo, yhi) = self._ranges()
        m = 0.02 * (yhi - ylo)
        color = self._color(b)
        a, c = float(x.min()), float(x.max())
        graph = self.plot.axes.plot(lambda t: float(np.clip(poly(t), ylo - m, yhi + m)),
                                    x_range=[a, c, (c - a) / 300], color=color, stroke_width=4)
        old = [m_ for m_ in self.fits.pop(f"curve:{key}", []) if b.get("replace", True)]
        for o in old:
            self.plot.remove(o)
        anims = [Create(graph)] + [FadeOut(o) for o in old]
        anims += self._status(f"degree {deg}: training error {fmt(sse)}", color)
        self.s.play(*anims, run_time=1.5)
        self._adopt(graph, self.status)
        self.fits[f"curve:{key}"] = [graph]

    # ---------------------------------------------------------------- PCA, projection, k-means
    def _pca_of(self, key, data):
        mu = data.mean(0)
        vals, vecs = np.linalg.eigh(np.cov(data.T, ddof=0))
        order = np.argsort(vals)[::-1]
        self.pcas[key] = (mu, vals[order], vecs[:, order])
        return self.pcas[key]

    def _do_pca(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        mu, vals, vecs = self._pca_of(key, data)
        share = vals / vals.sum() if vals.sum() > 0 else vals
        (xlo, xhi), (ylo, yhi) = self._ranges()
        v = vecs[:, 0]
        ts = []   # where the PC1 line through the mean leaves the plot box
        for k, (lo, hi) in enumerate(((xlo, xhi), (ylo, yhi))):
            if abs(v[k]) > 1e-12:
                ts += sorted(((lo - mu[k]) / v[k], (hi - mu[k]) / v[k]))
        t0 = max(t for t in ts[0::2]) if ts else -1.0
        t1 = min(t for t in ts[1::2]) if ts else 1.0
        axis = DashedLine(self._xy(*(mu + t0 * v)), self._xy(*(mu + t1 * v)), color=YELLOW,
                          dash_length=0.1, stroke_opacity=0.6)
        arrows = VGroup()
        for k, color in ((0, YELLOW), (1, "#FF86C8")):
            tip = mu + 2 * np.sqrt(max(vals[k], 0)) * vecs[:, k]
            arrows.add(Line(self._xy(*mu), self._xy(*tip), color=color, stroke_width=6).add_tip(tip_length=0.18))
        anims = [Create(axis), Create(arrows)] + self._status(
            f"PC1 holds {share[0] * 100:.0f}% of the spread, PC2 {share[1] * 100:.0f}%", YELLOW)
        self.s.play(*anims, run_time=1.5)
        self._adopt(*(a.mobject for a in anims))

    def _do_project(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        mu, vals, vecs = self.pcas.get(key) or self._pca_of(key, data)
        v = vecs[:, 0]
        proj = mu + ((data - mu) @ v)[:, None] * v
        drops = VGroup(*(DashedLine(self._xy(*p), self._xy(*q), dash_length=0.05, color=GREY_B)
                         for p, q in zip(data, proj)))
        self.s.play(Create(drops), run_time=0.9)
        kept = vals[0] / vals.sum() if vals.sum() > 0 else 1.0
        self.s.play(self.plot.move_dots(dots, proj), FadeOut(drops),
                    *self._status(f"1 number per point keeps {kept * 100:.0f}% of the spread", YELLOW))
        self._adopt(self.status)
        self.series[key] = (dots, proj)

    def _do_kmeans(self, b):
        self._need("plot")
        key, (dots, data) = self._series(b)
        k = int(np.clip(int(b.get("k") or 3), 2, min(6, len(data))))
        iters = int(np.clip(int(b.get("iterations") or 3), 1, 6))
        cent = [data[0]]
        while len(cent) < k:   # farthest-point start: deterministic and well spread
            d = np.min([np.linalg.norm(data - c, axis=1) for c in cent], axis=0)
            cent.append(data[int(np.argmax(d))])
        cent = np.array(cent, float)
        marks = VGroup(*(Circle(radius=0.16, color=WHITE, fill_color=CLASS_COLORS[j], fill_opacity=1, stroke_width=3)
                         .move_to(self._xy(*cent[j])) for j in range(k)))
        self.s.play(FadeIn(marks, scale=1.4), *self._status(f"start: {k} centers"), run_time=0.9)
        for it in range(1, iters + 1):
            assign = np.argmin([np.linalg.norm(data - c, axis=1) for c in cent], axis=0)
            self.s.play(*(d.animate.set_color(CLASS_COLORS[a]) for d, a in zip(dots, assign)),
                        *self._status(f"step {it}: every point joins its nearest center"), run_time=0.9)
            new = np.array([data[assign == j].mean(0) if np.any(assign == j) else cent[j] for j in range(k)])
            spread = float(sum(((data[assign == j] - new[j]) ** 2).sum() for j in range(k)))
            self.s.play(*(m.animate.move_to(self._xy(*c)) for m, c in zip(marks, new)),
                        *self._status(f"step {it}: centers move to their group's mean · spread {fmt(spread)}"),
                        run_time=0.9)
            if np.allclose(new, cent):
                break
            cent = new
        self._adopt(marks, self.status)
        self.labels[key] = np.array([f"cluster {a + 1}" for a in assign])

    # ---------------------------------------------------------------- distributions and curves over time
    def _bins(self, b, v):
        return int(np.clip(int(b.get("bins") or np.ceil(np.log2(max(len(v), 2)) + 1)), 2, 40))

    def _do_histogram(self, b):
        self._need("plot")
        v = self._arr(b.get("values")).ravel()
        counts, edges = np.histogram(v, bins=self._bins(b, v), density=bool(b.get("density")))
        color = self._color(b, "#58C4DD")
        bars = VGroup()
        for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
            p0, p1 = self._xy(e0, 0), self._xy(e1, c)
            bar = Rectangle(width=abs(p1[0] - p0[0]) * 0.96, height=max(abs(p1[1] - p0[1]), 0.005), fill_color=color,
                            fill_opacity=0.7, stroke_color=color, stroke_width=1.5)
            bar.move_to(self._xy((e0 + e1) / 2, 0), aligned_edge=DOWN)
            bars.add(bar)
        self.s.play(AnimationGroup(*(GrowFromEdge(r, DOWN) for r in bars), lag_ratio=0.06), run_time=1.4)
        self._adopt(bars)

    def _series_ys(self, b) -> dict:
        ys = b.get("ys")
        if isinstance(ys, dict):
            return {str(k): self._arr(v).ravel() for k, v in ys.items()}
        return {str(b.get("label") or ""): self._arr(b.get("y")).ravel()}

    def _series_x(self, b, n):
        return self._arr(b["x"]).ravel()[:n] if b.get("x") is not None else np.arange(1, n + 1, dtype=float)

    def _do_series(self, b):
        """Line chart(s), e.g. {"x": epochs, "ys": {"training loss": [...], "validation loss": [...]}}."""
        self._need("plot")
        lines, items = VGroup(), []
        for k, (name, y) in enumerate(self._series_ys(b).items()):
            x = self._series_x(b, len(y))
            color = CLASS_COLORS[k % len(CLASS_COLORS)] if len(self._series_ys(b)) > 1 else self._color(b, "#58C4DD")
            line = VMobject(stroke_color=color, stroke_width=4).set_points_as_corners(
                [self._xy(xi, yi) for xi, yi in zip(x, y)])
            lines.add(line)
            if name:
                items.append((name, color))
        anims = [AnimationGroup(*(Create(m) for m in lines), lag_ratio=0.3)]
        if items:
            anims.append(FadeIn(self._legend_box(items)))
        mark = b.get("mark_min")
        if isinstance(mark, str) and mark in self._series_ys(b):
            y = self._series_ys(b)[mark]
            x = self._series_x(b, len(y))
            i = int(np.argmin(y))
            dot = Dot(self._xy(x[i], y[i]), radius=0.1, color=YELLOW)
            anims += [FadeIn(dot, scale=1.6)] + self._status(f"lowest {mark}: {fmt(y[i])} at x = {fmt(x[i])}", YELLOW)
        self.s.play(*anims, run_time=1.8)
        self._adopt(*(a.mobject for a in anims[1:]), *lines)

    def _do_descent(self, b):
        """Gradient descent on a 1-D curve: x ← x − lr·f'(x), step by step."""
        self._need("plot")
        from studio_lesson import formula
        f = formula(b["f"])
        if not self.graphs:   # the curve is drawn once; later descents walk on it
            self._do_curve({"f": b["f"], "label": b.get("label"), "color": b.get("curve_color")})
        (xlo, xhi), _ = self._ranges()
        x, lr = float(b.get("start", xlo + 0.85 * (xhi - xlo))), float(b.get("lr", 0.1))
        steps, h = int(np.clip(int(b.get("steps") or 8), 1, 30)), 1e-4 * (xhi - xlo)
        fx = lambda t: float(f(np.array([t]))[0])
        color = self._color(b, YELLOW)
        dot = Dot(self._xy(x, fx(x)), radius=0.1, color=color)
        self.s.play(FadeIn(dot, scale=1.6), *self._status(f"start: x = {fmt(x)}, f(x) = {fmt(fx(x))}", color), run_time=0.7)
        trail = VGroup()
        for i in range(1, steps + 1):
            g = (fx(x + h) - fx(x - h)) / (2 * h)
            nx = x - lr * g
            if not np.isfinite(nx) or not xlo <= nx <= xhi:
                self.s.play(*self._status(f"step {i}: the step jumps out of view: the learning rate is too big", "#FC6255"))
                break
            seg = Line(self._xy(x, fx(x)), self._xy(nx, fx(nx)), color=color, stroke_width=3, stroke_opacity=0.7)
            trail.add(seg)
            self.s.play(Create(seg), dot.animate.move_to(self._xy(nx, fx(nx))),
                        *self._status(f"step {i}: slope {fmt(g)}, x = {fmt(nx)}, f(x) = {fmt(fx(nx))}", color),
                        run_time=0.5 if steps > 10 else 0.7)
            x = nx
        self._adopt(dot, trail, self.status)

    # ---------------------------------------------------------------- contour view (loss surfaces)
    def _build_contour(self, view: dict) -> None:
        from studio_lesson import _nice_range
        f = formula_xy(view.get("f") or "x^2 + y^2")
        xr, yr = view.get("x_range") or [-3, 3], view.get("y_range") or view.get("x_range") or [-3, 3]
        xlo, xhi, xs_ = _nice_range(*xr[:2], max_ticks=6)
        ylo, yhi, ys_ = _nice_range(*yr[:2], max_ticks=6)
        self.plot = DataPlot(x_range=(xlo, xhi, xs_), y_range=(ylo, yhi, ys_), width=6.2, height=4.1,
                             x_label=str(view.get("x_label") or "x"), y_label=str(view.get("y_label") or "y"))
        fit_to(self.plot, "main")
        xs, ys = np.linspace(xlo, xhi, 70), np.linspace(ylo, yhi, 70)
        F = f(*np.meshgrid(xs, ys))
        finite = F[np.isfinite(F)]
        n = int(np.clip(int(view.get("levels") or 9), 3, 16))
        levels = np.unique(np.quantile(finite, np.linspace(0.04, 0.92, n))) if finite.size else []
        lines = VGroup()
        for k, lev in enumerate(levels):
            segs = _marching(F, xs, ys, lev)
            if segs:
                color = interpolate_color(LOW, HIGH, k / max(1, len(levels) - 1))
                lines.add(_segments([(self._xy(*a), self._xy(*b_)) for a, b_ in segs], stroke_color=color,
                                    stroke_width=2.2))
        self.s.play(Create(self.plot.axes), FadeIn(self.plot.ticks), FadeIn(self.plot.x_label),
                    FadeIn(self.plot.y_label), run_time=1.0)
        self.s.remove(*self.plot.submobjects)
        self.s.add(self.plot)
        self.s.play(Create(lines, lag_ratio=0.15), run_time=1.6)
        self._adopt(lines)
        self.surface = f

    def _do_descent2d(self, b):
        """Gradient descent on the contour map: p ← p − lr·∇f(p)."""
        self._need("contour")
        f = self.surface
        (xlo, xhi), (ylo, yhi) = self._ranges()
        p = np.asarray(b.get("start") or [xlo + 0.8 * (xhi - xlo), ylo + 0.8 * (yhi - ylo)], float)[:2]
        lr, steps = float(b.get("lr", 0.1)), int(np.clip(int(b.get("steps") or 10), 1, 40))
        h = 1e-4 * (xhi - xlo)
        fv = lambda q: float(f(np.array([q[0]]), np.array([q[1]]))[0])
        color = self._color(b, WHITE)
        dot = Dot(self._xy(*p), radius=0.1, color=color).set_stroke(BLACK, 2)
        self.s.play(FadeIn(dot, scale=1.6), *self._status(f"start: f = {fmt(fv(p))}", color), run_time=0.7)
        trail = VGroup()
        for i in range(1, steps + 1):
            g = np.array([(fv(p + [h, 0]) - fv(p - [h, 0])) / (2 * h), (fv(p + [0, h]) - fv(p - [0, h])) / (2 * h)])
            q = p - lr * g
            if not np.all(np.isfinite(q)) or not (xlo <= q[0] <= xhi and ylo <= q[1] <= yhi):
                self.s.play(*self._status(f"step {i}: jumped out of view: the learning rate is too big", "#FC6255"))
                break
            seg = Line(self._xy(*p), self._xy(*q), color=color, stroke_width=3)
            trail.add(seg)
            self.s.play(Create(seg), dot.animate.move_to(self._xy(*q)),
                        *self._status(f"step {i}: f = {fmt(fv(q))}", color), run_time=0.35 if steps > 15 else 0.55)
            p = q
        self._adopt(dot, trail, self.status)

    # ---------------------------------------------------------------- network view
    def _build_network(self, view: dict) -> None:
        sizes = [int(np.clip(int(n), 1, 8)) for n in (view.get("layers") or [3, 4, 2])][:6]
        names = [str(n) for n in (view.get("names") or [])]
        xs = np.linspace(-4.6, 4.6, len(sizes)) if len(sizes) > 1 else [0.0]
        nodes = []
        for x, n in zip(xs, sizes):
            gap = min(0.95, 3.6 / max(n - 1, 1))
            col = [Circle(radius=0.27, color=WHITE, stroke_width=2.5, fill_color=BLACK, fill_opacity=1)
                   .move_to([x, 0.6 + (n - 1) / 2 * gap - i * gap, 0]) for i in range(n)]
            nodes.append(VGroup(*col))
        edges = [_segments([(a.get_center(), c.get_center()) for a in nodes[l] for c in nodes[l + 1]],
                           stroke_color=GREY_B, stroke_width=1.4, stroke_opacity=0.55) for l in range(len(nodes) - 1)]
        labels = VGroup(*(Text(n, font_size=24, color=GREY_B).move_to([x, -2.0, 0]) for x, n in zip(xs, names)))
        self.s.play(*(Create(e) for e in edges), run_time=1.0)
        self.s.play(*(FadeIn(col, lag_ratio=0.1) for col in nodes), FadeIn(labels), run_time=1.0)
        self.net = {"nodes": nodes, "edges": edges, "values": {}}

    def _layer(self, b) -> int:
        n = len(self.net["nodes"])
        layer = b.get("layer", n - 1)
        if layer in ("input", "first"):
            return 0
        if layer in ("output", "last"):
            return n - 1
        return int(np.clip(int(layer), 0, n - 1))

    def _write_values(self, layer: int, values):
        anims = []
        old = self.net["values"].pop(layer, None)
        if old is not None:
            anims.append(FadeOut(old))
        texts = VGroup()
        for node, v in zip(self.net["nodes"][layer], np.asarray(values, float).ravel()):
            t = Text(fmt(v), font_size=18, color=BLACK if abs(v) >= 0.5 else WHITE)
            if t.width > 0.46:
                t.scale_to_fit_width(0.46)
            texts.add(t.move_to(node))
            anims.append(node.animate.set_fill(interpolate_color(ManimColor(BLACK), HIGH, float(np.clip(abs(v), 0, 1))),
                                               opacity=1))
        self.net["values"][layer] = texts
        return anims + [FadeIn(texts)]

    def _do_activate(self, b):
        self._need("network")
        self.s.play(*self._write_values(self._layer(b), b.get("values") or []), run_time=0.8)

    def _do_forward(self, b):
        """The signal flows from the inputs through every layer to the outputs."""
        self._need("network")
        nodes, edges = self.net["nodes"], self.net["edges"]
        if b.get("values") is not None:
            self.s.play(*self._write_values(0, b["values"]), run_time=0.7)
        for l, e in enumerate(edges):
            flash = e.copy().set_stroke(YELLOW, width=3, opacity=1)
            self.s.play(ShowPassingFlash(flash, time_width=0.7),
                        *(Indicate(n, color=YELLOW, scale_factor=1.15) for n in nodes[l + 1]), run_time=1.0)
        if b.get("outputs") is not None:
            self.s.play(*self._write_values(len(nodes) - 1, b["outputs"]), run_time=0.7)

    def _do_focus(self, b):
        self._need("network")
        col = self.net["nodes"][self._layer(b)]
        targets = [col[int(b["node"])]] if b.get("node") is not None and int(b["node"]) < len(col) else list(col)
        self.s.play(*(Indicate(n, color=YELLOW, scale_factor=1.25) for n in targets), run_time=1.0)

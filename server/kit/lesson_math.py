"""Exact math for lesson plans (numpy only, no Manim): shared by the renderer (studio_lesson) and the
server's plan checker (server/lesson.py), so both compute every value the same way.

* compute(kind, spec, named)  -> a matrix the renderer shows ("correlation", "covariance", ...)
* fill(text, named)           -> a caption with {det(R)}, {corr(X, Y)}, {R[0][1]} ... filled in
* template_names(text)        -> the names a caption's templates need (for the checker)
"""
from __future__ import annotations

import ast
import re
from fractions import Fraction

import numpy as np

COMPUTE_KINDS = ("correlation", "covariance", "centered", "standardized", "transpose", "inverse", "product",
                 "gram", "pseudoinverse", "svd_u", "svd_s", "svd_vt", "eigenvectors", "eigenvalues", "projection")
TEMPLATE_FUNCS = ("det", "corr", "cov", "mean", "var", "std", "norm", "trace", "rank", "eig", "inv", "abs",
                  "sqrt", "round", "sv", "proj", "pinv", "dot", "solve", "T")
_TEMPLATE = re.compile(r"\{([^{}]{1,80})\}")


def fmt(x, decimals: int = 2) -> str:
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.{decimals}f}"
    return "0" if s.strip("-0.") == "" else s


def show(v) -> str:
    """A number, vector or matrix as text."""
    a = np.asarray(v, float)
    if a.ndim == 0:
        return nice(a)
    if a.ndim == 1:
        return "[" + ", ".join(fmt(x) for x in a) + "]"
    return "[" + ", ".join(show(r) for r in a) + "]"


def as_array(x, named: dict) -> np.ndarray:
    if isinstance(x, str):
        if x not in named:
            raise KeyError(f"'{x}' is not defined yet")
        return np.asarray(named[x], float)
    return np.asarray(x, float)


def dataset(of, named: dict) -> np.ndarray:
    """Rows = samples, columns = variables: a dataset/matrix name, a list of column names, or numbers."""
    if isinstance(of, list) and of and all(isinstance(c, str) for c in of):
        return np.column_stack([as_array(c, named).ravel() for c in of])
    return np.atleast_2d(as_array(of, named))


def compute(kind: str, spec: dict, named: dict) -> np.ndarray:
    kind = str(kind).lower()
    ddof = 1 if spec.get("sample") else 0
    if kind in ("correlation", "corr"):
        return np.corrcoef(dataset(spec.get("of", "data"), named), rowvar=False)
    if kind in ("covariance", "cov"):
        return np.cov(dataset(spec.get("of", "data"), named), rowvar=False, ddof=ddof)
    if kind in ("centered", "center"):
        D = dataset(spec.get("of", "data"), named)
        return D - D.mean(axis=0)
    if kind in ("standardized", "standardize"):
        D = dataset(spec.get("of", "data"), named)
        sd = D.std(axis=0, ddof=ddof)
        return (D - D.mean(axis=0)) / np.where(sd > 0, sd, 1)
    if kind == "transpose":
        return np.atleast_2d(as_array(spec.get("of"), named)).T
    if kind == "inverse":
        return np.linalg.inv(np.atleast_2d(as_array(spec.get("of"), named)))
    if kind == "gram":                      # XᵀX
        X = dataset(spec.get("of", "data"), named)
        return X.T @ X
    if kind in ("pseudoinverse", "pinv"):
        return np.linalg.pinv(np.atleast_2d(as_array(spec.get("of"), named)))
    if kind in ("svd_u", "svd_s", "svd_vt"):
        return dict(zip(("svd_u", "svd_s", "svd_vt"), svd_parts(as_array(spec.get("of"), named))))[kind]
    if kind in ("eigenvectors", "eigenvalues"):
        Q, L, _, _ = eigen_parts(as_array(spec.get("of"), named))
        return Q if kind == "eigenvectors" else L
    if kind == "projection":                # the matrix that projects onto the column space of `of`
        A = np.atleast_2d(as_array(spec.get("of"), named))
        A = A.T if A.shape[0] == 1 else A   # a single vector given as a row
        return A @ np.linalg.pinv(A)
    if kind == "product":
        return np.atleast_2d(as_array(spec.get("a"), named)) @ np.atleast_2d(as_array(spec.get("b"), named))
    raise ValueError(f"unknown compute '{kind}' (use one of {', '.join(COMPUTE_KINDS)})")


# ------------------------------------------------------------------ caption templates --
def _corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def _cov(a, b=None):
    if b is None:
        return np.cov(np.atleast_2d(a), rowvar=False, ddof=0)
    return float(np.cov(np.ravel(a), np.ravel(b), ddof=0)[0, 1])


def _eig(a):
    vals = np.linalg.eigvals(np.atleast_2d(a))
    return np.sort(vals.real)[::-1] if np.all(np.abs(vals.imag) < 1e-9) else vals


_FUNCS = {"det": lambda a: float(np.linalg.det(np.atleast_2d(a))), "corr": _corr, "cov": _cov,
          "mean": lambda a: float(np.mean(a)), "var": lambda a: float(np.var(a)), "std": lambda a: float(np.std(a)),
          "norm": lambda a: float(np.linalg.norm(a)), "trace": lambda a: float(np.trace(np.atleast_2d(a))),
          "rank": lambda a: float(np.linalg.matrix_rank(np.atleast_2d(a), tol=1e-9)), "eig": _eig,
          "inv": lambda a: np.linalg.inv(np.atleast_2d(a)), "abs": np.abs, "sqrt": np.sqrt,
          "sv": lambda a: np.linalg.svd(np.atleast_2d(a), compute_uv=False), "proj": lambda b, a: project(b, a),
          "pinv": lambda a: np.linalg.pinv(np.atleast_2d(a)), "dot": lambda a, b: float(np.ravel(a) @ np.ravel(b)),
          "solve": lambda a, b: np.linalg.lstsq(np.atleast_2d(a), np.ravel(b), rcond=None)[0],
          "T": lambda a: np.atleast_2d(a).T,
          "round": lambda a, n=2: np.round(a, int(n))}


def evaluate(expr: str, named: dict):
    tree = ast.parse(expr.strip(), mode="eval")
    ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return as_array(n.id, named)
        if isinstance(n, (ast.List, ast.Tuple)):
            return np.array([ev(e) for e in n.elts], float)
        if isinstance(n, ast.Subscript):
            base, idx = ev(n.value), ev(n.slice)
            return np.asarray(base)[int(idx)]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*(ev(a) for a in n.args))
        raise ValueError(f"cannot compute '{expr}'")

    return ev(tree.body)


def template_names(text: str) -> list[tuple[str, set]]:
    """[(expression, names it uses)] for every {...} in a caption. Raises on a malformed template."""
    out = []
    for expr in _TEMPLATE.findall(str(text)):
        tree = ast.parse(expr.strip(), mode="eval")
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(_FUNCS)
        out.append((expr, names))
    return out


def fill(text: str, named: dict) -> str:
    """Replace {expr} with its exact value; a template that cannot be computed is left as it was."""
    def sub(m):
        try:
            return show(evaluate(m.group(1), named))
        except Exception:
            return m.group(0)
    return _TEMPLATE.sub(sub, str(text))


# ------------------------------------------------------------------ number formatting --
def snap(x: float) -> float:
    """0.1667 -> 1/6 exactly: a value typed to 3-4 digits that is clearly a simple fraction."""
    x = float(x)
    fr = Fraction(x).limit_denominator(12)
    return float(fr) if abs(float(fr) - x) < 6e-4 * max(1.0, abs(x)) else x


def nice(x, decimals: int = 2, approx: bool = True) -> str:
    """A number as people write it: 3, 0.5, 1/6, 35/12 ≈ 2.92 (simple fractions instead of 0.17)."""
    x = float(x)
    if not np.isfinite(x):
        return "∞" if x > 0 else ("−∞" if x < 0 else "undefined")
    fr = Fraction(x).limit_denominator(12)
    if abs(float(fr) - x) < 6e-4 * max(1.0, abs(x)):
        if fr.denominator in (1, 2, 4, 5, 10):          # a short decimal: 3, 3.5, 0.25
            return f"{float(fr):.2f}".rstrip("0").rstrip(".")
        frac = f"{fr.numerator}/{fr.denominator}"
        return frac if not approx or abs(fr.numerator) < fr.denominator else f"{frac} ≈ {fmt(x, decimals)}"
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".") if abs(x - round(x, decimals)) < 1e-9 else fmt(x, decimals)


def _vec(v) -> str:
    return "[" + ", ".join(fmt(x) for x in np.asarray(v, float).ravel()) + "]"


def _matrix_text(M) -> str:
    M = np.atleast_2d(np.asarray(M, float))
    return "[" + ", ".join(_vec(r) for r in M) + "]"


def _p(x) -> str:
    """A number inside a product: negative numbers in brackets."""
    s = fmt(x)
    return f"({s})" if s.startswith("-") else s


# ------------------------------------------------------------------ decompositions --
def svd_parts(A):
    """U, Σ (as a matrix), Vᵀ with A = U·Σ·Vᵀ. For square A, Vᵀ is made a pure rotation (det +1) and, when
    det(A) > 0, so is U: 'rotate, stretch, rotate' is then literally true on screen."""
    A = np.atleast_2d(np.asarray(A, float))
    U, s, Vt = np.linalg.svd(A)
    if A.shape[0] == A.shape[1] and np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
        U[:, -1] *= -1
    if A.shape == (2, 2) and np.trace(Vt) < 0:   # the same rotation the short way round (at most 90°)
        Vt, U = -Vt, -U
    S = np.zeros(A.shape)
    np.fill_diagonal(S, s)
    return U, S, Vt


def eigen_parts(A):
    """Q (eigenvectors as columns), Λ (diagonal), Q⁻¹ with A = Q·Λ·Q⁻¹. Symmetric A: Q is a rotation and
    Q⁻¹ = Qᵀ. Raises when A has no full set of real eigenvectors."""
    A = np.atleast_2d(np.asarray(A, float))
    if A.shape[0] != A.shape[1]:
        raise ValueError("eigendecomposition needs a square matrix")
    symmetric = np.allclose(A, A.T, atol=1e-9)
    vals, Q = np.linalg.eigh(A) if symmetric else np.linalg.eig(A)
    if np.any(np.abs(np.imag(vals)) > 1e-9):
        raise ValueError("this matrix has complex eigenvalues (it rotates), so it has no real eigendecomposition")
    vals, Q = np.real(vals), np.real(Q)
    order = np.argsort(-vals)
    vals, Q = vals[order], Q[:, order]
    if abs(np.linalg.det(Q)) < 1e-9:
        raise ValueError("this matrix is defective: it does not have enough independent eigenvectors")
    for j in range(Q.shape[1]):   # readable signs: the largest entry of each eigenvector is positive
        if Q[np.argmax(np.abs(Q[:, j])), j] < 0:
            Q[:, j] *= -1
    if symmetric and np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q, np.diag(vals), (Q.T if symmetric else np.linalg.inv(Q)), symmetric


def decomposition_steps(kind: str, A):
    """[(matrix, name), ...] in the order they are APPLIED (rightmost factor first)."""
    kind = str(kind).lower()
    if kind == "svd":
        U, S, Vt = svd_parts(A)
        if S.shape[0] != S.shape[1]:
            raise ValueError("the animated SVD needs a square matrix")
        return [(Vt, "Vᵀ"), (S, "Σ"), (U, "U")]
    if kind in ("eigen", "eig", "eigendecomposition", "diagonalization"):
        Q, L, Qi, symmetric = eigen_parts(A)
        return [(Qi, "Qᵀ" if symmetric else "Q⁻¹"), (L, "Λ"), (Q, "Q")]
    raise ValueError(f"unknown decomposition '{kind}' (use svd or eigen)")


def ridge_polyfit(x, y, degree: int, lam: float = 0.0):
    """Least-squares polynomial with an L2 penalty lam·Σw² on all weights except the constant.
    x is scaled to [-1, 1] inside (so the penalty treats every power fairly). Returns (predict, weights)."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    mid, half = (x.max() + x.min()) / 2, max((x.max() - x.min()) / 2, 1e-12)
    V = np.vander((x - mid) / half, degree + 1, increasing=True)
    P = float(lam) * np.eye(degree + 1)
    P[0, 0] = 0.0
    w = np.linalg.lstsq(V.T @ V + P, V.T @ y, rcond=None)[0]

    def predict(t):
        return np.vander((np.asarray(t, float).ravel() - mid) / half, degree + 1, increasing=True) @ w
    return predict, w


def project(b, a):
    """Projection of vector b onto the line through a."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if not np.any(a):
        raise ValueError("cannot project onto the zero vector")
    return (a @ b) / (a @ a) * a


# ------------------------------------------------------------------ formulas --
FORMULA_FUNCS = {"exp": np.exp, "log": np.log, "ln": np.log, "log2": np.log2, "log10": np.log10, "sqrt": np.sqrt,
          "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh, "arctan": np.arctan, "atan": np.arctan,
          "abs": np.abs, "floor": np.floor, "max": np.maximum, "min": np.minimum, "maximum": np.maximum,
          "minimum": np.minimum, "sign": np.sign}
FORMULA_CONSTS = {"pi": np.pi, "e": np.e}


def formula(src: str):
    """A safe f(x) from text like '1/(1+exp(-x))' or 'x^2 - 3' (no eval: the tree is walked by hand)."""
    tree = ast.parse(str(src).replace("^", "**").replace("·", "*").replace("−", "-"), mode="eval")
    ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
           ast.Pow: np.power, ast.Mod: np.mod}

    def ev(n, x):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id == "x":
                return x
            if n.id in FORMULA_CONSTS:
                return FORMULA_CONSTS[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left, x), ev(n.right, x))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand, x)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FORMULA_FUNCS and not n.keywords:
            return FORMULA_FUNCS[n.func.id](*(ev(a, x) for a in n.args))
        raise ValueError(f"unsupported formula part: {ast.dump(n)[:60]}")

    def f(x):
        with np.errstate(all="ignore"):
            return np.asarray(ev(tree.body, np.asarray(x, float)), float) * np.ones_like(np.asarray(x, float))
    f(np.array([0.5, 1.0]))  # fail now, not mid-animation
    return f


# ------------------------------------------------------------------ calculations --
def calc_text(b: dict, named: dict) -> str:
    """A calculation written out step by step with exact numbers (the 'calc' action)."""
    _arr = lambda x: as_array(x, named)  # noqa: E731
    what = str(b.get("what") or "").lower()
    name = str(b.get("name") or "")

    def mat(key="matrix"):
        v = b.get(key) if b.get(key) is not None else b.get("of")
        return np.atleast_2d(_arr(v))

    def label(key, fallback):
        v = b.get(key)
        return v if isinstance(v, str) and len(v) <= 4 else fallback

    if what in ("det", "determinant"):
        A, n = mat(), label("matrix", name or "A")
        d = float(np.linalg.det(A))
        if A.shape == (2, 2):
            a, bb, c, dd = A.ravel()
            return f"det({n}) = a·d − b·c = {_p(a)}·{_p(dd)} − {_p(bb)}·{_p(c)} = {fmt(d)}"
        return f"det({n}) = {fmt(d)}"
    if what in ("matvec", "apply"):
        A, v = mat(), _arr(b.get("vector")).ravel()
        n, vn = label("matrix", "A"), label("vector", "v")
        r = A @ v
        if A.shape[1] <= 3:
            parts = " + ".join(f"{_p(x)}·{_vec(A[:, j])}" for j, x in enumerate(v))
            return f"{n}·{vn} = {parts} = {_vec(r)}"
        return f"{n}·{vn} = {_vec(r)}"
    if what in ("matmul", "product"):
        A, B = np.atleast_2d(_arr(b.get("a"))), np.atleast_2d(_arr(b.get("b")))
        return f"{label('a', 'A')}·{label('b', 'B')} = {_matrix_text(A @ B)}"
    if what == "dot":
        u, v = _arr(b.get("a")).ravel(), _arr(b.get("b")).ravel()
        terms = " + ".join(f"{_p(x)}·{_p(y)}" for x, y in zip(u, v))
        return f"{label('a', 'u')}·{label('b', 'v')} = {terms} = {fmt(float(u @ v))}"
    if what in ("norm", "length"):
        v = _arr(b.get("vector")).ravel()
        inside = " + ".join(f"{_p(x)}²" for x in v)
        return f"‖{label('vector', 'v')}‖ = √({inside}) = {fmt(float(np.linalg.norm(v)))}"
    if what == "inverse":
        A, n = mat(), label("matrix", "A")
        d = float(np.linalg.det(A))
        if abs(d) < 1e-12:
            return f"det({n}) = 0, so {n} has no inverse"
        inv = np.linalg.inv(A)
        if A.shape == (2, 2):
            a, bb, c, dd = A.ravel()
            return (f"{n}⁻¹ = 1/{fmt(d)} · [[{fmt(dd)}, {fmt(-bb)}], [{fmt(-c)}, {fmt(a)}]] = {_matrix_text(inv)}")
        return f"{n}⁻¹ = {_matrix_text(inv)}"
    if what in ("eigen", "eigenvalues"):
        A, n = mat(), label("matrix", "A")
        if b.get("vector") is not None:
            v = _arr(b.get("vector")).ravel()
            Av = A @ v
            k = int(np.argmax(np.abs(v)))
            lam = Av[k] / v[k] if abs(v[k]) > 1e-12 else 0.0
            ok = np.allclose(Av, lam * v, atol=1e-6)
            return (f"{n}·{_vec(v)} = {_vec(Av)} = {fmt(lam)}·{_vec(v)}" if ok
                    else f"{n}·{_vec(v)} = {_vec(Av)}: not a multiple of {_vec(v)}")
        vals = np.linalg.eigvals(A)
        if np.all(np.abs(vals.imag) < 1e-9):
            vals = sorted(vals.real, reverse=True)
            return ", ".join(f"λ{chr(0x2081 + i)} = {fmt(x)}" for i, x in enumerate(vals))
        return f"{n} has no real eigenvalues (it rotates every direction)"
    if what in ("mean", "average", "variance", "var", "std", "sd"):
        x = _arr(b.get("values")).ravel()
        m = float(x.mean())
        mean_txt = f"mean = ({' + '.join(fmt(v) for v in x)}) / {len(x)} = {nice(m)}"
        if what in ("mean", "average"):
            return mean_txt
        ddof = 1 if b.get("sample") else 0
        var = float(((x - m) ** 2).sum() / (len(x) - ddof))
        dev = " + ".join(f"({fmt(v)} − {fmt(m)})²" for v in x[:6]) + (" + …" if len(x) > 6 else "")
        var_txt = f"variance = ({dev}) / {len(x) - ddof} = {nice(var)}"
        if what in ("variance", "var"):
            return var_txt
        return f"std = √{fmt(var)} = {fmt(np.sqrt(var))}"
    if what in ("covariance", "cov", "correlation", "corr"):
        a, c = _arr(b.get("a")).ravel(), _arr(b.get("b")).ravel()
        an, cn = label("a", "X"), label("b", "Y")
        ma, mc = a.mean(), c.mean()
        terms = " + ".join(f"({fmt(x - ma)})·({fmt(y - mc)})" for x, y in zip(a[:4], c[:4]))
        terms += " + …" if len(a) > 4 else ""
        cov = float(((a - ma) * (c - mc)).mean())
        if what in ("covariance", "cov"):
            return f"cov({an}, {cn}) = ({terms}) / {len(a)} = {fmt(cov)}"
        sa, sc = float(a.std()), float(c.std())
        r = cov / (sa * sc) if sa * sc > 0 else 0.0
        return f"r({an}, {cn}) = cov({an}, {cn}) / (σ{an}·σ{cn}) = {fmt(cov)} / ({fmt(sa)}·{fmt(sc)}) = {fmt(r)}"
    if what in ("expectation", "expected", "expected_square", "second_moment", "variance_p", "var_p"):
        x, p = _arr(b.get("values")).ravel(), np.array([snap(v) for v in _arr(b.get("probs")).ravel()])
        terms = " + ".join(f"{fmt(v)}·{fmt(q)}" for v, q in zip(x, p))
        if what in ("expected_square", "second_moment"):
            terms = " + ".join(f"{fmt(v)}²·{nice(q)}" for v, q in zip(x, p))
            return f"E[X²] = {terms} = {nice(float((x ** 2) @ p))}"
        if what in ("variance_p", "var_p"):
            m = float(x @ p)
            terms = " + ".join(f"({fmt(v)} − {nice(m, approx=False)})²·{nice(q)}" for v, q in zip(x[:3], p[:3])) + (" + …" if len(x) > 3 else "")
            return f"Var(X) = {terms} = {nice(float(((x - m) ** 2) @ p))}"
        terms = " + ".join(f"{fmt(v)}·{nice(q)}" for v, q in zip(x, p))
        return f"E[X] = {terms} = {nice(float(x @ p))}"
    if what in ("sum", "total"):
        x = _arr(b.get("values")).ravel()
        return f"{' + '.join(fmt(v) for v in x)} = {fmt(float(x.sum()))}"
    if what in ("svd", "singular_values"):
        A, n = mat(), label("matrix", "A")
        sv = np.linalg.svd(A, compute_uv=False)
        ev = np.sort(np.linalg.eigvalsh(A.T @ A))[::-1]
        return (f"singular values of {n} = √(eigenvalues of {n}ᵀ{n}) = "
                + ", ".join(f"√{fmt(max(e, 0))} = {fmt(v)}" for e, v in zip(ev, sv)))
    if what in ("projection", "project"):
        bvec, a = _arr(b.get("vector")).ravel(), _arr(b.get("onto")).ravel()
        bn, an = label("vector", "b"), label("onto", "a")
        pr = project(bvec, a)
        return (f"proj = ({an}·{bn} / {an}·{an})·{an} = ({fmt(float(a @ bvec))} / {fmt(float(a @ a))})·{_vec(a)} = {_vec(pr)}")
    if what in ("least_squares", "normal_equations", "solve"):
        A, y = np.atleast_2d(_arr(b.get("a"))), _arr(b.get("b")).ravel()
        an, bn = label("a", "A"), label("b", "b")
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        exact = np.allclose(A @ x, y, atol=1e-9)
        return (f"{an}·x = {bn}  →  x = {_vec(x)}" if exact
                else f"no exact solution; least squares: x = ({an}ᵀ{an})⁻¹{an}ᵀ{bn} = {_vec(x)}, error ‖{an}x − {bn}‖ = {fmt(float(np.linalg.norm(A @ x - y)))}")
    if what == "rank":
        A, n = mat(), label("matrix", "A")
        return f"rank({n}) = {int(np.linalg.matrix_rank(A, tol=1e-9))} (independent columns out of {A.shape[1]})"
    if what == "trace":
        A, n = mat(), label("matrix", "A")
        return f"tr({n}) = {' + '.join(_p(v) for v in np.diag(A))} = {fmt(float(np.trace(A)))}"
    if what == "entropy":
        q = _arr(b.get("probs")).ravel()
        q = q[q > 0]
        h = float(-(q * np.log2(q)).sum())
        terms = " − ".join(f"{nice(v)}·log₂({nice(v)})" for v in q[:4]) + (" − …" if len(q) > 4 else "")
        return f"H = −{terms} = {fmt(h)} bits"
    if what in ("kl", "kl_divergence"):
        pp, qq = _arr(b.get("p")).ravel(), _arr(b.get("q")).ravel()
        m = (pp > 0)
        kl = float((pp[m] * np.log2(pp[m] / qq[m])).sum())
        return f"KL(P‖Q) = Σ p·log₂(p/q) = {fmt(kl, 3)} bits"
    if what == "bayes":
        prior, like, false_pos = float(b.get("prior")), float(b.get("likelihood")), float(b.get("false_positive"))
        ev = like * prior + false_pos * (1 - prior)
        return (f"P(A|B) = P(B|A)·P(A) / P(B) = {nice(like)}·{nice(prior)} / "
                f"({nice(like)}·{nice(prior)} + {nice(false_pos)}·{nice(1 - prior)}) = {fmt(like * prior / ev, 3)}")
    raise ValueError(f"unknown calc '{what}' (use det, inverse, matvec, matmul, dot, norm, eigen, svd, rank, trace, "
                     "projection, least_squares, mean, variance, std, covariance, correlation, expectation, "
                     "variance_p, entropy, kl, bayes, sum)")


# ------------------------------------------------------------------ free shapes --
SHAPES = ("circle", "ellipse", "polygon", "norm_ball", "arrow", "line", "label")


def shape_points(do: str, b: dict, named: dict):
    """(points as an N×2 array in data coordinates, closed?) for a shape action."""
    def pt(key, default=None):
        v = b.get(key, default)
        if v is None:
            raise ValueError(f"'{do}' needs '{key}': [x, y]")
        return as_array(v, named).ravel()[:2].astype(float)

    t = np.linspace(0, 2 * np.pi, 97)[:-1]
    if do == "circle":
        c, r = pt("center", [0, 0]), float(b.get("radius", 1))
        return c + r * np.column_stack([np.cos(t), np.sin(t)]), True
    if do == "ellipse":
        c = pt("center", [0, 0])
        if b.get("cov") is not None:      # the k-sigma contour of a Gaussian with this covariance
            C = np.atleast_2d(as_array(b["cov"], named))
            if C.shape != (2, 2) or not np.allclose(C, C.T, atol=1e-9):
                raise ValueError("'cov' must be a symmetric 2x2 matrix")
            vals, vecs = np.linalg.eigh(C)
            if vals.min() < -1e-12:
                raise ValueError("a covariance matrix cannot have negative eigenvalues")
            k = float(b.get("sigmas", 2))
            pts = np.column_stack([np.cos(t), np.sin(t)]) * (k * np.sqrt(np.clip(vals, 0, None)))
            return c + pts @ vecs.T, True
        a_, b_ = float(b.get("a", 2)), float(b.get("b", 1))
        ang = np.radians(float(b.get("angle", 0)))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        return c + (np.column_stack([a_ * np.cos(t), b_ * np.sin(t)]) @ R.T), True
    if do == "norm_ball":                 # all points with ‖x‖_p = radius
        r, pn = float(b.get("radius", 1)), str(b.get("p", 2)).lower()
        c = pt("center", [0, 0])
        if pn in ("1", "1.0", "l1"):
            return c + r * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float), True
        if pn in ("inf", "infinity", "∞", "max", "linf"):
            return c + r * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float), True
        q = float(pn.lstrip("l"))
        if q <= 0:
            raise ValueError("p must be positive")
        cs, sn = np.cos(t), np.sin(t)
        return c + r * np.column_stack([np.sign(cs) * np.abs(cs) ** (2 / q), np.sign(sn) * np.abs(sn) ** (2 / q)]), True
    if do == "polygon":
        pts = np.atleast_2d(as_array(b.get("points"), named))[:, :2]
        if len(pts) < 3:
            raise ValueError("a polygon needs at least 3 points")
        return pts, True
    if do in ("arrow", "line"):
        return np.array([pt("from"), pt("to")]), False
    if do == "label":
        return np.array([pt("at")]), False
    raise ValueError(f"unknown shape '{do}'")

"""Exact math for lesson plans (numpy only, no Manim): shared by the renderer (studio_lesson) and the
server's plan checker (server/lesson.py), so both compute every value the same way.

* compute(kind, spec, named)  -> a matrix the renderer shows ("correlation", "covariance", ...)
* fill(text, named)           -> a caption with {det(R)}, {corr(X, Y)}, {R[0][1]} ... filled in
* template_names(text)        -> the names a caption's templates need (for the checker)
"""
from __future__ import annotations

import ast
import re

import numpy as np

COMPUTE_KINDS = ("correlation", "covariance", "centered", "standardized", "transpose", "inverse", "product")
TEMPLATE_FUNCS = ("det", "corr", "cov", "mean", "var", "std", "norm", "trace", "rank", "eig", "inv", "abs",
                  "sqrt", "round")
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
        return fmt(a)
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
          "round": lambda a, n=2: np.round(a, int(n))}


def evaluate(expr: str, named: dict):
    tree = ast.parse(expr.strip(), mode="eval")
    ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return as_array(n.id, named)
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

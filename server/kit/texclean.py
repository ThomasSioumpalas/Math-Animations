"""LaTeX -> plain Unicode math (no Manim): used by studio_kit for every Text on screen and by the
server for chat answers. Models often write "$A$", "\\(\\det A > 0\\)" or "\\begin{pmatrix}..." even
when asked not to; readers should see A, det A > 0 and [[0, 1], [1, 0]] instead.
"""
from __future__ import annotations

import re

TEX_SYMBOLS = {
    "lambda": "λ", "mu": "μ", "sigma": "σ", "Sigma": "Σ", "alpha": "α", "beta": "β", "gamma": "γ",
    "delta": "δ", "Delta": "Δ", "epsilon": "ε", "varepsilon": "ε", "theta": "θ", "pi": "π", "phi": "φ",
    "omega": "ω", "rho": "ρ", "tau": "τ", "eta": "η", "nu": "ν", "xi": "ξ", "cdot": "·", "times": "×",
    "div": "÷", "to": "→", "rightarrow": "→", "Rightarrow": "⇒", "implies": "⇒", "iff": "⇔",
    "Leftrightarrow": "⇔", "leftarrow": "←", "mapsto": "↦", "neq": "≠", "ne": "≠", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "approx": "≈", "sim": "∼", "infty": "∞", "in": "∈", "sum": "Σ", "prod": "Π",
    "int": "∫", "partial": "∂", "nabla": "∇", "pm": "±", "ldots": "…", "dots": "…", "cdots": "⋯",
    "top": "ᵀ", "perp": "⊥", "circ": "∘", "forall": "∀", "exists": "∃", "det": "det", "log": "log",
    "ln": "ln", "exp": "exp", "sin": "sin", "cos": "cos", "max": "max", "min": "min", "quad": "  ",
    "qquad": "    ", ",": " ", ";": " ", ":": " ", "!": "", "left": "", "right": "", "big": "", "Big": "",
    "{": "{", "}": "}", "|": "‖", "%": "%", "&": "&", "displaystyle": "", "textstyle": "",
}
_SUP = str.maketrans("0123456789+-=()nTi", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵀⁱ")
_SUB = str.maketrans("0123456789+-=()aeijkmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₖₘₙₒₚᵣₛₜᵤᵥₓ")
_HATS = {"i": "î", "j": "ĵ", "k": "k̂", "x": "x̂", "y": "ŷ"}


def tex_to_unicode(s: str) -> str:
    """One piece of LaTeX math as Unicode text."""
    def matrix(m):
        rows = [[c.strip() for c in r.split("&")] for r in re.split(r"\\\\", m.group(1)) if r.strip()]
        if len(rows) == 1:
            return "[" + ", ".join(rows[0]) + "]"
        if all(len(r) == 1 for r in rows):
            return "[" + ", ".join(r[0] for r in rows) + "]ᵀ"
        return "[" + ", ".join("[" + ", ".join(r) + "]" for r in rows) + "]"

    s = re.sub(r"\\\\\[[^\]]*\]", r"\\\\", s)          # row break with spacing: \\[2pt]
    s = re.sub(r"\\begin\{[bpvBV]?matrix\}(.*?)\\end\{[bpvBV]?matrix\}", matrix, s, flags=re.S)
    s = re.sub(r"\\hat\s*\{?\s*\\?([ijkxy])\s*\}?", lambda m: _HATS[m.group(1)], s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname|boldsymbol|vec|hat|bar|mathcal)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\(([\w.]+)\)/", r"\1/", s)          # (1)/(n) -> 1/n, (a+b)/(2) -> (a+b)/2
    s = re.sub(r"/\(([\w.]+)\)", r"/\1", s)
    s = re.sub(r"\\sqrt\{([\w.]+)\}", r"√\1", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    s = re.sub(r"\^\{?\\top\}?", "ᵀ", s)

    def sup(m):
        body = m.group(1) or m.group(2)
        return body.translate(_SUP) if all(c in "0123456789+-=()nTi" for c in body) else "^" + body
    s = re.sub(r"\^\{([^{}]*)\}|\^(\w)", sup, s)

    def sub(m):
        body = m.group(1) or m.group(2)
        return body.translate(_SUB) if all(c in "0123456789+-=()aeijkmnoprstuvx" for c in body) else "_" + body
    s = re.sub(r"_\{([^{}]*)\}|_(\w)", sub, s)
    s = re.sub(r"\\([A-Za-z]+|[,;:!{}|%&])", lambda m: TEX_SYMBOLS.get(m.group(1), m.group(1)), s)
    return s.replace("{", "").replace("}", "")


def plain_math(text):
    """A short on-screen text: $...$ segments and stray commands become Unicode."""
    if not isinstance(text, str) or ("$" not in text and "\\" not in text):
        return text
    if text.count("$") >= 2:  # $...$ segments are math
        parts = re.split(r"(\$[^$]+\$)", text)
        text = "".join(tex_to_unicode(p[1:-1]) if p.startswith("$") and p.endswith("$") and len(p) > 1
                       else p for p in parts)
    if "\\" in text:
        text = re.sub(r"\\([A-Za-z]+)", lambda m: TEX_SYMBOLS.get(m.group(1), m.group(0)), text)
    return re.sub(r"  +", " ", text)


_MATH = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)|\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.S)


def plain_chat(text: str) -> str:
    """A chat answer: \\[...\\], \\(...\\), $$...$$ and $...$ become Unicode math; ``` code ``` stays as is."""
    if not isinstance(text, str) or ("$" not in text and "\\" not in text):
        return text
    out = []
    for part in re.split(r"(```.*?```)", text, flags=re.S):
        if part.startswith("```"):
            out.append(part)
            continue
        part = _MATH.sub(lambda m: tex_to_unicode(next(g for g in m.groups() if g is not None)).strip(), part)
        part = re.sub(r"\\([A-Za-z]+)", lambda m: TEX_SYMBOLS.get(m.group(1), m.group(0)), part)
        out.append(part)
    return "".join(out)

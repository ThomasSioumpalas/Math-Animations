"""Fast checks of the lesson pipeline (no rendering, no network):  uv run python tests/test_lessons.py
Also runs under pytest. Real renders: tools/render_plan.py."""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import lesson_math as LM  # noqa: E402
from lesson_plans import PLANS  # noqa: E402
from server import lesson, prompts, reference, sandbox, sources  # noqa: E402
from server.kit import texclean  # noqa: E402


def test_every_plan_passes_the_checker():
    for name, plan in PLANS.items():
        assert lesson.check(lesson.normalize(copy.deepcopy(plan))) == [], name


def test_prompt_examples_are_valid_plans():
    examples = re.findall(r"```json\s*(\{.*?\})\s*```", prompts.LESSON_SYSTEM, re.S)
    assert len(examples) >= 2
    for e in examples:
        assert lesson.check(lesson.normalize(json.loads(e))) == []


def test_generated_scene_files_pass_the_sandbox_rules():
    for name, plan in PLANS.items():
        p = lesson.normalize(copy.deepcopy(plan))
        assert sandbox.validate(lesson.to_code(p)) == [], name
        starts = lesson.split_parts(p, 4)
        assert starts[0] == 0 and starts == sorted(starts) and len(starts) <= 4
        for k in range(len(starts)):
            assert sandbox.validate(lesson.to_code(p, (k, starts))) == [], (name, k)


def test_checker_catches_what_would_be_wrong_on_screen():
    plan = lesson.normalize(copy.deepcopy(PLANS["decomp"]))
    last = plan["steps"][-1]["beats"]
    last[2]["ridge"] = last[1]["ridge"]                       # the same curve drawn twice
    assert any("exactly the same thing" in p for p in lesson.check(plan))
    plan = lesson.normalize({"kind": "lesson", "steps": [{"title": "t", "view": {"type": "plane"}, "beats": [
        {"say": "flatten", "do": "transform", "matrix": [[1, 2], [2, 4]]}, {"say": "undo", "do": "reset"},
        {"say": "bad calc", "do": "calc", "what": "nonsense"}, {"say": "bad f", "do": "label", "at": [0, 0]}]}]})
    problems = " | ".join(lesson.check(plan))
    assert "cannot be undone" in problems and "unknown calc" in problems and "missing" in problems
    pruned, dropped = lesson.prune(plan)
    assert dropped == 3 and lesson.check(pruned) == []


def test_wrong_numbers_in_captions_are_corrected():
    plan = lesson.normalize(copy.deepcopy(PLANS["corr"]))
    plan["answer"] = "Here det(R) = 2 and corr(X, Z) ≈ 0."
    assert lesson.correct_claims(plan) == 2
    assert "det(R) = 0" in plan["answer"] and "0.45" in plan["answer"]


def test_exact_math():
    steps = LM.decomposition_steps("svd", [[3, 0], [4, 5]])
    product = np.eye(2)
    for m, _ in steps:
        product = m @ product
    assert np.allclose(product, [[3, 0], [4, 5]])
    assert all(abs(np.linalg.det(m) - 1) < 1e-9 for m, n in steps if n != "Σ")     # pure rotations
    Q, L, Qi, symmetric = LM.eigen_parts([[2, 1], [1, 2]])
    assert symmetric and np.allclose(Q @ L @ Qi, [[2, 1], [1, 2]]) and np.allclose(np.diag(L), [3, 1])
    assert LM.nice(1 / 6) == "1/6" and LM.nice(3.5) == "3.5" and LM.nice(35 / 12) == "35/12 ≈ 2.92" and LM.nice(2) == "2"
    text = LM.calc_text({"what": "expectation", "values": [1, 2, 3, 4, 5, 6], "probs": [0.1667] * 6}, {})
    assert text.endswith("= 3.5") and "1/6" in text
    assert LM.fill("{det(A)} {sv(A)[0]}", {"A": np.array([[3.0, 0], [4, 5]])}) == "15 6.71"
    assert np.allclose(LM.project([1, 3], [2, 1]), [2, 1])
    f = LM.formula("1/(1+exp(-x))")
    assert abs(float(f([0.0])[0]) - 0.5) < 1e-12
    for bad in ("__import__('os')", "x.real", "open('f')"):
        try:
            LM.formula(bad)
            raise AssertionError(bad)
        except (ValueError, SyntaxError):
            pass


def test_reference_library():
    ids = [c["id"] for c in reference.CARDS]
    assert len(ids) == len(set(ids))
    for c in reference.CARDS:
        assert c["facts"] and c["keywords"] and all(book in reference.BOOKS for book, _ in c["read"]), c["id"]
        assert all(path in sources.D2L_CHAPTERS for path in c["d2l"]), c["id"]
    assert [c["id"] for c in sources.match_cards("Explain the SVD of a matrix")][0] == "svd"
    assert [c["id"] for c in sources.match_cards("what does the determinant mean?")][0] == "determinant"
    assert sources.match_cards("a ball bouncing on the floor") == []


def test_latex_is_turned_into_plain_math():
    out = texclean.plain_chat(r"\(\det A>0\) and \[C=\begin{pmatrix}0&1\\1&0\end{pmatrix}\] and $\lambda^2$")
    assert out == "det A>0 and C=[[0, 1], [1, 0]] and λ²"


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:
                failed += 1
                print("FAIL", name, "->", type(e).__name__, e)
    sys.exit(1 if failed else 0)

"""Reference material for the teacher model: so explanations rest on textbooks, not on model memory alone.

lookup(request) returns two things:
  * text:  what is added to the teacher's prompt:
           1. cards from the curated library (server/reference.py) that match the request;
           2. a passage from the open textbook "Dive into Deep Learning" (CC BY-SA 4.0), fetched from
              its public repository, cached, and cut down to the paragraphs closest to the request.
  * cited: the sources, shown to the user under the video.
The lookup never blocks a video for long (one short timeout) and never fails it: without network the
cards alone are used. Fetched text is reference DATA: it is cleaned, shortened and clearly fenced in
the prompt, and only this one fixed, trusted repository is ever contacted.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from dataclasses import dataclass, field

import httpx

from . import config, reference

D2L_RAW = "https://raw.githubusercontent.com/d2l-ai/d2l-en/master/"
D2L_SITE = "https://d2l.ai/"
CACHE_DIR = config.DATA_DIR / "refcache"
CACHE_SECONDS = 7 * 24 * 3600
FETCH_TIMEOUT = 3.0
MAX_CARDS = 2
MAX_PASSAGE_CHARS = 1500
_lock = threading.Lock()
_memory: dict[str, tuple[float, str]] = {}
_STOP = set("a an and are as at be by for from how in is it its of on or that the this to what when which why with "
            "explain show me using use each every step steps then first finally also into their them does do can "
            "matrix vector example video animation".split())


@dataclass
class Refs:
    text: str = ""
    cited: list = field(default_factory=list)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Zα-ωΑ-Ω][a-zA-Z\-']+", text.lower()) if w not in _STOP and len(w) > 2]


def match_cards(request: str) -> list[dict]:
    """Cards whose keywords occur in the request, best first (longer keyword matches count more)."""
    text = " " + re.sub(r"\s+", " ", request.lower()) + " "
    scored = []
    for card in reference.CARDS:
        score = sum(1 + kw.count(" ") for kw in card["keywords"] if kw in text)
        if card["title"].lower() in text:
            score += 3
        if score:
            scored.append((score, card))
    scored.sort(key=lambda sc: -sc[0])
    return [c for _, c in scored[:MAX_CARDS]]


def _clean(md: str) -> str:
    """Book markdown -> plain paragraphs: no code, figures, labels, citations or cross-reference markup."""
    md = re.sub(r"```.*?```", "", md, flags=re.S)                    # code blocks
    md = re.sub(r"^\s*:(?:begin_tab|end_tab):.*$", "", md, flags=re.M)
    md = re.sub(r"^\s*!\[.*?\]\(.*?\)\s*$", "", md, flags=re.M)      # figures
    md = re.sub(r"^\s*:(?:label|width|eqlabel|numref|bibliography):.*$", "", md, flags=re.M)
    md = re.sub(r":(?:numref|eqref|cite|citet|ref):`[^`]*`", "", md)
    md = re.sub(r":[a-z_]+:`([^`]*)`", r"\1", md)
    md = re.sub(r"\(\(\(|\)\)\)|\(\*\*|\*\*\)|\[\*\*|\*\*\]", "", md)   # d2l emphasis brackets
    md = re.sub(r"<[^>]+>", "", md)                                  # html
    return md


def _sections(md: str) -> list[tuple[str, str]]:
    """[(heading, text)] of a cleaned chapter."""
    out, head, buf = [], "", []
    for line in _clean(md).splitlines():
        m = re.match(r"^#{1,3}\s+(.*)$", line)
        if m:
            if buf:
                out.append((head, "\n".join(buf)))
            head, buf = m.group(1).strip(), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        out.append((head, "\n".join(buf)))
    return out


def _best_passage(md: str, request: str) -> tuple[str, str, float]:
    """(section heading, the paragraphs of that chapter closest to the request, shortened, relevance)."""
    terms = set(_words(request))
    if not terms:
        return "", "", 0.0
    paras = []   # (score, order, heading, paragraph)
    all_paras = [(h, p_) for h, body in _sections(md) for p_ in re.split(r"\n(?=[A-Z])", body) if len(p_) > 120]
    if not all_paras:
        return "", "", 0.0
    df = {t: sum(1 for _, p_ in all_paras if t in p_.lower()) for t in terms}
    for k, (h, p_) in enumerate(all_paras):
        low = p_.lower()
        score = sum(math.log(1 + len(all_paras) / (1 + df[t])) for t in terms if t in low)
        score += 2 * sum(1 for t in terms if t in h.lower())
        if "exercise" in h.lower() or "summary" == h.lower().strip():
            score *= 0.3
        paras.append((score, k, h, p_))
    best = sorted(paras, key=lambda x: -x[0])[:3]
    if not best or best[0][0] <= 0:
        return "", "", 0.0
    relevance = sum(x[0] for x in best)
    if len({t_ for t_ in terms if any(t_ in x[3].lower() for x in best)}) < min(2, len(terms)):
        return "", "", 0.0   # one stray word in common is not a relevant passage
    best.sort(key=lambda x: x[1])   # reading order
    text, total = [], 0
    for _, _, _, p_ in best:
        p_ = re.sub(r"\s+", " ", p_)[: MAX_PASSAGE_CHARS - total]
        if len(p_) < 80:
            break
        text.append(p_)
        total += len(p_)
    return best[0][2], "\n".join(text), relevance


def _fetch(path: str) -> str:
    """A chapter of the book as markdown: memory cache, then disk cache, then one short request."""
    now = time.time()
    with _lock:
        hit = _memory.get(path)
    if hit and now - hit[0] < CACHE_SECONDS:
        return hit[1]
    file = CACHE_DIR / (hashlib.sha1(path.encode()).hexdigest()[:16] + ".md")
    try:
        if file.is_file() and now - file.stat().st_mtime < CACHE_SECONDS:
            text = file.read_text("utf-8")
            with _lock:
                _memory[path] = (now, text)
            return text
    except OSError:
        pass
    if not re.fullmatch(r"chapter_[a-z0-9\-]+/[a-z0-9\-]+\.md", path):
        return ""
    try:
        r = httpx.get(D2L_RAW + path, timeout=FETCH_TIMEOUT, follow_redirects=False)
        text = r.text if r.status_code == 200 and len(r.text) < 400_000 else ""
    except httpx.HTTPError:
        text = ""
    with _lock:
        _memory[path] = (now if text else now - CACHE_SECONDS + 300, text)   # retry a failure after 5 minutes
    if text:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            file.write_text(text, "utf-8")
        except OSError:
            pass
    return text


def _d2l_url(path: str) -> str:
    return D2L_SITE + path[:-3] + ".html"


def _chapters_for(request: str, cards: list[dict]) -> list[str]:
    """Book chapters to read: those the matched cards point to, else chapters whose file name fits."""
    paths = [p_ for c in cards for p_ in c.get("d2l", [])]
    if not paths:
        terms = set(_words(request))
        scored = [(sum(1 for t in terms if t.rstrip("s") in name.replace("-", " ")), name) for name in D2L_CHAPTERS]
        paths = [name for score, name in sorted(scored, reverse=True)[:2] if score > 0]
    return list(dict.fromkeys(paths))[:3]


def lookup(request: str) -> Refs:
    """Reference text for the teacher + the list of sources for the user. Never raises."""
    try:
        cards = match_cards(request)
        parts, cited = [], []
        for c in cards:
            parts.append(f"[{c['title']}] {c['facts']}")
            for book, chapter in c["read"]:
                cited.append({"title": reference.BOOKS[book], "where": chapter, "url": ""})
        best = ("", "", "", 0.0)
        for path in _chapters_for(request, cards):
            md = _fetch(path)
            if not md:
                continue
            head, passage, relevance = _best_passage(md, request)
            if passage and relevance > best[3] * 1.15:   # earlier chapters (the cards' first choice) win ties
                title = next((ln[2:].strip() for ln in md.splitlines() if ln.startswith("# ")), path)
                best = (title, passage, path, relevance)
        if best[1]:
            parts.append(f"[Textbook passage: Dive into Deep Learning, \"{best[0]}\"] {best[1]}")
            cited.append({"title": reference.BOOKS["D2L"], "where": best[0], "url": _d2l_url(best[2])})
        seen, unique = set(), []
        for c in cited:
            key = (c["title"], c["where"])
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return Refs("\n\n".join(parts), unique[:8])
    except Exception as e:   # reference material is a bonus: never lose a video to it
        print(f"[sources] lookup failed: {type(e).__name__}: {e}", flush=True)
        return Refs()


# Chapters of Dive into Deep Learning that explain mathematics or core ML ideas (file names in its repository).
D2L_CHAPTERS = [
    "chapter_appendix-mathematics-for-deep-learning/distributions.md",
    "chapter_appendix-mathematics-for-deep-learning/eigendecomposition.md",
    "chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.md",
    "chapter_appendix-mathematics-for-deep-learning/information-theory.md",
    "chapter_appendix-mathematics-for-deep-learning/integral-calculus.md",
    "chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.md",
    "chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.md",
    "chapter_appendix-mathematics-for-deep-learning/naive-bayes.md",
    "chapter_appendix-mathematics-for-deep-learning/random-variables.md",
    "chapter_appendix-mathematics-for-deep-learning/single-variable-calculus.md",
    "chapter_appendix-mathematics-for-deep-learning/statistics.md",
    "chapter_attention-mechanisms-and-transformers/attention-pooling.md",
    "chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md",
    "chapter_attention-mechanisms-and-transformers/multihead-attention.md",
    "chapter_attention-mechanisms-and-transformers/queries-keys-values.md",
    "chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.md",
    "chapter_attention-mechanisms-and-transformers/transformer.md",
    "chapter_convolutional-neural-networks/conv-layer.md",
    "chapter_convolutional-neural-networks/why-conv.md",
    "chapter_gaussian-processes/gp-intro.md",
    "chapter_gaussian-processes/gp-priors.md",
    "chapter_linear-classification/classification.md",
    "chapter_linear-classification/generalization-classification.md",
    "chapter_linear-classification/softmax-regression.md",
    "chapter_linear-regression/generalization.md",
    "chapter_linear-regression/linear-regression.md",
    "chapter_linear-regression/weight-decay.md",
    "chapter_multilayer-perceptrons/backprop.md",
    "chapter_multilayer-perceptrons/dropout.md",
    "chapter_multilayer-perceptrons/generalization-deep.md",
    "chapter_multilayer-perceptrons/mlp.md",
    "chapter_multilayer-perceptrons/numerical-stability-and-init.md",
    "chapter_optimization/adagrad.md",
    "chapter_optimization/adam.md",
    "chapter_optimization/convexity.md",
    "chapter_optimization/gd.md",
    "chapter_optimization/lr-scheduler.md",
    "chapter_optimization/minibatch-sgd.md",
    "chapter_optimization/momentum.md",
    "chapter_optimization/optimization-intro.md",
    "chapter_optimization/rmsprop.md",
    "chapter_optimization/sgd.md",
    "chapter_preliminaries/autograd.md",
    "chapter_preliminaries/calculus.md",
    "chapter_preliminaries/linear-algebra.md",
    "chapter_preliminaries/probability.md",
    "chapter_recurrent-neural-networks/bptt.md",
    "chapter_recurrent-neural-networks/rnn.md",
]

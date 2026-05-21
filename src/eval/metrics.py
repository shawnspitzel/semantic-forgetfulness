from __future__ import annotations
import re
import string
from collections import Counter


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def token_f1(prediction: str, references: list[str]) -> float:
    pred = _tokens(prediction)
    best = 0.0
    for ref in references:
        ref_t = _tokens(ref)
        common = sum((Counter(pred) & Counter(ref_t)).values())
        if common == 0:
            continue
        p = common / len(pred) if pred else 0.0
        r = common / len(ref_t) if ref_t else 0.0
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


def exact_match(prediction: str, references: list[str]) -> float:
    norm = _normalize(prediction)
    return float(any(norm == _normalize(r) for r in references))


def needle_hit(prediction: str, answer: str) -> float:
    return float(answer.lower() in prediction.lower())


def rouge_l(prediction: str, references: list[str]) -> float:
    def lcs(a: list[str], b: list[str]) -> int:
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        for i in range(m):
            curr = [0] * (n + 1)
            for j in range(n):
                curr[j + 1] = prev[j] + 1 if a[i] == b[j] else max(curr[j], prev[j + 1])
            prev = curr
        return prev[n]

    pred = _tokens(prediction)
    best = 0.0
    for ref in references:
        ref_t = _tokens(ref)
        if not pred or not ref_t:
            continue
        l = lcs(pred, ref_t)
        p, r = l / len(pred), l / len(ref_t)
        if p + r == 0:
            continue
        best = max(best, 2 * p * r / (p + r))
    return best

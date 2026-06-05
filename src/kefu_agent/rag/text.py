import math
import re
from collections import Counter
from functools import lru_cache
from typing import Iterable


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_][a-z0-9_\-./]*", lowered)
    cjk = [ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"]
    cjk_bigrams = ["".join(cjk[i : i + 2]) for i in range(max(0, len(cjk) - 1))]
    model_like = re.findall(r"[a-z]*\d+[a-z0-9\-]*", lowered)
    return words + cjk + cjk_bigrams + model_like


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    left = a[:length]
    right = b[:length]
    dot = sum(x * y for x, y in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(x * x for x in left)) or 1.0
    right_norm = math.sqrt(sum(y * y for y in right)) or 1.0
    return dot / (left_norm * right_norm)


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    return frozenset(
        {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "on",
            "for",
            "with",
            "how",
            "what",
            "why",
            "when",
            "where",
            "is",
            "are",
            "do",
            "does",
            "can",
            "i",
            "my",
            "it",
            "this",
            "that",
        }
    )


def term_counts(text: str) -> Counter[str]:
    return Counter(token for token in tokenize(text) if token not in stopwords())

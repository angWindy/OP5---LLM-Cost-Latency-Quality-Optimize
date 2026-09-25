"""token_f1 scorer: token-level F1 over free-text answers (Track 2 QA)."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _tokenize(s: str) -> list[str]:
    import re as _re

    return _re.findall(r"\w+|[^\w\s]", s.lower(), flags=_re.UNICODE)


def token_f1(pred: Any, ref: Any) -> dict[str, float]:
    p = _tokenize(str(pred) if pred is not None else "")
    r = _tokenize(str(ref) if ref is not None else "")
    if not p and not r:
        return {"token_f1": 1.0, "precision": 1.0, "recall": 1.0}
    if not p or not r:
        return {"token_f1": 0.0, "precision": 0.0, "recall": 0.0}
    pc = Counter(p)
    rc = Counter(r)
    common = pc & rc
    tp = sum(common.values())
    precision = tp / len(p)
    recall = tp / len(r)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"token_f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4)}

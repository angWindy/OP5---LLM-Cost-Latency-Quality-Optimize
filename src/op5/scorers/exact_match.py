"""exact_match scorer: simple 1/0 exact string match (case/whitespace normalized)."""

from __future__ import annotations

import re
from typing import Any


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def exact_match(pred: Any, ref: Any) -> dict[str, float]:
    p = _to_text(pred)
    r = _to_text(ref)
    return {"exact_match": 1.0 if p == r else 0.0}

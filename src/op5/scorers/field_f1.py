"""field_f1 scorer: macro-F1 over key-value field extraction."""

from __future__ import annotations

from typing import Any


def _normalize(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return str(v)
    return str(v).strip().lower()


def field_f1(pred: dict[str, Any], ref: dict[str, Any]) -> dict[str, float]:
    keys = set(ref.keys()) | set(pred.keys())
    if not keys:
        return {"field_f1": 1.0, "precision": 1.0, "recall": 1.0, "n_correct": 0, "n_total": 0}
    tp = fp = fn = 0
    for k in keys:
        p = _normalize(pred.get(k))
        r = _normalize(ref.get(k))
        if p and r and p == r:
            tp += 1
        elif p and not r:
            fp += 1
        elif r and not p:
            fn += 1
        elif p and r and p != r:
            fp += 1
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "field_f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "n_correct": tp,
        "n_total": len(keys),
    }

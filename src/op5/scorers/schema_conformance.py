"""schema_conformance scorer: validate that pred matches the JSON schema for Track 1 output."""

from __future__ import annotations

from typing import Any


def schema_conformance(pred: dict[str, Any], ref: dict[str, Any]) -> dict[str, float]:
    expected = set(ref.keys()) if isinstance(ref, dict) else set()
    if not expected:
        return {"schema_conformance": 1.0, "n_expected": 0, "n_present": 0}
    if not isinstance(pred, dict):
        return {"schema_conformance": 0.0, "n_expected": len(expected), "n_present": 0}
    n_present = sum(1 for k in expected if k in pred)
    score = n_present / len(expected)
    return {
        "schema_conformance": round(score, 4),
        "n_expected": len(expected),
        "n_present": n_present,
    }

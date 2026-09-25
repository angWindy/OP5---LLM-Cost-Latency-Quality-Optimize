"""citation_precision scorer: fraction of cited chunk-IDs that exist in retrieved chunks."""

from __future__ import annotations

from typing import Any


def citation_precision(pred: Any, ref: Any) -> dict[str, float]:
    pred_dict = pred if isinstance(pred, dict) else {}
    ref_dict = ref if isinstance(ref, dict) else {}
    citations = list(pred_dict.get("citations", []))
    valid = set(ref_dict.get("valid_chunk_ids", []) or [])
    if not citations:
        return {"citation_precision": 0.0, "n_cited": 0, "n_valid": len(valid), "n_correct": 0}
    if not valid:
        return {"citation_precision": 0.0, "n_cited": len(citations), "n_valid": 0, "n_correct": 0}
    n_correct = sum(1 for c in citations if c in valid)
    return {
        "citation_precision": round(n_correct / len(citations), 4),
        "n_cited": len(citations),
        "n_valid": len(valid),
        "n_correct": n_correct,
    }

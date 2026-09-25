"""pii_precision_recall scorer: do redaction spans cover all GT PII types? (type-level)"""

from __future__ import annotations

from typing import Any


def pii_precision_recall(pred: Any, ref: Any) -> dict[str, float]:
    pred_spans = pred if isinstance(pred, list) else []
    ref_spans = ref if isinstance(ref, list) else []
    pred_types = {s.get("type") for s in pred_spans if s.get("action") != "keep_token"}
    ref_types = {s.get("type") for s in ref_spans}
    tp = len(pred_types & ref_types)
    fp = len(pred_types - ref_types)
    fn = len(ref_types - pred_types)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "pii_precision": round(precision, 4),
        "pii_recall": round(recall, 4),
        "pii_f1": round(f1, 4),
        "n_pred_types": len(pred_types),
        "n_ref_types": len(ref_types),
    }

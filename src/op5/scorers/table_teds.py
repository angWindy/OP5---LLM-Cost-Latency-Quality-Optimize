"""table_teds scorer: simplified Table Edit Distance Score for Phase 03."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def _flatten_cells(tables: list[dict]) -> list[str]:
    cells: list[str] = []
    for t in tables:
        for c in t.get("cells", []) or t.get("rows", []) or []:
            if isinstance(c, list):
                cells.extend(str(x).strip().lower() for x in c if x)
            elif isinstance(c, dict):
                cells.append(str(c.get("text", "")).strip().lower())
            else:
                cells.append(str(c).strip().lower())
    return [c for c in cells if c]


def table_teds(pred: Any, ref: Any) -> dict[str, float]:
    if isinstance(pred, dict):
        pred_cells = _flatten_cells(pred.get("tables") or [])
    elif isinstance(pred, list):
        pred_cells = _flatten_cells(pred)
    else:
        pred_cells = []

    if isinstance(ref, dict):
        ref_cells = _flatten_cells(ref.get("tables") or [])
    elif isinstance(ref, list):
        ref_cells = _flatten_cells(ref)
    else:
        ref_cells = []

    if not ref_cells and not pred_cells:
        return {"teds": 1.0, "n_pred": 0, "n_ref": 0}
    if not ref_cells:
        return {"teds": 0.0, "n_pred": len(pred_cells), "n_ref": 0}
    if not pred_cells:
        return {"teds": 0.0, "n_pred": 0, "n_ref": len(ref_cells)}

    sm = SequenceMatcher(None, pred_cells, ref_cells)
    matching = sum(block.size for block in sm.get_matching_blocks())
    teds = 2.0 * matching / (len(pred_cells) + len(ref_cells))
    return {"teds": round(teds, 4), "n_pred": len(pred_cells), "n_ref": len(ref_cells)}

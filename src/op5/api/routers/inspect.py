"""Inspect endpoint — read JSONL eval logs for a given case_id and return predictions + scores.

GET /inspect/{case_id}
  Reads results/phase-03-track1.jsonl and results/phase-03-track2.jsonl,
  filters to the given case_id, and returns rows + a summary block.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException

from op5.api.schemas import InspectResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inspect", tags=["inspect"])

DEFAULT_TRACK1 = Path("results/phase-03-track1.jsonl")
DEFAULT_TRACK2 = Path("results/phase-03-track2.jsonl")


def _load_rows(path: Path, case_id: str) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("case_id") == case_id:
                out.append(rec)
    return out


@router.get("/{case_id}", response_model=InspectResponse)
def inspect_case(case_id: str, track1_path: str | None = None, track2_path: str | None = None) -> InspectResponse:
    p1 = Path(track1_path) if track1_path else DEFAULT_TRACK1
    p2 = Path(track2_path) if track2_path else DEFAULT_TRACK2
    if not p1.exists() and not p2.exists():
        raise HTTPException(status_code=404, detail=f"Neither {p1} nor {p2} exists; run a Track runner first.")

    t1_rows = _load_rows(p1, case_id)
    t2_rows = _load_rows(p2, case_id)

    summary: dict[str, dict] = {}
    for track_name, rows in [("track1", t1_rows), ("track2", t2_rows)]:
        by_cfg: dict[str, dict] = defaultdict(lambda: {"n": 0, "cost": 0.0, "lat": 0.0})
        for r in rows:
            cfg = r.get("config", "?")
            by_cfg[cfg]["n"] += 1
            by_cfg[cfg]["cost"] += float(r.get("cost_usd") or 0)
            by_cfg[cfg]["lat"] += float(r.get("latency_ms") or 0)
        for cfg, agg in by_cfg.items():
            n = max(1, agg["n"])
            summary[f"{track_name}.{cfg}"] = {
                "n": agg["n"],
                "avg_cost_usd": round(agg["cost"] / n, 6),
                "avg_latency_ms": round(agg["lat"] / n, 1),
                "deployment_id": (rows[0].get("deployment_id") if rows else ""),
            }

    return InspectResponse(case_id=case_id, track1=t1_rows, track2=t2_rows, summary=summary)

"""Deterministic router for Phase 03.

Routes an input (Track 1 extraction or Track 2 RAG) to a model tier
(cheap / mid / strong) for observability and cost tracking.

IMPORTANT: As of 2026-09-26 all tiers route to the SAME model:
  gemini-3.5-flash-lite ($0.075 / $0.30 per 1M tokens).
The tier labels are kept for logging and cost analytics only —
do NOT change DEPLOYMENTS to add stronger models without a paired experiment.

The rules are pure functions of input features (no LLM calls, fully reproducible).
Per master plan §4.2:

Track 1 (extraction):
  T1-R1: n_fields <= 10 AND no tables -> cheap
  T1-R2: n_fields <= 30 OR small tables -> mid
  T1-R3: n_fields > 30 OR complex tables -> strong

Track 2 (RAG):
  T2-R1: est_tokens < 2000 AND no multi-clause synthesis -> cheap
  T2-R2: est_tokens < 8000 OR moderate complexity -> mid
  T2-R3: est_tokens >= 8000 OR multi-clause synthesis -> strong
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEPLOYMENTS = {
    # Single model for all tiers — gemini-3.5-flash-lite is fast, cheap,
    # and strong enough for both extraction (Track 1) and RAG (Track 2).
    # Tier labels are kept for observability / cost tracking only.
    "cheap": "gemini-3.5-flash-lite",
    "mid": "gemini-3.5-flash-lite",
    "strong": "gemini-3.5-flash-lite",
}


@dataclass(frozen=True)
class RoutingFeatures:
    n_fields: int = 0
    has_tables: bool = False
    n_table_rows: int = 0
    n_table_cols: int = 0
    est_tokens: int = 0
    requires_multi_clause_synthesis: bool = False
    is_refusable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_fields": self.n_fields,
            "has_tables": self.has_tables,
            "n_table_rows": self.n_table_rows,
            "n_table_cols": self.n_table_cols,
            "est_tokens": self.est_tokens,
            "requires_multi_clause_synthesis": self.requires_multi_clause_synthesis,
            "is_refusable": self.is_refusable,
        }


def route_track1(features: RoutingFeatures) -> str:
    if not features.has_tables and features.n_fields <= 10:
        return "cheap"
    if features.n_fields <= 30 and not (features.n_table_rows > 10 or features.n_table_cols > 6):
        return "mid"
    return "strong"


def route_track2(features: RoutingFeatures) -> str:
    if features.est_tokens < 2000 and not features.requires_multi_clause_synthesis:
        return "cheap"
    if features.est_tokens < 8000 and not features.requires_multi_clause_synthesis:
        return "mid"
    return "strong"


def deployment_id_for(tier: str) -> str:
    return DEPLOYMENTS.get(tier, DEPLOYMENTS["cheap"])


def route(features: RoutingFeatures, track: str) -> dict[str, Any]:
    track = track.lower()
    if track == "track1":
        tier = route_track1(features)
        rule = "T1-R1" if tier == "cheap" else ("T1-R2" if tier == "mid" else "T1-R3")
    elif track == "track2":
        tier = route_track2(features)
        rule = "T2-R1" if tier == "cheap" else ("T2-R2" if tier == "mid" else "T2-R3")
    else:
        raise ValueError(f"Unknown track: {track!r}")
    return {
        "tier": tier,
        "deployment_id": deployment_id_for(tier),
        "features": features.to_dict(),
        "rule": rule,
    }


def extract_features_track1(ocr_redacted: dict[str, Any], expected_fields: list[str]) -> RoutingFeatures:
    n_fields = len(expected_fields)
    tables = ocr_redacted.get("tables", []) or []
    has_tables = len(tables) > 0
    n_rows = max((len(t.get("cells", [])) for t in tables), default=0)
    n_cols = max(((max((c.get("col", 0) for c in t.get("cells", [])), default=0) + 1) for t in tables), default=0)
    redacted_text = ocr_redacted.get("redacted_text", "") or " ".join(b.get("text", "") for b in ocr_redacted.get("text_blocks", []) or [])
    est_tokens = max(1, len(redacted_text.split()))
    return RoutingFeatures(
        n_fields=n_fields,
        has_tables=has_tables,
        n_table_rows=n_rows,
        n_table_cols=n_cols,
        est_tokens=est_tokens,
    )


def extract_features_track2(question: str, chunks: list[str], is_refusable: bool = False) -> RoutingFeatures:
    q_tokens = len(question.split())
    c_tokens = sum(len(c.split()) for c in chunks)
    est_tokens = q_tokens + c_tokens
    synth = sum(1 for c in chunks if any(k in c.lower() for k in ["điều khoản", "khoản", "mục", "điều", "section", "clause"])) >= 2
    return RoutingFeatures(
        est_tokens=est_tokens,
        requires_multi_clause_synthesis=synth,
        is_refusable=is_refusable,
    )


if __name__ == "__main__":
    cases = [
        ("T1-R1", RoutingFeatures(n_fields=8, has_tables=False), "track1", "cheap"),
        ("T1-R1", RoutingFeatures(n_fields=10, has_tables=False), "track1", "cheap"),
        ("T1-R2", RoutingFeatures(n_fields=15, has_tables=True, n_table_rows=5, n_table_cols=4), "track1", "mid"),
        ("T1-R3", RoutingFeatures(n_fields=50, has_tables=True, n_table_rows=20, n_table_cols=8), "track1", "strong"),
        ("T1-R3", RoutingFeatures(n_fields=5, has_tables=True, n_table_rows=15, n_table_cols=2), "track1", "strong"),
        ("T2-R1", RoutingFeatures(est_tokens=1500, requires_multi_clause_synthesis=False), "track2", "cheap"),
        ("T2-R2", RoutingFeatures(est_tokens=5000, requires_multi_clause_synthesis=False), "track2", "mid"),
        ("T2-R3", RoutingFeatures(est_tokens=10000, requires_multi_clause_synthesis=True), "track2", "strong"),
        ("T2-R2", RoutingFeatures(est_tokens=7000, requires_multi_clause_synthesis=False), "track2", "mid"),
        ("T2-R3", RoutingFeatures(est_tokens=1500, requires_multi_clause_synthesis=True), "track2", "strong"),
    ]
    fails = 0
    for name, feat, track, expected_tier in cases:
        result = route(feat, track)
        ok = result["tier"] == expected_tier
        if not ok:
            fails += 1
            print(f"  FAIL {name}: expected={expected_tier} got={result['tier']}")
    print(f"\n{'OK' if fails == 0 else 'FAIL'}: {len(cases) - fails}/{len(cases)} cases passed")

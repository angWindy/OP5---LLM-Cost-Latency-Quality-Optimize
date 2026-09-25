"""Audit helpers for redaction runs.

Writes per-case audit records (case_id, policy_version, span counts, total redactions)
to STORAGE.postgres() and returns the AuditRecord that was inserted.
"""

from __future__ import annotations

from typing import Any

from op5.storage import STORAGE, AuditRecord


def log_redaction(
    case_id: str,
    policy_version: str,
    span_count_by_type: dict[str, int],
    source: str,
) -> AuditRecord:
    record = AuditRecord(
        case_id=case_id,
        policy_version=policy_version,
        span_count=span_count_by_type,
        total_redactions=sum(span_count_by_type.values()),
        source=source,
    )
    STORAGE().postgres().insert_audit(record)
    return record


def fetch_redaction_audit(case_id: str) -> list[dict[str, Any]]:
    return STORAGE().postgres().query_audit(case_id)

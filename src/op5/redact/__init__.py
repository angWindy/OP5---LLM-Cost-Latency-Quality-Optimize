"""Sensitive-PII Redaction Layer for Phase 03.

Public API:
- Redactor (in redactor.py) — apply policy to OCRResult or plain text.
- load_policy / default_policy (in policy.py) — read or build a policy.
- log_redaction (in audit.py) — write per-case audit record to storage.
"""

from op5.redact.audit import fetch_redaction_audit, log_redaction
from op5.redact.policy import CategoryPolicy, Policy, default_policy, load_policy
from op5.redact.redactor import PIISpan, RedactionResult, Redactor, redactor_from_policy_file

__all__ = [
    "CategoryPolicy",
    "PIISpan",
    "Policy",
    "RedactionResult",
    "Redactor",
    "default_policy",
    "fetch_redaction_audit",
    "load_policy",
    "log_redaction",
    "redactor_from_policy_file",
]

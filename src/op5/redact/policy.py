"""Loader for redaction policy YAML.

A policy file controls which PII categories get redacted and how (mask, hash, or keep_token).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CategoryPolicy:
    name: str
    action: str
    mask: str = "[REDACTED]"

    def __post_init__(self) -> None:
        if self.action not in {"mask", "hash", "keep_token"}:
            raise ValueError(f"Unknown action {self.action!r} for category {self.name}")


@dataclass
class Policy:
    policy_version: str
    default_action: str = "mask"
    categories: dict[str, CategoryPolicy] = field(default_factory=dict)
    exempt_context_keys: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        cats = {
            name: CategoryPolicy(
                name=name,
                action=cfg.get("action", "mask"),
                mask=cfg.get("mask", f"[REDACTED:{name.upper()}]"),
            )
            for name, cfg in (data.get("categories") or {}).items()
        }
        return cls(
            policy_version=data.get("policy_version", "phase03.redact.v1"),
            default_action=data.get("default_action", "mask"),
            categories=cats,
            exempt_context_keys=list(data.get("exempt_context_keys", [])),
        )


def load_policy(path: str | Path) -> Policy:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Policy file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Policy.from_dict(data or {})


def default_policy() -> Policy:
    return Policy.from_dict(
        {
            "policy_version": "phase03.redact.v1",
            "default_action": "mask",
            "categories": {
                "cccd": {"action": "mask", "mask": "[REDACTED:CCCD]"},
                "phone_vn": {"action": "mask", "mask": "[REDACTED:PHONE]"},
                "email": {"action": "mask", "mask": "[REDACTED:EMAIL]"},
                "tax_id": {"action": "mask", "mask": "[REDACTED:MST]"},
                "bank_account": {"action": "mask", "mask": "[REDACTED:STK]"},
                "vin": {"action": "mask", "mask": "[REDACTED:VIN]"},
                "license_plate": {"action": "mask", "mask": "[REDACTED:PLATE]"},
                "person_name": {"action": "keep_token", "mask": "[REDACTED:NAME]"},
            },
            "exempt_context_keys": [
                "contract_no",
                "model",
                "version",
                "color",
                "delivery_date",
            ],
        }
    )

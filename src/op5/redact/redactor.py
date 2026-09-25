"""Redactor: applies policy to OCRResult text + tables, returns OCRResult_redacted.

For each PII span detected by op5.redact.patterns, it can either:
- MASK: replace span text with the category-specific mask string.
- HASH: replace span text with sha256(original)[:8] (still traceable but irreversible).
- KEEP_TOKEN: leave the original (used for person_name which is needed for extraction).

Each OCRResult in/out preserves bbox + table structure; the only changes are to the
text fields, plus a new top-level redacted: true flag and pii_spans[] audit list.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from op5.redact.patterns import all_patterns
from op5.redact.policy import CategoryPolicy, Policy, default_policy


@dataclass
class PIISpan:
    type: str
    start: int
    end: int
    original_hash: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "original_hash": self.original_hash,
            "action": self.action,
        }


@dataclass
class RedactionResult:
    text: str
    spans: list[PIISpan] = field(default_factory=list)
    span_count_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def total_redactions(self) -> int:
        return sum(1 for s in self.spans if s.action != "keep_token")


class Redactor:
    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or default_policy()
        self._patterns = all_patterns()

    def redact_text(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(text=text)

        raw_spans: list[PIISpan] = []

        for entry in self._patterns:
            cat = self.policy.categories.get(entry.name)
            if cat is None:
                continue
            for m in entry.regex.finditer(text):
                original = m.group(0)
                raw_spans.append(
                    PIISpan(
                        type=entry.name,
                        start=m.start(),
                        end=m.end(),
                        original_hash=hashlib.sha256(original.encode("utf-8")).hexdigest()[:12],
                        action=cat.action,
                    )
                )

        # Resolve overlapping spans: keep longest-first, then earliest start.
        # If two spans have the same length, keep the one whose category name comes first
        # in the patterns list (giving priority order to more specific patterns).
        priority = {entry.name: i for i, entry in enumerate(self._patterns)}
        raw_spans.sort(key=lambda s: (-(s.end - s.start), s.start, priority.get(s.type, 999)))

        accepted: list[PIISpan] = []
        for sp in raw_spans:
            overlap = False
            for prev in accepted:
                if not (sp.end <= prev.start or sp.start >= prev.end):
                    overlap = True
                    break
            if not overlap:
                accepted.append(sp)

        accepted.sort(key=lambda s: (s.start, s.end))

        out = text
        for span in reversed(accepted):
            if span.action == "keep_token":
                continue
            cat = self.policy.categories.get(span.type)
            if cat is None:
                continue
            original = out[span.start : span.end]
            out = out[: span.start] + self._render(cat, original) + out[span.end :]

        count_by_type: dict[str, int] = {}
        for s in accepted:
            if s.action == "keep_token":
                continue
            count_by_type[s.type] = count_by_type.get(s.type, 0) + 1

        return RedactionResult(text=out, spans=accepted, span_count_by_type=count_by_type)

    def _render(self, cat: CategoryPolicy, original: str) -> str:
        if cat.action == "mask":
            return cat.mask
        if cat.action == "hash":
            return "[HASH:" + hashlib.sha256(original.encode("utf-8")).hexdigest()[:8] + "]"
        return original

    def redact_ocr_result(self, ocr_result: dict[str, Any]) -> dict[str, Any]:
        new = {k: v for k, v in ocr_result.items()}
        new["redacted"] = True
        all_spans: list[dict[str, Any]] = []

        for blk in new.get("text_blocks", []) or []:
            if "text" in blk and blk["text"]:
                res = self.redact_text(blk["text"])
                blk["text"] = res.text
                for s in res.spans:
                    d = s.to_dict()
                    d["block_page"] = blk.get("page")
                    all_spans.append(d)

        for tbl in new.get("tables", []) or []:
            for cell in tbl.get("cells", []) or []:
                if "text" in cell and cell["text"]:
                    res = self.redact_text(cell["text"])
                    cell["text"] = res.text
                    for s in res.spans:
                        d = s.to_dict()
                        d["block_page"] = tbl.get("page")
                        all_spans.append(d)

        new["redacted_text"] = "\n".join(b.get("text", "") for b in new.get("text_blocks", []) or [])
        new["pii_spans"] = all_spans
        new["policy_version"] = self.policy.policy_version
        new["pii_span_count_by_type"] = self._aggregate_counts(all_spans)
        return new

    @staticmethod
    def _aggregate_counts(spans: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in spans:
            if s.get("action") == "keep_token":
                continue
            t = s.get("type", "?")
            out[t] = out.get(t, 0) + 1
        return out


def redactor_from_policy_file(path: str) -> Redactor:
    from op5.redact.policy import load_policy

    return Redactor(load_policy(path))

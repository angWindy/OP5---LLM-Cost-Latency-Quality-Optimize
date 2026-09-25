"""Regex + dictionary patterns for Sensitive-PII redaction.

8 categories covered:
- cccd          (9 hoặc 12 chữ số)
- phone_vn      (+84-... hoặc 0xxx..., bỏ qua 0xxx-xxx-xxx ngắn)
- email         (RFC 5322 simplified)
- tax_id        (10-13 chữ số có prefix)
- bank_account  (8-16 chữ số liền, có thể có spaces/dashes)
- vin           (17 ký tự [A-HJ-NPR-Z0-9])
- license_plate (Vietnamese biển số)
- person_name   (heuristic 2-4 word VN — chỉ match khi ở gần keyword)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


@dataclass(frozen=True)
class PatternEntry:
    name: str
    regex: Pattern[str]
    description: str


_PATTERNS: list[PatternEntry] = [
    PatternEntry(
        "cccd",
        re.compile(r"\b(?:\d{9}|\d{12})\b"),
        "Vietnam CCCD/CMND: 9 hoặc 12 chữ số",
    ),
    PatternEntry(
        "phone_vn",
        re.compile(r"(?:\+84[\s\-]?|0)(?:\d{2,3}[\s\-]?)\d{3}[\s\-]?\d{3,4}\b"),
        "Số điện thoại VN: +84-... hoặc 0xxx.xxx.xxx (10-11 chữ số sau 0)",
    ),
    PatternEntry(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "Địa chỉ email",
    ),
    PatternEntry(
        "tax_id",
        re.compile(r"(?:MST|TAX\s*ID|Mã\s*số\s*thuế|Ma\s*so\s*thue)[:\s\-]+\d{10}(?:[\-\s]?\d{1,3})?\b", re.IGNORECASE),
        "MST: 10-13 chữ số (preceded by MST/Tax ID keyword)",
    ),
    PatternEntry(
        "bank_account",
        re.compile(r"(?:STK|so\s*tk|số\s*tk|tai\s*khoan|tài\s*khoản)[:\s]+\d[\d\s\-]{7,18}\d"),
        "Số tài khoản ngân hàng (8-16 chữ số, đứng sau keyword)",
    ),
    PatternEntry(
        "vin",
        re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
        "Vehicle Identification Number (17 ký tự VIN)",
    ),
    PatternEntry(
        "license_plate",
        re.compile(r"\b\d{2}[A-Z][\-\s]?\d{4,5}\b"),
        "Biển số xe VN: 59X-12345 hoặc 59X 1234",
    ),
    PatternEntry(
        "person_name",
        re.compile(
            r"(?:ông|bà|anh|chị|cô|chú|bác|em|khách\s*hàng|người\s*mua|"
            r"người\s*bán|đại\s*diện|người\s*đại\s*diện|buyer|seller|representative|"
            r"người\s*ký|người\s*lập)\s*[:\-]?\s*"
            r"([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịòóỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+"
            r"(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịòóỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+){0,3})"
        ),
        "Họ tên người đại diện/mua/bán (heuristic VN)",
    ),
]


def all_patterns() -> list[PatternEntry]:
    """Return all compiled patterns (used by Redactor)."""
    return list(_PATTERNS)


def get_pattern(name: str) -> PatternEntry | None:
    """Look up one pattern by name."""
    for p in _PATTERNS:
        if p.name == name:
            return p
    return None

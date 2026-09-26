"""OCR Adapter abstract base class for Phase 03.

Every OCR provider implements the same run(pdf_bytes) -> OCRResult interface
so the rest of the pipeline can swap providers transparently.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class OCRBlock:
    page: int
    bbox: list[float]
    text: str
    confidence: float


@dataclass
class OCRTable:
    page: int
    bbox: list[float]
    cells: list[dict[str, Any]]


@dataclass
class OCRResult:
    case_id: str
    ocr_provider: str
    ocr_version: str
    page_count: int
    text_blocks: list[OCRBlock]
    tables: list[OCRTable]
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ocr_provider": self.ocr_provider,
            "ocr_version": self.ocr_version,
            "page_count": self.page_count,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "text_blocks": [
                {
                    "page": b.page,
                    "bbox": b.bbox,
                    "text": b.text,
                    "confidence": b.confidence,
                }
                for b in self.text_blocks
            ],
            "tables": [
                {
                    "page": t.page,
                    "bbox": t.bbox,
                    "cells": t.cells,
                }
                for t in self.tables
            ],
        }


class OCRAdapter(ABC):
    name: ClassVar[str] = ""
    version: ClassVar[str] = "unknown"

    @abstractmethod
    def run(self, pdf_path: Path) -> OCRResult: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    def __call__(self, pdf_path: Path) -> OCRResult:
        if not self.is_available():
            return OCRResult(
                case_id=pdf_path.stem,
                ocr_provider=self.name,
                ocr_version=self.version,
                page_count=0,
                text_blocks=[],
                tables=[],
                error=f"{self.name} not available on this host",
            )
        t0 = time.time()
        try:
            result = self.run(pdf_path)
        except Exception as e:
            return OCRResult(
                case_id=pdf_path.stem,
                ocr_provider=self.name,
                ocr_version=self.version,
                page_count=0,
                text_blocks=[],
                tables=[],
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )
        result.latency_ms = (time.time() - t0) * 1000
        return result

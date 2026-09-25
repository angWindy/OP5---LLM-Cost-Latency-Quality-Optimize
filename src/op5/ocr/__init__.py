"""OCR adapter registry for Phase 03.

Usage:
    from op5.ocr import ADAPTERS
    result = ADAPTERS["easyocr"](Path("contract.pdf"))
"""

from op5.ocr.base import OCRAdapter, OCRBlock, OCRResult, OCRTable
from op5.ocr.easyocr_adapter import EasyOCRAdapter
from op5.ocr.paddleocr_adapter import PaddleOCRAdapter
from op5.ocr.tesseract_adapter import TesseractAdapter

ADAPTERS: dict[str, OCRAdapter] = {
    "paddleocr": PaddleOCRAdapter(),
    "tesseract": TesseractAdapter(),
    "easyocr": EasyOCRAdapter(),
}


def get_adapter(name: str) -> OCRAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown OCR adapter: {name!r}. Available: {list(ADAPTERS)}")
    return ADAPTERS[name]


__all__ = [
    "ADAPTERS",
    "EasyOCRAdapter",
    "OCRAdapter",
    "OCRBlock",
    "OCRResult",
    "OCRTable",
    "PaddleOCRAdapter",
    "TesseractAdapter",
    "get_adapter",
]

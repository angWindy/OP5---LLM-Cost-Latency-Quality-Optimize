"""Tesseract OCR adapter (via pytesseract wrapper, CPU-only).

Requires the tesseract binary installed on the host (apt install tesseract-ocr tesseract-ocr-vie).
Returns empty OCRResult with error if binary missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from op5.ocr.base import OCRAdapter, OCRBlock, OCRResult


class TesseractAdapter(OCRAdapter):
    name = "tesseract"
    version = "5.x"

    def __init__(self, lang: str = "vie+eng") -> None:
        self.lang = lang

    def is_available(self) -> bool:
        return shutil.which("tesseract") is not None

    def run(self, pdf_path: Path) -> OCRResult:
        import pypdfium2 as pdfium
        import pytesseract
        from PIL import Image

        pdf = pdfium.PdfDocument(str(pdf_path))
        page_count = len(pdf)
        blocks: list[OCRBlock] = []

        for i in range(page_count):
            img = pdf[i].render(scale=2).to_pil()
            tmp = pdf_path.with_suffix(f".tes_p{i}.png")
            try:
                img.save(tmp)
                data = pytesseract.image_to_data(Image.open(tmp), lang=self.lang, output_type=pytesseract.Output.DICT)
                for j, text in enumerate(data["text"]):
                    if not text.strip():
                        continue
                    blocks.append(
                        OCRBlock(
                            page=i + 1,
                            bbox=[
                                float(data["left"][j]),
                                float(data["top"][j]),
                                float(data["left"][j] + data["width"][j]),
                                float(data["top"][j] + data["height"][j]),
                            ],
                            text=text,
                            confidence=float(data["conf"][j]) / 100.0 if int(data["conf"][j]) > 0 else 0.0,
                        )
                    )
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

        return OCRResult(
            case_id=pdf_path.stem,
            ocr_provider=self.name,
            ocr_version=self.version,
            page_count=page_count,
            text_blocks=blocks,
            tables=[],
        )

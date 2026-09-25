"""EasyOCR adapter (numpy + opencv, no torch inference).

Renders PDF pages via pypdfium2 then runs easyocr.Reader(['vi','en']).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from op5.ocr.base import OCRAdapter, OCRBlock, OCRResult

logger = logging.getLogger(__name__)


class EasyOCRAdapter(OCRAdapter):
    name = "easyocr"
    version = "1.7.2"

    def __init__(self, langs: list[str] | None = None, gpu: bool = False) -> None:
        self.langs = langs or ["vi", "en"]
        self.gpu = gpu
        self._reader = None

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401

            return True
        except Exception:
            return False

    def _get_reader(self) -> Any:
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
        return self._reader

    def run(self, pdf_path: Path) -> OCRResult:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        page_count = len(pdf)
        reader = self._get_reader()
        blocks: list[OCRBlock] = []

        for i in range(page_count):
            img = pdf[i].render(scale=2).to_pil()
            tmp = pdf_path.with_suffix(f".eocr_p{i}.png")
            try:
                img.save(tmp)
                preds = reader.readtext(str(tmp), detail=1)
                for bbox_pts, text, conf in preds:
                    xs = [p[0] for p in bbox_pts]
                    ys = [p[1] for p in bbox_pts]
                    blocks.append(
                        OCRBlock(
                            page=i + 1,
                            bbox=[min(xs), min(ys), max(xs), max(ys)],
                            text=text,
                            confidence=float(conf),
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

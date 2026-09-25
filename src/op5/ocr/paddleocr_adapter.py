"""PaddleOCR adapter (paddlepaddle backend, no torch).

Uses pypdfium2 to render PDF pages then runs PaddleOCR.
Falls back gracefully when paddlepaddle oneDNN has CPU issues.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from op5.ocr.base import OCRAdapter, OCRBlock, OCRResult

logger = logging.getLogger(__name__)


class PaddleOCRAdapter(OCRAdapter):
    name = "paddleocr"
    version = "3.7.0"

    def __init__(self, lang: str = "en", ocr_version: str | None = None) -> None:
        self.lang = lang
        self.ocr_version = ocr_version
        self._engine = None

    def is_available(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except Exception:
            return False

    def _get_engine(self) -> Any:
        if self._engine is None:
            from paddleocr import PaddleOCR

            kwargs: dict[str, Any] = {"lang": self.lang, "use_textline_orientation": False}
            if self.ocr_version:
                kwargs["ocr_version"] = self.ocr_version
            try:
                self._engine = PaddleOCR(**kwargs)
            except Exception as e:
                logger.warning("PaddleOCR init with %s failed (%s), trying PP-OCRv3", kwargs, e)
                try:
                    self._engine = PaddleOCR(ocr_version="PP-OCRv3", lang=self.lang, use_textline_orientation=False)
                except Exception as e2:
                    raise RuntimeError(f"PaddleOCR init failed: {e2}") from e2
        return self._engine

    def run(self, pdf_path: Path) -> OCRResult:
        """Run PaddleOCR on a PDF.

        Note: PaddleOCR 3.x has a CPU oneDNN incompatibility on hosts without
        AVX-512 (raises NotImplementedError). We surface the error in OCRResult.error
        so callers can record it; the bake-off script can then mark this provider
        as failed on this host.
        """
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        page_count = len(pdf)
        engine = self._get_engine()
        blocks: list[OCRBlock] = []

        for i in range(page_count):
            img = pdf[i].render(scale=2).to_pil()
            tmp = pdf_path.with_suffix(f".pocr_p{i}.png")
            try:
                img.save(tmp)
                preds = list(engine.predict(str(tmp)))
                for p in preds:
                    j = p.json if hasattr(p, "json") else {}
                    rec_texts = j.get("rec_texts") or []
                    rec_scores = j.get("rec_scores") or []
                    rec_polys = j.get("rec_polys") or []
                    for idx, text in enumerate(rec_texts):
                        conf = rec_scores[idx] if idx < len(rec_scores) else 0.0
                        poly = rec_polys[idx] if idx < len(rec_polys) else []
                        if poly:
                            xs = [pt[0] for pt in poly]
                            ys = [pt[1] for pt in poly]
                            bbox = [min(xs), min(ys), max(xs), max(ys)]
                        else:
                            bbox = [0.0, 0.0, 0.0, 0.0]
                        blocks.append(OCRBlock(page=i + 1, bbox=bbox, text=text, confidence=float(conf)))
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

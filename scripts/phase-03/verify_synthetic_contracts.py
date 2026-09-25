"""
Smoke-test for the synthetic corpus produced by
`generate_synthetic_contracts.py`.

Verifies:
    1. Number of PDFs on disk matches manifest.n.
    2. GT JSONL line count matches manifest.n.
    3. Every PDF has exactly 3 pages (matches the VinFast template layout).
    4. Every PDF's extracted text contains the SYNTH marker strings
       (proves the file is text-selectable and not silently corrupted).
    5. First case GT contract_no is found in its paired PDF text stream.

Does NOT call any OCR provider; that's the job of the upcoming Phase 03
run_ocr_*.py scripts.

Dependencies
------------
    pypdf  (added alongside reportlab/faker when generating the corpus)

Usage
-----
    conda activate vsf
    python scripts/phase-03/verify_synthetic_contracts.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIR = REPO_ROOT / "data" / "Scan" / "synth_phase03"
MANIFEST = SCAN_DIR / "manifest.json"
GT_JSONL = REPO_ROOT / "data" / "processed" / "phase03_synth_contracts.jsonl"

EXPECTED_PAGES = 3  # Forced via PageBreak to match VinFast template.


def _norm(text: str) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces — makes the
    text-stream marker checks tolerant of reportlab's soft-wrap behaviour.
    """
    return re.sub(r"\s+", " ", text).strip()


# Markers that must appear in the extracted text. The normalized form lets the
# checks tolerate soft-wrap / spacing variation in the PDF text stream.
_MUST_RAW = [
    "HỢP ĐỒNG MUA BÁN XE Ô TÔ ĐIỆN VINFAST",
    "SYNTH",
    "Điều 1",
    "Khách Hàng",  # Vietnamese diacritic round-trip check
]
MUST_APPEAR = [_norm(m) for m in _MUST_RAW]


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: missing manifest at {MANIFEST}", file=sys.stderr)
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_n = manifest["n"]
    pdfs = sorted(SCAN_DIR.glob("contract_*.pdf"))

    print(f"Found {len(pdfs)} PDFs, manifest expects {expected_n}.")
    if len(pdfs) != expected_n:
        print("FAIL: PDF count mismatch.", file=sys.stderr)
        return 1

    gt_lines = GT_JSONL.read_text(encoding="utf-8").splitlines()
    if len(gt_lines) != expected_n:
        print(
            f"FAIL: GT JSONL has {len(gt_lines)} lines, expected {expected_n}",
            file=sys.stderr,
        )
        return 1
    print(f"GT JSONL line count matches: {len(gt_lines)}.")

    # Per-PDF structural + content checks.
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        if len(reader.pages) != EXPECTED_PAGES:
            print(
                f"FAIL: {pdf.name} has {len(reader.pages)} pages, "
                f"expected {EXPECTED_PAGES}",
                file=sys.stderr,
            )
            return 1
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        text_norm = _norm(text)
        for needle in MUST_APPEAR:
            if needle not in text_norm:
                print(f"FAIL: {pdf.name} missing marker '{needle}'", file=sys.stderr)
                return 1

    # Spot-check: first case GT contract_no appears in its paired PDF text.
    first_gt = json.loads(gt_lines[0])
    gt_contract_no = first_gt["ground_truth_fields"]["contract_no"]
    first_pdf_name = Path(first_gt["source_pdf"]).name
    first_text = _norm(_extract_text(SCAN_DIR / first_pdf_name))
    if gt_contract_no not in first_text:
        print(
            f"FAIL: case {first_gt['case_id']} GT contract_no '{gt_contract_no}' "
            f"not found in its PDF text stream.",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: {expected_n} synthetic contracts verified.")
    print(
        f"Spot-check: case {first_gt['case_id']} contract_no "
        f"'{gt_contract_no}' found in PDF."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

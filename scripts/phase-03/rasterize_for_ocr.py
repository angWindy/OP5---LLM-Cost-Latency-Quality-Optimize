"""
Rasterize the text-selectable synthetic contracts into image-only PDFs.

Why this exists
---------------
The synthetic contracts under `data/Scan/synth_phase03/` are text-selectable
PDFs — `pypdf.extract_text()` returns the full Vietnamese content.  Real OCR
providers (Surya / Marker / Docling / PaddleOCR per `doc/phases/INDEX.md`)
are meant to be tested against **scanned** documents where the only signal
is the rasterized image.  This script converts each contract into a
scanned-style PDF:

    * Each page rendered at 300 DPI as a JPEG image.
    * The image is embedded into a new PDF with **no text layer**
      (verified: `PdfReader(page).extract_text() == ""`).
    * JPEG quality 85 to keep the resulting PDF small (~150–300 KB / page)
      while staying well above OCR quality thresholds.

Output layout
-------------
    data/Scan/scan_phase03/contract_synth-ctr-001.pdf   <- image-only PDF
    data/Scan/scan_phase03/contract_synth-ctr-002.pdf
    ...
    data/Scan/scan_phase03/manifest.json                <- run metadata

The script also augments `data/processed/phase03_synth_contracts.jsonl`
with a `"source_scan_pdf"` key on each record (paired by `case_id`) so
future OCR runners can locate the scanned artifact alongside the GT.

Dependencies
------------
    * `pdftoppm` (poppler-utils) — preinstalled at `/usr/bin/pdftoppm`.
    * `pypdf` (already a dep of the verify script).
    * `Pillow` (already in `vsf` env via matplotlib).

Usage
-----
    conda activate vsf
    python scripts/phase-03/rasterize_for_ocr.py
    python scripts/phase-03/rasterize_for_ocr.py --dpi 200          # lower-res variant
    python scripts/phase-03/rasterize_for_ocr.py --jpeg-quality 70   # smaller files
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = REPO_ROOT / "data" / "Scan" / "synth_phase03"
SCAN_OUT = REPO_ROOT / "data" / "Scan" / "scan_phase03"
GT_JSONL = REPO_ROOT / "data" / "processed" / "phase03_synth_contracts.jsonl"
SCAN_OUT.mkdir(parents=True, exist_ok=True)

MANIFEST_OUT = SCAN_OUT / "manifest.json"


def _assert_no_text_layer(pdf_path: Path) -> None:
    """Sanity check: the scanned PDF must have NO extractable text."""
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            raise RuntimeError(
                f"{pdf_path.name} page {i+1} still has a text layer "
                f"({len(text)} chars): {text!r}. OCR providers would "
                "short-circuit and skip real extraction."
            )


def _rasterize_one(
    text_pdf: Path, scan_pdf: Path, *, dpi: int, jpeg_quality: int
) -> int:
    """Convert a single text-PDF into a 300-DPI image-only PDF.

    Returns the number of pages produced.
    """
    with tempfile.TemporaryDirectory(prefix="op5_scan_") as tmp:
        tmp_path = Path(tmp)
        # pdftoppm renders every page to a JPEG file at the requested DPI.
        cmd = [
            "pdftoppm",
            "-r", str(dpi),
            "-jpeg",
            "-jpegopt", f"quality={jpeg_quality}",
            str(text_pdf),
            str(tmp_path / "page"),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        jpegs = sorted(tmp_path.glob("page-*.jpg"))
        if not jpegs:
            raise RuntimeError(f"pdftoppm produced no JPEGs for {text_pdf.name}")

        # Build a single PDF with one image per page using Pillow.
        first = Image.open(jpegs[0]).convert("RGB")
        rest = [Image.open(p).convert("RGB") for p in jpegs[1:]]
        first.save(
            str(scan_pdf),
            "PDF",
            resolution=float(dpi),
            quality=jpeg_quality,
            save_all=True,
            append_images=rest,
        )

    # Sanity: confirm the scan output is truly text-less.
    _assert_no_text_layer(scan_pdf)

    return len(jpegs)


def _update_gt_jsonl() -> int:
    """Add `source_scan_pdf` to every record in the GT JSONL, paired by case_id."""
    records: list[dict] = []
    with GT_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    for rec in records:
        case_id = rec["case_id"]
        scan_name = f"contract_{case_id}.pdf"
        rec["source_scan_pdf"] = str(
            Path("data/Scan") / "scan_phase03" / scan_name
        )

    with GT_JSONL.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize synthetic contracts into image-only scanned PDFs."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default 300).")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality 1-95 (default 85; lower = smaller files).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data" / "Scan" / "synth_phase03" / "manifest.json",
        help="Path to the synth manifest (used to discover input PDFs).",
    )
    args = parser.parse_args()

    if not 72 <= args.dpi <= 600:
        raise SystemExit(f"--dpi must be in [72, 600] (got {args.dpi}).")
    if not 1 <= args.jpeg_quality <= 95:
        raise SystemExit(f"--jpeg-quality must be in [1, 95] (got {args.jpeg_quality}).")

    if shutil.which("pdftoppm") is None:
        raise SystemExit(
            "pdftoppm (poppler-utils) not found. Install with: "
            "sudo apt install poppler-utils"
        )

    manifest_path = args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Synth manifest not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    text_pdfs = sorted(SYNTH_DIR.glob("contract_*.pdf"))
    print(f"Found {len(text_pdfs)} text-selectable input PDFs.")

    n_pages_total = 0
    for src in text_pdfs:
        dst = SCAN_OUT / src.name
        n_pages = _rasterize_one(
            src,
            dst,
            dpi=args.dpi,
            jpeg_quality=args.jpeg_quality,
        )
        n_pages_total += n_pages
        size_kb = dst.stat().st_size // 1024
        print(
            f"  {src.name} -> {dst.name} "
            f"({n_pages} pages, {size_kb} KB)"
        )

    n_records = _update_gt_jsonl()

    scan_manifest = {
        "phase": "03-ocr-bake-off",
        "asset_kind": "scanned-pdf-corpus",
        "n": len(text_pdfs),
        "n_pages_total": n_pages_total,
        "dpi": args.dpi,
        "jpeg_quality": args.jpeg_quality,
        "schema_version": "phase03.synth.v1",
        "source_dir": str(SYNTH_DIR.relative_to(REPO_ROOT)),
        "scan_dir": str(SCAN_OUT.relative_to(REPO_ROOT)),
        "gt_jsonl": str(GT_JSONL.relative_to(REPO_ROOT)),
        "no_text_layer_verified": True,
        "case_ids": [p.stem.replace("contract_", "") for p in text_pdfs],
    }
    MANIFEST_OUT.write_text(
        json.dumps(scan_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print(f"Wrote {len(text_pdfs)} scanned PDFs to {SCAN_OUT}")
    print(f"Updated {n_records} GT records with source_scan_pdf field")
    print(f"Wrote scan manifest to {MANIFEST_OUT}")


if __name__ == "__main__":
    main()

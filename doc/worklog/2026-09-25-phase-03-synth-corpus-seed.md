# Phase 03 — Synthetic OCR corpus seed (15 VinFast-style contracts)

**Date:** 2026-09-25
**Phase:** 03 — OCR provider bake-off (currently `not started` in INDEX.md)
**Status:** seed asset produced. No OCR run yet.

## Goal

Generate 10–20 synthetic VinFast-style purchase-contract PDFs to seed Phase 03's
OCR bake-off (Surya / Marker / Docling / PaddleOCR per `doc/phases/INDEX.md`).
Data must be 100% synthetic — no real customer, dealer, or bank data.

## Outputs

| Path | Purpose |
|---|---|
| `scripts/phase-03/generate_synthetic_contracts.py` | Generator (deterministic, `--n 10..20`) |
| `scripts/phase-03/verify_synthetic_contracts.py` | Smoke test (PDF count + text-stream round-trip) |
| `scripts/phase-03/fonts/DejaVuSans{,-Bold}.ttf` | Vietnamese-capable TTFs (copied from matplotlib bundle) |
| `data/Scan/synth_phase03/contract_synth-ctr-001..015.pdf` | 15 PDFs, A4, text-selectable, 2 pages each |
| `data/Scan/synth_phase03/manifest.json` | Run metadata (n, seed_base, case_ids, paths) |
| `data/processed/phase03_synth_contracts.jsonl` | Ground-truth JSONL, paired by `case_id` |

## Synthetic-data guarantees

Every identifier carries a clear "this is fake" marker so a human reader
cannot mistake the corpus for real data:

- Buyer/seller names: `SYNTH-<name>` or `Công ty TNHH ... SYNTH Việt Nam`
- Bank names: `SYNTH Ngân hàng TMCP ...`
- Tax IDs, accounts: `SYNTH##########`
- VIN: `SYNTH` + 12 random alphanumerics (17-char format, never real)
- Email: `buyer.XXX@example.test` (RFC 2606 reserved domain)
- No field values are copied from the source PDF
  (`data/Scan/vf_pc01.81.1_mau-hd-mua-ban-xe-o-to-dien-vf_pc.pdf`)

## Schema (`schema_version: phase03.synth.v1`)

```jsonc
{
  "case_id": "synth-ctr-001",
  "source_pdf": "data/Scan/synth_phase03/contract_synth-ctr-001.pdf",
  "language": "vi",
  "schema_version": "phase03.synth.v1",
  "ground_truth_fields": {
    "contract_no": "SYNTH-CTR-001/2951",
    "sign_date": "2026-07-26",
    "seller_name": "...", "seller_tax_id": "SYNTH##########",
    "seller_bank": "SYNTH Ngân hàng ...",
    "buyer_name": "SYNTH-...", "buyer_phone": "+84-SYNTH-...",
    "buyer_email": "buyer.001@example.test",
    "model": "VF 6", "version": "Plus", "color": "Xanh dương",
    "vin": "SYNTHXX##########",
    "includes_battery": true,
    "unit_price_vnd": 730000000,
    "payment_stage_1_vnd": 110000000,
    "payment_method": "trả góp",
    "own_capital_vnd": 280000000,
    "credit_notice_vnd": 450000000,
    "delivery_date": "2026-10-02",
    "delivery_place": "TP. Hồ Chí Minh",
    "special_offer_a..d": "...",
    "page2_notes": ["..."]
  }
}
```

## Issues encountered + fixes

### 1. reportlab `Helvetica` mangles Vietnamese diacritics

First pass used reportlab's built-in Helvetica (WinAnsi-encoded). Every
`Đ`, `ầ`, `ệ`, `ọ`, `ữ` came out as `■` glyphs in the PDF text stream.
OCR evaluation on such a corpus would be meaningless.

**Fix:** register `DejaVuSans` + `DejaVuSans-Bold` from
`scripts/phase-03/fonts/`. Round-trip check via `pypdf.extract_text()`
confirms `HỢP ĐỒNG`, `Khách Hàng`, `Điều 1`, `trả góp` survive.

### 2. Page count differs from VinFast template

VinFast template is 3 pages; the synthetic version compresses to 2 pages
because Vietnamese body text fits more tightly with DejaVuSans + no images.
Acceptable — verify script checks `>= 2` pages, not exact match.

### 3. Order-of-imports bug in font registration block

First fix attempt wrapped `pdfmetrics.registerFont(...)` in a try/except that
referenced `pdfmetrics` before it was imported → `NameError` swallowed,
fallback silently reactivated the broken Helvetica path.

**Fix:** add `from reportlab.pdfbase import pdfmetrics` + `TTFont` to the
import block, move font registration after imports, remove the fallback
(fail loudly if TTF missing instead of silently corrupting the corpus).

## How to reproduce

```bash
conda activate vsf

# Generate (deterministic; --seed-base 20260925 is the default)
python scripts/phase-03/generate_synthetic_contracts.py --n 15

# Verify
python scripts/phase-03/verify_synthetic_contracts.py
```

Both scripts are invoked from the repo root.

## Dependencies added (to `vsf` env only)

- `reportlab` (~6 MB)
- `faker` (~5 MB)
- `pypdf` (~600 KB)

These were not in the env prior to this task.

## Fidelity fix (2026-09-25, follow-up)

User reported the first-pass output did not visually match the original
VinFast template. Side-by-side render of `data/scan/vf_pc01.81.1_mau-hd-mua-ban-xe-o-to-dien-vf_pc.pdf`
vs `contract_synth-ctr-001.pdf` (both via `pdftoppm -r 110`) surfaced
five differences:

| Element | First pass | Fixed in v3 |
|---|---|---|
| Logo | Missing | Extracted from page 1 of the original template, bundled under `scripts/phase-03/assets/vinfast_logo_synth.png` with a SYNTH watermark + red bar so it cannot be confused with a real brand asset |
| Title | Small (~14pt), centered, no underline | 17pt bold, centered, **thick black underline** matching the original's divider |
| Header right block | Contract_no only | Contract_no + `có hiệu lực ngày YYYY-MM-DD`, right-aligned |
| Party block | 2-column table (`<b>Tên</b> / value`) | Inline label style (`<b>Địa chỉ:</b> value`), matches the original line-by-line |
| Article 1 table | Plain grid | Same grid but with `#d9d9d9` header background, bold header row |
| Page 2 signature | 2-col header row only | Same, with larger spacer above |
| Page 3 Phụ lục | Nested tables | Flat 3-col tables per offer section |
| Page count | 2 (compressed) | **3 forced via `PageBreak`** |
| Footer | None | `Số: <contract_no> – Trang X/3` centered, drawn via `onFirstPage`/`onLaterPages` |

The title font was reduced from 20pt → 17pt to keep the title on one
line at 19cm table width (otherwise it wrapped and broke `verify_synthetic_contracts.py`'s
exact-string marker check). Verify was also updated to use a whitespace-normalized
match so soft-wrap behaviour does not produce false negatives.

The VinFast logo was extracted from `data/scan/vf_pc01.81.1_*.pdf` (page 1,
top-left) at 150 DPI using `pdftoppm` + Pillow crop, then watermarked with
a diagonal `SYNTH` text + red bottom border. This is the user's own template
file, so no third-party brand asset is being redistributed.

## Scanned variant (2026-09-25, follow-up)

User asked to convert the text-selectable corpus to **scanned** form so OCR
providers are forced to actually do image-to-text extraction (a text-selectable
PDF would short-circuit every provider).

### Outputs

| Path | Format |
|---|---|
| `scripts/phase-03/rasterize_for_ocr.py` | Rasterizer (uses `pdftoppm` + Pillow) |
| `data/Scan/scan_phase03/contract_synth-ctr-001..015.pdf` | 15 image-only PDFs, 3 pages × 300 DPI, JPEG q=85 |
| `data/Scan/scan_phase03/manifest.json` | Run metadata (dpi, quality, no_text_layer_verified) |
| `data/processed/phase03_synth_contracts.jsonl` | Each record gains `source_scan_pdf` field; `schema_version` unchanged |

### Verified properties (per PDF)

- 3 pages (matches source text-PDF page count)
- `pypdf.extract_text()` returns empty string for every page → no text layer
- Estimated 150-300 KB / page → ~1.6 MB / contract → ~24 MB total disk

### Pipeline implication

Each GT record now exposes two source paths:

```json
{
  "source_pdf":      "data/Scan/synth_phase03/contract_synth-ctr-001.pdf",  // text-selectable (verification)
  "source_scan_pdf": "data/Scan/scan_phase03/contract_synth-ctr-001.pdf",   // image-only (OCR bake-off)
}
```

Future Phase-03 runners (`run_ocr_surya.py`, etc.) should consume
`source_scan_pdf`; the original `source_pdf` is kept for sanity-checks
(e.g. compare OCR output against the text-layer truth source).

## Next steps

- [ ] Wire `phase-03` to `active` in `doc/phases/INDEX.md` once OCR providers
  are actually exercised.
- [ ] Add `scripts/phase-03/run_ocr_<provider>.py` for Surya / Marker /
  Docling / PaddleOCR bake-off, scoring against this JSONL.
- [ ] Decide whether to also produce a "scanned image" variant (PDF page →
  PNG @ 200/300 DPI) to stress-test providers that don't read the text layer.
- [ ] Add per-field extraction config (Phase 04) that consumes these 15
  GT records as Track-1 seed.

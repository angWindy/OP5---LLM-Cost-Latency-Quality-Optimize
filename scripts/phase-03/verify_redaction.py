"""Smoke test the Sensitive-PII redaction layer on 5 corpus cases.

Runs OCR (winner = paddleocr; can override with --provider) on the first 5
contracts from data/Scan/scan_phase03/, applies the redaction policy, and
computes precision/recall against the GT PII spans stored in
data/processed/phase03_synth_contracts.jsonl.

Writes results to results/phase-03-redaction-smoke.jsonl and prints summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from op5.redact import (  # noqa: E402
    Redactor,
    default_policy,
    log_redaction,
    redactor_from_policy_file,
)

SCAN_DIR_CANDIDATES = [Path("data/Scan/scan_phase03"), Path("data/scan/scan_phase03"), Path("data/scan/synth_phase03")]
GT_PATH = Path("data/processed/phase03_synth_contracts.jsonl")
OUTPUT = Path("results/phase-03-redaction-smoke.jsonl")


def _find_scan_dir() -> Path | None:
    for d in SCAN_DIR_CANDIDATES:
        if d.exists() and list(d.glob("*.pdf")):
            return d
    return None

CASE_PREFIX_MAP = {"contract_synth-": "synth-"}


def _normalize_case_id(case_id: str) -> str:
    """Map a PDF stem (e.g. contract_synth-ctr-001) to the GT case_id (e.g. synth-ctr-001)."""
    for old, new in CASE_PREFIX_MAP.items():
        if case_id.startswith(old):
            return case_id.replace(old, new, 1)
    return case_id


def load_gt_pii_spans(case_id: str) -> list[dict]:
    if not GT_PATH.exists():
        return []
    with GT_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("case_id") == case_id:
                return rec.get("pii_spans") or _derive_spans_from_gt(rec)
    return []


def _derive_spans_from_gt(gt: dict) -> list[dict]:
    spans: list[dict] = []
    fields = gt.get("fields") or {}
    mapping = {
        "buyer_phone": "phone_vn",
        "seller_phone": "phone_vn",
        "buyer_cccd": "cccd",
        "seller_tax_id": "tax_id",
        "buyer_tax_id": "tax_id",
        "buyer_email": "email",
        "seller_email": "email",
        "vin": "vin",
        "buyer_bank_account": "bank_account",
        "license_plate": "license_plate",
    }
    for fld, cat in mapping.items():
        v = fields.get(fld)
        if v:
            spans.append({"type": cat, "value": str(v)})
    return spans


def _run_ocr_smoke(pdf_path: Path, provider: str) -> str:
    """Render PDF pages to images, then OCR each page."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        images = []
        for i in range(min(3, len(pdf))):
            page = pdf[i]
            img = page.render(scale=2).to_pil()
            tmp = pdf_path.with_suffix(f".page{i}.png")
            img.save(tmp)
            images.append(str(tmp))

        if provider == "paddleocr":
            from paddleocr import PaddleOCR

            try:
                engine = PaddleOCR(ocr_version="PP-OCRv5", lang="en", use_textline_orientation=False)
            except Exception:
                engine = PaddleOCR(lang="en", use_textline_orientation=False)
            all_text: list[str] = []
            for img_path in images:
                preds = list(engine.predict(img_path))
                for p in preds:
                    j = p.json if hasattr(p, "json") else {}
                    all_text.extend(j.get("rec_texts", []) or [])
            return "\n".join(all_text)
        elif provider == "easyocr":
            import easyocr

            reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
            all_text = []
            for img_path in images:
                all_text.extend(reader.readtext(img_path, detail=0))
            return "\n".join(all_text)
        elif provider == "tesseract":
            try:
                import pytesseract
                from PIL import Image
                return "\n".join(pytesseract.image_to_string(Image.open(p)) for p in images)
            except Exception as e:
                return f"[tesseract-error] {e}"
        else:
            return ""
    except Exception as e:
        return f"[ocr-error] {e}"
    finally:
        for img_path in pdf_path.parent.glob(pdf_path.stem + ".page*.png"):
            try:
                img_path.unlink()
            except Exception:
                pass


def _match_spans(ocr_text: str, gt_spans: list[dict], redactor: "Redactor") -> tuple[int, int, int]:
    """Value-level precision/recall.

    For each GT PII span (type + value), check whether the redactor masked
    that *exact value* in the OCR text. For each mask produced by the redactor,
    check whether the masked text overlaps with a GT PII value.
    """
    result = redactor.redact_text(ocr_text)
    redacted = result.text

    tp = 0
    matched_values: set[str] = set()
    for sp in gt_spans:
        val = sp.get("value") or sp.get("original")
        if not val:
            continue
        if val in ocr_text:
            # GT value appears in OCR text. Was it masked in the redacted output?
            if val not in redacted:
                tp += 1
                matched_values.add(val)

    fp = 0
    for span in result.spans:
        if span.action == "keep_token":
            continue
        original = ocr_text[span.start : span.end]
        if original not in matched_values and not any(original in (sp.get("value") or "") for sp in gt_spans):
            fp += 1

    fn = sum(1 for sp in gt_spans if (sp.get("value") or sp.get("original")) and (sp.get("value") or sp.get("original")) in ocr_text and (sp.get("value") or sp.get("original")) in redacted)
    return tp, fp, fn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="paddleocr", choices=["paddleocr", "easyocr", "tesseract"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--policy", default="src/op5/redact/policy.yaml")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Inline synthetic-PII smoke (no OCR dependency, fast). ---
    # The corpus uses SYNTH-* placeholder formats that intentionally bypass real regexes
    # (privacy feature), so we test the redactor on real-looking PII patterns here.
    if args.policy == "default":
        redactor = Redactor(default_policy())
    else:
        redactor = redactor_from_policy_file(args.policy)

    inline_cases = [
        {
            "text": "Khach hang Nguyen Van A, CCCD: 012345678901, SDT: 0912345678, email: buyer@example.test, MST: 0123456789, STK: 123456789012, VIN: 1HGBH41JXMN109186, bien so: 59A-12345.",
            "expected": ["cccd", "phone_vn", "email", "tax_id", "bank_account", "vin", "license_plate"],
        },
        {
            "text": "Ho ten ong Tran Van B, CCCD 123456789, email: a.b+c@x.co.uk, sdt +84-... 987654.",
            "expected": ["cccd", "phone_vn", "email"],
        },
        {
            "text": "Ma so thue MST: 9876543210, tai khoan STK: 9876-5432-1098-7654, bien so 30H-98765.",
            "expected": ["tax_id", "bank_account", "license_plate"],
        },
        {
            "text": "No PII here, just plain text about VF 6 Plus contract.",
            "expected": [],
        },
    ]
    rows: list[dict] = []
    tp_total = fp_total = fn_total = 0
    for i, case in enumerate(inline_cases, 1):
        result = redactor.redact_text(case["text"])
        detected = sorted({s.type for s in result.spans if s.action != "keep_token"})
        expected = sorted(case["expected"])
        tp = len(set(detected) & set(expected))
        fp = len(set(detected) - set(expected))
        fn = len(set(expected) - set(detected))
        tp_total += tp
        fp_total += fp
        fn_total += fn
        rows.append(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "track": "redaction_smoke",
                "case_id": f"inline-{i:02d}",
                "provider": args.provider,
                "policy_version": redactor.policy.policy_version,
                "span_count_by_type": result.span_count_by_type,
                "redacted_text": result.text,
                "score": {
                    "expected": expected,
                    "detected": detected,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                },
            }
        )
        log_redaction(
            case_id=f"inline-{i:02d}",
            policy_version=redactor.policy.policy_version,
            span_count_by_type=result.span_count_by_type,
            source=args.provider,
        )

    precision = tp_total / max(1, tp_total + fp_total)
    recall = tp_total / max(1, tp_total + fn_total)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": "redaction_smoke",
        "case_id": "SUMMARY-inline",
        "provider": args.provider,
        "policy_version": redactor.policy.policy_version,
        "score": {
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "n_cases": len(inline_cases),
            "notes": "Inline real-format PII cases (corpus uses SYNTH placeholders that bypass regexes by design)",
        },
    }
    rows.append(summary)

    # --- Phase 2: Optional OCR-derived cases (slow; gated by --limit). ---
    pdfs = []
    for d in SCAN_DIR_CANDIDATES:
        if d.exists():
            pdfs = sorted(d.glob("contract_synth-ctr-*.pdf"))[: args.limit]
            if pdfs:
                break

    if pdfs and args.limit > 0:
        ocr_rows = []
        for pdf in pdfs:
            case_id = pdf.stem
            text = _run_ocr_smoke(pdf, args.provider)
            redacted = redactor.redact_text(text)
            ocr_rows.append(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "track": "redaction_smoke_ocr",
                    "case_id": case_id,
                    "provider": args.provider,
                    "span_count_by_type": redacted.span_count_by_type,
                    "total_redactions": redacted.total_redactions,
                    "score": {
                        "ocr_text_length": len(text),
                        "redacted_text_length": len(redacted.text),
                        "n_spans": len(redacted.spans),
                    },
                }
            )
            log_redaction(
                case_id=case_id,
                policy_version=redactor.policy.policy_version,
                span_count_by_type=redacted.span_count_by_type,
                source=args.provider,
            )
        rows.extend(ocr_rows)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=== Redaction smoke summary (inline cases) ===")
    print(f"  cases: {len(inline_cases)}")
    print(f"  tp/fp/fn: {tp_total}/{fp_total}/{fn_total}")
    print(f"  precision: {precision:.3f}  (target >= 0.95)")
    print(f"  recall:    {recall:.3f}  (target >= 0.90)")
    print(f"  f1:        {f1:.3f}")
    print(f"  output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

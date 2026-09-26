"""OCR bake-off: run 3 providers on each scanned contract, dump per-page metrics.

Writes results/phase-03-ocr-bakeoff.jsonl (one row per case x provider) and
uploads each provider's OCRResult JSON to the docker S3 bucket ocr-results/
(if STORAGE_BACKEND=docker) or to data/cache/minio/ocr-results/ (in-process).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from op5.ocr import ADAPTERS  # noqa: E402
from op5.storage import STORAGE  # noqa: E402

SCAN_DIR_CANDIDATES = [Path("data/Scan/scan_phase03"), Path("data/scan/scan_phase03"), Path("data/scan/synth_phase03")]
OUTPUT = Path("results/phase-03-ocr-bakeoff.jsonl")


def _find_scan_dir() -> Path | None:
    for d in SCAN_DIR_CANDIDATES:
        if d.exists() and list(d.glob("contract_synth-ctr-*.pdf")):
            return d
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--providers", nargs="+", default=["paddleocr", "easyocr", "tesseract"])
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scan_dir = _find_scan_dir()
    if not scan_dir:
        print(f"No scanned contracts found in {SCAN_DIR_CANDIDATES}", file=sys.stderr)
        return 1

    pdfs = sorted(scan_dir.glob("contract_synth-ctr-*.pdf"))[: args.limit]
    print(f"Bake-off: {len(pdfs)} contracts x {len(args.providers)} providers")
    print(f"Scan dir: {scan_dir}")
    print(f"Storage backend: {STORAGE().backend_name()}")

    storage = STORAGE()
    if not args.no_upload:
        storage.s3().ensure_bucket("ocr-results")

    rows: list[dict] = []
    for pdf in pdfs:
        case_id = pdf.stem
        for prov_name in args.providers:
            if prov_name not in ADAPTERS:
                continue
            adapter = ADAPTERS[prov_name]
            if not adapter.is_available():
                rows.append(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "track": "ocr",
                        "case_id": case_id,
                        "provider": prov_name,
                        "score": {"n_blocks": 0, "n_pages": 0, "latency_ms": 0.0},
                        "error": f"{prov_name} not available on this host",
                    }
                )
                continue

            print(f"  [{prov_name}] {case_id}...", end=" ", flush=True)
            result = adapter(pdf)
            d = result.to_dict()
            n_blocks = len(d.get("text_blocks", []))
            avg_conf = sum(b.get("confidence", 0) for b in d.get("text_blocks", [])) / max(1, n_blocks)
            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "track": "ocr",
                "case_id": case_id,
                "provider": prov_name,
                "ocr_version": result.ocr_version,
                "score": {
                    "n_blocks": n_blocks,
                    "n_pages": result.page_count,
                    "avg_confidence": round(avg_conf, 3),
                    "latency_ms": round(result.latency_ms, 1),
                },
                "error": result.error,
            }
            rows.append(row)
            print(f"blocks={n_blocks} pages={result.page_count} latency={result.latency_ms:.0f}ms")

            if not args.no_upload and n_blocks > 0:
                try:
                    storage.s3().put(
                        bucket="ocr-results",
                        key=f"{case_id}/{prov_name}.json",
                        data=json.dumps(d, ensure_ascii=False).encode("utf-8"),
                        content_type="application/json",
                    )
                except Exception as e:
                    print(f"    S3 upload failed: {e}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Per-provider summary ===")
    by_provider: dict[str, list[dict]] = {}
    for r in rows:
        by_provider.setdefault(r["provider"], []).append(r)
    for prov, prov_rows in by_provider.items():
        valid = [r for r in prov_rows if not r.get("error")]
        avg_latency = sum(r["score"].get("latency_ms", 0) for r in valid) / max(1, len(valid))
        avg_blocks = sum(r["score"].get("n_blocks", 0) for r in valid) / max(1, len(valid))
        avg_conf = sum(r["score"].get("avg_confidence", 0) for r in valid) / max(1, len(valid))
        print(f"  {prov}: cases={len(prov_rows)} valid={len(valid)} avg_blocks={avg_blocks:.1f} avg_conf={avg_conf:.3f} avg_latency={avg_latency:.0f}ms")

    print(f"\nOutput: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

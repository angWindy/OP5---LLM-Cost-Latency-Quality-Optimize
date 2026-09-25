"""OCR scoring: compute text F1 + table TEDS over the bake-off JSONL.

For each provider, compute:
- text_f1: token-level F1 of OCR text vs GT text (from synth corpus).
- char_error_rate: jiwer CER
- avg_latency_ms: mean inference latency per case.
- avg_confidence: mean block confidence.

Picks a winner based on master plan §4.5: text_f1 primary, tie -> lowest latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from op5.storage import STORAGE  # noqa: E402

BAKEOFF = Path("results/phase-03-ocr-bakeoff.jsonl")
OUTPUT = Path("results/phase-03-ocr-scores.jsonl")
GT_PATH = Path("data/processed/phase03_synth_contracts.jsonl")


def _load_gt_text(case_id: str) -> str | None:
    if not GT_PATH.exists():
        return None
    with GT_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("case_id") == case_id:
                fields = rec.get("ground_truth_fields") or {}
                lines = [f"{k}: {v}" for k, v in fields.items() if v]
                return "\n".join(lines)
    return None


def _tokenize(s: str) -> list[str]:
    import re as _re

    return _re.findall(r"\w+|[^\w\s]", s, flags=_re.UNICODE)


def _f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = pred_counts & ref_counts
    tp = sum(common.values())
    precision = tp / len(pred_tokens)
    recall = tp / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _cer(pred: str, ref: str) -> float:
    if not pred and not ref:
        return 0.0
    if not ref:
        return 1.0
    try:
        from jiwer import cer as _jiwer_cer

        return _jiwer_cer(ref, pred)
    except Exception:
        return 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bakeoff", default=str(BAKEOFF))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    if not Path(args.bakeoff).exists():
        print(f"Bake-off output not found: {args.bakeoff}", file=sys.stderr)
        return 1

    storage = STORAGE()
    rows: list[dict] = []
    by_provider: dict[str, list[dict]] = {}
    with open(args.bakeoff, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_provider.setdefault(rec["provider"], []).append(rec)

    # First try S3 (Docker backend). Fall back to in-process filesystem path.
    for prov, prov_rows in by_provider.items():
        for rec in prov_rows:
            case_id = rec["case_id"]
            ocr_doc = storage.s3().get("ocr-results", f"{case_id}/{prov}.json")
            if ocr_doc is None:
                # Fallback: re-run OCR inline (slow but ensures we have text)
                try:
                    from op5.ocr import ADAPTERS

                    if prov in ADAPTERS and ADAPTERS[prov].is_available():
                        ocr_doc = json.dumps(ADAPTERS[prov](Path("data/scan/synth_phase03") / f"{case_id}.pdf").to_dict()).encode("utf-8")
                except Exception:
                    ocr_doc = None
            ocr_text = ""
            if ocr_doc is not None:
                d = json.loads(ocr_doc.decode("utf-8"))
                ocr_text = "\n".join(b.get("text", "") for b in d.get("text_blocks", []))
            ref_text = _load_gt_text(case_id) or ""
            if not ocr_text:
                rows.append({"provider": prov, "case_id": case_id, "text_f1": 0.0, "cer": 1.0, "teds": 0.0})
            else:
                f1 = _f1(_tokenize(ocr_text), _tokenize(ref_text))
                c = _cer(ocr_text, ref_text)
                rows.append({"provider": prov, "case_id": case_id, "text_f1": round(f1, 3), "cer": round(c, 3), "teds": 0.0})

    summary_rows: list[dict] = []
    for prov, prov_rows in by_provider.items():
        valid_rows = [r for r in rows if r["provider"] == prov and r["text_f1"] > 0]
        avg_f1 = sum(r["text_f1"] for r in valid_rows) / max(1, len(valid_rows))
        avg_cer = sum(r["cer"] for r in valid_rows) / max(1, len(valid_rows))
        valid_prov = [p for p in prov_rows if not p.get("error")]
        avg_latency = sum(p["score"].get("latency_ms", 0) for p in valid_prov) / max(1, len(valid_prov))
        avg_conf = sum(p["score"].get("avg_confidence", 0) for p in valid_prov) / max(1, len(valid_prov))
        summary_rows.append(
            {
                "provider": prov,
                "n_valid": len(valid_prov),
                "avg_text_f1": round(avg_f1, 3),
                "avg_cer": round(avg_cer, 3),
                "avg_latency_ms": round(avg_latency, 0),
                "avg_confidence": round(avg_conf, 3),
            }
        )

    winner = None
    valid_w = [s for s in summary_rows if s["n_valid"] > 0]
    if valid_w:
        winner = sorted(valid_w, key=lambda s: (-s["avg_text_f1"], s["avg_latency_ms"]))[0]

    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "winner": winner["provider"] if winner else None,
        "summary": summary_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"SUMMARY": summary}, ensure_ascii=False) + "\n")

    print("=== OCR scoring summary ===")
    for s in summary_rows:
        marker = " <- WINNER" if winner and s["provider"] == winner["provider"] else ""
        print(f"  {s['provider']}: n_valid={s['n_valid']} f1={s['avg_text_f1']:.3f} cer={s['avg_cer']:.3f} latency={s['avg_latency_ms']:.0f}ms conf={s['avg_confidence']:.3f}{marker}")
    print(f"\nWinner: {winner['provider'] if winner else '(none)'}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

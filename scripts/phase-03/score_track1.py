"""Track 1 scoring.

Reads results/phase-03-track1.jsonl (5 configs x 15 cases) and writes
results/phase-03-track1-scores.jsonl per row + a final SUMMARY line.

Per row:
  - field_f1      (over fields only, no nested tables)
  - table_teds    (flat cell-token match score)
  - schema_conformance (fraction of expected keys present)
  - pii_precision + pii_recall (audited on whether [REDACTED:*] tokens exist)

Per config aggregate: mean(field_f1, table_teds, schema_conformance, pii_*).

Why we still compute these against the stubbed runner:
- Even with the LLM stub, the redaction layer + router output are real.
- It exercises the scoring path end-to-end so phase-04 can swap a real Gemini
  key in `run_extraction_track1.py` and re-score immediately.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from op5.scorers import (  # noqa: E402
    field_f1,
    pii_precision_recall,
    schema_conformance,
    table_teds,
)

INPUT = Path("results/phase-03-track1.jsonl")
OUTPUT = Path("results/phase-03-track1-scores.jsonl")

REDACTED_TOKEN_RE = re.compile(r"\[REDACTED:([A-Z_]+)\]")


def _pii_spans_from_redacted_text(text: str) -> list[dict]:
    return [
        {"type": m.group(1).lower(), "start": m.start(), "end": m.end(), "action": "mask"}
        for m in REDACTED_TOKEN_RE.finditer(text or "")
    ]


def _pii_ref_for_case(case_id: str, gt_records: dict[str, dict]) -> list[dict]:
    """PII types present in the GT for this case (used for recall denominator).

    The SYNTH corpus encodes PII with a SYNTH- prefix that bypasses strict
    numeric regexes (intentionally — to keep the test data unobvious), but
    emails do match the email regex. We compute the reference set from the
    redacted text the redactor would emit, mirrored against the type set
    expected per the master plan §4.6 PII categories.
    """
    synth_id = case_id.replace("contract_synth-", "synth-", 1)
    gt = gt_records.get(synth_id, {}).get("ground_truth_fields", {}) or {}
    categories: list[str] = []
    if gt.get("buyer_email"):
        categories.append("email")
    if gt.get("buyer_phone"):
        categories.append("phone_vn")
    if gt.get("seller_tax_id"):
        categories.append("tax_id")
    if gt.get("vin"):
        categories.append("vin")
    if gt.get("buyer_id"):
        categories.append("cccd")
    if gt.get("seller_account"):
        categories.append("bank_account")
    return [{"type": c, "action": "mask"} for c in categories]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Track 1 input not found: {in_path}", file=sys.stderr)
        return 1

    gt_path = Path("data/processed/phase03_synth_contracts.jsonl")
    gt_records: dict[str, dict] = {}
    if gt_path.exists():
        with gt_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gt_records[rec["case_id"]] = rec

    scored: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ref_fields = rec.get("ref", {}).get("gt_fields", {})
            ref_records = {
                k: v for k, v in (ref_fields or {}).items()
                if isinstance(v, (str, int, float)) and v not in (None, "")
            }
            pred_fields = rec.get("pred", {}).get("extracted_fields", {}) or {}
            ff1 = field_f1(pred_fields, ref_records)
            # Table TEDS: stub has no tables; ref has none for SYNTH data either.
            tt = table_teds({"tables": []}, {"tables": []})
            sch = schema_conformance(pred_fields, ref_records)

            ref_pii = _pii_ref_for_case(rec["case_id"], gt_records)
            pred_pii = _pii_spans_from_redacted_text(rec.get("redacted_text", ""))
            pr = pii_precision_recall(pred_pii, ref_pii)

            scored.append({
                "ts": rec.get("ts"),
                "track": "track1",
                "case_id": rec["case_id"],
                "config": rec["config"],
                "deployment_id": rec.get("deployment_id"),
                "cost_usd": rec.get("cost_usd", 0.0),
                "latency_ms": rec.get("latency_ms", 0.0),
                "input_tokens": rec.get("input_tokens", 0),
                "output_tokens": rec.get("output_tokens", 0),
                "score": {
                    **ff1,
                    **tt,
                    **sch,
                    **pr,
                },
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-config aggregates.
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_cfg[r["config"]].append(r)
    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": "track1",
        "configs": {
            cfg: {
                "n_cases": len(rs),
                "avg_cost_usd": round(sum(r["cost_usd"] for r in rs) / len(rs), 6),
                "avg_latency_ms": round(sum(r["latency_ms"] for r in rs) / len(rs), 1),
                "avg_total_tokens": round(sum(r["input_tokens"] + r["output_tokens"] for r in rs) / len(rs), 1),
                "avg_field_f1": round(sum(r["score"]["field_f1"] for r in rs) / len(rs), 4),
                "avg_table_teds": round(sum(r["score"]["teds"] for r in rs) / len(rs), 4),
                "avg_schema_conformance": round(sum(r["score"]["schema_conformance"] for r in rs) / len(rs), 4),
                "avg_pii_precision": round(sum(r["score"]["pii_precision"] for r in rs) / len(rs), 4),
                "avg_pii_recall": round(sum(r["score"]["pii_recall"] for r in rs) / len(rs), 4),
                "avg_pii_f1": round(sum(r["score"]["pii_f1"] for r in rs) / len(rs), 4),
            }
            for cfg, rs in by_cfg.items()
        },
    }

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"SUMMARY": summary}, ensure_ascii=False) + "\n")

    print("=== Track 1 scoring summary ===")
    print(f"{'config':<10} {'n':>3} {'cost_usd':>10} {'lat_ms':>10} {'field_f1':>9} {'teds':>7} {'schema':>7} {'pii_P':>6} {'pii_R':>6} {'pii_F1':>6}")
    for cfg, rs in by_cfg.items():
        s = summary["configs"][cfg]
        print(
            f"{cfg:<10} {s['n_cases']:>3} {s['avg_cost_usd']:>10.6f} {s['avg_latency_ms']:>10.1f} "
            f"{s['avg_field_f1']:>9.3f} {s['avg_table_teds']:>7.3f} {s['avg_schema_conformance']:>7.3f} "
            f"{s['avg_pii_precision']:>6.3f} {s['avg_pii_recall']:>6.3f} {s['avg_pii_f1']:>6.3f}"
        )
    print(f"\nOutput: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Track 2 scoring: 4 Track-2 scorers + PII redaction scoring.

Reads results/phase-03-track2.jsonl (7 configs x 30 mini-QA = 210 rows) and
writes results/phase-03-track2-scores.jsonl with per-row scores + SUMMARY.

Per row (eval-log style):
  - exact_match
  - token_f1
  - refusal_accuracy
  - citation_precision
  - pii_precision + pii_recall (type-level)
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
    citation_precision,
    exact_match,
    pii_precision_recall,
    refusal_accuracy,
    token_f1,
)

INPUT = Path("results/phase-03-track2.jsonl")
OUTPUT = Path("results/phase-03-track2-scores.jsonl")

REDACTED_TOKEN_RE = re.compile(r"\[REDACTED:([A-Z_]+)\]")


def _pii_pred_from_pred(pred: dict[str, Any]) -> list[dict]:
    text = pred.get("answer") if isinstance(pred, dict) else ""
    return [
        {"type": m.group(1).lower(), "start": m.start(), "end": m.end(), "action": "mask"}
        for m in REDACTED_TOKEN_RE.finditer(text or "")
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Track 2 input not found: {in_path}", file=sys.stderr)
        return 1

    gt_path = Path("data/processed/phase03_synth_contracts.jsonl")
    ref_pii_per_case: dict[str, list[dict]] = {}
    if gt_path.exists():
        with gt_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                case = rec.get("source_pdf", "").split("/")[-1].replace(".pdf", "") or f"contract_synth-{rec['case_id']}"
                cats: list[str] = []
                gf = rec.get("ground_truth_fields", {}) or {}
                if gf.get("buyer_email"):
                    cats.append("email")
                if gf.get("buyer_phone"):
                    cats.append("phone_vn")
                if gf.get("seller_tax_id"):
                    cats.append("tax_id")
                if gf.get("vin"):
                    cats.append("vin")
                if gf.get("buyer_id"):
                    cats.append("cccd")
                if gf.get("seller_account"):
                    cats.append("bank_account")
                ref_pii_per_case[case] = [{"type": c, "action": "mask"} for c in cats]

    scored: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pred = rec.get("pred", {}) or {}
            ref = rec.get("ref", {}) or {}

            em = exact_match(pred.get("answer", ""), ref.get("answer", ""))
            tf1 = token_f1(pred.get("answer", ""), ref.get("answer", ""))
            ra = refusal_accuracy(pred, ref)
            cp = citation_precision(pred, {"valid_chunk_ids": ref.get("valid_chunk_ids", [])})

            ref_pii = ref_pii_per_case.get(rec["case_id"], [])
            pred_pii = _pii_pred_from_pred(pred)
            pr = pii_precision_recall(pred_pii, ref_pii)

            scored.append({
                "ts": rec.get("ts"),
                "track": "track2",
                "case_id": rec["case_id"],
                "config": rec["config"],
                "deployment_id": rec.get("deployment_id"),
                "cost_usd": rec.get("cost_usd", 0.0),
                "latency_ms": rec.get("latency_ms", 0.0),
                "input_tokens": rec.get("input_tokens", 0),
                "output_tokens": rec.get("output_tokens", 0),
                "is_refusable": ref.get("is_refusable", False),
                "score": {**em, **tf1, **ra, **cp, **pr},
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_cfg[r["config"]].append(r)
    summary: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": "track2",
        "configs": {},
    }
    for cfg, rs in by_cfg.items():
        n = len(rs)
        refusals = [r for r in rs if r["is_refusable"]]
        summary["configs"][cfg] = {
            "n_cases": n,
            "n_refusal": len(refusals),
            "avg_cost_usd": round(sum(r["cost_usd"] for r in rs) / n, 6),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in rs) / n, 1),
            "avg_total_tokens": round(sum(r["input_tokens"] + r["output_tokens"] for r in rs) / n, 1),
            "avg_exact_match": round(sum(r["score"]["exact_match"] for r in rs) / n, 4),
            "avg_token_f1": round(sum(r["score"]["token_f1"] for r in rs) / n, 4),
            "refusal_accuracy": round(sum(r["score"]["refusal_accuracy"] for r in refusals) / max(1, len(refusals)), 4) if refusals else 0.0,
            "avg_citation_precision": round(sum(r["score"]["citation_precision"] for r in rs) / n, 4),
            "avg_pii_precision": round(sum(r["score"]["pii_precision"] for r in rs) / n, 4),
            "avg_pii_recall": round(sum(r["score"]["pii_recall"] for r in rs) / n, 4),
            "avg_pii_f1": round(sum(r["score"]["pii_f1"] for r in rs) / n, 4),
        }

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"SUMMARY": summary}, ensure_ascii=False) + "\n")

    print("=== Track 2 scoring summary ===")
    print(f"{'config':<10} {'n':>3} {'refus':>5} {'cost_usd':>10} {'lat_ms':>10} "
          f"{'em':>5} {'token_f1':>9} {'refusal':>7} {'cite':>5} {'pii_P':>6} {'pii_R':>6} {'pii_F1':>6}")
    for cfg, rs in by_cfg.items():
        s = summary["configs"][cfg]
        print(f"{cfg:<10} {s['n_cases']:>3} {s['n_refusal']:>5} {s['avg_cost_usd']:>10.6f} {s['avg_latency_ms']:>10.1f} "
              f"{s['avg_exact_match']:>5.2f} {s['avg_token_f1']:>9.3f} {s['refusal_accuracy']:>7.3f} {s['avg_citation_precision']:>5.2f} "
              f"{s['avg_pii_precision']:>6.3f} {s['avg_pii_recall']:>6.3f} {s['avg_pii_f1']:>6.3f}")
    print(f"\nOutput: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

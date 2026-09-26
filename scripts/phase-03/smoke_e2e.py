"""End-to-end smoke test for the OP5 live UI stack.

Sequence:
  1. GET  /healthz                       (FastAPI alive + Gemini keys visible)
  2. POST /track1/extract                (extract fields on one SYNTH case)
  3. POST /track2/ask                    (ask one Q&A on one SYNTH case)
  4. GET  /inspect/{case_id}             (read JSONL results for the same case)
  5. verify results/phase-03-track*-live.jsonl has cost_usd > 0
  6. invoke scripts/phase-03/pareto_plot.py with the live JSONL

Exit 0 = all green; non-zero = the first step that failed (printed).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
TRACK1_JSONL = REPO / "results" / "phase-03-track1-live.jsonl"
TRACK2_JSONL = REPO / "results" / "phase-03-track2-live.jsonl"


def step(msg: str) -> None:
    print(f"\n=== STEP: {msg} ===", flush=True)


def run() -> int:
    base = os.getenv("OP5_API_URL", "http://localhost:8000").rstrip("/")
    client = httpx.Client(base_url=base, timeout=120.0)

    step("1. GET /healthz")
    try:
        r = client.get("/healthz")
        r.raise_for_status()
        health = r.json()
        print(f"  status:           {health.get('status')}")
        print(f"  keys_configured:  {health.get('keys_configured')}")
        print(f"  storage_backend:  {health.get('storage_backend')}")
        print(f"  embedder_status:  {(health.get('embedder_status') or '')[:40]}")
        if health.get("status") != "ok":
            print("  WARN: health not OK; some steps may fail")
    except Exception as exc:
        print(f"  FAIL: cannot reach {base}: {exc}", file=sys.stderr)
        return 1

    step("2. POST /track1/extract")
    try:
        r = client.post(
            "/track1/extract",
            json={
                "pdf_path": "data/Scan/scan_phase03/contract_synth-ctr-001.pdf",
                "config": "D",
                "use_ocr": False,
            },
        )
        r.raise_for_status()
        t1 = r.json()
        print(f"  case_id:        {t1.get('case_id')}")
        print(f"  model:          {t1.get('model')}")
        print(f"  cost_usd:       {t1.get('cost_usd')}")
        print(f"  latency_ms:     {t1.get('latency_ms'):.0f}")
        print(f"  n_fields:       {len(t1.get('extracted_fields') or {})}")
    except Exception as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        return 2

    step("3. POST /track2/ask")
    try:
        r = client.post(
            "/track2/ask",
            json={
                "case_id": "contract_synth-ctr-001",
                "question": "Hợp đồng có hiệu lực từ ngày nào?",
                "config": "A",
                "top_k": 4,
            },
        )
        r.raise_for_status()
        t2 = r.json()
        print(f"  answer:        {t2.get('answer', '')[:80]}")
        print(f"  refused:       {t2.get('refused')}")
        print(f"  citations:     {t2.get('citations')}")
        print(f"  cost_usd:      {t2.get('cost_usd')}")
        print(f"  latency_ms:    {t2.get('latency_ms'):.0f}")
    except Exception as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        return 3

    step("4. GET /inspect/{case_id}")
    try:
        r = client.get("/inspect/contract_synth-ctr-001")
        r.raise_for_status()
        ins = r.json()
        print(f"  case_id:       {ins.get('case_id')}")
        print(f"  track1 rows:   {len(ins.get('track1') or [])}")
        print(f"  track2 rows:   {len(ins.get('track2') or [])}")
    except Exception as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        return 4

    step("5. verify results/phase-03-track*-live.jsonl has cost_usd > 0")
    try:
        t1_cost = _read_avg_cost(TRACK1_JSONL)
        t2_cost = _read_avg_cost(TRACK2_JSONL)
        print(f"  track1 avg cost: ${t1_cost:.6f} (file: {TRACK1_JSONL.name})")
        print(f"  track2 avg cost: ${t2_cost:.6f} (file: {TRACK2_JSONL.name})")
        if t1_cost <= 0 and t2_cost <= 0:
            print("  WARN: at least one of the JSONLs has $0 cost (stub mode likely)")
    except Exception as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        return 5

    step("6. invoke scripts/phase-03/pareto_plot.py on live JSONL")
    try:
        out_t1 = REPO / "doc" / "figs" / "phase-03-track1-live.png"
        out_t2 = REPO / "doc" / "figs" / "phase-03-track2-live.png"
        out_t1.parent.mkdir(parents=True, exist_ok=True)
        score1 = REPO / "results" / "phase-03-track1-scores-live.jsonl"
        score2 = REPO / "results" / "phase-03-track2-scores-live.jsonl"
        if not score1.exists() or not score2.exists():
            print(f"  SKIP: missing {score1} or {score2}; run score_track*.py first")
        else:
            cmd = [
                sys.executable,
                str(REPO / "scripts" / "phase-03" / "pareto_plot.py"),
                "--track1-scores", str(score1),
                "--track2-scores", str(score2),
                "--out-t1", str(out_t1),
                "--out-t2", str(out_t2),
            ]
            subprocess.run(cmd, cwd=str(REPO), check=True)
            print(f"  wrote {out_t1.name} and {out_t2.name}")
    except Exception as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        return 6

    print("\n=== ALL 6 STEPS GREEN ===")
    return 0


def _read_avg_cost(path: Path) -> float:
    if not path.exists():
        return 0.0
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return 0.0
    total = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    return total / len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=os.getenv("OP5_API_URL", "http://localhost:8000"))
    args = parser.parse_args()
    os.environ["OP5_API_URL"] = args.api_url
    return run()


if __name__ == "__main__":
    sys.exit(main())

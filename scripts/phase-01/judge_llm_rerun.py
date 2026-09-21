#!/usr/bin/env python3
"""
Re-judge existing results using Gemini-as-judge.

Reads JSONL results (any phase-01-longllmlingua-opt*.jsonl), filters cases
where `judge_correct` == False, sends (context_question, gold, pred) to Gemini
3.1-flash-lite, and writes new JSONL with updated `llm_judge_correct` /
`llm_judge_reason` fields.

Output:
  results/phase-01-{basename}-llm-judge.jsonl   (re-judged subset)
  results/phase-01-{basename}-llm-judge-summary.json

Usage:
  conda activate vsf
  python scripts/phase-01/judge_llm_rerun.py \
      --input results/phase-01-longllmlingua-opt-n20.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------
LLM_JUDGE_PROMPT = """Ban la mot truong gia phan bien (expert evaluator).

Ngu canh (tuy chon, co the trong):
{context}

Cau hoi: {question}

Dap an chuan (ground truth): {gold}

Dap an cua he thong (prediction): {pred}

Nhiem vu: Danh gia dap an cua he thong co chinh xac khong.

Tieu chi danh gia:
- "correct" neu dap an bat dau bang cung thong tin dung tu ground truth, hoac
  co y nghia tuong duong (vi du: "Poland" = "Ba Lan", "70%" = "70 percent")
- "ambiguous" neu co the dung nhung khong chac chan (confidence < 0.7)
- "incorrect" neu thong tin chinh hoac dac biet la thong tin SO DO vi pham ground truth

Tra loi CHI VOI JSON (khong giai thich them):
{{"verdict": "correct"|"incorrect"|"ambiguous", "reason": "... (1-2 cau tieng Viet)", "confidence": 0.0-1.0}}
"""


# ---------------------------------------------------------------
# Gemini client (lazy)
# ---------------------------------------------------------------
_gemini = None


def get_gemini(model_name: str = "gemini-3.1-flash-lite"):
    global _gemini
    if _gemini is None:
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY missing — set GOOGLE_API_KEY")
        genai.configure(api_key=key)
        _gemini = genai.GenerativeModel(model_name)
        print(f"  [init] Gemini-as-judge ready: {model_name}")
    return _gemini


# ---------------------------------------------------------------
# LLM-as-judge call
# ---------------------------------------------------------------
def llm_judge(model, question: str, gold: str, pred: str, context: str = "",
              max_retries: int = 3) -> dict:
    prompt = LLM_JUDGE_PROMPT.format(
        context=context[:2000] if context else "(khong co nghi canh)",
        question=question or "(khong co cau hoi — task tu dong)",
        gold=gold[:500],
        pred=pred[:500],
    )
    for attempt in range(max_retries):
        try:
            r = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 200,
                },
            )
            raw = (r.text or "").strip()
            # Try to extract JSON
            m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                verdict = obj.get("verdict", "ambiguous").lower()
                confidence = float(obj.get("confidence", 0.5))
                reason = str(obj.get("reason", ""))
                return {
                    "verdict": verdict,
                    "correct": verdict == "correct",
                    "reason": reason,
                    "confidence": confidence,
                    "raw": raw,
                }
            # Fallback: keyword heuristic on raw response
            lower = raw.lower()
            if any(w in lower for w in ["\"correct\"", "correct", "chinh xac", "dung"]):
                return {"verdict": "correct", "correct": True,
                        "reason": raw[:80], "confidence": 0.6, "raw": raw}
            elif any(w in lower for w in ["\"incorrect\"", "incorrect", "sai", "khong dung"]):
                return {"verdict": "incorrect", "correct": False,
                        "reason": raw[:80], "confidence": 0.6, "raw": raw}
            return {"verdict": "ambiguous", "correct": None,
                    "reason": raw[:80], "confidence": 0.5, "raw": raw}
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"    [warn] 429, sleeping {wait}s (try {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                return {"verdict": "error", "correct": None,
                        "reason": f"{type(exc).__name__}: {err[:100]}", "confidence": 0.0}
    return {"verdict": "timeout", "correct": None,
            "reason": "max retries exceeded", "confidence": 0.0}


# ---------------------------------------------------------------
# Load JSONL helper
# ---------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    decoder = json.JSONDecoder()
    rows = []
    buf = path.read_text(encoding="utf-8")
    pos = 0
    while pos < len(buf):
        while pos < len(buf) and buf[pos] in " \t\n\r":
            pos += 1
        if pos >= len(buf):
            break
        try:
            obj, end = decoder.raw_decode(buf, pos)
            rows.append(obj)
            pos = end
        except json.JSONDecodeError:
            break
    return rows


def extract_dataset_id(case_id: str) -> str:
    """Extract the dataset ID from a case_id like 'S-00-3hop1__1' or 'L00-3hop1__1'."""
    m = re.search(r"(?:S-\d+-|L\d+-)(.+?)(?:__|$)", case_id)
    return m.group(1) if m else case_id


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-judge re-run on existing results")
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input JSONL (e.g. results/phase-01-longllmlingua-opt-n20.jsonl)")
    parser.add_argument("--only-failed", action="store_true", default=True,
                        help="Only re-judge heuristic failures (default True)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        return 1

    rows = load_jsonl(args.input)
    print(f"\n  Loaded {len(rows)} rows from {args.input.name}")

    # Classify by heuristic result
    heuristic_ok = [r for r in rows if r.get("judge_correct") is True]
    heuristic_miss = [r for r in rows if r.get("judge_correct") is not True]

    print(f"  Heuristic correct:   {len(heuristic_ok)}")
    print(f"  Heuristic miss:      {len(heuristic_miss)}  <- will re-judge")

    # Load context+question from dataset
    dataset_path = REPO_ROOT / "data" / "processed" / "zero_scrolls_dev95.jsonl"
    ds_rows = load_jsonl(dataset_path)
    ds_map = {}
    for ds_r in ds_rows:
        did = ds_r.get("id", "")
        inp = ds_r.get("input", "")
        ds = int(ds_r.get("document_start_index", 0))
        de = int(ds_r.get("document_end_index", 0))
        qs = int(ds_r.get("query_start_index", 0))
        qe = int(ds_r.get("query_end_index", 0))
        ctx = inp[ds:de]
        q = inp[qs:qe]
        ds_map[did] = {"context": ctx, "question": q}
    print(f"  Dataset context loaded: {len(ds_map)} entries")

    # Match each heuristic-miss row to dataset context
    cases_to_judge = []
    skipped = []
    for rec in heuristic_miss:
        cid = rec.get("case_id", "")
        did = extract_dataset_id(cid)
        # Try exact, then partial
        ds_info = ds_map.get(did)
        if not ds_info:
            ds_info = ds_map.get(cid)
        if not ds_info:
            for k, v in ds_map.items():
                if did in k or k in did:
                    ds_info = v
                    break
        if ds_info:
            cases_to_judge.append((rec, ds_info))
        else:
            skipped.append(cid)
            rec["llm_judge_verdict"] = "no_context"
            rec["llm_judge_correct"] = None
            rec["llm_judge_reason"] = "no dataset context found"

    if skipped:
        print(f"  [warn] {len(skipped)} cases skipped (no dataset context): {skipped[:3]}")

    print(f"  Cases to LLM-judge: {len(cases_to_judge)}")

    if not cases_to_judge:
        print("  Nothing to judge. Exiting.")
        return 0

    # Run LLM judge
    model = get_gemini()
    out_all = list(heuristic_ok)  # copy heuristic-ok rows unchanged
    heuristic_miss_ids = set(r["case_id"] for r in heuristic_miss)

    total = len(cases_to_judge)
    for i, (rec, ds_info) in enumerate(cases_to_judge):
        cid = rec["case_id"]
        pred = rec.get("pred", "")
        gold = rec.get("gold", "")
        question = ds_info.get("question", "")
        context = ds_info.get("context", "")

        print(f"\n  [{i+1}/{total}] {cid}")
        print(f"    Q: {question[:80]!r}")
        print(f"    Gold: {gold[:60]!r}")
        print(f"    Pred: {pred[:60]!r}")

        jj = llm_judge(model, question, gold, pred, context=context)
        rec["llm_judge_verdict"] = jj["verdict"]
        rec["llm_judge_correct"] = jj["correct"]
        rec["llm_judge_reason"] = jj["reason"]
        rec["llm_judge_confidence"] = jj.get("confidence", 0.0)
        rec["llm_judge_raw"] = jj.get("raw", "")[:200]

        verdict_str = jj["verdict"]
        conf = jj.get("confidence", 0.0)
        print(f"    -> {verdict_str} (conf={conf:.2f}) {jj['reason'][:60]}")

        out_all.append(rec)

        # Rate limit
        time.sleep(1.0)

    # Write outputs
    in_name = args.input.stem
    out_path = REPO_ROOT / "results" / f"phase-01-{in_name}-llm-judge.jsonl"
    summary_path = REPO_ROOT / "results" / f"phase-01-{in_name}-llm-judge-summary.json"

    out_all.sort(key=lambda r: r["case_id"])
    with out_path.open("w", encoding="utf-8") as fh:
        for r in out_all:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    n_total = len(out_all)
    n_heuristic_ok = len(heuristic_ok)
    n_llm_correct = sum(1 for r in out_all if r.get("llm_judge_correct") is True)
    n_llm_incorrect = sum(1 for r in out_all if r.get("llm_judge_correct") is False)
    n_llm_ambiguous = sum(1 for r in out_all if r.get("llm_judge_verdict") == "ambiguous")
    n_skipped = len(skipped)

    # Improvement from LLM judge
    improved = sum(1 for r in cases_to_judge
                   if r[0].get("llm_judge_correct") is True)
    still_wrong = sum(1 for r in cases_to_judge
                      if r[0].get("llm_judge_correct") is False)

    acc_heuristic = round(n_heuristic_ok / n_total, 3) if n_total else None
    acc_llm = round((n_heuristic_ok + n_llm_correct) / n_total, 3) if n_total else None

    # Per-task breakdown
    per_task = {}
    for r in out_all:
        t = r.get("task", "?")
        per_task.setdefault(t, {"n": 0, "h_ok": 0, "llm_ok": 0})
        per_task[t]["n"] += 1
        if r.get("judge_correct") is True:
            per_task[t]["h_ok"] += 1
        if r.get("llm_judge_correct") is True:
            per_task[t]["llm_ok"] += 1

    summary = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": str(args.input),
        "n_total": n_total,
        "heuristic_correct": n_heuristic_ok,
        "llm_judge": {
            "n_judged": len(cases_to_judge),
            "correct": n_llm_correct,
            "incorrect": n_llm_incorrect,
            "ambiguous": n_llm_ambiguous,
            "skipped": n_skipped,
        },
        "improvement": {
            "was_miss_now_correct": improved,
            "was_miss_still_wrong": still_wrong,
        },
        "accuracy": {
            "heuristic_only": acc_heuristic,
            "heuristic_plus_llm": acc_llm,
        },
        "per_task": {
            t: {
                "n": v["n"],
                "heuristic_acc": round(v["h_ok"] / v["n"], 3) if v["n"] else None,
                "llm_judge_acc": round((v["h_ok"] + v["llm_ok"]) / v["n"], 3) if v["n"] else None,
            }
            for t, v in sorted(per_task.items())
        },
    }

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print(" LLM-AS-JUDGE SUMMARY")
    print("=" * 80)
    print(f"  Input:              {args.input.name}")
    print(f"  Total cases:        {n_total}")
    print(f"  Heuristic OK:      {n_heuristic_ok} (acc={acc_heuristic:.1%})")
    print(f"  LLM-judged:         {len(cases_to_judge)}")
    print(f"    -> correct:      {n_llm_correct}")
    print(f"    -> incorrect:    {n_llm_incorrect}")
    print(f"    -> ambiguous:    {n_llm_ambiguous}")
    print(f"    -> skipped:     {n_skipped}")
    print(f"  Improvement:        {improved}/{len(cases_to_judge)} heuristic-miss -> now correct")
    print(f"  Accuracy (heuristic only):   {acc_heuristic:.1%}")
    print(f"  Accuracy (heuristic + LLM):   {acc_llm:.1%}")
    print(f"\n  Per-task:")
    for t, v in sorted(per_task.items()):
        ha = round(v["h_ok"] / v["n"], 3) if v["n"] else 0
        la = round((v["h_ok"] + v["llm_ok"]) / v["n"], 3) if v["n"] else 0
        print(f"    {t:<20s} n={v['n']:>2}  heuristic={ha:.1%}  llm={la:.1%}")
    print(f"\n  Output: {out_path}")
    print(f"          {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

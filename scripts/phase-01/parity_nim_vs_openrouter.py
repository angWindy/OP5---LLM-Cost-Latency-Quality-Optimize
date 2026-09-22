#!/usr/bin/env python3
"""
NIM-only smoke test on 5 zero_scrolls test cases.
Tests exact, paraphrase, wrong variants per case.
Faster than full parity (skips slow OpenRouter).

Run: conda activate vsf && python scripts/phase-01/parity_nim_vs_openrouter.py
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(ROOT / "src"))

from op5.llm import NIMJudge

DEV = ROOT / "data/processed/zero_scrolls_test5.jsonl"

def paraphrase(txt):
    for s,d in [(" and "," & "),(" percent","%"),(" million","M"),
                (" is "," = "),("1 ","one "),("2 ","two ")]:
        txt = txt.replace(s,d)
    return txt

def main():
    cases = [json.loads(l) for l in DEV.open()]
    judge = NIMJudge()

    n_ok, n_total = 0, 0
    print(f"{'#':<2} {'variant':<14} {'verdict':<12} {'correct':<8} {'conf':<5} {'ms':<7}")
    print("-"*55)

    for idx, row in enumerate(cases, 1):
        gold = row["output"].strip()
        text = row["input"]
        q = text.split("Question:")[1].split("Answer:")[0].strip()[:300] if "Question:" in text else text[:200]

        for vname, pred in [
            ("exact", gold),
            ("paraphrase", paraphrase(gold)),
            ("wrong", "xyz123_not_matching"),
        ]:
            r = judge.judge(question=q, gold=gold, pred=pred, context="")
            ok = r.correct is True if vname != "wrong" else r.correct is False
            n_total += 1
            if ok: n_ok += 1
            print(f"{idx:<2} {vname:<14} {r.verdict:<12} {str(r.correct):<8} {r.confidence:.1f} {r.latency_ms:>5.0f}ms  {r.model_used.split('/')[-1][:25]}")

    print("-"*55)
    print(f"  Pass: {n_ok}/{n_total} ({n_ok/n_total*100:.0f}%)")

    out = ROOT / "results" / "nim_vs_openrouter_parity.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as fh:
        for idx, row in enumerate(cases, 1):
            gold = row["output"].strip()
            text = row["input"]
            q = text.split("Question:")[1].split("Answer:")[0].strip()[:300] if "Question:" in text else text[:200]
            for vname, pred in [
                ("exact", gold),
                ("paraphrase", paraphrase(gold)),
                ("wrong", "xyz123_not_matching"),
            ]:
                r = judge.judge(question=q, gold=gold, pred=pred, context="")
                fh.write(json.dumps({"case": idx, "variant": vname, "gold": gold[:80], "pred": pred[:80], **r.to_dict()}, ensure_ascii=False) + "\n")
    print(f"  Wrote: {out}")

if __name__ == "__main__":
    main()

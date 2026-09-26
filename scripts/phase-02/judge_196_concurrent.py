"""
judge_196_concurrent.py — Concurrent LLM-as-judge runner for the 5 phase-02 files.

Reuses `LLMJudge` from src/op5/llm/judge.py. Reads from `results/runs/phase-02/`,
writes judged outputs to `results/judges/{profile}/phase-02/`, and cross-rate
summary to `results/summaries/phase-02/`.

Why concurrent: 5 files x 196 = 980 judge calls; serial ~30 min,
12 workers ~3 min (DeepSeek-V3 API handles concurrency well).

    Output convention:
      results/runs/phase-02/phase-02-run-{slug}.jsonl           — raw predictions
      results/judges/{profile}/phase-02/phase-02-judged-{slug}.jsonl  — judged
      results/judges/{profile}/phase-02/phase-02-judged-{slug}-summary.json
      results/summaries/phase-02/phase-02-{profile_key}-196-summary.json

Usage:
  conda activate vsf
  python scripts/phase-02/judge_196_concurrent.py                  # all 5 files, deepseek
  python scripts/phase-02/judge_196_concurrent.py --profile deepseek_pro --workers 12
  python scripts/phase-02/judge_196_concurrent.py --files baseline_n196 rate60
  python scripts/phase-02/judge_196_concurrent.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# ── Path convention ────────────────────────────────────────────────────────────
RUNS_DIR      = REPO_ROOT / "results" / "runs"      / "phase-02"
JUDGES_DIR    = REPO_ROOT / "results" / "judges"
SUMMARIES_DIR = REPO_ROOT / "results" / "summaries" / "phase-02"
BACKUP_DIR    = REPO_ROOT / "results" / "_backups"

# Profile name → subfolder name
_PROFILE_SUBDIR = {
    "deepseek":     "deepseek-flash",
    "deepseek_pro": "deepseek-v4-pro",
}


def judge_dir_for_profile(profile: str) -> Path:
    sub = _PROFILE_SUBDIR.get(profile, profile)
    return JUDGES_DIR / sub / "phase-02"


def profile_key(profile: str) -> str:
    known = {
        "deepseek":     "deepseek-flash",
        "deepseek_pro": "deepseek-v4-pro",
    }
    return known.get(profile, profile.lower().replace("-", "").replace("_", ""))


# Canonical run slugs
DEFAULT_RUNS = [
    ("baseline_n196", "baseline (no compression)"),
    ("rate40",        "LongLLMLingua rate=0.4"),
    ("rate50",        "LongLLMLingua rate=0.5"),
    ("rate60",        "LongLLMLingua rate=0.6"),
    ("rate70",        "LongLLMLingua rate=0.7"),
]


# ── Dedup (in-place on RUNS_DIR) ─────────────────────────────────────────────
def dedup_inplace(path: Path) -> tuple[int, int]:
    """Returns (n_input, n_kept). Keeps row with latest `ts` per case_id.
    Writes a sibling .dedup-report.json."""
    rows = [json.loads(l) for l in path.open() if l.strip()]
    n_input = len(rows)
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_id[r.get("case_id", f"_no_id_{len(by_id)}")].append(r)
    kept, dropped_ids = [], []
    for cid, group in by_id.items():
        group_sorted = sorted(group, key=lambda r: r.get("ts", ""), reverse=True)
        kept.append(group_sorted[0])
        if len(group_sorted) > 1:
            dropped_ids.append(cid)
    kept.sort(key=lambda r: r.get("case_id", ""))
    with path.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "input": str(path),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_input": n_input,
        "n_kept": len(kept),
        "n_dropped": len(dropped_ids),
        "dropped_case_ids": dropped_ids[:50],
        "strategy": "keep_latest_ts",
    }
    with path.with_suffix(".dedup-report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return n_input, len(kept)


# ── Judge ─────────────────────────────────────────────────────────────────────
def judge_one(judge, question: str, gold: str, pred: str) -> dict:
    """Run LLMJudge, return judge_* field dict."""
    try:
        result = judge.judge(question=question, gold=gold, pred=pred)
        return {
            "judge_verdict": result.verdict,
            "judge_correct": (result.verdict == "correct") if result.verdict != "ambiguous" else None,
            "judge_reason": (result.reason or "")[:200],
            "judge_model": result.model_used,
            "judge_latency_ms": round(float(result.latency_ms), 1),
            "judge_confidence": round(float(result.confidence), 3),
        }
    except Exception as exc:
        return {
            "judge_verdict": "error",
            "judge_correct": None,
            "judge_reason": f"judge exception: {str(exc)[:160]}",
            "judge_model": "",
            "judge_latency_ms": 0.0,
            "judge_confidence": 0.0,
        }


def extract_pred(rec: dict) -> str:
    pred = rec.get("pred") or rec.get("text") or ""
    if isinstance(pred, dict):
        pred = pred.get("text", str(pred))
    return pred or ""


# ── Per-file judge loop ───────────────────────────────────────────────────────
def judge_file(
    slug: str,
    run_path: Path,
    judge_path: Path,
    profile: str,
    workers: int,
    skip_existing: bool,
    dry_run: bool,
    verbose: bool,
) -> dict:
    print(f"\n=== {slug} ===")
    print(f"  Run  : {run_path.relative_to(REPO_ROOT)}")
    print(f"  Judge: {judge_path.relative_to(REPO_ROOT)}")

    if not run_path.exists():
        print(f"  MISSING - skipped")
        return {"slug": slug, "status": "missing"}

    rows = [json.loads(l) for l in run_path.open() if l.strip()]
    print(f"  Records: {len(rows)}")

    # Load existing judged rows if output already exists (resume / skip-existing mode)
    if judge_path.exists():
        existing = {json.loads(l).get("case_id"): json.loads(l)
                    for l in judge_path.open() if l.strip()}
        print(f"  Existing judged: {len(existing)}")
    else:
        existing = {}

    # Decide which need judging
    if skip_existing:
        todo_cids = [
            r.get("case_id") for r in rows
            if r.get("case_id") not in existing
            or existing[r.get("case_id")].get("judge_verdict")
               not in ("correct", "incorrect", "ambiguous")
        ]
    else:
        todo_cids = [r.get("case_id") for r in rows]
    print(f"  Need judging: {len(todo_cids)} (skip_existing={skip_existing})")

    if not todo_cids and existing:
        final_rows = sorted(existing.values(), key=lambda r: r.get("case_id", ""))
        return _build_summary(slug, final_rows)

    if dry_run:
        print(f"  DRY-RUN - would judge {len(todo_cids)} records")
        return {"slug": slug, "status": "dry-run", "n_todo": len(todo_cids)}

    if not todo_cids:
        print(f"  All records already judged")
        final_rows = sorted(existing.values(), key=lambda r: r.get("case_id", ""))
        return _build_summary(slug, final_rows)

    rows_by_id = {r.get("case_id"): r for r in rows}
    judged: dict[str, dict] = dict(existing)

    from op5.llm import LLMJudge
    worker_judges: dict[int, LLMJudge] = {}

    def get_judge(wid: int) -> LLMJudge:
        if wid not in worker_judges:
            worker_judges[wid] = LLMJudge(profile=profile)
        return worker_judges[wid]

    t0 = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for cid in todo_cids:
            rec = rows_by_id.get(cid)
            if not rec:
                continue
            wid = len(futures) % workers
            fut = ex.submit(
                judge_one,
                get_judge(wid),
                rec.get("question", ""),
                rec.get("gold", ""),
                extract_pred(rec),
            )
            futures[fut] = (cid, rec)

        for fut in as_completed(futures):
            cid, rec = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "judge_verdict": "error", "judge_correct": None,
                    "judge_reason": f"future exception: {str(exc)[:160]}",
                    "judge_model": "", "judge_latency_ms": 0.0, "judge_confidence": 0.0,
                }
            merged = dict(rec)
            merged.update(result)
            judged[cid] = merged
            completed += 1
            if verbose or completed % 25 == 0 or completed == len(todo_cids):
                v = result["judge_verdict"]
                flag = "T" if result["judge_correct"] is True else ("F" if result["judge_correct"] is False else "?")
                print(f"  [{completed:>3}/{len(todo_cids)}] {flag} {cid[:35]:<35} "
                      f"v={v:<10} ms={result['judge_latency_ms']:>5.0f}")

    elapsed = time.perf_counter() - t0

    # Persist judged output — ordered by case_id
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    final_rows = sorted(judged.values(), key=lambda r: r.get("case_id", ""))
    with judge_path.open("w", encoding="utf-8") as fh:
        for r in final_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = _build_summary(slug, final_rows)
    summary["judge_wallclock_s"] = round(elapsed, 1)
    summary["judge_throughput_per_s"] = round(completed / elapsed, 2) if elapsed else None
    summary["judge_profile"] = profile
    summary["judge_workers"] = workers
    summary["n_source"] = len(rows)
    summary["n_judged_new"] = completed

    summary_path = judge_path.with_name(judge_path.stem + "-summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"  Wrote: {summary_path.name}")

    print(f"  Done: {completed} new judged in {elapsed:.1f}s ({summary['judge_throughput_per_s']} calls/s)")
    print(f"  Accuracy: {summary['accuracy_excl_ambig']}")
    return summary


def _build_summary(slug: str, rows: list[dict]) -> dict:
    judged = [r for r in rows if r.get("judge_verdict") in ("correct", "incorrect", "ambiguous")]
    corr   = sum(1 for r in judged if r.get("judge_correct") is True)
    incorr = sum(1 for r in judged if r.get("judge_correct") is False)
    ambig  = sum(1 for r in judged if r.get("judge_verdict") == "ambiguous")
    model  = next((r.get("judge_model") for r in rows if r.get("judge_model")), None)
    lats   = [r["judge_latency_ms"] for r in rows if r.get("judge_latency_ms")]

    per_task: dict[str, dict] = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in judged:
        t = r.get("task", "?")
        per_task[t]["n"] += 1
        if r.get("judge_correct"):
            per_task[t]["ok"] += 1

    return {
        "slug": slug,
        "status": "ok",
        "n_records": len(rows),
        "n_judged": len(judged),
        "correct": corr,
        "incorrect": incorr,
        "ambiguous": ambig,
        "accuracy_excl_ambig": round(corr / (corr + incorr), 4) if (corr + incorr) else None,
        "judge_model": model,
        "judge_latency_ms_median": round(statistics.median(lats), 1) if lats else None,
        "per_task": {t: dict(d) for t, d in per_task.items()},
    }


def print_cross_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("CROSS-RATE SUMMARY")
    print("=" * 90)
    header = f"{'Slug':<14}  {'N':>3}  {'Corr':>4}  {'Inc':>4}  {'Amb':>3}  {'Acc':>7}  {'LatMed':>7}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        acc = s.get("accuracy_excl_ambig")
        acc_s = f"{acc*100:.1f}%" if acc is not None else "n/a"
        lat = s.get("judge_latency_ms_median")
        lat_s = f"{lat:.0f}ms" if lat is not None else "n/a"
        print(f"{s.get('slug','?'):<14}  {s.get('n_records','?'):>3}  "
              f"{s.get('correct',0):>4}  {s.get('incorrect',0):>4}  "
              f"{s.get('ambiguous',0):>3}  {acc_s:>7}  {lat_s:>7}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="deepseek",
                   help="LLMJudge profile name (default: deepseek = deepseek-chat / flash)")
    p.add_argument("--workers", type=int, default=12,
                   help="ThreadPoolExecutor max_workers (default: 12)")
    p.add_argument("--files", nargs="*", default=None,
                   help="Subset of slugs to judge (default: all 5)")
    p.add_argument("--skip-existing", dest="skip_existing",
                   action="store_true", default=True,
                   help="Skip records that already have judge_verdict (default: True)")
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false",
                   help="Re-judge all records (overwrites existing judge fields)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be judged without making API calls")
    p.add_argument("--no-dedup", dest="dedup", action="store_false", default=True,
                   help="Skip the dedup step for files with duplicate case_ids")
    p.add_argument("--no-backup", dest="backup", action="store_false", default=True,
                   help="Skip backup of run files before judging")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-record progress")
    args = p.parse_args()

    chosen = [(slug, desc) for slug, desc in DEFAULT_RUNS
              if args.files is None or slug in args.files]
    if not chosen:
        print("ERROR: no files selected", file=sys.stderr)
        return 1

    jdir = judge_dir_for_profile(args.profile)
    print("=" * 90)
    print(f"  Concurrent judge-196  |  profile={args.profile}  workers={args.workers}")
    print(f"  Judge output dir: {jdir.relative_to(REPO_ROOT)}")
    print(f"  Runs: {[s for s, _ in chosen]}")
    print(f"  skip_existing={args.skip_existing}  dedup={args.dedup}  backup={args.backup}  dry_run={args.dry_run}")
    print("=" * 90)

    # Phase 0: backup run files
    if args.backup and not args.dry_run:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for slug, _ in chosen:
            src = RUNS_DIR / f"phase-02-run-{slug}.jsonl"
            if src.exists():
                dst = BACKUP_DIR / f"phase-02-run-{slug}.{ts}.jsonl"
                shutil.copy2(src, dst)
                print(f"  backup: {dst.relative_to(REPO_ROOT)}")

    # Phase 1: dedup
    if args.dedup and not args.dry_run:
        print("\n--- dedup ---")
        for slug, _ in chosen:
            path = RUNS_DIR / f"phase-02-run-{slug}.jsonl"
            if not path.exists():
                continue
            n_in, n_kept = dedup_inplace(path)
            tag = "" if n_in == n_kept else f"  (dropped {n_in - n_kept} dupes)"
            print(f"  {slug:<14}: {n_in} -> {n_kept}{tag}")

    # Phase 2: judge each run file → judged output
    print("\n--- judge ---")
    summaries = []
    for slug, desc in chosen:
        run_path   = RUNS_DIR / f"phase-02-run-{slug}.jsonl"
        judge_path = jdir  / f"phase-02-judged-{slug}.jsonl"
        s = judge_file(
            slug, run_path, judge_path,
            args.profile, args.workers,
            args.skip_existing, args.dry_run,
            not args.quiet,
        )
        s["description"] = desc
        summaries.append(s)

    # Phase 3: cross-rate summary
    if not args.dry_run:
        print_cross_table(summaries)
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        pkey = profile_key(args.profile)
        out = SUMMARIES_DIR / f"phase-02-{pkey}-196-summary.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump({
                "profile": args.profile,
                "judge_dir": str(jdir.relative_to(REPO_ROOT)),
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "summaries": summaries,
            }, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

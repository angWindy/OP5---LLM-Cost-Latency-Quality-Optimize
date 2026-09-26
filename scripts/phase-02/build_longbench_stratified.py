"""
Build a stratified 200-case eval set from LongBench (7 QA tasks).

Criteria:
  - Question is explicit (non-empty, >= 10 chars)
  - Answer is short (1-100 chars) for exact-match / short-judge friendliness
  - Context length >= 500 tokens (so compression has room to operate)
  - Strata: 4 equal-width buckets by context_length within each task,
            so we sample across the min->max range, not just medians.

Output:
  data/processed/longbench_200_stratified.jsonl
    Fields per record: id, task, dataset, language, length,
                       question, context, answer, stratum_idx

Usage:
  conda activate vsf
  python scripts/phase-02/build_longbench_stratified.py --n 200
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_OUT = PROCESSED_DIR / "longbench_200_stratified.jsonl"

# Tasks chosen for short-answer, explicit-question coverage (Sep 23 2026).
# See doc/worklog/2026-09-23-phase-02-dataset-switch-longbench.md.
LONGBENCH_TASKS = [
    "hotpotqa", "2wikimqa", "musique", "narrativeqa",
    "multifieldqa_en", "qasper", "triviaqa",
]

# Filter criteria
MIN_QUESTION_LEN = 10
MIN_ANSWER_LEN = 1
MAX_ANSWER_LEN = 100
MIN_CONTEXT_LEN = 500

# Stratification: split each task's filtered records into 4 equal-width buckets
# by context_length, then sample quota proportionally to the per-task total.
N_STRATA = 4


def load_and_filter(task: str) -> list[dict]:
    """Load a LongBench task file and apply answer-length + question filters."""
    fpath = PROCESSED_DIR / f"longbench_{task}.jsonl"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing {fpath}. Run download_longbench.py first.")

    out = []
    with fpath.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            length = int(r.get("length") or 0)
            if len(q) < MIN_QUESTION_LEN:
                continue
            if not (MIN_ANSWER_LEN <= len(a) <= MAX_ANSWER_LEN):
                continue
            if length < MIN_CONTEXT_LEN:
                continue
            out.append(r)
    return out


def stratified_sample(records: list[dict], n_target: int, seed: int) -> list[dict]:
    """Pick n_target records, evenly distributed across N_STRATA length buckets."""
    if not records:
        return []
    lengths = sorted(int(r.get("length") or 0) for r in records)
    q1 = lengths[len(lengths) // 4]
    q2 = lengths[len(lengths) // 2]
    q3 = lengths[3 * len(lengths) // 4]
    boundaries = [q1, q2, q3]

    def stratum_of(rec: dict) -> int:
        l = int(rec.get("length") or 0)
        for i, b in enumerate(boundaries):
            if l <= b:
                return i
        return N_STRATA - 1

    buckets: dict[int, list[dict]] = {i: [] for i in range(N_STRATA)}
    for r in records:
        buckets[stratum_of(r)].append(r)

    rng = random.Random(seed)
    per_stratum = max(1, n_target // N_STRATA)
    selected: list[dict] = []
    leftovers: list[dict] = []
    for i in range(N_STRATA):
        bucket = buckets[i]
        if len(bucket) <= per_stratum:
            selected.extend(bucket)
        else:
            rng.shuffle(bucket)
            selected.extend(bucket[:per_stratum])
            leftovers.extend(bucket[per_stratum:])

    # Fill shortfall with leftovers from any bucket
    deficit = n_target - len(selected)
    if deficit > 0 and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:deficit])

    # Tag each record with its stratum index for downstream inspection
    for r in selected:
        r["stratum_idx"] = stratum_of(r)
    return selected


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200,
                   help="Total cases across all tasks (default 200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", default="",
                   help="Comma-separated task subset; empty = all 7 default")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or LONGBENCH_TASKS

    print(f"=== Building stratified LongBench eval set ===")
    print(f"  Target: {args.n} cases")
    print(f"  Tasks:  {tasks}")
    print(f"  Seed:   {args.seed}")
    print(f"  Out:    {args.out}")
    print()

    # Per-task quotas: equal split (28-29 per task for 200 cases, 7 tasks)
    per_task = max(1, args.n // len(tasks))

    print(f"  Per-task target: {per_task}")
    print()
    print(f"  {'Task':<18} {'Avail':>7} {'Pick':>5} {'ctx_med':>8} {'ctx_range':>12}")
    print("  " + "-" * 55)

    all_selected: list[dict] = []
    for t in tasks:
        recs = load_and_filter(t)
        if not recs:
            print(f"  {t:<18} {0:>7} {0:>5} (no records passed filter)")
            continue
        lens = sorted(int(r.get("length") or 0) for r in recs)
        ctx_med = statistics.median(lens)
        ctx_range = f"{min(lens)}-{max(lens)}"
        pick = min(per_task, len(recs))
        chosen = stratified_sample(recs, pick, args.seed)
        for c in chosen:
            c["task"] = t
        all_selected.extend(chosen)
        print(f"  {t:<18} {len(recs):>7} {len(chosen):>5} {ctx_med:>8.0f} {ctx_range:>12}")

    print()
    print(f"Total selected: {len(all_selected)}")

    # Sort by length for easier stratified inspection downstream
    all_selected.sort(key=lambda r: int(r.get("length") or 0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in all_selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote: {args.out}")

    # Quick stats
    lens = [int(r.get("length") or 0) for r in all_selected]
    print()
    print(f"  Length min/med/max: {min(lens)} / {statistics.median(lens):.0f} / {max(lens)}")
    print(f"  Length p25/p75:     {lens[len(lens)//4]} / {lens[3*len(lens)//4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

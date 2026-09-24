"""
Download and prepare LongBench dataset (QA tasks only).

LongBench: https://huggingface.co/datasets/THUDM/LongBench
Filters to QA tasks only (excludes: musiang, vcr, lsht, leetcode, bhfile).
"""
import json
import os
from pathlib import Path
from datasets import load_dataset

# Output paths
DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# LongBench tasks - only QA tasks (exclude musiang, vcr, lsht, leetcode, bhfile)
# All have: input (context+question), context, question, answer
QA_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "vcsum",
    "en_sum",
]


def load_longbench_qa():
    """Load LongBench, keep only QA tasks, output JSONL by task."""
    print("Loading LongBench dataset...")
    ds = load_dataset("THUDM/LongBench")

    all_records = []
    task_counts = {}

    for split in ["test", "val"]:
        if split not in ds:
            continue
        print(f"\nProcessing split: {split}")
        for example in ds[split]:
            dataset_name = example.get("_dataset_name_", "")
            if dataset_name not in QA_TASKS:
                continue

            # Normalize to standard format: {id, task, dataset, language, length,
            # input, question, context, answer}
            record = {
                "id": f"{dataset_name}_{example.get('id', all_records.__len__())}",
                "task": dataset_name,
                "dataset": "LongBench",
                "input": example.get("input", ""),
                "context": example.get("context", ""),
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
                "length": example.get("length", 0),
            }

            all_records.append(record)
            task_counts[dataset_name] = task_counts.get(dataset_name, 0) + 1

    print(f"\nTotal records: {len(all_records)}")
    print("Records by task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task:<20}: {count}")

    return all_records


def save_by_task(records: list[dict]):
    """Save combined JSONL + per-task JSONL files."""
    # Combined
    out_path = DATA_DIR / "longbench_qa.jsonl"
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_path} ({len(records)} records)")

    # Per-task
    by_task: dict[str, list] = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)

    for task, task_records in by_task.items():
        task_path = DATA_DIR / f"longbench_{task}.jsonl"
        with task_path.open("w") as f:
            for r in task_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved: {task_path} ({len(task_records)} records)")


def show_sample(records: list[dict], n: int = 3):
    """Show sample records."""
    print("\n" + "=" * 60)
    print("SAMPLE RECORDS")
    print("=" * 60)
    for r in records[:n]:
        print(f"\nTask: {r['task']}")
        print(f"ID: {r['id']}")
        ctx = r["context"][:300] + "..." if len(r["context"]) > 300 else r["context"]
        print(f"Context: {ctx}")
        print(f"Question: {r['question']}")
        print(f"Answer: {r['answer']}")
        print("-" * 40)


def main():
    records = load_longbench_qa()
    save_by_task(records)
    show_sample(records)


if __name__ == "__main__":
    main()

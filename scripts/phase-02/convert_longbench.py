"""
Convert LongBench JSONL files to standard format for OP5.

Expected fields:
- id: unique identifier
- task: task/dataset name
- context: full context
- question: question
- answer: gold answer
"""
import json
from pathlib import Path
from collections import defaultdict

DATA_RAW = Path("data/raw")
DATA_OUT = Path("data/processed")
DATA_OUT.mkdir(parents=True, exist_ok=True)

# QA tasks (have question field in context)
QA_TASKS = {
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "gov_report", "multi_news", "vcsum", "qmsum", "samsum",
    "triviaqa", "musique", "passage_count", "passage_retrieval_en",
    "trec", "lsht", "lcc", "dureader"
}

# Non-QA tasks (summarization, code, etc.)
SKIP_TASKS = {
    "repobench-p",  # code completion
}

# Check a sample file for format
def check_format(file_path: Path) -> dict:
    """Check JSONL format by reading first record."""
    with file_path.open() as f:
        line = f.readline()
        return json.loads(line)


def convert_record(record: dict, dataset: str) -> dict:
    """Convert LongBench record to standard format."""
    # Extract answer (list -> string)
    answers = record.get("answers", [])
    if isinstance(answers, list) and len(answers) > 0:
        answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
    else:
        answer = str(answers)

    # Extract question from context if not separate
    # LongBench format: context contains both context and question
    context = record.get("context", "")
    input_text = record.get("input", "")

    # Try to find question marker
    if "\n\nQuestion:" in context:
        parts = context.split("\n\nQuestion:", 1)
        ctx = parts[0].replace("Context: ", "")
        question = parts[1].replace("Question: ", "").split("\n\nAnswer:")[0].strip()
    elif "\nQuestion:" in input_text:
        parts = input_text.split("\nQuestion:", 1)
        ctx = parts[0].replace("Context: ", "")
        question = parts[1].strip()
    else:
        # Try to extract from input
        ctx = context
        question = input_text.replace(context, "").strip()

    return {
        "id": f"{dataset}_{record.get('_id', '')}",
        "task": dataset,
        "dataset": "LongBench",
        "input": input_text,
        "context": ctx,
        "question": question,
        "answer": answer,
        "length": record.get("length", 0),
        "language": record.get("language", "en"),
    }


def process_files():
    """Process all JSONL files."""
    all_records = []
    task_counts = defaultdict(int)
    errors = []

    for jsonl_file in sorted(DATA_RAW.glob("*.jsonl")):
        dataset = jsonl_file.stem
        print(f"Processing: {dataset}...", end=" ")

        # Skip non-QA tasks
        if dataset in SKIP_TASKS:
            print("SKIP (non-QA)")
            continue

        records = []
        with jsonl_file.open() as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    converted = convert_record(record, dataset)
                    records.append(converted)
                    all_records.append(converted)
                    task_counts[dataset] += 1
                except Exception as e:
                    errors.append(f"{dataset}:{i}: {e}")
                    continue

        print(f"{len(records)} records")

    return all_records, task_counts, errors


def save_results(records: list[dict], task_counts: dict):
    """Save results to JSONL files."""
    # Combined
    out_path = DATA_OUT / "longbench_qa.jsonl"
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_path} ({len(records)} total)")

    # Per-task
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    for task, task_records in sorted(by_task.items()):
        task_path = DATA_OUT / f"longbench_{task}.jsonl"
        with task_path.open("w") as f:
            for r in task_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved: {task_path} ({len(task_records)} records)")


def show_samples(records: list[dict], n: int = 3):
    """Show sample records."""
    print("\n" + "=" * 70)
    print("SAMPLE RECORDS")
    print("=" * 70)

    # Show one from each task
    shown = set()
    for r in records:
        if r["task"] in shown:
            continue
        shown.add(r["task"])

        print(f"\nTask: {r['task']}")
        print(f"ID: {r['id']}")
        ctx = r["context"][:200] + "..." if len(r["context"]) > 200 else r["context"]
        print(f"Context: {ctx}")
        print(f"Question: {r['question'][:100]}...")
        print(f"Answer: {r['answer'][:100]}...")
        print("-" * 40)

        if len(shown) >= n:
            break


def main():
    print("=" * 60)
    print("CONVERT LongBench TO OP5 FORMAT")
    print("=" * 60)

    records, task_counts, errors = process_files()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(records)}")
    print("\nRecords by task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task:<25}: {count:>5}")

    if errors:
        print(f"\nErrors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e}")

    save_results(records, task_counts)
    show_samples(records)


if __name__ == "__main__":
    main()

"""
Shared utilities for Phase 1 PoC scripts.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def get_gemini_model(model_name: str | None = None):
    """Configure + return a GenerativeModel instance."""
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY chua duoc set trong .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name or os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite"))


def call_gemini(model, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> dict:
    """Single Gemini call. Returns dict with text + usage + latency."""
    t0 = time.perf_counter()
    response = model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    return {
        "text": text,
        "latency_ms": elapsed_ms,
        "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
    }


def count_tokens_approx(text: str) -> int:
    """Approximate token count (chars/4) when Gemini usage metadata is not available."""
    return max(1, len(text) // 4)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------
_ZEROSCROLLS_CACHE = Path("/tmp/zero_scrolls")

# Dataset registry — OP5 đã chuyển sang ZeroSCROLLS (2026-09-21) vì:
#   - LongBench-v2 context trung bình ~120k tokens → accuracy chỉ 25-45%
#   - ZeroSCROLLS context trung bình ~10k tokens → accuracy kỳ vọng 60-70%
#   - Multi-domain 10 tasks, có gold answer (F1/EM/Rouge đều đo được)
#   - Public, không cần HF token, đã được LongLLMLingua paper benchmark
DATASETS = {
    "zero_scrolls": "tau/zero_scrolls",
    "longbench_v2": "zai-org/LongBench-v2",  # legacy, giữ cho tương thích ngược
}

# Schema thật của ZeroSCROLLS (sau khi inspect thực tế 2026-09-21):
#
#   {
#     "id": str,                          # unique row id (vd "2hop__546800_262512")
#     "pid": str,                         # passage id (group nhiều instance của 1 passage)
#     "input": str,                       # prompt + document + separator + question + postfix
#     "output": str,                      # gold answer (string, không phải list)
#     "document_start_index": int,        # chars offset của document trong input
#     "document_end_index":   int,        # chars offset end
#     "query_start_index":    int,        # chars offset của question
#     "query_end_index":      int,
#     "truncation_seperator": str,        # "[typo: separator]" marker cho truncation
#     "inner_docs_start_indices": list    # (multi-hop: musique, space_digest, book_sum_sort)
#   }
#
# → context = input[document_start_index:document_end_index]
# → question = input[query_start_index:query_end_index]
# → answer = output
# → task name lấy từ zip filename (qasper, musique, gov_report, ...)

# 10 tasks của ZeroSCROLLS. narrative_qa BỊ LOẠI khỏi default vì context
# quá lớn (~315k chars / ~80k tokens), lớn hơn cả LongBench-v2 và làm hỏng
# mục tiêu "context ngắn để test compressor hiệu quả".
ZERO_SCROLLS_TASKS = [
    "qasper",           # QA trên NLP papers, ~23k chars
    "musique",          # Multi-hop QA, ~10k chars
    "gov_report",       # Long summary, ~49k chars
    "space_digest",     # Sentiment aggregation, ~30k chars
    "summ_screen_fd",   # TV transcript summary, ~31k chars
    "qmsum",            # Meeting summary, ~58k chars
    "squality",         # Question-focused summary, ~29k chars
    "quality",          # MCQ trên long passage, ~25k chars
    "book_sum_sort",    # Sort chapter summaries, ~39k chars
]
ZERO_SCROLLS_TASKS_ALL = ZERO_SCROLLS_TASKS + ["narrative_qa"]  # full set


def _download_zero_scrolls_task(task: str, cache_dir: Path) -> Path:
    """Download một task `.zip` từ HF và extract vào cache_dir.

    Returns path đến `<cache_dir>/<task>/<split>.jsonl`.
    Idempotent — nếu đã download thì skip.
    """
    import requests

    out_dir = cache_dir / task
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "test.jsonl"
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        return jsonl_path  # already cached

    zip_path = cache_dir / f"{task}.zip"
    url = f"https://huggingface.co/datasets/tau/zero_scrolls/resolve/main/{task}.zip"
    print(f"[zs] Downloading {task} from HF...")
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
    except Exception as exc:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Failed to download {task}: {exc}") from exc

    # Extract
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(cache_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract {task}: {exc}") from exc
    finally:
        if zip_path.exists():
            zip_path.unlink()  # save disk

    if not jsonl_path.exists():
        raise RuntimeError(f"{task}.jsonl not found after extract")
    return jsonl_path


def stream_zero_scrolls(split: str = "test", tasks: list[str] | None = None,
                        cache_dir: Path | None = None):
    """
    Yield rows from tau/zero_scrolls (multi-task: qasper, musique, gov_report, ...).

    Schema mỗi row (đã được normalized qua detect_fields):
        {
            "id": str,
            "pid": str,
            "task": str,           # thêm: tên task
            "context": str,        # extract từ input[document_start:end]
            "question": str,       # extract từ input[query_start:end]
            "answer": str,         # gold = output
            "_raw": dict           # original row (giữ để debug)
        }

    Args:
        split: 'test' (default) hoặc 'validation'
        tasks: subset của ZERO_SCROLLS_TASKS. None → default 9 tasks
               (bỏ narrative_qa quá lớn).
        cache_dir: nơi download/extract. Default `/tmp/zero_scrolls`.
    """
    if tasks is None:
        tasks = ZERO_SCROLLS_TASKS
    if cache_dir is None:
        cache_dir = _ZEROSCROLLS_CACHE

    for task in tasks:
        try:
            jsonl_path = _download_zero_scrolls_task(task, cache_dir)
        except Exception as exc:
            print(f"[zs] WARNING: skip task {task}: {exc}")
            continue
        with open(jsonl_path) as f:
            n_rows = 0
            for line in f:
                raw = json.loads(line)
                # Tag with task name
                raw["task"] = task
                yield raw
                n_rows += 1
            print(f"[zs] {task}: {n_rows} rows")


# Giữ tên cũ làm alias để các script đã viết không cần sửa nhiều
def stream_longbench_v2(split: str = "test"):
    """DEPRECATED alias cho stream_zero_scrolls — đổi sang ZeroSCROLLS 2026-09-21."""
    return stream_zero_scrolls(split=split)


def stream_squad_v2(split: str = "train"):
    """
    Yield rows from SQuAD v2.0 (public, no auth needed).
    Schema: {id, title, context, question, answers: [{text, answer_start}], is_impossible}
    """
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split=split, streaming=True)
    for row in ds:
        yield row


def detect_fields(row: dict) -> dict[str, str]:
    """
    Best-effort mapping from dataset row to canonical fields.

    Handles:
      - ZeroSCROLLS {id, pid, input, output, document_start/end_index,
                     query_start/end_index, truncation_seperator, task}
      - LongBench-v2 (legacy) {context, question, answer, choice}
      - SQuAD v2 {context, question, answers: [{text}], is_impossible}

    Trả về canonical {context, question, answer, choice, task} cho mọi schema.

    ZeroSCROLLS note: input chứa cả prompt+document+separator+question+postfix.
    Trích context từ input[document_start_index:document_end_index] và
    question từ input[query_start_index:query_end_index].
    """
    out = {"context": "", "question": "", "answer": "", "choice": "", "task": ""}

    # Task name (ZeroSCROLLS — được gắn vào row bởi stream_zero_scrolls)
    if "task" in row:
        out["task"] = str(row["task"])

    # ZeroSCROLLS: input + *_index
    if "input" in row and "document_start_index" in row:
        inp = str(row["input"])
        ds = int(row.get("document_start_index", 0))
        de = int(row.get("document_end_index", 0))
        qs = int(row.get("query_start_index", 0))
        qe = int(row.get("query_end_index", 0))
        out["context"] = inp[ds:de]
        out["question"] = inp[qs:qe]
    elif "passage" in row:
        out["context"] = str(row["passage"])
    elif "context" in row:
        out["context"] = str(row["context"])

    # Standalone question (legacy)
    if not out["question"] and "question" in row:
        out["question"] = str(row["question"])

    # Gold answer
    if "answers" in row and isinstance(row["answers"], list) and row["answers"]:
        out["answer"] = str(row["answers"][0].get("text", ""))
    elif "answer" in row:
        ans = row["answer"]
        if isinstance(ans, list) and ans:
            out["answer"] = " ".join(str(x) for x in ans)
        else:
            out["answer"] = str(ans)
    elif "output" in row:
        # ZeroSCROLLS gold
        ans = row["output"]
        if isinstance(ans, list) and ans:
            out["answer"] = " ".join(str(x) for x in ans)
        else:
            out["answer"] = str(ans)

    if "choice" in row:
        out["choice"] = str(row["choice"])
    elif "label" in row:
        out["choice"] = str(row["label"])

    return out


def build_prompt_track1(context: str) -> str:
    """Prompt for Track 1 (extraction): no question, extract all fields."""
    return (
        "Hay trich xuat cac truong thong tin quan trong tu hop dong sau. "
        "Tra ve JSON voi cac key: party_a, party_b, effective_date, contract_value, "
        "termination_clause. Neu khong co, dat null.\n\n"
        f"HOP DONG:\n{context}"
    )


def build_prompt_track2(context: str, question: str) -> str:
    """Prompt for Track 2 (RAG QA): context + question."""
    return (
        f"NGU CANH (hop dong):\n{context}\n\n"
        f"CAU HOI:\n{question}\n\n"
        "Tra loi ngan gon, dung thong tin trong ngu canh. Neu khong co trong ngu canh, "
        "tra 'Khong ro'."
    )


def judge_answer_gemini(model, question: str, gold: str, pred: str) -> dict:
    """LLM-as-judge: ask Gemini if pred matches gold."""
    judge_prompt = (
        "Ban la mot tro ly danh gia. So sanh cau tra loi cua he thong voi dap an tham chieu.\n\n"
        f"Cau hoi: {question}\n"
        f"Dap an tham chieu: {gold}\n"
        f"Cau tra loi he thong: {pred}\n\n"
        "Neu he thong tra loi dung hoac tuong duong ve nghia (khong can khop ky tu), "
        "tra ve JSON: {\"correct\": true, \"reason\": \"...\"}.\n"
        "Neu sai hoac khong dap ung, tra ve: {\"correct\": false, \"reason\": \"...\"}.\n"
        "Chi tra JSON, khong giai thich them."
    )
    try:
        result = call_gemini(model, judge_prompt, max_tokens=200, temperature=0.0)
        text = result["text"]
        try:
            parsed = json.loads(text)
            return {"judge_correct": bool(parsed.get("correct")), "judge_reason": str(parsed.get("reason", ""))[:200]}
        except Exception:
            return {"judge_correct": "true" in text.lower(), "judge_reason": text[:200]}
    except Exception as exc:
        return {"judge_correct": None, "judge_reason": f"judge error: {exc}"}

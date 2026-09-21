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
_ZEROSCROLLS_PATH = "/tmp/zero_scrolls.json"

# Dataset registry — OP5 đã chuyển sang ZeroSCROLLS (2026-09-21) vì:
#   - LongBench-v2 context trung bình ~120k tokens → accuracy chỉ 25-45%
#   - ZeroSCROLLS context trung bình ~10k tokens → accuracy kỳ vọng 60-70%
#   - Multi-domain 10 tasks, có gold answer (F1/EM/Rouge đều đo được)
#   - Public, không cần HF token, đã được LongLLMLingua paper benchmark
DATASETS = {
    "zero_scrolls": "tau/zero_scrolls",
    "longbench_v2": "zai-org/LongBench-v2",  # legacy, giữ cho tương thích ngược
}


def stream_zero_scrolls(split: str = "test"):
    """
    Yield rows from tau/zero_scrolls.

    ZeroSCROLLS gồm 10 tasks: NarrativeQA, Qasper, QuALITY, SpaceDigest, MuSiQue,
    GovReport, SummScreenFD, QMSum, BookSumSort, TriviaQA. Avg context ~10k tokens.

    Tries to load from pre-downloaded file first (/tmp/zero_scrolls.json),
    then falls back to HuggingFace streaming.

    Schema (per row): {id, pid, passage, question, answer, options, task, ...}
      - "passage"      : str — long context (~10k tokens)
      - "question"     : str — câu hỏi
      - "answer"       : str — gold answer (string ngắn cho QA, list câu cho summary)
      - "task"         : str — task name (narrative_qa, qasper, ...)
    """
    p = Path(_ZEROSCROLLS_PATH)
    if p.exists() and p.stat().st_size > 500_000:
        try:
            with p.open() as f:
                data = json.load(f)
            print(f"[dataset] Loaded {len(data)} rows from {p}")
            for row in data:
                yield row
            return
        except Exception as exc:
            print(f"[dataset] Pre-downloaded file failed ({exc}), trying HF...")

    from datasets import load_dataset
    return load_dataset("tau/zero_scrolls", split=split, streaming=True)


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
      - ZeroSCROLLS {id, passage, question, answer, task, ...}
      - LongBench-v2 (legacy) {context, question, answer, choice}
      - SQuAD v2 {context, question, answers: [{text}], is_impossible}

    Trả về canonical {context, question, answer, choice, task} cho mọi schema.
    """
    out = {"context": "", "question": "", "answer": "", "choice": "", "task": ""}

    # Context — ZeroSCROLLS dùng "passage", các dataset khác dùng "context"/"input"
    if "passage" in row:
        out["context"] = str(row["passage"])
    elif "context" in row:
        out["context"] = str(row["context"])
    elif "input" in row:
        out["context"] = str(row["input"])

    if "question" in row:
        out["question"] = str(row["question"])

    # Task name (ZeroSCROLLS có field này)
    if "task" in row:
        out["task"] = str(row["task"])

    # Gold answer
    if "answers" in row and isinstance(row["answers"], list) and row["answers"]:
        out["answer"] = str(row["answers"][0].get("text", ""))
    elif "answer" in row:
        ans = row["answer"]
        # ZeroSCROLLS summary tasks có answer là list các câu
        if isinstance(ans, list) and ans:
            out["answer"] = " ".join(str(x) for x in ans)
        else:
            out["answer"] = str(ans)
    elif "output" in row:
        out["answer"] = str(row["output"])

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

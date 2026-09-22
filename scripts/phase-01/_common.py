"""
Shared utilities for Phase 1 PoC scripts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
load_dotenv(REPO_ROOT / ".env")
from _prompts import (
    format_eval_prompt as _format_eval_prompt,  # canonical Track 2 prompt
    _format_gemini_judge_prompt,  # canonical Gemini-direct judge prompt
)


def get_gemini_model(model_name: str | None = None):
    """Configure + return a GenerativeModel instance."""
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
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

# Dataset registry — OP5 switched to ZeroSCROLLS on 2026-09-21 because:
#   - LongBench-v2 context averages ~120k tokens → accuracy only 25-45%
#   - ZeroSCROLLS context averages ~10k tokens → expected accuracy 60-70%
#   - Multi-domain 10 tasks, gold answers available (F1/EM/Rouge all measurable)
#   - Public, no HF token needed, benchmarked by LongLLMLingua paper
DATASETS = {
    "zero_scrolls": "tau/zero_scrolls",
    "longbench_v2": "zai-org/LongBench-v2",  # legacy, kept for backward compat
}

# Actual schema of ZeroSCROLLS (verified 2026-09-21):
#
#   {
#     "id": str,                          # unique row id (e.g. "2hop__546800_262512")
#     "pid": str,                         # passage id (groups multiple instances)
#     "input": str,                       # prompt + document + separator + question + postfix
#     "output": str,                      # gold answer (string, not list)
#     "document_start_index": int,        # char offset of document in input
#     "document_end_index":   int,        # char offset end
#     "query_start_index":    int,        # char offset of question
#     "query_end_index":      int,
#     "truncation_seperator": str,        # marker for truncation
#     "inner_docs_start_indices": list    # (multi-hop: musique, space_digest, book_sum_sort)
#   }
#
# → context  = input[document_start_index:document_end_index]
# → question = input[query_start_index:query_end_index]
# → answer   = output
# → task name comes from the zip filename (qasper, musique, gov_report, ...)

# 10 tasks of ZeroSCROLLS. narrative_qa is EXCLUDED from default because its
# context is too large (~315k chars / ~80k tokens), larger than LongBench-v2
# itself, defeating the goal of "short context to test compressor effectiveness".
ZERO_SCROLLS_TASKS = [
    "qasper",           # QA on NLP papers, ~23k chars
    "musique",          # Multi-hop QA, ~10k chars
    "gov_report",       # Long summary, ~49k chars
    "space_digest",     # Sentiment aggregation, ~30k chars
    "summ_screen_fd",   # TV transcript summary, ~31k chars
    "qmsum",            # Meeting summary, ~58k chars
    "squality",         # Question-focused summary, ~29k chars
    "quality",          # MCQ on long passage, ~25k chars
    "book_sum_sort",    # Sort chapter summaries, ~39k chars
]
ZERO_SCROLLS_TASKS_ALL = ZERO_SCROLLS_TASKS + ["narrative_qa"]  # full set


def _download_zero_scrolls_task(task: str, cache_dir: Path) -> Path:
    """Download one task `.zip` from HF and extract into cache_dir.

    Returns path to `<cache_dir>/<task>/<split>.jsonl`.
    Idempotent — skip if already downloaded.
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

    Per-row schema (normalized via detect_fields):
        {
            "id": str,
            "pid": str,
            "task": str,           # added: task name
            "context": str,        # extracted from input[document_start:end]
            "question": str,       # extracted from input[query_start:end]
            "answer": str,         # gold = output
            "_raw": dict           # original row (kept for debugging)
        }

    Args:
        split: 'test' (default) or 'validation'
        tasks: subset of ZERO_SCROLLS_TASKS. None → default 9 tasks
               (excluding narrative_qa which is too large).
        cache_dir: download/extract location. Default `/tmp/zero_scrolls`.
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


# Keep old name as alias so existing scripts don't need much rewriting
def stream_longbench_v2(split: str = "test"):
    """DEPRECATED alias for stream_zero_scrolls — switched to ZeroSCROLLS 2026-09-21."""
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

    Returns canonical {context, question, answer, choice, task} for any schema.

    ZeroSCROLLS note: input contains prompt + document + separator + question
    + postfix. Extract context from input[document_start_index:document_end_index]
    and question from input[query_start_index:query_end_index].
    """
    out = {"context": "", "question": "", "answer": "", "choice": "", "task": ""}

    # Task name (ZeroSCROLLS — attached to row by stream_zero_scrolls)
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
        "Extract the key fields from the contract below. "
        "Return JSON with keys: party_a, party_b, effective_date, contract_value, "
        "termination_clause. If a field is missing, set it to null.\n\n"
        f"CONTRACT:\n{context}"
    )


def build_prompt_track2(context: str, question: str) -> str:
    """Prompt for Track 2 (RAG QA): context + question.

    Delegates to canonical prompt in `scripts/_prompts.py` — do NOT edit
    the prompt string here. Improvements come from compression / routing,
    not prompt tweaks.
    """
    return _format_eval_prompt(context, question)


def judge_answer_gemini(model, question: str, gold: str, pred: str) -> dict:
    """LLM-as-judge: ask Gemini if pred matches gold.

    Uses canonical judge prompt from scripts/_prompts.py (kept in sync with
    src/op5/llm/judge_prompt.py). Output schema simplified to {correct, reason}
    for backward compatibility with older scripts.
    """
    judge_prompt = _format_gemini_judge_prompt(question, gold, pred)
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


def judge_answer_openrouter(
    question: str = "",
    gold: str = "",
    pred: str = "",
    context: str = "",
    log_path: str | Path | None = None,
) -> dict:
    """
    DEPRECATED — use judge_answer_llm_judge() instead.

    LLM-as-judge via OpenRouter with automatic fallback chain.

    Primary model:  nvidia/nemotron-3-ultra-550b-a55b:free
    Fallback models: deepseek/deepseek-chat-v3:free → openrouter/free

    Auto-fallback triggers: HTTP 429, 503, timeout, parse failure.
    Results include model_used for reproducibility.

    Returns dict with keys: verdict, correct, reason, confidence, model_used, raw, latency_ms

    Env vars:
        OPENROUTER_API_KEY — required
        OP5_JUDGE_TEMPERATURE — default 0.1
    """
    return judge_answer_llm_judge(
        question=question,
        gold=gold,
        pred=pred,
        context=context,
        profile="openrouter",
        log_path=log_path,
    )


def judge_answer_llm_judge(
    question: str = "",
    gold: str = "",
    pred: str = "",
    context: str = "",
    profile: str | None = None,
    log_path: str | Path | None = None,
) -> dict:
    """
    LLM-as-judge via the unified LLMJudge with profile routing.

    Chain (default): NIM → OpenRouter → Gemini
    Single profile:  specify profile="nim" (fastest) or profile="openrouter"

    Args:
        question: evaluation question
        gold: ground truth answer
        pred: model prediction
        context: optional supporting context
        profile: single profile name (e.g. "nim", "openrouter", "gemini").
                 None → auto-chain from config.yaml
        log_path: optional ops log path

    Returns dict with keys:
        verdict, correct, reason, confidence, model_used, raw, latency_ms

    Env vars (loaded from .env):
        NVIDIA_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY
    """
    from op5.llm import LLMJudge
    judge = LLMJudge(profile=profile, log_path=Path(log_path) if log_path else None)
    result = judge.judge(
        question=question,
        gold=gold,
        pred=pred,
        context=context,
    )
    return result.to_dict()

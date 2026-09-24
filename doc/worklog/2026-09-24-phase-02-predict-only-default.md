# Worklog — 2026-09-24 default predict-only in run scripts

## Context

User muốn tách biệt rõ ràng giữa **phase predict** và **phase judge**.

Trước đây cả `run_baseline.py` và `run_compressed.py` đều chạy LLM-as-judge ngay sau khi gọi Gemini (default behavior), dù user muốn giữ hai phase riêng biệt.

User confirmed: option B — đổi default `--no-judge` (tức predict-only), vẫn giữ `--judge` flag opt-in cho debug.

## Changes

### `scripts/phase-02/run_baseline.py`

| Change | Detail |
|---|---|
| Docstring | Thêm giải thích predict-only default + hướng dẫn dùng `judge_jsonl.py` |
| `--no-judge` flag | **Xoá** (không còn cần) |
| `--judge` flag (new, positive) | Opt-in chạy judge inline (debug only) |
| Default behavior | Predict-only: `[1/2]` compress+Gemini → `[2/2]` SKIPPED with instructions |
| Resume logic | `done_ids` skip condition: predict-only compatible |

### `scripts/phase-02/run_compressed.py`

Same changes as above (same refactor).

## Usage workflow

```bash
# Step 1: Predict (default, no judge)
conda activate vsf
python scripts/phase-02/run_baseline.py
python scripts/phase-02/run_compressed.py --rate 0.4
python scripts/phase-02/run_compressed.py --rate 0.5
python scripts/phase-02/run_compressed.py --rate 0.6
python scripts/phase-02/run_compressed.py --rate 0.7

# Step 2: Judge (separate phase)
conda activate vsf
python scripts/phase-02/judge_jsonl.py \
    --in results/phase-02-run-baseline.jsonl \
    --out results/phase-02-run-baseline-judged.jsonl \
    --profile deepseek

python scripts/phase-02/judge_jsonl.py \
    --in results/phase-02-run-rate40.jsonl \
    --out results/phase-02-run-rate40-judged.jsonl \
    --profile deepseek

# Debug: run judge inline (not recommended for normal workflow)
python scripts/phase-02/run_baseline.py --judge
```

## Notes

- `judge_jsonl.py` đã tồn tại, dùng `op5.llm.LLMJudge` với profile support.
- JSONL output fields không đổi: `judge_correct`, `judge_reason`, `judge_model`, `judge_latency_ms` vẫn được ghi (set null khi judge skipped).
- Summary JSON: `judge_model` = null khi predict-only.

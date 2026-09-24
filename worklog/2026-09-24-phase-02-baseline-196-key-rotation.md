# Worklog — 2026-09-24 Baseline 196 + key-rotation

## Context

User yêu cầu (verbatim): *"chạy baseline với bộ 200 câu hỏi. API Key dùng
Google API Key trong .env (có 4 Key API tất cả, dùng lẫn nhau khi retry
quá hạn)"*.

State trước session:
- `run_baseline.py` hardcode `longbench_20_stratified.jsonl` (20 cases) — sai
  với yêu cầu 200.
- `.env` có 4 GOOGLE_API_KEY nhưng 3 đang comment + 1 active (đã hết hạn 401).
- `call_gemini()` inline chỉ support single-key, không có rotation.

## 1. Status 4 keys trong .env

| Key (position) | Status |
|---|---|
| Line 6 (active) | ❌ 401 — invalid |
| Line 3 (comment) | ❌ 401 — invalid |
| Line 4 (comment) | ✅ 200 — alive (k2 trong rotation) |
| Line 5 (comment) | ✅ 200 — alive (k1 trong rotation) |

Tức là 1 active chết + 1 commented chết + 2 commented sống = **2 keys working**.

## 2. Quyết định về env handling

AGENTS.md cấm commit `.env` keys. Setup 4 keys qua env inline:

```bash
export GOOGLE_API_KEYS="AQ.Ab8RN6IiV-ALoB1XH4f...,AQ.Ab8RN6KktVLYPb6..."
python scripts/phase-02/run_baseline.py --out-tag baseline_n196 --sleep 3
```

Không persist keys xuống file. Nếu muốn persist an toàn, cần `gitignore .env.local`
chứa 4 keys và `.env.example` template (chưa làm trong session này).

## 3. Codebase changes (theo AGENTS.md §"Codebase-first rule")

### `src/op5/llm/gemini_client.py` (new, 195 lines)

Module-level helper:
- Class `GeminiClient(timeout_s, max_retries_per_key, global_retries, sleep_between_keys_s)`
- Key pool từ `GOOGLE_API_KEYS` (comma-separated, priority) hoặc `GOOGLE_API_KEY`
  (single fallback).
- Rotation strategy: round-robin success-path; on 429/503/Timeout → rotate
  immediately + retry (không backoff dài).
- Return dict gồm `status`, `text`, `latency_ms`, `input_tokens`,
  `output_tokens`, `key_id` (k1/k2/...), `attempts`.
- Stats per-key success/error counters qua `stats_summary()`.

### `src/op5/llm/__init__.py` (edited)

Re-export `GeminiClient` cùng với các judge class hiện có.

### `scripts/phase-02/run_baseline.py` (refactored)

- **Sample path:** `longbench_20_stratified.jsonl` → `longbench_200_stratified.jsonl`
  (196 rows, như worklog dataset chỉ định).
- **Inline `call_gemini()`:** xóa, thay bằng `gemini.generate(...)` qua
  `GeminiClient`.
- **Resume bug fix:** điều kiện skip cũ là
  `(args.run_judge or r.get("judge_correct") is not None)` — sai trong
  predict-only mode. Đổi thành 2 nhánh:
  - predict-only: skip khi `status=="ok" and pred truthy`
  - judge mode: skip khi `status=="ok" and pred truthy and judge_correct is not None`
  Bug tương tự có trong `run_compressed.py:350-352` (worklog
  2026-09-24-phase-02c-dedup-latency-eval.md đã flag, CHƯA fix).
- **New record fields:** `gemini_key_id` (k1/k2), `gemini_attempts` (số HTTP
  attempts).
- **Summary:** thêm `gemini_key_pool` field để thấy per-key usage.

## 4. Kết quả chạy baseline

### Output

- File: `results/phase-02-run-baseline_n196.jsonl` (196 records, OK 100%)
- File: `results/phase-02-run-baseline_n196-summary.json`
- Wallclock: **14.5 phút** (07:27:04 → 07:41:36 UTC+7)
- Resume hỗ trợ: nếu interrupt, lần sau skip case_id đã có pred.

### Latency (N=196, không compress)

```
gemini_s:
  mean   : 1.4s
  median : 1.3s
  p95    : 1.9s
  max    : 3.1s
```

Gemini flash-lite ổn định ~1.3s/call cho 196 cases, **không thấy
back-end overload** như session trước (worklog
2026-09-23-phase-02-sweep-retry-gemini-outage.md). Có thể nhờ key rotation
giảm tải cho mỗi key.

### Key usage pool

```
k1: 98 ok / 1 err
k2: 98 ok / 0 err
```

Round-robin chia đều 50/50. Có 1 err transient ở k1 (đã rotate sang k2).

### Tokens

```
input_median: 8824 tok
input_p95   : 31,815 tok
input_max   : 68,618 tok (narrativeqa)
output_med  : 31 tok
```

### Per-task (28 cases mỗi task)

| task | median input | max input |
|---|---|---|
| triviaqa | 627 | 1,815 |
| qasper | 4,869 | 19,973 |
| 2wikimqa | 6,391 | 16,655 |
| multifieldqa_en | 7,631 | 15,293 |
| hotpotqa | 14,981 | 17,120 |
| musique | 16,649 | 17,494 |
| narrativeqa | 31,757 | 68,618 |

Baseline 196 input_tokens ≈ 2× so với rate40 trong sweep (3709 median).
Compression thực sự giảm được ~40% tokens cho case ngắn, ~50%+ cho case
dài. Cần judge accuracy để biết trade-off chất lượng.

## 5. Open follow-ups

- [ ] **Chạy judge** để có accuracy baseline. `judge_jsonl.py` đã có
  sẵn, gọi DeepSeek-V3 làm judge (~$0.27/M input). 196 cases × ~9k tok =
  ~1.8M input, cost ~$0.49.
- [ ] **Fix resume bug tương tự trong `run_compressed.py:350-352`** — chưa
  làm trong session này.
- [ ] **Optional:** chuyển key rotation config vào `.env.local` + gitignore
  + viết `.env.example` template để không phải inline `export` mỗi lần.
- [ ] **INDEX.md** chưa cập nhật status (phase-02c vẫn active cho tới khi
  accuracy pass).

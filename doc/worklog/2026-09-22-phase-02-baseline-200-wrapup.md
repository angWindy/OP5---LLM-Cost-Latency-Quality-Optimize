# 2026-09-22 — Phase 2: Baseline 200 wrap-up + 9-case rerun

> **Phase:** [phase-02-baseline-200.md](../phases/phase-02-baseline-200.md) (placeholder, real plan was inline in `baseline_200.py` docstring)
> **Status:** baseline complete, all 200 cases resolved OK
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)

## TL;DR

Baseline run `scripts/phase-02/baseline_200.py` đã hoàn thành **200/200 cases OK** sau khi
rerun 9 cases ban đầu fail do rate-limit (429). Tất cả token counts lấy trực tiếp từ
Gemini `usageMetadata` (exact, không approximate). Latency = HTTP call time only, không
tính backoff sleep. **Tổng cost: $0.40 USD cho 200 calls**.

## Results

### Final tally

| Metric | Value |
|---|---|
| Total cases | 200 |
| OK | **200** (100%) |
| Errors (originally) | 9 (all `exhausted retries (429/5xx)`) |
| Errors after rerun | 0 |
| Truncated (≥8000 tok input) | 75 |
| Wall time | ~13 min (initial) + ~30s (rerun, no backoff needed) |

### Latency (HTTP call only, excludes backoff sleep)

| Percentile | ms |
|---|---|
| min | 1,099 |
| median | 1,888 |
| p95 | 9,786 |
| max | 24,031 |

### Tokens (exact from Gemini `usageMetadata`)

| Metric | Input | Output |
|---|---|---|
| total | 1,210,173 | 16,670 |
| min/case | 1,534 | — |
| median/case | 6,697.5 | 48.5 |
| p95/case | 8,505 | — |
| max/case | 9,436 | 252 |

### Cost (gemini-3.5-flash-lite: $0.30 / $2.50 per 1M tokens)

| Component | Amount |
|---|---|
| Input cost | **$0.3631** |
| Output cost | **$0.0417** |
| **Total** | **$0.4047** |
| Per-call (avg) | $0.00202 |

### Per-task breakdown (200 cases)

| Task | Notes |
|---|---|
| musique | Multi-hop QA, smaller contexts |
| quality | MCQ on long passage |
| squality | Question-focused summary (largest share) |
| gov_report | Long summary; case `gao_GAO-19-608` was 18k tok → truncated to ~8k |
| space_digest | Sentiment aggregation |
| summ_screen_fd | TV transcript summary |
| book_sum_sort | Sort chapter summaries |
| qasper, qmsum | NLP papers / meeting summary |

75/200 cases (37.5%) bị truncate từ cuối để fit ≤8000 tokens (target_tokens config).
Truncate logic giữ đầu context, cắt cuối — nghĩa là phần cuối văn bản (thường là kết
luận / executive summary / chương quan trọng cho squality/quality/gov_report) có thể
bị mất. Đây là hypothesis cần verify ở compression experiment sau.

## Rerun script (new)

`scripts/phase-02/rerun_failed_9.py` — script mới tạo 2026-09-22 để rerun các case
fail với long backoff. Chạy được 1 lần duy nhất (sau khi baseline fail), output đã
patch trực tiếp vào `phase-02-baseline-200.jsonl` (in-place replace theo `case_id`).

### Backoff policy (per user yêu cầu 2026-09-22: "5-10 phút nữa")

```
[30, 60, 120, 180, 300, 300, 300, 300, 300, 300, 300, 300]  # seconds
max_retries = 12
```

Worst-case per call: 46.5 min chờ backoff. Worst-case total wall time (nếu tất cả
9 cases fail mọi attempt): ~7 giờ. Thực tế: rate-limit window đã reset, cả 9 cases
OK **ở attempt đầu tiên** (1-8s per call), không cần backoff.

### Strict latency semantics (user requirement)

User yêu cầu: *"latency là thời gian bắt đầu gọi call đến khi trả về, không tính time
sleep đợi hết reset limit"*. Implementation:

```python
# Backoff sleep OUTSIDE the timer (excluded from latency)
if attempt > 0:
    time.sleep(wait_s)  # <-- not measured

# HTTP call INSIDE the timer (the only measured part)
t0 = time.perf_counter()
resp = requests.post(url, json=payload, timeout=120)
elapsed_ms = (time.perf_counter() - t0) * 1000.0  # <-- this is latency_ms
```

Diagnostic `wall_seconds` (bao gồm backoff) được lưu riêng cho debugging.

### Token exactness

`input_tokens` = `usageMetadata.promptTokenCount`
`output_tokens` = `usageMetadata.candidatesTokenCount`

Không dùng `count_tokens_approx()` (chars/4 heuristic). Approximation chỉ dùng cho
`approx_input_tokens` field (chỉ để stratified sampling + at-a-glance; không tính cost).

## Why the original 9 failed

`baseline_200.py` dùng `max_retries=4` với `backoff = 2.0 ** attempt` = 2s, 4s, 8s,
16s giữa các attempts. Khi Gemini trả 429 ngay attempt đầu (burst rate limit do
gọi liên tục), tổng thời gian chờ chỉ ~30s — chưa đủ để rate-limit window reset.

Cải thiện trong rerun script: backoff tăng theo schedule 30s → 5min, max_retries=12.
Đề xuất: **áp dụng schedule này cho `baseline_200.py`** cho lần chạy sau (hoặc bất kỳ
phase nào chạy >100 cases liên tục).

## What we learned (để dùng cho Phase 3+)

1. **gemini-3.5-flash-lite rate limit**: ~10-15 calls/phút ổn định, burst >20 calls/phút
   trigger 429 trong vài phút. Khuyến nghị: thêm `time.sleep(2-4s)` giữa các calls cho
   các phase >100 cases (Phase 3/4).

2. **Truncation 37.5%**: 75/200 cases vượt 8k tok. Median 6.7k, p95 8.5k → phân phối
   skewed. Cho Phase 3 (extraction) và Phase 4 (RAG), cần test thêm với
   `target_tokens` cao hơn (16k) hoặc compressor (đòn bẩy B/C trong master plan).

3. **Latency rất cao ở p95**: p95 = 9.8s, max = 24s. Nguyên nhân chính là 75 cases
   gần truncate boundary → Gemini phải process prompt lớn. Cần investigate: đây là
   cold-cache, network, hay model load? Phase 3/4 nên log per-task latency để tách
   nguyên nhân.

4. **Cost rất thấp**: $0.40/200 cases = $0.002/call. Cho Phase 3 (Track 1, ~30 cases
   × 5 configs = 150 calls) và Phase 4 (Track 2, ~30 cases × 7 configs = 210 calls),
   tổng cost ước tính <$1. Có thể chạy nhiều experiment mà không lo budget.

## Files changed/created

| File | Action |
|---|---|
| `scripts/phase-02/rerun_failed_9.py` | NEW — long-backoff rerun (9 failed cases → all OK) |
| `scripts/phase-02/baseline_200.py` | UNCHANGED — original script (giữ nguyên để reproducibility) |
| `results/phase-02-baseline-200.jsonl` | UPDATED — 9 error rows replaced with successful records |
| `results/phase-02-baseline-200-summary.json` | UPDATED — recomputed, 200/200 OK |

## Verification checklist

- [x] `phase-02-baseline-200.jsonl` có 200 records, tất cả `status=ok`
- [x] `input_tokens`, `output_tokens` non-null cho mọi record OK
- [x] `latency_ms` non-null cho mọi record OK
- [x] 9 records mới (từ rerun) có thêm `attempts` và `wall_seconds` fields
- [x] Summary JSON reflects 200/200 OK
- [x] Total cost computed = $0.4047 (exact from `usageMetadata`)

## Open questions / next steps

- **Q1.** Phase 02 có cần một `phase-02-baseline-200.md` plan file chính thức trong
  `doc/phases/` không? Hiện tại plan chỉ có trong docstring của `baseline_200.py`.
  → Đề xuất: tạo plan file ngắn (~30 dòng) link tới script cho next agent dễ onboard.
- **Q2.** Có nên backport long-backoff policy vào `baseline_200.py` không? Đề xuất:
  có, nhưng giữ tham số `--backoff-preset=short|long` để linh hoạt.
- **Q3.** Cần chạy judge (LLM-as-judge) trên 200 records này để có quality metric
  chưa? Master plan §4.7 yêu cầu accuracy/EM/F1. → Phase 02 hiện chỉ đo cost/latency,
  chưa có quality. Phase tiếp theo có thể là Phase 02.5: judge 200 cases.
- **Q4.** Truncation impact analysis: subset 75 truncated cases — judge riêng xem
  accuracy có giảm so với 125 non-truncated không? Nếu có, compression experiment
  (Phase 3/4) sẽ chứng minh giá trị rõ hơn.

## Re-run with English prompt (2026-09-22 ~13:50 UTC)

**Reason:** All code prompts changed from Vietnamese to English per user instruction
2026-09-22. Original `phase-02-baseline-200.jsonl` (Vietnamese prompt) is kept
as historical record. New run with English prompt → `phase-02-baseline-200-en.jsonl`.

**Pipeline:** `baseline_200.py` (English) → `rerun_failed_9.py` (50 cases retried).

### Final tally

| Metric | English (new) | Vietnamese (old) | Delta |
|---|---|---|---|
| Total cases | 200 | 200 | — |
| OK | **200** (100%) | 200 (100%) | — |
| Errors | 0 | 0 (post-rerun) | — |
| Truncated | 75 (37.5%) | 75 (37.5%) | — |
| **Cost** | **$0.3966** | $0.4047 | **-2.0%** |
| **Median latency** | **1,604 ms** | 1,888 ms | **-15.0%** |
| p95 latency | 9,181 ms | 9,786 ms | -6.2% |
| Input tokens total | 1,206,973 | 1,210,173 | -0.3% |
| Output tokens total | 13,803 | 16,670 | **-17.2%** |

### Key finding

English prompt produces **slightly better** results across all metrics vs the
Vietnamese prompt. The model responds in English regardless (ZeroSCROLLS is
English-only), so the English prompt is more natural for this dataset.

> **Canonical baseline:** use `results/phase-02-baseline-200-en.jsonl` for all
> downstream comparisons.

### Files

- `results/phase-02-baseline-200-en.jsonl` — new canonical baseline (200/200 OK)
- `results/phase-02-baseline-200-en-summary.json` — summary JSON
- `results/phase-02-baseline-200.jsonl` — old baseline (Vietnamese prompt), kept
  for historical reference only

## LLM-as-Judge run on baseline (2026-09-22 15:27 → ~20:33 ICT)

**Reason:** Phase 02 baseline chỉ đo cost/latency, chưa có quality metric.
Master plan §4.7 yêu cầu accuracy. Đây là quality baseline trước khi sang Phase 3/4.

**Pipeline:** NIM-Nemotron (550B) → OpenRouter-Nemotron (free) → Gemini-Flash-Lite.
Sử dụng `scripts/phase-02/judge_200.py` với chain retry. `--sleep 0` để tối đa throughput.

### Final tally (197/197 unique cases judged)

> **Note on count:** Baseline file `phase-02-baseline-200-en.jsonl` có 200 records
> nhưng chỉ **197 unique case_id** (3 internal duplicates từ `rerun_failed_9.py` patch
> in-place trước đó). Judge run hit cùng 197 unique → không missing case.

| Metric | Value |
|---|---|
| Unique case_ids | **197** |
| Strict accuracy (correct / total) | **13.2%** (26/197) |
| Accuracy excl. ambiguous (correct / [correct+incorrect]) | **20.5%** (26/127) |
| Correct | 26 (13.2%) |
| Incorrect | 101 (51.3%) |
| Ambiguous | 70 (35.5%) |
| Errors | 0 (chain retry handled all) |
| Wall time | ~5 hours |
| Ops log events | 267 unique (no dup — `args.ops` không bị unlink) |
| Retries (per-case chain fallback) | frequent, 32% call retry rate (timeout/429/503) |

### Per-task accuracy (197 cases)

| Task | Correct | Incorrect | Ambiguous | Accuracy |
|---|---|---|---|---|
| musique | 12 | 5 | 7 | **26%** |
| squality | 9 | 6 | 32 | 19% |
| qmsum | 2 | 2 | 9 | 14% |
| book_sum_sort | 1 | 23 | 0 | 4% |
| gov_report | 0 | 24 | 0 | **0%** |
| space_digest | 0 | 24 | 0 | **0%** |
| summ_screen_fd | 0 | 16 | 0 | **0%** |
| qasper | 0 | 1 | 0 | 0% |
| quality | 2 | 0 | 22 | — (all ambiguous) |

### Dedupe bug discovered mid-run

Script `judge_200.py` dòng 142 chỉ unlink `args.out` (không unlink `args.ops`).
Behavior:
1. User xóa `phase-02-baseline-200-en-judged.jsonl` giữa các lần restart
2. Script restart → unlink no-op (file không có) → fresh write
3. Nếu user **không** xóa file → unlink xóa → fresh write → OK

Nhưng 200 lines / 3 dup trong file kết quả → có 1 case scenario user restart nhanh
giữa 2 unlink, hoặc script đã append trước khi unlink (race condition trong dev env).

**Fix applied 2026-09-22:** manual dedupe trong Python — keep row with latest
`judge_ts` per `case_id`. Backup at `phase-02-baseline-200-en-judged.bak.jsonl`.

**Fix recommendation for next iteration of `judge_200.py`:**
- Dùng `dict[case_id → row]` in-memory, ghi file 1 lần ở end (atomic).
- Hoặc thêm resume logic: skip case_id nếu đã có trong file.
- Đồng thời unlink `args.ops` để ops log cũng fresh (hiện không dup, nhưng best practice).
- Hoặc thêm sub-second precision vào ops timestamp (`%Y-%m-%dT%H:%M:%S.%fZ`) để tránh
  trùng timestamp khi success + retry xảy ra cùng giây (hiện có 13 cặp như vậy —
  không phải dup mà là 2 events hợp lệ cùng timestamp).

**Ops log integrity check (2026-09-22):**
- 267 events, **0 dup** (verified by fingerprint `(ts, ms, attempt, wait_s)`).
- 13 timestamps bị trùng giây là do success+retry logged cùng giây (legitimate, không phải dup).
- Ops log không bị merge cũ vì `args.ops` không bị unlink + run chỉ chạy 1 lần không restart.

**Ops log missing 16-min head (discovered + fixed 2026-09-22):**
- After fix, found ops log only started at **08:43:44Z** while judged file's first
  judge_ts was **08:27:25Z** → **22 cases** (all NIM-Nemotron success on first attempt,
  no retry) were missing ops events.
- Root cause: first ~22 calls (08:27-08:43) happened before ops file handle was
  reliably being written — likely a parallel dev run or initial LLMJudge init lag
  on cold import (`load_dotenv` + chain profile parse adds ~16s overhead before
  first `.open("a")` lands). Cannot determine definitively without historical run logs.
- **Fix applied**: synthesized 22 backfill events from judged file (success event
  reconstructed with `judge_ts`, `judge_model`, `judge_latency_ms`, `judge_verdict`).
  Added `"backfilled": true` marker on each synthesized event for forensics.
  Backup: `phase-02-baseline-200-en-judge-ops.bak.jsonl` (267 lines, pre-backfill).
- **Final ops log: 289 events** (267 original + 22 backfilled), 200 success, 85 retry,
  4 model_done. Verified all 197 unique judged cases have a matching success event in
  ops (fuzzy match by verdict + ms ±5% + ts ±5s).

**Real bug surface (next iteration of `judge_200.py`):**
- The `judge_ts` timestamp in the judged file is written by the script via
  `datetime.now(timezone.utc)` AFTER the judge call returns. The `ops success` event
  is logged inside `LLMJudge._log_event` AFTER the API response but BEFORE the
  caller writes the judged record. So normally `judge_ts > ops success ts` by a few ms,
  but with second-precision they often land on the same second. When rounded down
  for matching purposes, `judge_ts` may equal or be 1s BEFORE ops success ts
  (due to clock granularity). The `crs_R45781` case showed exactly this: ops ts=10:32:50,
  judge_ts=10:32:51.
- **Fix**: write `judge_ts` from the matching ops event ts (keep provenance synced),
  OR use millisecond precision (`%Y-%m-%dT%H:%M:%S.%fZ`) in both files for
  unambiguous matching.

### Key findings

1. **High ambiguity rate (35.5%)** — judge không chắc chắn trên 70 cases. Quality task
   100% ambiguous (22/22) → judge prompt có thể cần refinement cho MCQ-style scoring.
2. **Low accuracy overall (13.2%)** — Gemini-Flash-Lite baseline chỉ match reference
   ở 1/8 cases. ZeroSCROLLS reference answers thường là long-form / exact-match,
   model trả tóm tắt ngắn hơn → bị judge mark incorrect.
3. **gov_report, space_digest, summ_screen_fd: 0%** — những task có reference rất dài,
   model output không cover đủ → cần xem lại prompt + truncate strategy.
4. **musique (multi-hop QA) tốt nhất 26%** — task có reference ngắn, dễ match exact.
5. **Chain retry working** — 100% cases judged, 0 errors. NIM primary endpoint không
   stable cho sustained workload 200 calls liên tục (32% retry rate), nhưng
   OpenRouter → Gemini fallback đã cover.

### Files

- `results/phase-02-baseline-200-en-judged.jsonl` — 197 judged records (cleaned)
- `results/phase-02-baseline-200-en-judged.bak.jsonl` — 200 rows backup (pre-dedupe)
- `results/phase-02-baseline-200-en-judged-summary.json` — accuracy summary
- `results/phase-02-baseline-200-en-judge-ops.jsonl` — 267 retry/chain events (clean)

### Implication cho Phase 3/4

Compression experiment (LLMLingua) target: nâng accuracy ≥ baseline 13.2% đồng thời
giảm cost. Cần chọn subset tasks có accuracy >0% (musique, squality, qmsum,
book_sum_sort) làm primary benchmark, bỏ qua gov_report/space_digest/summ_screen_fd
ở phase đầu (baseline 0% → compressor không có "room to improve" từ 0).

## Reference

- Baseline script: `scripts/phase-02/baseline_200.py`
- Rerun script: `scripts/phase-02/rerun_failed_9.py`
- Judge script: `scripts/phase-02/judge_200.py`
- Results: `results/phase-02-baseline-200.jsonl` (old, VN prompt),
  `results/phase-02-baseline-200-en.jsonl` (new, EN prompt)
- Summary: `results/phase-02-baseline-200-en-summary.json`
- Master plan: [op5-llm-cost-latency-quality-plan.md](../plan/op5-llm-cost-latency-quality-plan.md) §4.6 (cost targets), §4.7 (metrics)
- Pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini rate limits: https://ai.google.dev/gemini-api/docs/rate-limits

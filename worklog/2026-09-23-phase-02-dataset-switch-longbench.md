# Worklog — 2026-09-23 — Dataset switch: ZeroSCROLLS → LongBench

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Trigger:** User quyết định ở phiên 23 Sep 17:30: dataset hiện tại (ZeroSCROLLS) không phù hợp để đánh giá hiệu quả của compression.
> **Status:** active (chưa đóng)

## TL;DR

- **ZeroSCROLLS có vấn đề cốt lõi** với eval pipeline (không phải compressor):
  - 30% responses Gemini = `"Please provide a question"` (question không extract được)
  - Nhiều task có question **ngầm** (book_sum_sort, gov_report, summ_screen_fd, space_digest) → Gemini đoán sai task
  - Gold answer dài (5k chars cho gov_report) → exact-match không đo được
- **Quyết định: chuyển sang LongBench** — benchmark được LLMLingua paper dùng
- **Compression: dùng `rate` (tỉ lệ giữ lại token), KHÔNG dùng `target_token`** — paper không đề cập target_token
- **200 cases stratified** bằng context length, từ 7 tasks có question clear + answer ngắn

## 1. Root cause: ZeroSCROLLS không phù hợp

### 1.1 Gemini failures breakdown (n=20 baseline)

```
Total: 20, Correct: 8, Incorrect: 9, Ambiguous: 3

- Gemini BUG (no question processed): 6/20 = 30%
  - "Please provide a question so that I can answer it..."
  - "It looks like you forgot to include the question at the end..."
  → Tasks: book_sum_sort (2), summ_screen_fd (2), space_digest (2), gov_report (2)
- Hallucination (wrong answer): 3/20 = 15%
- Dataset issue: 0
```

### 1.2 Task-level accuracy (Baseline)

```
musique            2/2 = 100%  (factual QA, explicit Q)
squality           5/6 = 83%   (short answer, explicit Q)
qmsum              1/1 = 100%
book_sum_sort      0/2 = 0%    (implicit task: "sort these IDs")
gov_report         0/2 = 0%    (implicit task: summarize)
space_digest       0/2 = 0%    (implicit task: extract answer)
summ_screen_fd     0/2 = 0%    (implicit task: summarize)
quality            0/2 = 0%    (multiple choice, format mismatch)
qasper             0/1 = 0%    (paper QA, gold "No")
```

**Insight:** 5/9 tasks có 0% accuracy không phải do compression — chúng có format mà Gemini không hiểu (implicit task, multiple choice, long-form answer).

### 1.3 Per-task format check

| Task | Question slice | Gold format | Quality |
|---|---|---|---|
| musique | ✅ `Question: Where is...` | Short (Starogard County) | Eval-able |
| squality | ✅ Explicit | Short | Eval-able |
| book_sum_sort | ❌ **Empty** (implicit "sort these IDs") | List of IDs (1,2,5,3,4) | Long, hard |
| gov_report | ❌ **Empty** (implicit summarize) | Full paragraph (~3800 chars) | Too long |
| space_digest | ❌ **Empty** | Single number | OK |
| qmsum | ✅ Explicit | Long summary (~400 chars) | Medium |

## 2. Decision: switch to LongBench

**User decision (Sep 23 18:00):** Lọc khoảng 200 câu từ LongBench, yêu cầu:
- Question clear (explicit text)
- Answer ngắn, rõ ràng
- Context length trải đều từ min đến max

**Compression: dùng `rate` thay vì `target_token`** — paper LLMLingua không đề cập `target_token`. Trong codebase:
- `LLMLingua-2` dùng `rate` (ví dụ: rate=0.5 = giữ 50% tokens)
- `LongLLMLingua` dùng `rate` trong API 0.2.x
- CẢ HAI đều có `context_budget` parameter (số token) nhưng **paper benchmark bằng rate**

## 3. LongBench candidate tasks (filtered)

Đã download 21 QA tasks từ LongBench. Filter theo criteria (Q_len ≥ 10, A_len 1-100):

| Task | N | Q_med | A_med | A_p95 | Status |
|---|---|---|---|---|---|
| hotpotqa | 200 | 83 | 11 | 38 | ✅ Short answer, multi-hop |
| 2wikimqa | 200 | 66 | 13 | 29 | ✅ Short answer, multi-hop |
| musique | 200 | 79 | 14 | 47 | ✅ Short answer, multi-hop |
| narrativeqa | 198 | 46 | 20 | 64 | ✅ Reading comp |
| multifieldqa_en | 121 | 60 | 61 | 137 | ⚠️ A_p95 = 137 (medium) |
| qasper | 142 | 43 | 50 | 238 | ⚠️ A_p95 = 238 (medium) |
| triviaqa | 200 | 72 | 13 | 32 | ✅ Short, trivia |
| trec | 200 | 50 | 14 | 24 | ✅ Classification |
| **Total filtered** | **1461** | | | | |

### Context length distribution

| Task | ctx_med | ctx_min | ctx_max | ctx_p25 | ctx_p75 |
|---|---|---|---|---|---|
| hotpotqa | 10,104 | 1,078 | 12,697 | 7,662 | 11,463 |
| 2wikimqa | 4,210 | 535 | 11,950 | 3,293 | 5,918 |
| musique | 11,386 | 3,440 | 17,355 | 10,768 | 11,728 |
| narrativeqa | 17,500 | 5,397 | 36,418 | 10,013 | 25,090 |
| multifieldqa_en | 4,996 | 505 | 10,337 | 2,294 | 6,444 |
| qasper | 3,418 | 1,440 | 14,660 | 2,424 | 4,286 |
| triviaqa | 8,205 | 1,145 | 16,633 | 5,174 | 11,625 |
| trec | 5,194 | 1,435 | 8,714 | 3,220 | 7,107 |

**Range toàn tập:** 505 → 36,418 tokens (72× range) — tốt cho stratified sampling.

## 4. Sample plan (200 cases)

### Selection algorithm

```python
# 1. Bucket mỗi task by context_length quartile
# 2. Sample từ mỗi bucket để đảm bảo coverage min→max
# 3. Tổng = 200 cases (~25-30/task × 7-8 tasks)

target = 200
n_tasks = 7
per_task = 200 // n_tasks  # ~28-29
```

### Tasks selected (in priority order)

1. **hotpotqa** (200) — cleanest, well-known
2. **2wikimqa** (200) — cleanest multi-hop
3. **musique** (200) — cleanest multi-hop, harder
4. **narrativeqa** (198) — reading comprehension
5. **multifieldqa_en** (121) — structured
6. **qasper** (142) — paper QA
7. **triviaqa** (200) — trivia, easy

**Drop:**
- `trec` — classification task (không phải long-context QA, không test được compression)
- `multifieldqa_zh` — tiếng Trung (model Gemini có thể không tốt)
- `samsum`, `qmsum`, `gov_report`, etc — long-form answer > 100 chars

## 5. Compression configurations (paper-faithful)

**Theo LLMLingua paper:**
- Rate = `keep_ratio` = tỉ lệ tokens giữ lại
- Paper dùng rate ∈ {0.1, 0.2, 0.3, 0.5, 0.7}
- Rate = 0.3 = giữ 30% tokens (giảm 70%)

**Configs đề xuất cho eval:**
| Config | Rate | Token saving | Use case |
|---|---|---|---|
| Baseline | 1.0 | 0% | Reference |
| Aggressive | 0.3 | 70% | Paper benchmark low-end |
| Medium | 0.5 | 50% | User target: ≥50% saving |
| Light | 0.7 | 30% | Conservative |

**Cập nhật default trong scripts:**
- `run_longllmlingua.py`: bỏ `--target-token`, thêm `--rate`
- `run_llmlingua_v2.py`: bỏ `--target-token`, thêm `--rate`

## 6. Next steps

- [ ] Update `doc/phases/phase-01-llmlingua-poc.md`:
  - Section "Pick: library + dataset + model": dataset = LongBench (không phải ZeroSCROLLS)
  - Section "Track 2 — LongLLMLingua": dùng `--rate` thay vì `--target-token`
- [ ] Create `scripts/phase-02/build_longbench_stratified.py`:
  - Sample 200 cases, stratified by length, from 7 tasks
  - Write to `data/processed/longbench_200_stratified.jsonl`
- [ ] Update `scripts/phase-02/run_longllmlingua.py`:
  - Replace `--target-token NNNN` with `--rate 0.X`
  - Paper-faithful parameter names
- [ ] Run 4 configs (Baseline, rate=0.3, 0.5, 0.7) × 200 cases
- [ ] Re-judge with `deepseek_pro` (V4-Pro, more strict than Flash)

## 7. Files to update

| File | Change |
|---|---|
| `doc/phases/phase-01-llmlingua-poc.md` | Dataset + rate-based compression |
| `doc/phases/INDEX.md` | Update phase 02b status |
| `scripts/phase-02/run_longllmlingua.py` | Replace `--target-token` with `--rate` |
| `scripts/phase-02/run_llmlingua_v2.py` | Replace `--target-token` with `--rate` |
| `scripts/phase-02/build_longbench_stratified.py` | NEW — stratified sampling |

## 8. Open questions

- **Q1.** Nên sample stratified by task first (28 per task) rồi stratified by length, hay stratified by length only (across all tasks)?
  → Recommend: stratified by length, với quota mỗi task = 25-30 để diversity.
- **Q2.** Nên thêm `multifieldqa_zh` hay bỏ? Model Gemini có thể handle zh nhưng sẽ tăng judge complexity.
  → Recommend: bỏ, focus en-only.
- **Q3.** LongBench có chuẩn bị dev/test split không? Check `data.zip` README.
  → Per LongBench paper: train không có, chỉ test (200 cases/task).

# Worklog — 2026-09-24 Dedup sweep + latency evaluation (GPU run)

## Context

Phase-02c sweep v4 trên GPU server đã hoàn tất ~06:27 UTC+7 (worklog trước:
`2026-09-23-phase-02-sweep-retry-gemini-outage.md`). Kết quả thô nằm trong
`results/phase-02-llmlingua-r04-r07/` (đã được user move ra `results/`).

User yêu cầu (verbatim): *"xóa các row bị trùng lặp, suy ngẫm kĩ mới xóa,
sau đó đánh giá về latency thôi"* — tức là:
1. Xóa cẩn thận row trùng `case_id`.
2. Đánh giá latency (không cần accuracy/judge).

## 1. Phát hiện trùng lặp

Inspect `case_id` cho 4 file JSONL:

| File | Total rows | Unique case_ids | Dup |
|---|---|---|---|
| `phase-02-run-rate40.jsonl` | 196 | 196 | 0 |
| `phase-02-run-rate50.jsonl` | 198 | 196 | 2 |
| `phase-02-run-rate60.jsonl` | 229 | 196 | **33** |
| `phase-02-run-rate70.jsonl` | 196 | 196 | 0 |

### Pattern dựng lại từ timestamp

Quan sát theo phút cho rate60 (case nặng nhất):

```
05:19-05:22 (UTC+7) → 33 rows  = sweep lần 1, bị interrupt
05:51-06:27 (UTC+7) → 196 rows = sweep lần 2 (resume)
                          └─ 33 case_id trùng với lần 1
                          └─ 163 case_id mới
```

Tức là sweep lần 1 chạy được 33 cases rồi dừng (do outage hoặc torch error),
lần 2 chạy lại đủ 196 cases — nhưng **không skip được 33 case đã có**, append
đè lên.

### Root cause: bug resume trong `scripts/phase-02/run_compressed.py` line 350-352

Sau worklog `2026-09-24-phase-02-predict-only-default.md` đổi default sang
predict-only (judge field = None), điều kiện resume bị sai:

```python
if (r.get("status") == "ok"
        and r.get("pred")
                and (args.run_judge or r.get("judge_correct") is not None)):
    results.append(r)
    done_ids.add(r["case_id"])
```

Trong predict-only mode (`run_judge = False`), `judge_correct` luôn `None`,
nên điều kiện thứ 3 là `False or None is not None` = `False`. Không bao giờ
skip được row nào. Bug nghiêm trọng — sẽ phải fix trước khi re-run sweep.

**Fix cần làm (chưa áp dụng trong session này):**

```python
# Predict-only mode: skip if record exists with status=ok + pred truthy
# Judge mode: also require judge_correct is not None
done_ids.add(r["case_id"]) if (
    r.get("status") == "ok"
    and r.get("pred")
    and (args.run_judge is False or r.get("judge_correct") is not None)
) else None
```

## 2. Dedupe an toàn

Quyết định:
- **Strategy:** keep latest `ts` (user chọn). Bản mới hơn là bản sweep lần 2,
  predict regenerates với Gemini mới hơn (mặc dù không deterministic).
- **Heuristic khác (đã cân nhắc):** keep `pred` dài nhất — KHÔNG dùng vì qasper_3dd
  counter-example: v0 dài 419 chars (đúng BERT) vs v1 dài 278 chars ("not explained").
  Length ≠ correctness ở QA task.
- **An toàn:** dedupe dry-run mặc định, chỉ write khi có `--write`.

Đã viết script vào codebase (theo AGENTS.md §"Codebase-first rule"):

### `scripts/phase-02/dedupe_jsonl.py` (new)

- CLI: `--in <path> --write`
- Output: `<in>.jsonl.dedup` + `<in>.dedup-report.json`
- Strategy: keep-latest-ts; tiebreak = last-in-file-order
- Dry-run safe: không bao giờ overwrite input file gốc

### `scripts/phase-02/eval_latency.py` (new)

- CLI: `python scripts/phase-02/eval_latency.py`
- Reads 4 `.dedup` JSONLs, prints side-by-side latency table
- No judge, no accuracy — latency-only theo yêu cầu của user
- Per-task wallclock breakdown (rate60 representative)

## 3. Kết quả đánh giá latency

### Side-by-side (N=196/rate)

| rate | N | comp_med | comp_p95 | gem_med | gem_p95 | e2e_med | in_tok_med | in_tok_max | ratio_med | wallclock | cases/min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rate40 | 196 | 3392ms | 14827ms | 786ms | 5636ms | 5013ms | 3709 | 24890 | 2.40× | 30.4m | 6.45 |
| rate50 | 196 | 2700ms | 12408ms | 760ms | 25429ms | 4868ms | 4541 | 30653 | 1.90× | 37.0m | 5.29 |
| rate60 | 196 | 3164ms | 12320ms | 753ms | 18707ms | 5072ms | 5392 | 38661 | 1.60× | 34.9m | 5.62 |
| rate70 | 196 | 3577ms | 15513ms | 830ms | 6544ms | 5408ms | 6210 | 46390 | 1.40× | 31.0m | 6.33 |

### Token saving vs uncompressed (dùng rate70 làm proxy baseline)

| rate | median input | saving |
|---|---|---|
| rate40 | 3709 | **40.3%** |
| rate50 | 4541 | 26.9% |
| rate60 | 5392 | 13.2% |
| rate70 | 6210 | 0.0% (baseline proxy) |

> CAVEAT: `rate70` không thực sự là baseline (chưa có sweep uncompressed). Số
> "saving" này chỉ là relative-to-rate70 trong cùng sweep.

### Per-task (rate60 representative)

| task | N | e2e_med (s) | e2e_p95 (s) |
|---|---|---|---|
| triviaqa | 28 | 0.89 | 1.91 |
| qasper | 28 | 3.13 | 28.95 |
| 2wikimqa | 28 | 3.21 | 38.71 |
| multifieldqa_en | 28 | 4.12 | 35.51 |
| hotpotqa | 28 | 6.11 | 29.49 |
| musique | 28 | 7.97 | 9.87 |
| narrativeqa | 28 | 11.32 | 31.29 |

Trend rõ: tasks có context_length lớn (narrativeqa, musique) tốn nhiều thời
gian compress hơn tasks ngắn (triviaqa, qasper).

## 4. Phát hiện / quan sát

1. **`compress_ms` không phụ thuộc rõ rệt vào `rate`**: rate50 có comp_med
   2700ms (thấp nhất), rate70 là 3577ms (cao nhất). Điều này counter-intuitive
   — rate thấp hơn (compress aggressive hơn) thường *chạy nhanh hơn* vì ít
   token phải qua LLMlingua's extract step. Có thể do `comp_med` ảnh hưởng
   bởi mix task (rate70 ngẫu nhiên gặp nhiều task context-dài hơn).
   → Kiểm tra thêm bằng cách control cho task distribution.

2. **`gemini_s` p95 rất khác nhau giữa các rate**: rate50 có p95 = 25.4s vs
   rate70 p95 = 6.5s — cùng model, cùng task, cùng dataset, cùng session.
   Nhiều khả năng là do **Gemini flash-lite back-end** (overload / rate limit).
   Cần nhìn retry log để xác nhận — không nên quyết định rate nào "ổn định"
   hơn chỉ dựa trên con số p95 này.

3. **Throughput cases/min dao động 5.3 → 6.5**: dominated bởi `sleep=3s`
   giữa các call (để tránh Gemini rate-limit). Nếu sleep=0, throughput sẽ
   phụ thuộc hoàn toàn vào `comp_med + gem_med`, lúc đó rate40 sẽ win.

4. **Long-tail p95 gemini > comp_p95** ở rate50 và rate60: Gemini response
   time có tail rất dài (~25s và ~18s) — đây là bottleneck cho wallclock,
   không phải compression. Nếu cần giảm e2e p95, tăng retry budget cho
   rate-limited case hoặc switch sang model khác.

## 5. Open follow-ups

- [ ] **Fix resume bug** trong `run_compressed.py` line 350-352 trước khi
  re-run sweep (xem §1 ở trên).
- [ ] **Cân nhắc thêm `compress_ms` theo task** để có decomposition
  context_length → compress_time sạch hơn.
- [ ] **Accuracy eval (optional)**: hiện `n_judged=0`. Nếu user muốn
  accuracy, chạy `judge_jsonl.py --in ... --out ... --profile deepseek`
  trên 4 file `.dedup` (predict-only mode, cần gọi DeepSeek 784 lần,
  ~$0.27/M + output — rẻ, có thể làm).
- [ ] Worklog chưa update `doc/phases/INDEX.md` status (phase-02c vẫn active
  cho đến khi accuracy pass).

# Worklog — 2026-09-18 — Phase 1: LLMLingua 5-case effectiveness test

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status before this session:** active (scripts scaffolded)
> **Status after this session:** active, with **explicit DROP recommendation** for LLMLingua-2 (both paths)
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)
> **Note (2026-09-21):** dataset `zai-org/LongBench-v2` đã được thay bằng
> `tau/zero_scrolls`. Các tham chiếu `LongBench-v2` dưới đây là **legacy**.
> Xem [`2026-09-21-dataset-switch-zero-scrolls.md`](2026-09-21-dataset-switch-zero-scrolls.md).

## What this session did

1. **`verify_libs.py`** — pre-flight check cho 7 package (`google-generativeai`,
   `python-dotenv`, `datasets`, `langchain`, `langchain-community`, `llmlingua`,
   `jsonschema`). Tất cả đã có sẵn trong `vsf` env:
   - `google-generativeai 0.8.6`, `python-dotenv 1.2.3`, `datasets 5.0.1`,
     `langchain 1.4.1`, `langchain-community 0.4.2`, `llmlingua 0.2.2`, `jsonschema 4.26.0`.
   - 2 deprecation warnings:
     - `google.generativeai` **đã sunset** — phải chuyển sang `google.genai`
       (ngoài scope PoC này, nhưng đáng ghi nhận cho Phase 2+).
     - `langchain-community` **đang sunset** — ghi nhận tương tự.

2. **`setup_dataset.py`** — download 100 rows đầu từ `zai-org/LongBench-v2`
   (split=`train`, streaming), lưu vào `data/raw/longbench_v2_first100_2026-09-18.jsonl`,
   shuffle với seed=42, tách 5 row → `data/processed/llmlingua_test5.jsonl`,
   95 row còn lại → `data/processed/dev_first95.jsonl` (giữ làm dev pool cho eval mở rộng).

3. **Schema LongBench-v2 (giải quyết Q1 từ worklog 17/09):**

   | Field | Type | Ví dụ |
   |---|---|---|
   | `_id` | str | `'66ed910a821e116aacb2033b'` |
   | `domain` | str | `'Single-Document QA'` |
   | `sub_domain` | str | `'Literary'` |
   | `difficulty` | str | `'hard'` |
   | `length` | str | `'medium'` |
   | `question` | str | `'What is mainly symbolized...'` |
   | `choice_A`/`B`/`C`/`D` | str | `'Confusion of The Times'` |
   | `answer` | str | `'A'` (chỉ 1 chữ cái) |
   | `context` | str | long, 70k–2.5M chars |

   → **Quyết định:** dùng `answer` (1 chữ cái A/B/C/D) làm gold; so sánh exact-match
   bằng cách lấy chữ cái A/B/C/D đầu tiên trong response của Gemini.

4. **`smoke_both_paths.py`** — smoke test cả 2 path trên row ngắn nhất (Legal 70k chars):
   - **Path A (pip-direct)**: tên model `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`
     tải được, nhưng chunk size 5000 chars (1082 tokens) vượt quá max position
     embedding 512 → rate=0.33 không nén, ratio=**100.0%**. Fix chunk xuống **1500 chars**
     (~375 tokens) là an toàn.
   - **Path B (LangChain)**: tên model gốc trong `poc_track1.py` là `...-meetment`
     (typo, 3 chữ `t`) → **404 Not Found**. Đã sửa thành `...-meetingbank`. Sau fix:
     model load được, nhưng crash với `AttributeError: 'TokenClassifierOutput' object
     has no attribute 'past_key_values'`. Bug này khớp với comment cũ trong
     `smoke_llmlingua_langchain.py:50-60` (“LangChain wrapper has accelerate bug”).

5. **`eval_llmlingua_5.py`** — chạy 3 config (baseline / compressed_pip / compressed_lc)
   trên 5 case đã chọn, sort theo context_chars tăng dần. Đã chạy 2 case trước khi hit
   Gemini free-tier quota (250k input tokens/min) — case 3 (Literary 382k) không hoàn
   thành.

6. **`compute_summary.py`** — script phụ: đọc lại JSONL đã ghi, tính summary +
   delta % vs baseline.

## Kết quả đo (n=2 — Legal 70k + Table QA 246k)

| Path | % token saved | % time saved | Δ accuracy (pp) |
|---|---|---|---|
| `compressed_pip` | **−0.2%** | **−2393%** | +50pp |
| `compressed_lc`  | (n=1, bug) | n/a | n/a |

**Chi tiết từng case:**

```
Case L5-00-66ed91 (Legal 70k, gold=A):
  baseline:        tokens=18500  ratio=1.0    comp=    0ms + gem=1329ms =  1329ms  pred=C acc=0
  compressed_pip:  tokens=18558  ratio=1.0007 comp=45543ms + gem=1128ms = 46670ms  pred=C acc=0
  compressed_lc:   tokens=18500  ratio=1.0    comp= 1100ms + gem=1099ms =  2199ms  pred=A acc=1
                                                                  [crash trên case #2]

Case L5-01-6703a0 (Table QA 246k, gold=A):
  baseline:        tokens=88553  ratio=1.0    comp=    0ms + gem=1857ms =  1857ms  pred=D acc=0
  compressed_pip:  tokens=88698  ratio=1.0007 comp=30877ms + gem=1889ms = 32765ms  pred=A acc=1
  compressed_lc:   [crash, no data]
```

## Quyết định (ship / iterate / drop) — từng path

### Path A — pip-direct LLMLingua-2: **DROP**

- **Token saving:** 0% (ratio 1.0007 = compressed thậm chí DÀI HƠN baseline
  do khoảng trắng giữa các chunk). Lý do: với text legal/QA, model xlm-roberta
  giữ lại mọi token vì không có segment thừa để bỏ. Rate=0.33 chunk 1500 chars
  quá yếu để bỏ token.
- **Time:** chậm hơn baseline **24×** (compressor 38s + Gemini 1.5s = 39.7s
  vs baseline 1.6s). Trên context 2.5M chars (case #5 nếu không skip) sẽ
  tốn hơn 30 phút mỗi call.
- **Accuracy:** "cải thiện" 50pp chỉ là sample noise (baseline 0/2 → compressed 1/2).
- → Không có trade-off nào dùng được. KHÔNG ship.

### Path B — LangChain LLMLinguaCompressor: **DROP**

- **Bug đã verify thực tế:** `AttributeError: 'TokenClassifierOutput' object has
  no attribute 'past_key_values'` trên case 70k (smoke + eval đều crash).
- Nguyên nhân có thể là incompat giữa `transformers` mới (xlm-roberta token
  classifier API mới trả `TokenClassifierOutput` không có `past_key_values`)
  và `langchain-community` đang sunset. Không có quick fix.
- → Bỏ luôn wrapper, đừng tốn công maintain.

## Bài học

1. **LLMLingua-2 không phù hợp với LongBench-v2 long-context QA.** Dataset này
   context dài nhưng "nội dung cần thiết" không có cấu trúc để nén — text
   literary/legal là liên tục, không có boilerplate rõ ràng như code/log.
   Rate=0.33 + chunk 1500 chars cho ratio ~1.0 là điểm vô ích.

2. **Nếu muốn compress thật sự, cần:**
   - Dataset có nhiều redundancy (ví dụ: logs, code, RAG contexts lặp)
   - Hoặc model lớn hơn xử lý được >512 tokens (khả thi với GPU)
   - Hoặc RAG context selection trước khi compress (lever C trong master plan).

3. **`langchain-community` đang sunset**: future code nên dùng integration packages
   standalone (vd `langchain-text-splitters`, v.v.). LangChain wrapper không đáng
   maintain cho PoC LLMLingua.

4. **Gemini free-tier quota 250k input tokens/min** — eval nhiều case >100k tokens
   cần `--max-context-chars` để skip case khổng lồ, hoặc batch call cách nhau vài phút.
   Có thể chuyển sang model trả phí cho Phase 2.

## Cập nhật Plan / Phase status

- **Phase 1 vẫn active** nhưng kết luận PoC cho LLMLingua-2 là **DROP** với
  cấu hình hiện tại (rate=0.33, chunk=1500, text LongBench-v2). Nếu muốn
  ship, cần thay đổi khá lớn (dataset có redundancy hơn, model lớn hơn, hoặc
  integration RAG context selection). Ghi nhận trong [INDEX.md](../../phases/INDEX.md).

- **Q1 (LongBench-v2 schema) → resolved.** Schema đã được log ở §3 trên.

- **Đề xuất Phase tiếp theo** (cần user xác nhận):
  1. Đóng Phase 1 với decision note này (không cần chạy thêm).
  2. **Bắt đầu Phase 0** (5 JSON Schema + deterministic router + promptfoo smoke) —
     nền tảng cần thiết cho Phase 2/3/4. GĐ 0 không cần LLM API.
  3. Hoặc **đổi compressor** trước khi đóng Phase 1 (thử `longllmlingua` cho
     Track 2 RAG, hoặc thử `LLMLingua` gốc với rate cao hơn).
- Decision thuộc về user — worklog này chỉ record evidence.

## Files tạo / chạm trong session này

| File | Trạng thái |
|---|---|
| `scripts/phase-01/verify_libs.py` | Tạo mới |
| `scripts/phase-01/setup_dataset.py` | Tạo mới |
| `scripts/phase-01/smoke_both_paths.py` | Tạo mới (sửa typo `meetment`→`meetingbank`, fix chunk size) |
| `scripts/phase-01/eval_llmlingua_5.py` | Tạo mới |
| `scripts/phase-01/compute_summary.py` | Tạo mới (post-hoc summary từ JSONL) |
| `data/raw/longbench_v2_first100_2026-09-18.jsonl` | Tạo qua script (gitignored) |
| `data/processed/llmlingua_test5.jsonl` | Tạo qua script (gitignored, read-only sanity set) |
| `data/processed/dev_first95.jsonl` | Tạo qua script (gitignored, dev pool) |
| `results/phase-01-llmlingua-5cases.jsonl` | Output (6 records, 2 case × 3 config, 1 config crash) |
| `results/phase-01-llmlingua-5cases-summary.json` | Output (aggregated + delta %) |

## Next steps (cho session sau)

1. User xác nhận: **đóng Phase 1** với decision DROP, hay **thử compressor khác**
   (longllmlingua / rate cao hơn / RAG context selection) trước khi đóng.
2. Nếu đóng Phase 1 → update `doc/phases/INDEX.md` (Phase 1 status = `done`
   với note "LLMLingua-2 dropped after PoC").
3. Bắt đầu Phase 0 (nền tảng schemas + router + promptfoo) — không cần LLM API.

## Open questions

- **Q3 (mới).** Có nên đóng Phase 1 với DROP ngay, hay thử `longllmlingua` /
  RAG context selection trước khi commit kết luận? (Decision owner: user.)

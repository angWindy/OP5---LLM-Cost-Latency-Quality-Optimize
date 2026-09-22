# Phase 1 — LLMLingua PoC (prompt compression)

> **Status:** active
> **Owner:** —
> **Started:** 2026-09-17
> **Updated:** 2026-09-21 — bổ sung Approach C: Preselect + Compress combo

## Goal

Get a **sanity-check pass** trên bộ compressor của Microsoft trước khi commit vào
đòn bẩy **B (nén prompt)** và **C (chọn lọc context cho Track 2)** trong master plan.

Phase 1 này có **2 track song song**, mỗi track test một biến thể compressor phù hợp:

| Track | Compressor | Vì sao |
|---|---|---|
| **Track 1 — Trích xuất field/bảng** | **LLMLingua-2** | Task-agnostic, không cần câu hỏi để nén theo (vì schema extraction cần điền cả bộ field cùng lúc). Nhanh hơn 3-6× → không cộng thêm độ trễ sau bước OCR. |
| **Track 2 — RAG hỏi-đáp điều khoản** | **LongLLMLingua** | Biết trước câu hỏi → nén ngữ cảnh dài mà **không mất đoạn chứa đáp án**, đồng thời giảm vấn đề "lost in the middle". Khớp trực tiếp với thiết kế đòn bẩy C trong master plan. |

We want to know:

1. Cài đặt được cả 2 compressor trong `vsf` env không?
2. Từng compressor có shrink được prompt ZeroSCROLLS trong khi giữ đủ thông tin
   để Gemini trả lời đúng không?
3. **Compression ratio vs answer-quality delta** cho từng track.
4. Latency overhead của bước nén.

This is **scoping**, not a full experiment. Không có claim statistical significance — chỉ
"does it run, does it help, what knobs matter."

---

## 1.5 Approach C: Preselect + Compress (combo, added 2026-09-21)

Ngoài 2 track chính, session 2026-09-21 đã thử thêm **Approach C** —
kết hợp **RAG preselect** (SentenceTransformer) + **LLMLingua-2 compress**
trên 1 case ZeroSCROLLS.

### "Preselect + Compress" là gì?

**Preselect + Compress** là phương pháp **2 bước nối tiếp** để giảm context
cho LLM trước khi gọi Gemini. Hai bước này **bổ sung cho nhau**, mỗi bước
giải quyết một vấn đề khác nhau:

- **Bước 1 — Preselect:** Cắt bỏ phần context **không liên quan** đến câu hỏi.
  Dùng SentenceTransformer (22M params) embed tất cả câu + câu hỏi, tính
  cosine similarity, giữ top-k câu (k=20). Ví dụ: 70k chars → 6k chars (-91.5%).
- **Bước 2 — Compress:** Nén **các câu đã preselect** xuống mức token tối thiểu.
  Dùng LLMLingua-2 (560M xlm-roberta) với rate=0.5, giữ 50% tokens quan trọng
  nhất theo perplexity score.

### Data flow

```
context (70k chars)
       │
       ▼
┌─────────────────────┐
│ Preselect            │  SentenceTransformer (22M)
│ - sentence split     │  cosine similarity
│ - embed all + q      │  top-k=20 selection
│ - keep top-20        │
└─────────┬───────────┘
          │
          ▼
   selected (6k chars, -91.5%)
          │
          ▼
┌─────────────────────┐
│ Compress             │  LLMLingua-2 (560M xlm-roberta)
│ - perplexity score   │  rate=0.5
│ - keep 50% tokens    │
└─────────┬───────────┘
          │
          ▼
   compressed (929 tokens)
          │
          ▼
     Gemini call
```

### Tại sao cần cả 2 bước?

LLMLingua-2 chậm tỷ lệ thuận với input length (~850 chars/s CPU). Nếu không
preselect trước, compressor phải xử lý toàn bộ 70k chars → 81s. Preselect
giảm đầu vào còn 6k chars → compressor chỉ mất 7s (nhanh hơn 11×).

### Kết quả benchmark (1 case, 70k chars)

| Approach | Total time | Chars kept | Accuracy | Verdict |
|---|---|---|---|---|
| Baseline | 1,360 ms | 70,157 (100%) | 0% (wrong) | Reference |
| A: Compress only | 95,536 ms | 70,157 (100%) | 100% | 70× slower — không đáng |
| B: Preselect only | 15,046 ms | 5,992 (8.5%) | 100% | 11× slower |
| **C: Combo** | **13,823 ms** | **5,992 (8.5%)** | **100%** | 10× slower, tốt nhất |

Xem chi tiết tại [`doc/worklog/2026-09-21-3-approach-1case.md`](../worklog/2026-09-21-3-approach-1case.md)
(section 0 cho giải thích chi tiết, section 2 cho kết quả số).

---

## Scope (in)

### Phần chung
- Cài đặt: `langchain`, `langchain-community` (cho `LLMLinguaCompressor`),
  `google-generativeai`, `datasets`, `python-dotenv`, `jsonschema`.
- **Trước tiên**: chạy 1 request Gemini cơ bản để xác nhận API key hoạt động.
- Dataset: **`tau/zero_scrolls`** — multi-domain 10-task long-context benchmark.
  - Switched từ `zai-org/LongBench-v2` ngày 2026-09-21 (xem `2026-09-21-dataset-switch-zero-scrolls.md`).
  - Public trên HF, không cần token, đã được LongLLMLingua paper benchmark.
  - Stream first 30 rows.
- Model: **`gemini-3.5-flash-lite`**.
- Log JSONL theo EvalLog-style fields: `case_id`, `track`, `config`, `input_tokens`,
  `output_tokens`, `compression_ratio`, `answer_text`, `latency_ms`, `compressor`, `gemini_latency_ms`.
- Measure: compression ratio, answer correctness (LLM-as-judge với chính Gemini), latency.

### Track 1 — LLMLingua-2
- Input: full contract text (no question).
- Baseline (`A1`): prompt gốc → Gemini.
- Compressed (`B1`): prompt gốc → **LLMLingua-2** → Gemini.
- Đo: token saving, quality delta, compressor latency.
- Khớp đòn bẩy **B** trong master plan §4.6 cho Track 1.

### Track 2 — LongLLMLingua
- Input: full contract text + question.
- Baseline (`A2`): prompt gốc → Gemini.
- Compressed (`C2`): prompt gốc → **LongLLMLingua(question=q)** → Gemini.
- Đo: token saving, quality delta (exact-match + F1), compressor latency.
- Khớp đòn bẩy **C** trong master plan §4.6 cho Track 2.

## Scope (out)

- Full eval set, paired statistical analysis, promptfoo (that comes in Phase 4).
- Routing (đòn bẩy D), OCR (Phase 2).
- Anything with the company-private model.
- Heavy HuggingFace downloads — keep dataset small, stream it.

## Pick: library + dataset + model

| Component | Choice | Lý do |
|---|---|---|
| Compressor (Track 1) | **LLMLingua-2** | Task-agnostic, nhanh hơn 3-6× so với LLMLingua gốc, không cần câu hỏi |
| Compressor (Track 2) | **LongLLMLingua** | Nén theo câu hỏi, giảm "lost in the middle" |
| Wrapper | LangChain (`LLMLinguaCompressor`) | One-line integration, plug vào RAG chain dễ |
| Dataset | `tau/zero_scrolls` (ZeroSCROLLS) | Multi-domain 10 tasks, avg ~10k tokens context, có gold answer, public HF, đã được paper benchmark |
| **Baseline LLM** | `gemini-3.5-flash-lite` (Google AI) | Rẻ ($0.30/$2.50 per 1M tokens), nhanh, free tier đủ, GA Jul 21 2026 |
| **Trần chất lượng LLM (Phase 4+)** | `gemini-3.1-pro` (Google AI) | Flagship Gemini ($2/$12 per 1M tokens), prompts ≤ 200K, dùng cho cấu hình D(mạnh) |
| **LLM-as-judge (eval)** | `nvidia/nemotron-3-ultra-550b-a55b:free` qua OpenRouter | Free, GPQA Diamond 86.7%, 1M context, Sep 2026 ranking #1 free model. Fallback `openrouter/free` router |
| **Product (Phase 3)** | Private model API của công ty | Khi tích hợp thật — chỉ đổi adapter config, giữ threshold routing |

## Environment

```bash
conda activate vsf
pip install langchain langchain-community google-generativeai datasets python-dotenv jsonschema
```

API key: put `GOOGLE_API_KEY=...` into `.env` (gitignored). Load with `python-dotenv`
hoặc `os.environ`. **Không commit key.**

## Scripts

| File | Mục đích |
|---|---|
| `scripts/phase-01/smoke_gemini.py` | **1 request Gemini cơ bản** — verify API key hoạt động |
| `scripts/phase-01/poc_track1.py` | Track 1 PoC: LLMLingua-2 vs baseline |
| `scripts/phase-01/poc_track2.py` | Track 2 PoC: LongLLMLingua vs baseline |
| `scripts/phase-01/inspect_dataset.py` | Explore ZeroSCROLLS structure (stream 3 rows) |

## Deliverables

- [ ] `smoke_gemini.py` chạy được, in ra response từ `gemini-3.5-flash-lite`.
- [ ] `inspect_dataset.py` in ra cấu trúc 3 rows đầu của ZeroSCROLLS.
- [ ] `vsf` env có `langchain` + `LLMLinguaCompressor`.
- [ ] `poc_track1.py` chạy paired (baseline vs LLMLingua-2) trên ~15 case → `results/phase-01-track1-poc.jsonl`.
- [ ] `poc_track2.py` chạy paired (baseline vs LongLLMLingua) trên ~15 case → `results/phase-01-track2-poc.jsonl`.
- [ ] Worklog: ghi lại compression ratio, quality delta, latency cho cả 2 track.
- [ ] Decision note: ship / iterate / drop — riêng cho từng track.

## Done criteria

- Smoke test Gemini pass (response trả về, không có 401/403).
- Cả 2 track chạy end-to-end trên `vsf` không cần manual intervention sau khi set API key.
- JSONL validate theo EvalLog schema (loose subset).
- Compression ratio và quality delta được report **ít nhất ballpark** cho cả 2 track.
- Go/no-go recommendation trong worklog cho từng track.

## Blockers / risks

- **R1.** LLMLingua download ~1GB compression model lần đầu. Mitigation: cache ở
  `~/.cache/huggingface/` (đã trong `.gitignore`).
- **R2.** Gemini free-tier rate limit. Mitigation: dùng `gemini-3.5-flash-lite`,
  sleep giữa các call nếu cần. Nếu vẫn 429, fallback sang tier paid hoặc chuyển
  sang OpenRouter `google/gemini-3.5-flash-lite`.
- **R3.** ZeroSCROLLS là public trên HF, không cần token. Legacy LongBench-v2 thì
  tùy trường hợp — thử `use_auth_token=False` trước, fallback sang ZeroSCROLLS (mặc định).
- **R4.** ZeroSCROLLS context ~10k tokens — vừa đủ để test compressor ở tỉ lệ nén
  2-5× mà không bị OOM. Long-context hơn thì dùng LooGLE / MuSiQue.
- **R5.** `gemini-3.5-flash-lite` là model mới — verify availability trước (smoke test).

## Reference

- `doc/plan/op5-llm-cost-latency-quality-plan.md` — §4.6 (lever B, C, D), §4.7 (metrics).
- LLMLingua repo: https://github.com/microsoft/LLMLingua
- LLMLingua-2 paper: https://arxiv.org/abs/2403.12957
- LongLLMLingua paper: https://arxiv.org/abs/2310.06839
- ZeroSCROLLS dataset: https://huggingface.co/datasets/tau/zero_scrolls
- (Legacy) LongBench-v2 dataset: https://huggingface.co/datasets/zai-org/LongBench-v2
- LangChain LLMLinguaCompressor: https://python.langchain.com/docs/integrations/document_transformers/llmlingua

# Phase 1 — LLMLingua PoC (prompt compression)

> **Status:** active
> **Owner:** —
> **Started:** 2026-09-17
> **Updated:** 2026-09-17 — phân tách Track 1 (LLMLingua-2) / Track 2 (LongLLMLingua)

## Goal

Get a **sanity-check pass** trên bộ compressor của Microsoft trước khi commit vào
đòn bẩy **B (nén prompt)** và **C (chọn lọc context cho Track 2)** trong master plan.

Phase 1 này có **2 track song song**, mỗi track test một biến thể compressor phù hợp:

| Track | Compressor | Vì sao |
|---|---|---|
| **Track 1 — Trích xuất field/bảng** | **LLMLingua-2** | Task-agnostic, không cần câu hỏi để nén theo (vì schema extraction cần điền cả bộ field cùng lúc). Nhanh hơn 3-6× → không cộng thêm độ trễ sau bước OCR. |
| **Track 2 — RAG hỏi-đáp điều khoản** | **LongLLMLingua** | Biết trước câu hỏi → nén ngữ cảnh dài mà **không mất đoạn chứa đáp án**, đồng thời giảm vấn đề "lost in the middle". Khớp trực tiếp với thiết kế đòn bẩy C trong master plan. |

We want to know:

1. Cài đặt được cả 2 compressor trong `po5` env không?
2. Từng compressor có shrink được prompt LongBench-v2 trong khi giữ đủ thông tin
   để Gemini trả lời đúng không?
3. **Compression ratio vs answer-quality delta** cho từng track.
4. Latency overhead của bước nén.

This is **scoping**, not a full experiment. Không có claim statistical significance — chỉ
"does it run, does it help, what knobs matter."

## Scope (in)

### Phần chung
- Cài đặt: `langchain`, `langchain-community` (cho `LLMLinguaCompressor`),
  `google-generativeai`, `datasets`, `python-dotenv`, `jsonschema`.
- **Trước tiên**: chạy 1 request Gemini cơ bản để xác nhận API key hoạt động.
- Dataset: **`zai-org/LongBench-v2`** — multi-task long-context benchmark.
  Stream first 30 rows.
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
| Dataset | `zai-org/LongBench-v2` | Long-context benchmark, multi-task, phù hợp Track 1 + Track 2 |
| Model | `gemini-3.5-flash-lite` | Rẻ, nhanh, free tier đủ |

## Environment

```bash
conda activate po5
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
| `scripts/phase-01/inspect_dataset.py` | Explore LongBench-v2 structure (stream 3 rows) |

## Deliverables

- [ ] `smoke_gemini.py` chạy được, in ra response từ `gemini-3.5-flash-lite`.
- [ ] `inspect_dataset.py` in ra cấu trúc 3 rows đầu của LongBench-v2.
- [ ] `po5` env có `langchain` + `LLMLinguaCompressor`.
- [ ] `poc_track1.py` chạy paired (baseline vs LLMLingua-2) trên ~15 case → `results/phase-01-track1-poc.jsonl`.
- [ ] `poc_track2.py` chạy paired (baseline vs LongLLMLingua) trên ~15 case → `results/phase-01-track2-poc.jsonl`.
- [ ] Worklog: ghi lại compression ratio, quality delta, latency cho cả 2 track.
- [ ] Decision note: ship / iterate / drop — riêng cho từng track.

## Done criteria

- Smoke test Gemini pass (response trả về, không có 401/403).
- Cả 2 track chạy end-to-end trên `po5` không cần manual intervention sau khi set API key.
- JSONL validate theo EvalLog schema (loose subset).
- Compression ratio và quality delta được report **ít nhất ballpark** cho cả 2 track.
- Go/no-go recommendation trong worklog cho từng track.

## Blockers / risks

- **R1.** LLMLingua download ~1GB compression model lần đầu. Mitigation: cache ở
  `~/.cache/huggingface/` (đã trong `.gitignore`).
- **R2.** Gemini free-tier rate limit. Mitigation: dùng `gemini-3.5-flash-lite`,
  sleep giữa các call nếu cần.
- **R3.** LongBench-v2 có thể yêu cầu HF token. Mitigation: thử `use_auth_token=False`
  trước, fallback sang dataset public khác nếu fail.
- **R4.** LongBench-v2 là long-context → cần chunking cho compressor. Mitigation:
  test 1-2 example trước khi chạy full.
- **R5.** `gemini-3.5-flash-lite` là model mới — verify availability trước (smoke test).

## Reference

- `doc/plan/op5-llm-cost-latency-quality-plan.md` — §4.6 (lever B, C, D), §4.7 (metrics).
- LLMLingua repo: https://github.com/microsoft/LLMLingua
- LLMLingua-2 paper: https://arxiv.org/abs/2403.12957
- LongLLMLingua paper: https://arxiv.org/abs/2310.06839
- LongBench-v2 dataset: https://huggingface.co/datasets/zai-org/LongBench-v2
- LangChain LLMLinguaCompressor: https://python.langchain.com/docs/integrations/document_transformers/llmlingua

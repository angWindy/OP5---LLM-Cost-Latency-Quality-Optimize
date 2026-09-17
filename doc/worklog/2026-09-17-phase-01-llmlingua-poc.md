# Worklog — 2026-09-17 — Phase 1: LLMLingua PoC (full scripts ready)

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status when started:** active (scripts scaffolded)
> **Author:** —
> **Env:** conda `po5` (Python 3.11)

## Today (bản update)

- **Phân tách compressor theo track** trong plan:
  - **Track 1 (extraction field/bảng)** → `LLMLingua-2` (task-agnostic, nhanh hơn 3-6×)
  - **Track 2 (RAG hỏi-đáp)** → `LongLLMLingua` (question-aware, giảm "lost in the middle")
- **Smoke test Gemini:** `scripts/phase-01/smoke_gemini.py` — 1 request cơ bản
- **Inspect dataset:** `scripts/phase-01/inspect_dataset.py` — xem schema LongBench-v2
- **Track 1 PoC:** `scripts/phase-01/poc_track1.py` — paired (A1 baseline vs B1 LLMLingua-2)
- **Track 2 PoC:** `scripts/phase-01/poc_track2.py` — paired (A2 baseline vs C2 LongLLMLingua)
- **Shared module:** `scripts/phase-01/_common.py` — Gemini client + dataset loader + judge
- **Requirements:** `scripts/phase-01/requirements.txt`
- **Updated** `scripts/README.md` để liệt kê các scripts mới

## Pipeline chạy (end-to-end)

```bash
conda activate po5
pip install -r scripts/phase-01/requirements.txt
echo "GOOGLE_API_KEY=your-key" >> .env

# Step 1: verify API key
python scripts/phase-01/smoke_gemini.py

# Step 2: confirm dataset schema
python scripts/phase-01/inspect_dataset.py --n 3

# Step 3: Track 1 (LLMLingua-2)
python scripts/phase-01/poc_track1.py --n 15

# Step 4: Track 2 (LongLLMLingua)
python scripts/phase-01/poc_track2.py --n 15
```

## Decisions made today

1. **Compressor per track:** Track 1 → LLMLingua-2 (task-agnostic, nhanh),
   Track 2 → LongLLMLingua (question-aware).
2. **LLM-as-judge:** dùng chính Gemini để so sánh pred vs gold trên sample (5 case đầu).
   Biết bias, nhưng đủ cho PoC.
3. **Compression ratio:** approximate `len(compressed) / len(context)` (chars/4 heuristic),
   vì compressor không luôn trả token count chính xác.
4. **JSONL EvalLog-style fields** được giữ nguyên cho cả 2 track.
5. **Auto-resume:** scripts `unlink()` output file cũ trước khi chạy → chạy lại sạch sẽ.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LongBench-v2 yêu cầu HF token | `inspect_dataset.py` fail-fast trước khi chạy full PoC |
| `gemini-3.5-flash-lite` không tồn tại | Smoke test sẽ fail ngay với lỗi rõ ràng |
| LLMLingua model download ~1GB | Lần đầu hơi chậm; cache ở `~/.cache/huggingface/` |
| Compressor chậm trên CPU | PoC chỉ 15 case — chấp nhận được |

## Blockers / open questions

- **B1.** Chưa chạy được (cần user cài deps + set API key trước).
- **Q1.** Khi chạy thực tế, nếu LongBench-v2 schema khác với giả định
  (`context`, `question`, `answer`, `choice`), `detect_fields()` sẽ fallback sang
  `input`/`output`/`label`. Cần kiểm tra output của `inspect_dataset.py` để confirm.

## Next steps (for the next session, sau khi user chạy xong step 1-4)

1. Đọc output `results/phase-01-track1-poc.jsonl` và `results/phase-01-track2-poc.jsonl`.
2. Tính ballpark metrics: avg compression ratio, judge accuracy delta, latency overhead.
3. Ghi decision note (ship / iterate / drop) cho từng compressor.
4. Nếu ship → chuẩn bị integration vào Phase 0 (EvalLog schema, scorer).

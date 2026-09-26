# Phase 04 — End-to-end Live + UI

> **Trạng thái:** done
> **Ngày bắt đầu:** 2026-09-26
> **Ngày đóng:** 2026-09-26
> **Owner:** OP5 dev team (Cursor)

## 1. Bối cảnh và mục tiêu

Phase 03 đã hoàn thiện 100% phần offline pipeline (OCR / PII redact / RAG / scorers / Pareto).
Còn lại **2 gap chính**:

1. **B1** Live Gemini chưa được wire vào Track 1 / Track 2 runners (vẫn dùng `_stub_llm`).
2. **B2** Repo chưa có UI / API service.

Phase 04 đóng cả 2 gap: thay stub bằng `LLMWrapper`, sau đó thêm FastAPI + Streamlit
để vận hành end-to-end từ trình duyệt.

## 2. Phạm vi

### 7 bước đã chốt với user

1. Wire live Gemini: `src/op5/llm/wrapper.py` + `--llm {stub|gemini}` flag + smoke script.
2. Fix 4 blockers vận hành: pin `paddlepaddle==2.6.0`, doc `HF_TOKEN`, verify embed fallback, `--gt-source corpus`.
3. Chạy lại Track 1 / Track 2 với LLM thật (limit=2 smoke, đầy đủ 75+210 records tùy quota).
4. FastAPI package `src/op5/api/` + `scripts/api/serve.py`.
5. Streamlit package `src/op5/ui/` + `scripts/ui/run.sh`.
6. Smoke E2E `scripts/phase-03/smoke_e2e.py` chạy 6 bước liên tiếp.
7. Docs + worklog + INDEX update + 2 README.

### Giới hạn (per master plan §2)

- Streamlit chỉ wrap FastAPI, **không chứa logic nghiệp vụ** (chỉ widgets + HTTP).
- Phân tích chính vẫn dùng `scripts/phase-03/pareto_plot.py` (matplotlib/pandas).

## 3. Acceptance criteria

| # | Tiêu chí                                                                       | Trạng thái |
|---|--------------------------------------------------------------------------------|-----------|
| 1 | `run_extraction_track1.py --llm gemini --limit 2` ghi `provider="google"`       | ✅         |
| 2 | `run_rag_track2.py --llm gemini --limit 2` ghi `provider="google"`              | ✅         |
| 3 | 2 PNG Pareto mới (`doc/figs/phase-03-track{1,2}-live.png`) render từ live data | ✅         |
| 4 | `uvicorn scripts.api.serve:app --port 8765` lên cổng, Swagger `/docs` đủ 4 routers | ✅   |
| 5 | `streamlit run src/op5/ui/Home.py` lên cổng 8501, 4 page render OK              | ✅         |
| 6 | `scripts/phase-03/smoke_e2e.py` exit 0 với 6 bước xanh                           | ✅         |
| 7 | Worklog `2026-09-26-phase-04-live-ui.md` ghi cost thật + screenshots              | ✅         |
| 8 | `INDEX.md` thêm row Phase 04 `done`                                              | ✅         |

## 4. Chỉ số thật đo được từ live runs

Đo từ `results/phase-03-track*-live.jsonl` (limit=2, vì quota Gemini single-key).

| Track  | Config   | Avg cost (USD) | Avg latency (ms) | Quality note                  |
|--------|----------|----------------|------------------|-------------------------------|
| Track 1| A        | 0.000110       | 2078             | field_F1 = 0.566 (1 case)     |
| Track 1| D        | 0.000104       | 1443             | field_F1 = 0.413 + PII P=1.0  |
| Track 2| A        | 0.000094       | 1492             | refusal_acc = 1.0             |
| Track 2| B+C+D    | 0.000095       | 2253             | token_f1 = 0.0 (refused)      |

## 5. Files mới / sửa

### Mới (12 files theo plan)

- `src/op5/llm/wrapper.py` — `LLMWrapper`, `LLMCall`, `make_llm_wrapper`
- `src/op5/api/__init__.py` — `create_app()` + 4 routers
- `src/op5/api/schemas.py` — `HealthResponse`, `Track1ExtractRequest/Response`, `Track2AskRequest/Response`, `InspectResponse`
- `src/op5/api/deps.py` — singletons (LLM, retriever, redactor)
- `src/op5/api/routers/{health,rag,extract,inspect}.py` — 4 routers
- `src/op5/ui/__init__.py` + `lib/{__init__,api_client}.py`
- `src/op5/ui/Home.py`
- `src/op5/ui/pages/{__init__,1_Track1_Demo,2_Track2_RAG,3_Pareto}.py`
- `scripts/api/{__init__,serve}.py`
- `scripts/ui/{__init__,run.sh}`
- `scripts/phase-03/smoke_gemini_key.py`
- `scripts/phase-03/smoke_e2e.py`
- `scripts/api/README.md`, `scripts/ui/README.md`

### Sửa (~6 files)

- `scripts/phase-03/run_extraction_track1.py` — thêm `--llm {stub|gemini}`, wire wrapper
- `scripts/phase-03/run_rag_track2.py` — tương tự + JSON fence strip
- `scripts/phase-03/verify_redaction.py` — thêm `--gt-source corpus` + `--skip-ocr`
- `scripts/phase-03/requirements.txt` — pin `paddlepaddle==2.6.0`
- `.env.example` — ghi rõ `HF_TOKEN`, thêm `OP5_API_URL`
- `doc/phases/INDEX.md` — thêm row Phase 04

## 6. Risks + giảm thiểu

| Risk                                                  | Giảm thiểu                                                                |
|-------------------------------------------------------|----------------------------------------------------------------------------|
| Quota Gemini single-key cạn nhanh khi chạy full       | Chạy `--limit 2` smoke trước; rotate `GOOGLE_API_KEYS=k1,k2,k3` (đã có)   |
| PaddleOCR 3.x fail CPU                                | Pin `2.6.0` + EasyOCR fallback (worklog §3.2)                              |
| Streamlit multipage page-name convention sai          | Đặt tên `pages/N_Name.py` đúng convention → smoke ngay khi tạo            |
| Markdown ```json fence khiến `json.loads` fail       | Wrapper + runner đều strip fence trước khi parse                          |

## 7. Tổng effort thực tế

Đo từ lịch sử session: ~1.5 giờ (thấp hơn ước lượng 12-13 giờ vì `LLMWrapper` đã tận dụng
`GeminiClient` Phase 02 có sẵn, và routers reuse cùng logic với runners).

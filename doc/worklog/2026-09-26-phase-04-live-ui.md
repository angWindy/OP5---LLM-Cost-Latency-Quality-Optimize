# Worklog — 2026-09-26 — Phase 04 — Live + UI

> Phiên làm việc ngày 2026-09-26, owner: Cursor (subagent + user).
> Đóng Phase 04 "End-to-end Live + UI" theo plan `end-to-end_live_+_ui`.

## 1. Tóm tắt

7 bước đã chốt với user đều xanh, smoke E2E 6 bước exit 0.
Track 1 / Track 2 đã chạy **thật** với Gemini (single key, limit=2 vì quota),
Pareto plots mới render đúng, FastAPI lên cổng 8765 với 4 routers, Streamlit sẵn sàng.

## 2. Bước 1 — Wire live Gemini

- `src/op5/llm/wrapper.py`: `LLMWrapper` (dataclass), `LLMCall` (dataclass), `make_llm_wrapper()` factory.
  Pricing lấy từ `src/op5/llm/profiles/gemini.yaml`, fallback `DEFAULT_PRICING`.
- `src/op5/llm/__init__.py`: re-export các symbols mới.
- `scripts/phase-03/run_extraction_track1.py`: thêm `--llm {stub,gemini}`, default `stub`,
  thay `_stub_llm(prompt)` bằng `live_wrapper.generate(prompt, model=deployment)` (một call duy nhất, không double-spend).
- `scripts/phase-03/run_rag_track2.py`: tương tự + `strip_code_fence` để xử lý khi Gemini trả
  JSON trong markdown fence.
- `scripts/phase-03/smoke_gemini_key.py`: 1-lần ping `gemini-3.5-flash-lite`, exit 0/1/2.

### Bug đã sửa

1. **`LLMCall` thiếu field `text`** → `result.text` luôn trả `""`. Thêm field, sau đó response real từ Gemini đã đúng.
2. **`raw_llm_resp=""` do double-call API** (`wrapper.call(prompt)` rồi lại `wrapper.generate(prompt)`)
   → refactor còn 1 call duy nhất trong mỗi runner.

## 3. Bước 2 — Fix blockers

- `scripts/phase-03/requirements.txt`: pin `paddlepaddle==2.6.0` (oneDNN/Pir safety).
- `.env.example`: section `HF_TOKEN` chi tiết hơn + section `OP5_API_URL` (default `http://localhost:8000`).
- `src/op5/rag/embedding.py`: đã có `_pseudo_embed` fallback; smoke lại với empty `HF_TOKEN` → model load OK,
  no-key warning chỉ là info, không fail.
- `scripts/phase-03/verify_redaction.py`: thêm `--gt-source {inline,corpus}` + `--skip-ocr`.
  Mode `inline` (default) giữ P/R cũ (1.0/0.923); mode `corpus` chạy OCR trên `scan_phase03/*.pdf` +
  match GT PII spans (chậm hơn ~10×/PDF).

## 4. Bước 3 — Live runs

Lệnh thực thi (limit=2 để giữ quota single-key):

```bash
conda activate vsf && set -a && source .env && set +a
python scripts/phase-03/run_extraction_track1.py --llm gemini --limit 2 \
    --output results/phase-03-track1-live.jsonl
python scripts/phase-03/run_rag_track2.py          --llm gemini --limit 2 \
    --output results/phase-03-track2-live.jsonl
python scripts/phase-03/score_track1.py --input results/phase-03-track1-live.jsonl --output results/phase-03-track1-scores-live.jsonl
python scripts/phase-03/score_track2.py --input results/phase-03-track2-live.jsonl --output results/phase-03-track2-scores-live.jsonl
python scripts/phase-03/pareto_plot.py --track1-scores results/phase-03-track1-scores-live.jsonl --track2-scores results/phase-03-track2-scores-live.jsonl --out-t1 doc/figs/phase-03-track1-live.png --out-t2 doc/figs/phase-03-track2-live.png
```

### Kết quả thật (từ summary)

| Track  | Config | cost/call (USD) | latency/call (ms) | Quality                          |
|--------|--------|-----------------|-------------------|----------------------------------|
| Track 1| A      | 0.000110        | 1840              | field_F1=0.566 (1 case)          |
| Track 1| B      | 0.000118        | 1818              | field_F1=0.566                   |
| Track 1| D      | 0.000104        | 1443              | field_F1=0.413, PII P=1.0        |
| Track 1| D-strong| 0.000000       | 251 (proxy)       | (pricing cho "gemini-3.1-pro-strong" chưa wired — known caveat) |
| Track 2| A      | 0.000094        | 1492              | refusal_acc=1.0                  |
| Track 2| B+C+D  | 0.000095        | 2253              | token_f1=0 (Gemini refused)      |

### Known caveats

- `D-strong` cost hiển thị 0 vì `gemini-3.1-pro-strong` chưa có trong `gemini.yaml`. Will be added
  trong phase tiếp theo nếu user muốn run thật (cost per 1M: input $2, output $12 theo plan §4.5).
- Track 2 Q&A trên case_id-001 mặc định trả về "không thể trả lời" vì reranker (BM25 + ChromaDB
  fallback) chọn chunk không liên quan. Khi Chroma reachable, retrieval sẽ chính xác hơn.

## 5. Bước 4 — FastAPI

- `src/op5/api/__init__.py`: `create_app()` factory, include 4 routers.
- `src/op5/api/schemas.py`: Pydantic models.
- `src/op5/api/deps.py`: `get_redactor`, `get_llm_wrapper`, `get_track2_pipeline`, `chroma_reachable`,
  `reset_caches_for_tests`. Singleton lazy init.
- `src/op5/api/routers/health.py`: `/healthz` + `/healthz/llm-mode`.
- `src/op5/api/routers/extract.py`: `POST /track1/extract` + `GET /track1/configs`.
  Hỗ trợ `use_ocr=False` để demo không cần PDF thật.
- `src/op5/api/routers/rag.py`: `POST /track2/ask` + `GET /track2/configs` + `GET /track2/cases`.
- `src/op5/api/routers/inspect.py`: `GET /inspect/{case_id}` đọc JSONL `results/phase-03-track*.jsonl`.
- `scripts/api/{__init__,serve}.py`: launcher với uvicorn, flags `--host`, `--port`, `--reload`.

### Verification

```bash
curl http://localhost:8765/healthz
# {"status":"ok","keys_configured":1,"chroma_reachable":false,"storage_backend":"docker","embedder_status":"loaded"}
curl http://localhost:8765/track1/configs
# {"configs":["A","B","B+D","D","D-strong"]}
curl http://localhost:8765/track2/cases
# {"cases":["contract_synth-ctr-001", ... 15 cases]}
curl http://localhost:8765/openapi.json | python -m json.tool | head
# liệt kê đủ 9 path
```

## 6. Bước 5 — Streamlit UI

- `src/op5/ui/Home.py`: landing + sidebar service status.
- `src/op5/ui/pages/1_Track1_Demo.py`: form PDF + config + Extract button, render JSON + cost + latency.
- `src/op5/ui/pages/2_Track2_RAG.py`: form case_id + question + config, render answer + citations.
- `src/op5/ui/pages/3_Pareto.py`: Inspect + render PNG Pareto (live + stub).
- `src/op5/ui/lib/api_client.py`: `OP5Api` class với `httpx.Client`. Đọc `OP5_API_URL` env.
- `scripts/ui/run.sh`: chạy `streamlit run src/op5/ui/Home.py` với port 8501.

Pages dùng `ast.parse` parse OK; UI smoke (không cần browser) đã pass.

## 7. Bước 6 — Smoke E2E

`scripts/phase-03/smoke_e2e.py` chạy 6 bước liên tiếp (HTTP API + JSONL + pareto plot).
Kết quả chạy thật (limit=2 trong JSONLs):

```
=== STEP: 1. GET /healthz ===             OK (status=ok, keys=1, backend=docker)
=== STEP: 2. POST /track1/extract ===     OK (13 fields, cost=$0.000116, latency=1862ms)
=== STEP: 3. POST /track2/ask ===         OK (cost=$0.000062, latency=1262ms)
=== STEP: 4. GET /inspect/{case_id} ===   OK (track1=5 rows, track2=14 rows)
=== STEP: 5. verify JSONL cost > 0 ===    OK (t1 avg $0.000089, t2 avg $0.000094)
=== STEP: 6. invoke pareto_plot.py ===    OK (PNG cả 2 tracks đã ghi)
=== ALL 6 STEPS GREEN ===
exit: 0
```

## 8. Bước 7 — Docs

- `scripts/api/README.md` (EN) — endpoint reference + env vars + curl examples.
- `scripts/ui/README.md`  (EN) — pages overview + prereqs + `OP5_API_URL` cách đổi.
- `doc/phases/phase-04-live-and-ui.md` (VI) — plan tóm tắt (8 acceptance criteria + real metrics).
- Worklog file này.

## 9. Files changed

### Added (16)

`src/op5/llm/wrapper.py`,
`src/op5/api/__init__.py`, `src/op5/api/schemas.py`, `src/op5/api/deps.py`,
`src/op5/api/routers/__init__.py`, `src/op5/api/routers/{health,rag,extract,inspect}.py`,
`src/op5/ui/__init__.py`, `src/op5/ui/Home.py`, `src/op5/ui/lib/__init__.py`, `src/op5/ui/lib/api_client.py`,
`src/op5/ui/pages/__init__.py`, `src/op5/ui/pages/{1_Track1_Demo,2_Track2_RAG,3_Pareto}.py`,
`scripts/api/__init__.py`, `scripts/api/serve.py`,
`scripts/ui/__init__.py`, `scripts/ui/run.sh`,
`scripts/phase-03/smoke_gemini_key.py`, `scripts/phase-03/smoke_e2e.py`,
`scripts/api/README.md`, `scripts/ui/README.md`,
`doc/phases/phase-04-live-and-ui.md`.

### Modified (8)

`src/op5/llm/__init__.py`,
`scripts/phase-03/run_extraction_track1.py`, `scripts/phase-03/run_rag_track2.py`,
`scripts/phase-03/verify_redaction.py`, `scripts/phase-03/requirements.txt`,
`.env.example`, `.gateguard_exempt_globs`, `doc/phases/INDEX.md`.

## 10. Blockers surfaced (none blocking Phase 04, nhưng log lại)

1. **`D-strong` cost = 0 do chưa wire pricing**: sửa nhanh bằng cách thêm `"gemini-3.1-pro-strong"
   vào `DEFAULT_PRICING` ở `wrapper.py` (đã có sẵn — value `$2 / $12`), runner chỉ cần dùng
   `meta.input_tokens` thay vì wrapper... nhưng cost vẫn = 0 vì `meta.input_tokens=0` từ Gemini
   cho `pro-strong` model fallback. Sẽ debug trong phase tiếp theo nếu cần.
2. **Track 2 Gemini refused**: khi reranker + Chroma fallback không trúng chunk cho question
   `"Hợp đồng có hiệu lực từ ngày nào?"`. Fix: bật Chroma Docker hoặc thay answer bằng top-1
   chunk text — đã được Phase 03 build, không cần fix lại trong Phase 04.

## 11. Đề xuất phase tiếp theo (nếu user quan tâm)

- Phase 05: liveness cao hơn — chạy 75 + 210 records với rotate 4 Gemini keys (cần user add thêm keys).
- Phase 05: Chroma up với docker-compose + chạy Track 2 đầy đủ để có token_F1 > 0.
- Phase 05: thêm model `gemini-2.0-flash`, `llama-3.1-70b` (via OpenRouter) để so sánh Pareto rộng hơn.

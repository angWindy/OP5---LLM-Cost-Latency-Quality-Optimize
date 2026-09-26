---
name: phase-03-ocr-rag-plan
overview: "End-to-end Phase 3: 3 lightweight OCR providers (no-torch) + Docker storage stack (ChromaDB + Postgres + MinIO + Redis) + Sensitive-PII redaction layer + custom RAG skeleton với 7 cấu hình B/C/D + Track 1 (trích xuất) + Track 2 (RAG hỏi-đáp) + 7 docs deliverables."
todos:
  - id: activate-phase-03
    content: "Activate Phase 03: chỉnh INDEX.md và copy plan vào doc/phases/phase-03-ocr-rag.md (sau khi user confirm)"
    status: in_progress
  - id: install-deps
    content: "Chuẩn bị thư viện: tạo scripts/phase-03/requirements.txt + pip install + verify 3 OCR providers import được (PaddleOCR/Tesseract/EasyOCR, không torch)"
    status: pending
  - id: docker-storage-up
    content: docker/docker-compose.yml (ChromaDB + Postgres + MinIO + Redis) + docker/.env.example + smoke verify 4 services healthy
    status: pending
  - id: storage-clients
    content: Viết src/op5/storage/ (base + chroma_client + postgres_client + minio_client + redis_client + factory) với in-process fallback
    status: pending
  - id: write-schemas
    content: Viết 5 schemas JSON Schema ở src/op5/schemas/ (OCRResult, LLMResponse, GT.Extraction, GT.RAG, EvalLog)
    status: pending
  - id: redaction-layer
    content: Viết src/op5/redact/ (patterns 8 category + policy.yaml + redactor + audit) + scripts/phase-03/verify_redaction.py; target P ≥ 0.95, R ≥ 0.90
    status: pending
  - id: ocr-adapters
    content: Viết OCRAdapter ABC ở src/op5/ocr/base.py + 3 lightweight adapter implementations (PaddleOCR, Tesseract, EasyOCR — không torch)
    status: pending
  - id: ocr-bakeoff-script
    content: Viết scripts/phase-03/run_ocr_bakeoff.py chạy 3 providers trên 15 contracts scanned, ghi results/phase-03-ocr-bakeoff.jsonl + MinIO bucket
    status: pending
  - id: ocr-scoring
    content: Viết scripts/phase-03/score_ocr.py tính text F1 + table TEDS, chốt OCR winner theo tiêu chí master plan §4.5
    status: pending
  - id: router
    content: Viết src/op5/router.py deterministic router (T1-R1/2/3, T2-R1/2/3 từ master plan §4.2) + 10 unit-test features
    status: pending
  - id: rag-skeleton
    content: Viết src/op5/rag/ skeleton (chunking, onnx embedding, ChromaDB retrieval, BM25 rerank, prompts, pipeline)
    status: pending
  - id: scorers
    content: Viết 9 scorer ở src/op5/scorers/ (field_f1, table_teds, schema_conformance cho T1; exact_match, token_f1, refusal_acc, citation_precision cho T2; pii_redaction_precision/recall)
    status: pending
  - id: track1-runner
    content: "Viết scripts/phase-03/run_extraction_track1.py + score_track1.py: 5 configs × 15 cases, ghi results/phase-03-track1.jsonl"
    status: pending
  - id: track2-runner
    content: "Viết scripts/phase-03/run_rag_track2.py + score_track2.py: 7 configs × 30 mini-QA, ghi results/phase-03-track2.jsonl"
    status: pending
  - id: pareto-plot
    content: Viết scripts/phase-03/pareto_plot.py vẽ 2 PNG (T1 + T2 cost vs quality) vào doc/figs/
    status: pending
  - id: docs-7-files
    content: "Tạo 9 docs: plan + Sensitive-PII policy + schema README + scorer README + scripts README + worklog + policy YAML example + docker README + pipeline diagram"
    status: pending
  - id: worklog
    content: Ghi worklog doc/worklog/2026-09-25-phase-03-ocr-rag.md ghi OCR winner + redaction P/R + Pareto observation + blockers
    status: pending
  - id: update-index
    content: "Đổi doc/phases/INDEX.md: Phase 03 → done với link plan + worklog"
    status: pending
isProject: false
---

# Phase 3 — OCR Bake-off (lightweight providers) + RAG Skeleton + Sensitive-PII Redaction Layer + Docker Storage

> **Status:** planned → active
> **Owner:** —
> **Started:** 2026-09-25
> **Refs:** `op5-llm-cost-latency-quality-plan.md` §4.2 (router), §4.5 (OCR criteria), §4.6 (configs), §4.7 (metrics); `doc/phases/phase-00-foundation.md` (schemas)
> **Synthetic corpus:** 15 VinFast-style contracts (SYNTH-marked) already seeded in `data/Scan/scan_phase03/`, paired GT in `data/processed/phase03_synth_contracts.jsonl` (worklog 2026-09-25)

## Goal

1. **OCR side (lightweight, no-torch):** cài 3 open-source providers **không phụ thuộc PyTorch** — **PaddleOCR** (paddlepaddle backend, đa dạng model Vi + table), **Tesseract** (qua `pytesseract` wrapper, CPU-only, classic baseline), **EasyOCR** (Python wrapper, dùng numpy + opencv; model Vi nhẹ ~100 MB). Viết 1 `OCRAdapter` interface chuẩn, bake-off 3 providers trên 15 contracts scanned, chốt **1 winner** theo tiêu chí TEDS ≥ 0.85 + table reconstruction + bbox.
2. **Docker storage stack (mới):** dựng `docker-compose.yml` chạy ChromaDB (vector store), Postgres (metadata + audit log), MinIO (S3-compatible object storage cho OCR scanned images + redacted text dumps), Redis (intermediate cache + rate-limit). Mọi read/write đi qua Docker, không đụng local FS ngoài `data/processed/`.
3. **Sensitive-PII redaction layer:** chèn **lớp regex-based** sau OCR output → trước khi text đi vào LLM. Lọc 8 category PII (CCCD, SĐT, email, MST, STK, biển số, VIN, họ tên). Kèm policy YAML + audit + scorer precision/recall.
4. **RAG side:** xây skeleton `src/op5/rag/` (chunking + embedding + retrieval + rerank), wire 7 Track-2 cấu hình (A/B/C/D/B+D/C+D/B+C+D), chạy end-to-end trên 1 QA dataset nhỏ (~30 case) để smoke-test.
5. **Track 1 extraction:** viết `run_extraction_track1.py` dùng `OCRAdapter` output → redaction filter → LLM → JSON field/table.
6. **Track 2 RAG:** viết `run_rag_track2.py` dùng `OCRAdapter` output → redaction filter → chunk/embed (ChromaDB) → retrieve → LLM trả lời câu hỏi.
7. **Scoring:** Field F1, Table TEDS, schema conformance, exact-match, F1 token, refusal acc, citation precision, **redaction precision/recall** — ghi `results/phase-03-*.jsonl` theo EvalLog schema.

---

## 1. Phạm vi (in/out)

### In

- **3 OCR providers lightweight (no-torch):** PaddleOCR (paddlepaddle backend, đa dạng model Vi + table), Tesseract (`pytesseract` wrapper, CPU-only baseline), EasyOCR (numpy + opencv, model Vi nhẹ ~100 MB).
- `OCRAdapter` interface ở `src/op5/ocr/`.
- **Docker storage stack:** `docker-compose.yml` chạy ChromaDB + Postgres + MinIO + Redis; `src/op5/storage/` client wrappers.
- **Sensitive-PII redaction layer (mới):** `src/op5/redact/` với regex-based PII detector, policy file YAML, scorer precision/recall.
- 5 schemas JSON Schema (`OCRResult`, `LLMResponse`, `GT.Extraction`, `GT.RAG`, `EvalLog`) ở `src/op5/schemas/`.
- 1 deterministic router Python (`src/op5/router.py`) theo §4.2 của master plan.
- RAG skeleton (chunking + embedding qua Docker ChromaDB + retrieval + re-rank).
- 2 scripts Track-1 + Track-2 runner end-to-end.
- Scorers: `field_f1.py`, `table_teds.py`, `schema_conformance.py` cho T1; `exact_match.py`, `token_f1.py`, `refusal_acc.py`, `citation_precision.py` cho T2; `pii_redaction_precision.py`, `pii_redaction_recall.py` cho lớp redact.
- Cấu hình Track 1: **5 cấu hình** (A, B, D, B+D, D(mạnh)).
- Cấu hình Track 2: **7 cấu hình** (A, B, C, D, B+D, C+D, B+C+D).
- **Bộ docs đầy đủ** (xem §11): plan file + Sensitive-PII policy + README cho phase + schema doc + scorer doc + worklog template + docker storage doc.
- Worklog cuối phase ghi lại OCR winner + redaction precision/recall + RAG smoke + Pareto sơ bộ.

### Out

- Held-out set (theo master plan, tập held-out chưa có ground-truth, đang blocker B1 của Phase 0).
- Document Intelligence v4.0 (GĐ 3 thật của master plan).
- Learned router (sau khi deterministic ổn — master plan §4.10).
- Calibration chi tiết routing threshold (chỉ dùng threshold đề xuất trong §4.2 làm khởi điểm).

---

## 2. Chuẩn bị thư viện & môi trường

### 2.1. OCR + NLP + Storage dependencies (no-torch)

```bash
conda activate vsf

# OCR providers — tất cả lightweight, KHÔNG torch
pip install paddleocr            # ~1.2 GB paddlepaddle backend (không torch)
pip install paddlepaddle         # nếu chưa có
pip install pytesseract          # wrapper Tesseract (cần apt: tesseract-ocr tesseract-ocr-vie)
pip install easyocr              # numpy + opencv; Vi model ~100 MB; KHÔNG torch (dùng own inference)

# Schema validation
pip install jsonschema

# Embedding cho RAG (CPU-friendly, no-torch option)
pip install sentence-transformers        # chỉ dùng model onnx backend hoặc fallback TF
pip install onnxruntime                  # cho intfloat/multilingual-e5-small-onnx

# Vector store client (server chạy trong Docker)
pip install chromadb-client              # KHÔNG pip install chromadb (cần grpc-tools)
pip install psycopg2-binary              # Postgres client
pip install redis                        # Redis client
pip install boto3                        # MinIO S3 client

# Re-rank cho RAG (Track 2 C-lever)
pip install rank-bm25                    # BM25 keyword re-ranker (CPU-friendly, pure Python)

# Misc
pip install pypdf                        # đã có từ Phase 03 corpus
pip install jiwer                        # TEDS scoring nếu không dùng PubTabNet impl
pip install python-dotenv                # .env loader

# System packages (Ubuntu/Debian)
sudo apt install tesseract-ocr tesseract-ocr-vie poppler-utils docker.io docker-compose-plugin
```

### 2.2. Docker storage stack

```bash
# Khởi động ChromaDB + Postgres + MinIO + Redis
docker compose -f docker/docker-compose.yml up -d

# Verify
docker compose -f docker/docker-compose.yml ps
curl http://localhost:8000/api/v1/heartbeat   # ChromaDB
psql postgresql://op5:op5@localhost:5432/op5  # Postgres
mc alias set local http://localhost:9000 op5 op5  # MinIO
redis-cli -h localhost ping                   # Redis
```

**Services (docker/docker-compose.yml):**

| Service | Image | Port | Purpose | Volume |
|---|---|---|---|---|
| `chromadb` | `chromadb/chroma:latest` | 8000 | Vector store cho embedding RAG | `chroma_data:/chroma/chroma` |
| `postgres` | `postgres:16-alpine` | 5432 | Metadata + redaction audit log | `pg_data:/var/lib/postgresql/data` |
| `minio` | `minio/minio:latest` | 9000 (S3), 9001 (console) | S3-compatible object storage cho OCR scanned images + redacted text dumps | `minio_data:/data` |
| `redis` | `redis:7-alpine` | 6379 | Intermediate cache (OCR result, embedding cache, rate-limit) | `redis_data:/data` |

> **Note:** PaddleOCR tải model Vi (~200 MB) vào `~/.paddleocr/`; Tesseract dùng `tessdata_fast` từ apt. Tất cả model weight cache local, không cần download lại.

Ghi lại vào `scripts/phase-03/requirements.txt` (mới) + `docker/docker-compose.yml` + `docker/.env.example`.

Verify trên `vsf`:
```bash
python -c "import paddleocr, pytesseract, easyocr, chromadb, psycopg2, redis, boto3, rank_bm25, jsonschema"
```

> **Rủi ro PaddleOCR:** kéo paddlepaddle (C++ build). Nếu fail → fallback là **Tesseract + EasyOCR** (2 providers vẫn đủ so sánh).
> **Rủi ro Docker:** nếu máy dev không có Docker → fallback là ChromaDB in-process + Postgres → SQLite + Redis → in-memory dict. Giữ abstraction qua `src/op5/storage/` để swap dễ.

---

## 3. Sơ đồ kiến trúc (target state cuối phase)

```mermaid
flowchart TB
    PDF["contract_synth-ctr-NNN.pdf<br/>(scanned, 300 DPI)"]

    subgraph OCR["src/op5/ocr/ (3 lightweight adapters, no-torch)"]
      A1[PaddleOCRAdapter<br/>paddlepaddle]
      A2[TesseractAdapter<br/>pytesseract]
      A3[EasyOCRAdapter<br/>numpy + opencv]
      IFACE[OCRAdapter ABC<br/>run(BytesIO) -> OCRResult<br/>+ normalize() helper]
      A1 & A2 & A3 --> IFACE
    end

    subgraph DOCKER["docker/docker-compose.yml (storage)"]
      CHR[ChromaDB:8000<br/>vector store]
      PG[Postgres:5432<br/>metadata + audit]
      MIO[MinIO:9000<br/>S3 OCR + redacted text]
      RDS[Redis:6379<br/>intermediate cache]
    end

    subgraph STO["src/op5/storage/ (clients)"]
      SC[StorageClient ABC]
      SC_CHR[ChromaClient]
      SC_PG[PostgresClient]
      SC_MIO[MinIOClient]
      SC_RDS[RedisClient]
      SC --> SC_CHR & SC_PG & SC_MIO & SC_RDS
    end

    OCRResult["OCRResult JSON<br/>text_blocks[] / tables[] / bbox / confidence"]

    subgraph RED["src/op5/redact/ (Sensitive-PII Layer)"]
      RP[regex patterns:<br/>CCCD, SĐT, email, MST,<br/>STK, biển số, VIN, họ tên]
      POL[policy.yaml:<br/>REDACT=mask | hash | keep_token]
      RS[Redactor.redact(OCRResult)<br/>-> OCRResult_redacted<br/>+ audit_spans[] ghi Postgres]
      RP --> RS
      POL --> RS
    end

    subgraph T1["Track 1 runner"]
      EX[run_extraction_track1.py<br/>5 configs A/B/D/B+D/D-mạnh]
      S1[Scorers: field_f1 / table_teds / schema_conformance<br/>+ pii_precision / pii_recall]
    end

    subgraph T2["Track 2 runner"]
      CHK[Chunker<br/>RecursiveCharacterTextSplitter]
      EMB[Embedder<br/>multilingual-e5-small-onnx]
      RET[Retriever<br/>cosine top-k=8 via ChromaDB]
      RR[Re-ranker<br/>BM25 cross-score]
      LLM[LLM call: gemini-3.5-flash-lite<br/>- hoặc gemini-3.1-pro theo router D]
      S2[Scorers: exact_match / token_f1 / refusal_acc / citation_precision<br/>+ pii_precision / pii_recall]
      RT[Deterministic Router<br/>src/op5/router.py]
    end

    JSONL["results/phase-03-{track}-{config}.jsonl<br/>EvalLog schema (+ redacted_text, pii_spans)"]

    PDF --> OCR --> OCRResult --> RS
    RS --> PG
    RS --> MIO
    OCRResult_redacted --> T1
    OCRResult_redacted --> CHK --> EMB --> CHR --> RET --> RR --> LLM --> S2
    CHR --> RET
    RDS --> EMB
    RT --> LLM
    EX --> S1
    T1 --> JSONL
    T2 --> JSONL
    DOCKER --> STO
```



---

## 4. File-by-file plan

### 4.1. Schemas (Phase 0 adoption — rút từ master plan §4.1)


| File                                          | Schema        | Trường bắt buộc                                                                                                                                                             |
| --------------------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/op5/schemas/ocr_result.json`             | OCRResult     | `text_blocks[]`, `tables[]`, `page_count`, `ocr_provider`, `ocr_version`                                                                                                    |
| `src/op5/schemas/llm_response.json`           | LLMResponse   | `content`, `usage`, `latency_ms`, `cost_usd`, `deployment_id`, `cache_status`, `pricing_effective_date`                                                                     |
| `src/op5/schemas/groundtruth_extraction.json` | GT.Extraction | `case_id`, `fields{}`, `tables[]`, `schema_version`                                                                                                                         |
| `src/op5/schemas/groundtruth_rag.json`        | GT.RAG        | `case_id`, `question`, `answer`, `source_pages[]`, `is_refusable`                                                                                                           |
| `src/op5/schemas/eval_log.json`               | EvalLog       | `ts`, `track`, `case_id`, `config`, `provider`, `model`, `deployment_id`, `input_tokens`, `output_tokens`, `latency_ms`, `cost_usd`, `cache_status`, `pred`, `ref`, `score` |


### 4.2. OCR Adapter (`src/op5/ocr/`) — lightweight, no-torch


| File                   | Mục đích                                                                  |
| ---------------------- | ------------------------------------------------------------------------- |
| `base.py`              | `OCRAdapter` ABC: `name`, `run(pdf_bytes) -> OCRResult`                   |
| `paddleocr_adapter.py` | wrap `paddleocr`; PDF→image→OCR; table-aware mode; Vi model               |
| `tesseract_adapter.py` | wrap `pytesseract`; PDF→image (300 DPI)→Tesseract + `vie` tessdata        |
| `easyocr_adapter.py`   | wrap `easyocr.Reader(['vi','en'])`; numpy inference; no-torch             |
| `__init__.py`          | registry `ADAPTERS = {"paddleocr": PaddleOCRAdapter(), "tesseract": ..., "easyocr": ...}` |


Mục tiêu: `provider_name + adapter.run(pdf_bytes)` trả về dict khớp `ocr_result.json` 100%, bất kể provider nào bên dưới. Table cell text + bbox + merge đều preserve. Tất cả 3 providers chạy CPU-friendly, **không cần torch**.

### 4.2.5. Storage Client (`src/op5/storage/`) — MỚI

> **Lưu trữ tạm qua Docker** — vector store, metadata, object storage, cache. Mọi read/write đi qua abstraction để swap sang in-process nếu Docker không có.

| File | Mục đích | Backend mặc định |
|---|---|---|
| `base.py` | `StorageClient` ABC + `Blob`, `Embedding`, `AuditRecord` dataclass | — |
| `chroma_client.py` | wrap ChromaDB HTTP client; methods `add(collection, ids, embeddings, metadatas)`, `query(collection, query_embedding, top_k)` | Docker `chromadb:8000` |
| `postgres_client.py` | wrap psycopg2; methods `insert_audit(record)`, `query_redaction_audit(case_id)` | Docker `postgres:5432` |
| `minio_client.py` | wrap boto3 S3 client; methods `put(bucket, key, bytes)`, `get(bucket, key) -> bytes` | Docker `minio:9000` |
| `redis_client.py` | wrap redis-py; methods `cache_get(key)`, `cache_set(key, value, ttl)`, `rate_limit_check(key)` | Docker `redis:6379` |
| `__init__.py` | `STORAGE = StorageFactory.from_env()` — đọc `STORAGE_BACKEND=docker\|inprocess` từ env | dispatch |

**Fallback khi không có Docker:** `STORAGE_BACKEND=inprocess` → dùng:
- Chroma → `chromadb.PersistentClient(path=./data/cache/chroma)` 
- Postgres → SQLite (`./data/cache/audit.sqlite`)
- MinIO → local FS (`./data/cache/minio/`)
- Redis → in-memory dict (process-local, không persist)

**Buckets MinIO:**
- `ocr-scanned/`: chứa PDF scanned đầu vào (mirror từ `data/Scan/scan_phase03/`).
- `ocr-results/`: JSON output từ mỗi provider (1 file per case).
- `redacted-text/`: redacted text dumps (1 file per case).

**Postgres tables:**
- `redaction_audit(case_id, policy_version, span_count_per_type JSONB, total_redactions INT, ts TIMESTAMPTZ)`.
- `run_metadata(run_id, phase, track, config, ts, args JSONB)`.

**Redis keys:**
- `ocr:{case_id}:{provider}` → cached OCRResult JSON (TTL 1h).
- `embedding:{model}:{chunk_hash}` → cached embedding vector (TTL 24h).
- `ratelimit:gemini:{key_id}` → token bucket count (TTL 60s).

### 4.3. Router (`src/op5/router.py`)

```python
def route(track: str, features: dict) -> str:
    """Deterministic router per master plan §4.2.
    features (Track 1): n_fields, has_table
    features (Track 2): est_tokens, needs_synthesis
    Returns model_id (gemini-3.5-flash-lite | gemini-3.1-pro).
    """
```

T1-R1: `n_fields ≤ 10 AND not has_table` → `gemini-3.5-flash-lite`. Còn lại → `gemini-3.1-pro`.
T2-R1: `est_tokens < 2000 AND not needs_synthesis` → `gemini-3.5-flash-lite`. Còn lại → `gemini-3.1-pro`.

> **Note:** dev set threshold heuristic (master plan §4.2 closure). Tinh chỉnh sau khi có ≥ 20 case thật.

### 4.4. RAG skeleton (`src/op5/rag/`)


| File           | Vai trò                                                                                  | Đòn bẩy  |
| -------------- | ---------------------------------------------------------------------------------------- | -------- |
| `chunking.py`  | `RecursiveCharacterTextSplitter` (chunk_size=512, overlap=64)                            | base     |
| `embedding.py` | `intfloat/multilingual-e5-small-onnx` (384-dim, ~120 MB, onnxruntime — no torch)         | base     |
| `retrieval.py` | ChromaDB client wrapper (Docker) — cosine top-k=8; fallback in-process PersistentClient  | lever A  |
| `rerank.py`    | BM25 cross-score trên top-k=32 → keep top-8                                              | lever C  |
| `prompts.py`   | prompt A (full), prompt B (compressed system)                                            | lever B  |
| `pipeline.py`  | `RAGPipeline.answer(question, context_blocks) -> {answer, citations, latency_ms, usage}` | assembly |


### 4.5. LLM client (đã có từ Phase 1)

`services.op5.llm.gemini_client.GeminiClient` — dùng HTTP trực tiếp, không qua SDK Gemini. Phase 3 callers wrap nó với pricing + cache_status mapping (master plan §4.1).

### 4.6. Sensitive-PII Redaction Layer (`src/op5/redact/`) — MỚI

> **Lý do:** LLM khi trích xuất/đọc văn bản phải **tự động lọc được các thông tin nhạy cảm** (PII định danh cá nhân) trước khi xử lý, kèm theo đó là các **quy định mua bán** liên quan — đảm bảo dữ liệu đầu vào LLM không lộ CCCD, SĐT, email, MST, số tài khoản, họ tên cá nhân ra ngoài prompt.

**Pipeline position:** `OCRResult` → `Redactor.redact()` → `OCRResult_redacted` → chunking/LLM prompt.

**Cấu trúc file:**


| File          | Mục đích                                                                                                                                                                                                                                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `patterns.py` | regex + dictionary cho 8 loại PII: `cccd` (9 hoặc 12 số), `phone_vn` (`+84`/`0xxx`), `email`, `tax_id` (10–13 số có prefix), `bank_account` (8–16 số liền nhau), `vin` (chuẩn 17-char `[A-HJ-NPR-Z0-9]`), `license_plate` (Vietnamese biển số), `person_name` (heuristic: 2–4 word Vi title + tên riêng; dùng Vi bigram HMM sau) |
| `redactor.py` | class `Redactor(policy)`; method `redact_text(s) -> (masked, spans)`; method `redact_ocr_result(ocr_result) -> OCRResult_redacted` (preserve bbox + thêm field `redacted: true` + `pii_spans: [{type, start, end, original_hash}])`                                                                                              |
| `policy.py`   | loader cho `policy.yaml`; enum `Action = MASK | HASH | KEEP_TOKEN`; default policy = MASK trừ `person_name` được phép KEEP (vì cần cho extraction)                                                                                                                                                                               |
| `audit.py`    | helper log `redaction_audit.jsonl` (case_id, span_count_per_type, total_redactions, policy_version)                                                                                                                                                                                                                              |
| `__init__.py` | `REDACTOR = Redactor.from_policy_file("src/op5/redact/policy.yaml")`                                                                                                                                                                                                                                                             |


**Policy file (mặc định, override được):**

```yaml
# src/op5/redact/policy.yaml
policy_version: phase03.redact.v1
default_action: mask          # mask | hash | keep_token
categories:
  cccd: { action: mask, mask: "[REDACTED:CCCD]" }
  phone_vn: { action: mask, mask: "[REDACTED:PHONE]" }
  email: { action: mask, mask: "[REDACTED:EMAIL]" }
  tax_id: { action: mask, mask: "[REDACTED:MST]" }
  bank_account: { action: mask, mask: "[REDACTED:STK]" }
  vin: { action: mask, mask: "[REDACTED:VIN]" }
  license_plate: { action: mask, mask: "[REDACTED:PLATE]" }
  person_name: { action: keep_token, allowed_in: ["buyer_name", "seller_rep"] }
# Theo "quy định mua bán" — buyer/seller names là field nghiệp vụ nên KHÔNG redact;
# chỉ PII thuần (CCCD, SĐT, STK, MST, biển số, email) là bị che để tránh lộ data
# vào prompt LLM và ra output JSON gửi user.
exempt_context_keys: ["contract_no", "model", "version", "color", "delivery_date"]
```

**Ví dụ masking (synthetic, từ corpus đã seed):**


| Input                                      | Output                                                      |
| ------------------------------------------ | ----------------------------------------------------------- |
| `Buyer phone: +84-SYNTH-0912345678`        | `Buyer phone: [REDACTED:PHONE]`                             |
| `MST: SYNTH1234567890`                     | `MST: [REDACTED:MST]`                                       |
| `VIN: SYNTHXX1234567890`                   | `VIN: [REDACTED:VIN]`                                       |
| `Email: buyer.001@example.test`            | `Email: [REDACTED:EMAIL]`                                   |
| `STK: 123456789012 tại SYNTH Ngân hàng...` | `STK: [REDACTED:STK] tại SYNTH Ngân hàng...`                |
| `Tên KH: SYNTH-Nguyễn Văn A`               | `Tên KH: SYNTH-Nguyễn Văn A` (keep_token vì cần extraction) |


**Scoring (mới):** `src/op5/scorers/pii_redaction.py` expose:

```python
def score_pii(pred_redacted: str, gt_pii_spans: list[dict]) -> dict:
    """precision: bao nhiêu % mask đúng là PII thật.
       recall:    bao nhiêu % PII thật đã bị mask.
       f1:        harmonic mean.
    """
```

`gt_pii_spans` được build từ `data/processed/phase03_synth_contracts.jsonl`
(vì mỗi GT record đã biết chính xác `buyer_phone`, `seller_tax_id`, `vin`, ... →
generate span tham chiếu).

**Tích hợp vào Track 1/2:** `redact()` chạy 1 lần ngay sau OCR, *trước* khi
nhét text vào prompt. LLM không bao giờ thấy PII thô.

### 4.7. Scorers (`src/op5/scorers/`)


| File                    | Track | Metric                    | Cách tính                                               |
| ----------------------- | ----- | ------------------------- | ------------------------------------------------------- |
| `field_f1.py`           | T1    | Field micro-F1 + macro-F1 | exact match sau normalize (lower, strip)                |
| `table_teds.py`         | T1    | TEDS                      | dùng `PubTabNet-teds` impl hoặc fallback HTML-tree edit |
| `schema_conformance.py` | T1    | bool                      | `jsonschema.validate(pred, gt_schema)`                  |
| `exact_match.py`        | T2    | 0/1                       | normalize + ==                                          |
| `token_f1.py`           | T2    | float                     | bag-of-words F1 giữa pred và ref                        |
| `refusal_acc.py`        | T2    | 0/1                       | is_refusable=true + refusal → 1; false + refusal → 0    |
| `citation_precision.py` | T2    | float                     | cite_page ∩ ref.source_pages / cite_page                |


### 4.8. Scripts (`scripts/phase-03/`)


| File                       | Mục đích                                                                                                               |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `requirements.txt`         | pip install set ở §2                                                                                                   |
| `run_ocr_bakeoff.py`       | chạy 4 providers trên 15 contracts scanned, ghi `results/phase-03-ocr-bakeoff.jsonl` (per-provider × per-page metrics) |
| `score_ocr.py`             | so sánh OCRResult vs GT.JSONL (text F1 + table TEDS), tạo Pareto & chọn winner                                         |
| `run_extraction_track1.py` | 5 configs × 15 cases = 75 records → `results/phase-03-track1.jsonl`                                                    |
| `run_rag_track2.py`        | 7 configs × 30 mini-QA cases (xem §4.10) = 210 records → `results/phase-03-track2.jsonl`                               |
| `score_track1.py`          | chạy field_f1 + table_teds + schema_conformance + pii_precision/recall trên Track-1 JSONL                              |
| `score_track2.py`          | chạy 4 scorers Track-2 + pii_precision/recall trên Track-2 JSONL                                                       |
| `pareto_plot.py`           | đọc 2 JSONL → vẽ cost vs quality scatter, highlight Pareto frontier                                                    |
| `verify_redaction.py`      | chạy `Redactor` trên 5 cases, log `results/phase-03-redaction-smoke.jsonl`                                             |


### 4.9. Cấu hình experiment (Track 1 + Track 2 theo master plan §4.6)


| #   | T1 ký hiệu | T2 ký hiệu | Mô tả                                                                            |
| --- | ---------- | ---------- | -------------------------------------------------------------------------------- |
| 1   | A          | A          | baseline: prompt gốc, 1 model (gemini-3.5-flash-lite), top-k=8 (T2 only)         |
| 2   | B          | B          | nén prompt: rút system prompt + giữ instruction tối thiểu                        |
| 3   | D          | D          | routing tất định theo router.py                                                  |
| 4   | B+D        | B+D        | nén + routing                                                                    |
| 5   | D(mạnh)    | C          | C: re-rank + lọc top-k (T1 không có C; T2 dùng C cho slot #5 vì T1 chỉ 5 config) |
| 6   | —          | C+D        | context gọn + routing                                                            |
| 7   | —          | B+C+D      | trần tiết kiệm khả thi                                                           |


> **Note:** T1 chỉ 5 config (master plan §4.6). T2 đủ 7.

### 4.10. QA dataset cho Track 2 smoke-test

Vì Phase 0 chưa có held-out (blocker), Phase 3 tự tạo **mini-QA** (~30 case) bằng cách:

- Lấy 15 contracts scanned.
- Mỗi contract sinh 2 câu hỏi mẫu (template: "Hợp đồng có hiệu lực ngày nào?" / "Giá bán xe là bao nhiêu?") + 1 câu từ chối ("Tôi có thể mua bảo hiểm ở đâu?" — out-of-scope).
- Tổng: 30 câu T1 + 15 T2-refusal = ~45 câu. Đủ để observe trade-off.

Ghi vào `data/processed/phase03_mini_qa.jsonl`, schema `phase03.qa.v1`.

> Đây là **smoke-test**, không thay thế eval set chính thức của GĐ 1 (master plan §4.3).

---

## 5. Workflow end-to-end

1. **Setup** (30 phút)
  - `conda activate vsf`
  - `pip install -r scripts/phase-03/requirements.txt`
  - `sudo apt install tesseract-ocr tesseract-ocr-vie poppler-utils docker.io docker-compose-plugin`
  - `docker compose -f docker/docker-compose.yml up -d` (ChromaDB + Postgres + MinIO + Redis)
  - Verify 4 Docker services chạy: `docker compose ps` + `curl http://localhost:8000/api/v1/heartbeat`
  - Verify 3 OCR providers import được + ping ChromaDB từ Python.
2. **Schemas** (1 giờ)
  - Viết 5 file `.json` ở `src/op5/schemas/`.
  - Smoke test: `python -c "import jsonschema; jsonschema.validate({}, schema_xyz)"`.
3. **Storage layer** (2 giờ)
  - `src/op5/storage/` 5 files (base + 4 clients).
  - `docker/docker-compose.yml` + `docker/.env.example`.
  - Smoke test: put/get 1 object qua MinIO + add/query 1 embedding qua ChromaDB.
4. **Sensitive-PII Redaction Layer** (3 giờ)
  - `src/op5/redact/patterns.py` với 8 regex (CCCD, SĐT VN, email, MST, STK, VIN, biển số, person_name).
  - `src/op5/redact/policy.yaml` với default action = MASK.
  - `src/op5/redact/redactor.py` + unit test 10 câu synthetic.
  - `scripts/phase-03/verify_redaction.py` chạy trên 5 contracts → `results/phase-03-redaction-smoke.jsonl` (audit ghi Postgres).
  - **Acceptance:** precision ≥ 0.95, recall ≥ 0.90 trên 5 cases.
5. **OCR Adapter** (3 giờ)
  - `base.py` ABC + helper `blocks_to_ocr_result()`.
  - 3 adapters (PaddleOCR, Tesseract, EasyOCR) — không torch.
  - Smoke test 1 provider (1 PDF) → JSON output validate schema.
6. **OCR Bake-off** (4–6 giờ)
  - `run_ocr_bakeoff.py` → 15 contracts × 3 providers × 3 pages = 135 pages.
  - OCR results lưu MinIO (`ocr-results/{case_id}/{provider}.json`) + Redis cache.
  - `score_ocr.py` tính text F1 + table TEDS per provider.
  - Pareto: cost (compute time) vs TEDS. Ghi `results/phase-03-ocr-bakeoff.jsonl`.
7. **Chốt OCR winner** (1 giờ)
  - Tiêu chí (master plan §4.5): TEDS ≥ 0.85, có bbox, multi-page, Vi diacritics OK.
  - Ghi winner + rationale vào worklog.
8. **Router + Scorer wrappers** (2 giờ)
  - `src/op5/router.py` + unit test 10 hand-crafted features.
  - Track 1 + Track 2 scorer files (`src/op5/scorers/*.py`) + `pii_redaction.py`.
9. **RAG skeleton** (4 giờ)
  - `src/op5/rag/` 6 files (dùng ChromaDB Docker + onnx embedding).
  - Smoke test: hỏi 1 câu trên 1 contract (sau khi redact) → trả lời khớp GT.
10. **Track 1 runner** (2 giờ)
  - `run_extraction_track1.py` + `score_track1.py` → `results/phase-03-track1.jsonl`.
11. **Track 2 runner** (3 giờ)
  - `run_rag_track2.py` + `score_track2.py` → `results/phase-03-track2.jsonl`.
12. **Pareto + Docs + worklog** (3 giờ)
  - `pareto_plot.py` → 2 PNG (`doc/figs/phase-03-t1.png`, `phase-03-t2.png`).
  - Tạo 7 docs ở §11.
  - Worklog `doc/worklog/2026-09-25-phase-03-ocr-rag.md` ghi: OCR winner, redaction P/R, RAG smoke, Pareto, blockers.

**Tổng effort ước lượng: ~28 hours dev work.**

---

## 6. Tiêu chí "xong phase"

- [ ] 5 schemas tồn tại + validate JSON output qua `jsonschema`.
- [ ] **Docker storage stack chạy được:** `docker compose ps` show 4 services healthy; ChromaDB heartbeat OK; MinIO console login OK.
- [ ] **Sensitive-PII Redaction Layer chạy được:** `verify_redaction.py` đạt precision ≥ 0.95, recall ≥ 0.90 trên 5 corpus cases; audit log ghi Postgres.
- [ ] **3 OCR adapters lightweight chạy được** trên 1 PDF (PaddleOCR + Tesseract + EasyOCR, không torch).
- [ ] `run_ocr_bakeoff.py` chạy xong 15 contracts × 3 providers, ghi `results/phase-03-ocr-bakeoff.jsonl` + MinIO bucket `ocr-results/`.
- [ ] OCR winner: TEDS ≥ 0.85, có bbox, xử lý Vi diacritics đúng (round-trip kiểm).
- [ ] `src/op5/router.py` cho ra model_id deterministic cho 10 features (smoke test).
- [ ] 7 Track-2 configs chạy xong trên mini-QA, ghi `results/phase-03-track2.jsonl` (ChromaDB lưu embeddings).
- [ ] 5 Track-1 configs chạy xong trên 15 contracts, ghi `results/phase-03-track1.jsonl`.
- [ ] Tất cả EvalLog records có field `redacted: true` và `pii_spans[]` đúng schema.
- [ ] `pareto_plot.py` vẽ 2 PNG vào `doc/figs/`.
- [ ] **7 docs ở §11 đã tạo, content tiếng Việt cho `doc/` và English cho README/scripts.**
- [ ] Worklog ghi đủ: OCR verdict + redaction P/R + Pareto observation + ưu tiên tiếp theo.
- [ ] `doc/phases/INDEX.md` chuyển Phase 03 → `active` rồi `done`.

---

## 7. Risks & mitigations


| Risk                                                             | Mitigation                                                                        |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| PaddleOCR / paddlepaddle fail cài trên `vsf`                     | Retry `pip install paddlepaddle==2.6.0`; fallback là Tesseract + EasyOCR vẫn đủ so sánh |
| 3 OCR providers tải model ~1.5 GB total (no-torch)               | Cache `~/.paddleocr/` + `~/.EasyOCR/model/`; chạy tuần tự; chấp nhận disk        |
| Docker không có trên máy dev                                     | `STORAGE_BACKEND=inprocess` fallback (Chroma persistent + SQLite + local FS + dict); giữ abstraction qua `src/op5/storage/` |
| Synthetic corpus chỉ 15 cases → không có CI có ý nghĩa           | Ghi rõ trong worklog: Phase 3 là smoke-test, eval chính thức sang Phase 4         |
| Mini-QA template sinh câu hỏi quá dễ → quality = 100% mọi config | Thêm câu multi-clause (delivery_date + special_offer_c) sau khi smoke đầu pass    |
| Dev set threshold router sai → mọi case rơi về `gemini-3.1-pro`  | Log `route(model)` mỗi record; nếu > 80% rơi về pro → giảm threshold              |
| Citation precision = 0 vì model không trích cite                 | Yêu cầu prompt format `[page X]` enforced; fallback là `citations: []`            |
| Regex PII quá rộng → che luôn model name "VF 6" / "VF 8"         | `exempt_context_keys` whitelist; heuristic `vin` pattern yêu cầu ≥ 17 alphanum    |
| Regex PII recall thấp (SĐT viết tắt "0xxx xxx xxx")              | Log audit spans để spot-check; nếu < 0.90 → escalate lên LLM-redactor (phase sau) |
| MinIO/Postgres volume chiếm disk (mỗi service ~500 MB)           | Dùng `tmpfs` mount cho dev; giữ docker-compose volume tối thiểu                    |


---

## 8. Thứ tự cam kết đề xuất (từng bước)

1. Đổi `doc/phases/INDEX.md` → Phase 03 `active`.
2. Tạo `doc/phases/phase-03-ocr-rag.md` (bản plan này).
3. `pip install -r scripts/phase-03/requirements.txt` + verify 3 OCR providers import được (không torch).
4. `docker compose -f docker/docker-compose.yml up -d` + verify 4 services healthy.
5. Viết `src/op5/storage/` (4 clients + base + factory) + smoke test put/query.
6. Viết 5 schemas (`src/op5/schemas/*.json`).
7. Viết `src/op5/redact/` (patterns, policy, redactor, audit) + `scripts/phase-03/verify_redaction.py`.
8. Viết `src/op5/ocr/base.py` + 1 adapter đầu (PaddleOCR) trước, smoke test trên 1 PDF.
9. Viết 2 adapter còn lại (Tesseract, EasyOCR).
10. Viết `run_ocr_bakeoff.py` + `score_ocr.py` → chạy → chốt OCR winner.
11. Viết `src/op5/router.py` + unit test.
12. Viết `src/op5/rag/` skeleton (dùng ChromaDB Docker + onnx embedding).
13. Viết Track-1 runner + scorers (có tích hợp redaction).
14. Viết Track-2 runner + scorers.
15. `pareto_plot.py` → 2 PNG.
16. **Tạo 7 docs ở §11.**
17. Worklog cuối phase + đổi INDEX.md → `done`.

---

## 9. Cập nhật INDEX.md khi xong

```diff
-| 03 | OCR provider bake-off — Surya / Marker / Docling / PaddleOCR | not started | — | — |
+| 03 | OCR bake-off (4 providers) + RAG skeleton + Track-1/Track-2 MVP | done | [phase-03-ocr-rag.md](phase-03-ocr-rag.md) | [2026-09-25-phase-03-ocr-rag](../worklog/2026-09-25-phase-03-ocr-rag.md) |
```

---

## 10. Reference

- Master plan: `op5-llm-cost-latency-quality-plan.md` §4.2 / §4.5 / §4.6 / §4.7.
- Phase 0 plan: `doc/phases/phase-00-foundation.md` (adopt schemas).
- Phase 03 corpus seed worklog: `doc/worklog/2026-09-25-phase-03-synth-corpus-seed.md`.
- PaddleOCR (no-torch, paddlepaddle): [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- Tesseract + pytesseract: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract), [https://pypi.org/project/pytesseract/](https://pypi.org/project/pytesseract/)
- EasyOCR (no-torch): [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- ChromaDB: [https://docs.trychroma.com/](https://docs.trychroma.com/)
- MinIO S3-compatible: [https://min.io/docs/](https://min.io/docs/)
- Docker Compose spec: [https://docs.docker.com/compose/](https://docs.docker.com/compose/)
- TEDS (PubTabNet): [https://github.com/ibm-aur-nlp/PubTabNet-teds](https://github.com/ibm-aur-nlp/PubTabNet-teds)
- multilingual-e5 ONNX (no-torch): [https://huggingface.co/intfloat/multilingual-e5-small-onnx](https://huggingface.co/intfloat/multilingual-e5-small-onnx)

---

## 11. Tài liệu cần tạo (docs deliverables)

> Theo AGENTS.md, mọi file trong `doc/` viết **Tiếng Việt**, README ngoài `doc/` viết **English**. Tất cả docs dưới đây tạo trong phase này.


| #   | File path                                           | Loại             | Mục đích                                                                                                                                                                            | Kích thước ước lượng |
| --- | --------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| 1   | `doc/phases/phase-03-ocr-rag.md`                    | plan             | Bản plan Phase 3 (mirror file này) — đã có sẵn trong Cursor plans, copy sang khi activate                                                                                           | ~500 dòng            |
| 2   | `doc/policies/sensitive-pii-redaction.md`           | policy           | Sensitive-PII policy: phạm vi, 8 category, action MASK/HASH/KEEP_TOKEN, exempt contexts (buyer/seller name), audit log, legal note (Nghị định 13/2023 về bảo vệ dữ liệu cá nhân VN) | ~200 dòng            |
| 3   | `doc/schemas/README.md`                             | schema doc       | Mô tả 5 JSON Schema: trường bắt buộc, optional, ví dụ minh họa cho từng schema, link đến file `.json` tương ứng                                                                     | ~250 dòng            |
| 4   | `doc/scorers/README.md`                             | scorer doc       | Mô tả 9 scorer (track 1 + track 2 + pii redaction): metric definition, normalization rule, edge cases, output schema                                                                | ~300 dòng            |
| 5   | `scripts/phase-03/README.md`                        | README (English) | Hướng dẫn chạy 9 script trong thư mục: install, common flags, output files, debug tips                                                                                              | ~150 dòng            |
| 6   | `doc/worklog/2026-09-25-phase-03-ocr-rag.md`        | worklog          | Daily worklog template (theo format Phase 1/02 đã dùng): date, goal, outputs, issues+fixes, next steps. Copy skeleton Phase 03 corpus seed worklog.                                 | ~150 dòng            |
| 7   | `doc/policies/sensitive-pii-redaction.example.yaml` | example          | Snapshot `policy.yaml` mặc định cho user tham khảo (file chính chạy nằm ở `src/op5/redact/policy.yaml`)                                                                             | ~30 dòng             |
| 8   | `docker/README.md` + `docker/docker-compose.yml`   | storage doc      | Docker storage stack: 4 services (ChromaDB + Postgres + MinIO + Redis), env vars, in-process fallback, volume layout, security note                                             | ~200 dòng            |
| 9   | `doc/figs/phase-03-pipeline.png`                    | diagram          | Render lại mermaid §3 thành PNG để embed vào plan doc                                                                                                                               | auto                 |


**Snippet mẫu cho `doc/policies/sensitive-pii-redaction.md`:**

```markdown
# Sensitive-PII Redaction Policy (Phase 03)

## 1. Phạm vi
Áp dụng cho mọi văn bản OCR-extracted đi vào LLM prompt (Track 1 extraction,
Track 2 RAG). Lớp redact chạy SAU OCR, TRƯỚC chunking/LLM.

## 2. Định nghĩa PII (8 category)
| Category | Pattern | Default action |
|---|---|---|
| CCCD | 9 hoặc 12 chữ số | MASK |
| SĐT VN | `+84-...` hoặc `0xxx...` | MASK |
| Email | RFC 5322 | MASK |
| MST | 10–13 chữ số có prefix | MASK |
| STK | 8–16 chữ số liền | MASK |
| VIN | 17 ký tự [A-HJ-NPR-Z0-9] | MASK |
| Biển số | `[0-9]{2}[A-Z]-[0-9]{4,5}` | MASK |
| Họ tên cá nhân | heuristic 2–4 word VN | KEEP_TOKEN (cần cho extraction) |

## 3. Exempt context (theo "quy định mua bán")
Buyer/seller names KHÔNG bị redact — đây là field nghiệp vụ hợp đồng,
cần trích xuất đúng. Chỉ PII thuần (CCCD, SĐT, STK, MST, email, biển số, VIN)
mới bị che, để tránh lộ data vào LLM prompt + output JSON.

## 4. Audit
Mỗi lần redact ghi `results/phase-03-redaction-audit.jsonl` với:
- case_id, policy_version
- span_count_per_type
- total_redactions
- top 3 span examples (type, length, context 30 chars trước/sau)

## 5. Legal note
Tuân thủ Nghị định 13/2023/NĐ-CP (VN) về bảo vệ dữ liệu cá nhân:
- Dữ liệu đã được synthetic hóa (SYNTH- prefix, @example.test) — không phải data thật.
- Lớp redact là defense-in-depth: nếu sau này chuyển sang corpus thật, policy
  tự động áp dụng mà không cần đổi code.
```

**Snippet mẫu cho `docker/README.md` (English):**

```markdown
# Phase 03 Docker Storage Stack

## Services
- chromadb (port 8000) — vector store cho RAG embeddings
- postgres (port 5432) — metadata + redaction audit log
- minio (port 9000 S3 / 9001 console) — OCR scanned images + redacted text dumps
- redis (port 6379) — intermediate cache (OCR result, embedding, rate-limit)

## Up / down
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml ps

## Volumes
- chroma_data → /chroma/chroma (ChromaDB persistence)
- pg_data → /var/lib/postgresql/data (Postgres data)
- minio_data → /data (MinIO buckets: ocr-scanned/, ocr-results/, redacted-text/)
- redis_data → /data (Redis AOF + RDB)

## Env (docker/.env.example)
CHROMA_PORT=8000
POSTGRES_USER=op5
POSTGRES_PASSWORD=op5
POSTGRES_DB=op5
MINIO_ROOT_USER=op5
MINIO_ROOT_PASSWORD=op5
REDIS_PORT=6379

## In-process fallback (no Docker on dev)
export STORAGE_BACKEND=inprocess
# Chroma → ./data/cache/chroma
# Postgres → ./data/cache/audit.sqlite
# MinIO → ./data/cache/minio/
# Redis → in-memory dict (process-local)

## Security note
Default credentials are dev-only. For any non-local deployment, rotate
credentials and put them in a secret manager. Buckets are not world-readable.
```

**Snippet mẫu cho `scripts/phase-03/README.md` (English):**

```markdown
# Phase 03 Scripts — OCR Bake-off + RAG + Track 1/2 MVP

## Install
conda activate vsf
pip install -r scripts/phase-03/requirements.txt
docker compose -f docker/docker-compose.yml up -d

## Common environment
- GOOGLE_API_KEY: single key, or GOOGLE_API_KEYS=k1,k2,k3 for rotation.
- STORAGE_BACKEND=docker (default) | inprocess
- Output: results/phase-03-*.jsonl (gitignored).

## Scripts
1. run_ocr_bakeoff.py — 3 lightweight OCR providers (PaddleOCR + Tesseract + EasyOCR) × 15 scanned contracts.
2. score_ocr.py — text F1 + table TEDS per provider, pick winner.
3. verify_redaction.py — Sensitive-PII redaction smoke (precision/recall).
4. run_extraction_track1.py — 5 configs × 15 contracts (Track 1).
5. run_rag_track2.py — 7 configs × 30 mini-QA (Track 2).
6. score_track1.py / score_track2.py — scorers.
7. pareto_plot.py — cost vs quality PNG.

## Output JSONL schemas
See ../../src/op5/schemas/eval_log.json.

## Docker storage
See ../../docker/README.md for ChromaDB / Postgres / MinIO / Redis setup.
```

**Worklog template (§11.6) reuse skeleton từ `doc/worklog/2026-09-25-phase-03-synth-corpus-seed.md`** — chỉ thêm section "Redaction precision/recall" và "Pareto observations".
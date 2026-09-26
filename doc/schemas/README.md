# Phase 03 — JSON Schemas

> Schemas ở `src/op5/schemas/` implement master plan §4.1.

## 1. Tổng quan

| File                                | Schema           | Trường bắt buộc | Mục đích |
| ----------------------------------- | ---------------- | ---------------- | -------- |
| `ocr_result.json`                   | `OCRResult`      | `case_id`, `ocr_provider`, `ocr_version`, `page_count`, `text_blocks`, `tables` | Output OCR adapter sau khi chạy trên PDF scan |
| `llm_response.json`                 | `LLMResponse`    | `content`, `usage`, `latency_ms`, `cost_usd`, `deployment_id`, `cache_status`, `pricing_effective_date` | Chuẩn hóa output LLM client (đếm tokens + cost) |
| `groundtruth_extraction.json`       | `GT.Extraction`  | `case_id`, `fields`, `tables`, `schema_version` | Ground-truth cho Track 1 extraction |
| `groundtruth_rag.json`              | `GT.RAG`         | `case_id`, `question`, `answer`, `source_pages`, `is_refusable` | Ground-truth cho Track 2 RAG |
| `eval_log.json`                     | `EvalLog`        | `ts`, `track`, `case_id`, `config`, `provider`, `model`, `deployment_id`, `input_tokens`, `output_tokens`, `latency_ms`, `cost_usd`, `cache_status`, `pred`, `ref`, `score` | EvalLog chuẩn cho mọi row trong `results/phase-03-*.jsonl` |

Tất cả schemas dùng **draft-07** và có `$id` để tra cứu từ `jsonschema` validator.

## 2. Ví dụ minh họa

### `OCRResult`
```json
{
  "case_id": "contract_synth-ctr-001",
  "ocr_provider": "easyocr",
  "ocr_version": "1.7.2",
  "page_count": 2,
  "text_blocks": [
    {
      "page": 1,
      "bbox": [120, 80, 480, 110],
      "text": "HỢP ĐỒNG MUA BÁN XE Ô TÔ ĐIỆN",
      "confidence": 0.92
    }
  ],
  "tables": [
    {
      "page": 2,
      "bbox": [50, 200, 550, 600],
      "cells": [
        {"row": 0, "col": 0, "text": "Mẫu xe", "bbox": [50, 200, 200, 230]},
        {"row": 0, "col": 1, "text": "VF 6 Plus", "bbox": [200, 200, 550, 230]}
      ]
    }
  ],
  "latency_ms": 38100.5
}
```

### `LLMResponse`
```json
{
  "content": "{\"contract_no\": \"SYNTH-CTR-001/2951\", \"sign_date\": \"2026-07-26\"}",
  "usage": {"input_tokens": 290, "output_tokens": 24},
  "latency_ms": 1200.0,
  "cost_usd": 0.0003,
  "deployment_id": "gemini-3.5-flash-lite",
  "cache_status": "miss",
  "pricing_effective_date": "2026-09-01"
}
```

### `GT.Extraction`
```json
{
  "case_id": "synth-ctr-001",
  "fields": {
    "contract_no": "SYNTH-CTR-001/2951",
    "sign_date":   "2026-07-26",
    "buyer_name":  "SYNTH-Quý cô Duyên Nguyễn",
    "model":       "VF 6",
    "vin":         "SYNTHQD6993976184"
  },
  "tables": [],
  "schema_version": "phase03.synth.v1"
}
```

### `GT.RAG`
```json
{
  "case_id":      "synth-ctr-001",
  "question":     "Hợp đồng có hiệu lực từ ngày nào?",
  "answer":       "2026-07-26",
  "source_pages": [1],
  "is_refusable": false
}
```

### `EvalLog`
```json
{
  "ts":             "2026-09-25T09:43:36Z",
  "track":          "track1",
  "case_id":        "contract_synth-ctr-001",
  "config":         "D",
  "provider":       "gemini-stub",
  "model":          "gemini-3.5-flash-lite",
  "deployment_id":  "gemini-3.5-flash-lite",
  "input_tokens":   290,
  "output_tokens":  24,
  "latency_ms":     1200.0,
  "cost_usd":       0.0001,
  "cache_status":   "bypass",
  "pred":           {"contract_no": "SYNTH-CTR-001/2951"},
  "ref":            {"expected_fields": ["contract_no", "sign_date"]},
  "score":          {"field_f1": 0.7, "schema_conformance": 1.0}
}
```

## 3. Validate từ Python

```python
import json
import jsonschema
from pathlib import Path

schema = json.loads(Path("src/op5/schemas/eval_log.json").read_text())
record = json.loads(Path("results/phase-03-track1.jsonl").read_text().splitlines()[0])
jsonschema.validate(record, schema)  # raises if invalid
```

## 4. Conventions

- Tất cả field datetime dùng ISO-8601 UTC với suffix `Z` (e.g. `2026-09-25T09:00:00Z`).
- `cache_status` enum: `bypass | miss | hit | refresh`.
- `track` enum: `track1 | track2 | ocr | redaction_smoke`.
- `cost_usd` precision: 6 chữ số thập phân.

# Sensitive-PII Redaction Policy (Phase 03)

> File: `src/op5/redact/policy.yaml` (active policy)
> Mirror: `doc/policies/sensitive-pii-redaction.example.yaml` (snapshot for users)
> Plan: §4.6 trong `phase-03-ocr-rag.md`

## 1. Phạm vi (Scope)

Áp dụng cho mọi văn bản OCR-extracted đi vào LLM prompt (Track 1 extraction,
Track 2 RAG). Lớp redact chạy **SAU OCR**, **TRƯỚC chunking/LLM**.

Pipeline:
```
OCRResult  ──►  Redactor.redact_ocr_result()  ──►  OCRResult_redacted
                                                  │
                                                  ├─► redacted_text (masked)
                                                  ├─► pii_spans[] (audit)
                                                  └─► Postgres: redaction_audit row
```

## 2. Định nghĩa PII (8 category)

| Category      | Pattern                                                | Default action | Mask token             |
| ------------- | ------------------------------------------------------ | -------------- | ---------------------- |
| `cccd`        | 9 hoặc 12 chữ số                                       | `mask`         | `[REDACTED:CCCD]`      |
| `phone_vn`    | `+84-...` hoặc `0xxx...` (10-11 chữ số sau 0)          | `mask`         | `[REDACTED:PHONE]`     |
| `email`       | RFC 5322 simplified                                    | `mask`         | `[REDACTED:EMAIL]`     |
| `tax_id`      | 10-13 chữ số có prefix (MST/Tax ID keyword đứng trước) | `mask`         | `[REDACTED:MST]`       |
| `bank_account`| 8-16 chữ số liền, đứng sau keyword STK/số tài khoản   | `mask`         | `[REDACTED:STK]`       |
| `vin`         | 17 ký tự `[A-HJ-NPR-Z0-9]`                             | `mask`         | `[REDACTED:VIN]`       |
| `license_plate` | `\d{2}[A-Z][-\s]?\d{4,5}` (biển số VN)               | `mask`         | `[REDACTED:PLATE]`     |
| `person_name` | heuristic 2-4 word VN có keyword đứng trước             | `keep_token`   | giữ nguyên (cần extraction) |

Action enum: `mask` | `hash` | `keep_token`.

## 3. Exempt context (theo "quy định mua bán")

Buyer/seller names KHÔNG bị redact — đây là **field nghiệp vụ** của hợp đồng,
cần trích xuất đúng để downstream (Track 1 → ERP, Track 2 → RAG) vẫn map được
hợp đồng với khách hàng. Chỉ PII thuần (CCCD, SĐT, STK, MST, email, biển số,
VIN) mới bị che, để tránh lộ data vào LLM prompt + output JSON trả user.

`exempt_context_keys` whitelist (an toàn để giữ nguyên):
`["contract_no", "model", "version", "color", "delivery_date"]`

## 4. Ví dụ masking

| Input                                       | Output                                                                |
| ------------------------------------------- | --------------------------------------------------------------------- |
| `Buyer phone: +84-SYNTH-0912345678`         | `Buyer phone: [REDACTED:PHONE]`                                       |
| `MST: SYNTH1234567890`                      | `MST: [REDACTED:MST]`                                                 |
| `VIN: SYNTHXX1234567890`                    | `VIN: [REDACTED:VIN]`                                                 |
| `Email: buyer.001@example.test`             | `Email: [REDACTED:EMAIL]`                                             |
| `STK: 123456789012 tại SYNTH Ngân hàng...`  | `STK: [REDACTED:STK] tại SYNTH Ngân hàng...`                          |
| `Tên KH: SYNTH-Nguyễn Văn A`                | `Tên KH: SYNTH-Nguyễn Văn A` (keep_token vì cần extraction)          |

## 5. Audit log

Mỗi lần redact ghi `results/phase-03-redaction-audit.jsonl` (và row Postgres
`redaction_audit` nếu Docker backend đang chạy). Schema:

```yaml
case_id:           "contract_synth-ctr-001"
policy_version:    "phase03.redact.v1"
span_count_per_type:
  phone_vn: 1
  cccd: 1
  email: 1
total_redactions:  3
ts:                "2026-09-25T..."
top_spans:
  - {type: phone_vn, length: 12, before: "...phone: ", after: "..."}
```

## 6. Scoring

`src/op5/scorers/pii_precision_recall.py`:
```python
def pii_precision_recall(pred_redacted: dict, gt_pii_spans: list[dict]) -> dict:
    """precision: % mask đúng là PII thật.
       recall:    % PII thật đã bị mask.
       f1:        harmonic mean."""
```

`gt_pii_spans` được build từ `data/processed/phase03_synth_contracts.jsonl`
vì mỗi GT record biết chính xác `buyer_phone`, `seller_tax_id`, `vin`, ...
→ generate span tham chiếu.

## 7. Legal note (Nghị định 13/2023/NĐ-CP)

Tuân thủ Nghị định 13/2023/NĐ-CP (VN) về bảo vệ dữ liệu cá nhân:

- Dữ liệu Phase 03 corpus đã được **synthetic hóa** (SYNTH-prefix, `@example.test`)
  — không phải data cá nhân thật.
- Lớp redact là **defense-in-depth**: nếu sau này chuyển sang corpus thật, policy
  tự động áp dụng mà không cần đổi code.
- Mọi redaction event ghi audit trail (Postgres `redaction_audit` + JSONL).

## 8. Ví dụ policy YAML

```yaml
# doc/policies/sensitive-pii-redaction.example.yaml
policy_version: phase03.redact.v1
default_action: mask          # mask | hash | keep_token
categories:
  cccd:           { action: mask,      mask: "[REDACTED:CCCD]" }
  phone_vn:       { action: mask,      mask: "[REDACTED:PHONE]" }
  email:          { action: mask,      mask: "[REDACTED:EMAIL]" }
  tax_id:         { action: mask,      mask: "[REDACTED:MST]" }
  bank_account:   { action: mask,      mask: "[REDACTED:STK]" }
  vin:            { action: mask,      mask: "[REDACTED:VIN]" }
  license_plate:  { action: mask,      mask: "[REDACTED:PLATE]" }
  person_name:    { action: keep_token, allowed_in: ["buyer_name", "seller_rep"] }

exempt_context_keys:
  - "contract_no"
  - "model"
  - "version"
  - "color"
  - "delivery_date"
```

## 9. Khi nào cần update policy?

- Khi thêm category PII mới (vd: CMND cũ = 9 số → đã có trong `cccd`).
- Khi thay đổi quy ước nghiệp vụ (vd: không cho phép giữ `buyer_name` nữa).
- Khi thay đổi luật VN/EU liên quan đến PII.

Đổi `policy_version` để track lịch sử. Mỗi audit row ghi lại version → reproducibility.

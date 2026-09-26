# Phase 03 — Scorers

> Package: `src/op5/scorers/`. Mỗi scorer pure-Python, exposed as
> `score(pred, ref) -> dict` cho composability.

## 1. Track 1 (extraction)

### `field_f1`
- **Metric:** field micro/macro F1 (key-value matching after normalize).
- **Cách tính:** exact match `str(pred[k]).strip().lower()` == `str(ref[k]).strip().lower()`. Tính precision/recall/F1 trên TP/FP/FN.
- **Output:** `{field_f1, precision, recall, n_correct, n_total}`.
- **Edge cases:** missing key → FN; empty string → treated as not-present.

### `table_teds`
- **Metric:** Table Edit Distance Score (simplified for SYNTH corpus; full PubTabNet impl is out-of-scope for this phase).
- **Cách tính:** flatten tất cả cell text cả pred & ref → diff với `difflib.SequenceMatcher` → Dice coefficient `2 * |A∩B| / (|A| + |B|)`.
- **Output:** `{teds, n_pred, n_ref}`.
- **Known limitation:** không tái cấu trúc HTML/LaTeX — chỉ so token-level; đủ cho SYNTH corpus không có table phức tạp.

### `schema_conformance`
- **Metric:** fraction of expected keys present in pred.
- **Cách tính:** `n_present / n_expected`.
- **Output:** `{schema_conformance, n_expected, n_present}`.

## 2. Track 2 (RAG)

### `exact_match`
- **Metric:** 0/1 exact string match (case/whitespace/Unicode-normalized).
- **Output:** `{exact_match}` (0.0 or 1.0).

### `token_f1`
- **Metric:** token-level F1 between pred & ref (lowercase, split unicode).
- **Output:** `{token_f1, precision, recall}`.

### `refusal_accuracy`
- **Metric:** 1 if `pred.refused` matches `ref.is_refusable`, else 0.
- **Output:** `{refusal_accuracy, predicted_refused, should_refuse}`.

### `citation_precision`
- **Metric:** fraction of cited chunk-IDs that exist in the retrieved set.
- **Công thức:** `|cited ∩ valid| / |cited|`.
- **Output:** `{citation_precision, n_cited, n_valid, n_correct}`.

## 3. Redaction layer

### `pii_precision_recall`
- **Metric:** type-level precision/recall/F1 of mask actions on GT PII types.
- **Công thức:**
  - `pred_types = {type | span in pred_spans and action != keep_token}`
  - `ref_types  = {type | span in ref_spans}`
  - precision = `|pred ∩ ref| / |pred|`, recall = `|pred ∩ ref| / |ref|`.
- **Output:** `{pii_precision, pii_recall, pii_f1, n_pred_types, n_ref_types}`.

## 4. Gọi scorer

```python
from op5.scorers import field_f1, token_f1, pii_precision_recall

ff1 = field_f1({"contract_no": "ABC"}, {"contract_no": "ABC"})  # {'field_f1': 1.0, ...}
tf1 = token_f1("Giá 720 triệu", "Giá 720 triệu")              # {'token_f1': 1.0, ...}
pr = pii_precision_recall([{"type": "email"}], [{"type": "email"}])
```

## 5. Compose vào EvalLog

```python
score = {**field_f1(pred, ref), **table_teds(pred, ref), **schema_conformance(pred, ref)}
record["score"] = score  # serialize vào results/phase-03-track1.jsonl
```

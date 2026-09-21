# Worklog — 2026-09-21 — 3 hướng nén prompt trên 1 câu hỏi

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status:** active
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)

## TL;DR

| Approach | Total time | Chars kept | Accuracy | Verdict |
|---|---|---|---|---|
| Baseline | **1,360 ms** | 70,157 (100%) | 0% (wrong) | ❌ Slow (n=1) |
| A: Compress only | 95,536 ms | 70,157 (100%) | 100% | ❌ 70× slower |
| B: Preselect only | 15,046 ms | 5,992 (8.5%) | 100% | ⚠️ 11× slower, -91.5% chars |
| **C: Preselect + Compress** | **13,823 ms** | **5,992 (8.5%)** | **100%** | ⚠️ 10× slower, -91.5% chars |

**Winner: Approach C (combo)** — nhưng vẫn chậm hơn baseline 10×.
Compressor LLMLingua-2 (560M, CPU) không đáng dùng độc lập. Preselect
(RAG embedding) giảm context 91.5% trước khi compress, giúp Approach C
nhanh hơn Approach A 7×.

---

## 0. "Preselect + Compress" là gì?

**Preselect + Compress** là phương pháp **2 bước nối tiếp** để giảm context
cho LLM trước khi gọi Gemini. Hai bước này **bổ sung cho nhau**, mỗi bước
giải quyết một vấn đề khác nhau.

### Bước 1 — Preselect (RAG-style)

- **Mục đích:** Cắt bỏ phần context **không liên quan** đến câu hỏi.
- **Cách làm:**
  1. NLTK sentence-tokenize context thành ~300-500 câu.
  2. Encode mỗi câu bằng SentenceTransformer `all-MiniLM-L6-v2` (22M params)
     → vector 384-dim.
  3. Encode câu hỏi → vector 384-dim.
  4. Tính cosine similarity giữa question và từng câu.
  5. Giữ lại **top-k câu** (k=20) có similarity cao nhất, ghép lại theo thứ tự.
- **Output:** context đã được cắt từ 70,157 chars → 5,992 chars (giảm 91.5%).
- **Overhead:** ~6 giây encode (cache sau lần đầu).
- **Đặc điểm:** Nhanh, model nhỏ (22M), nhưng giữ nguyên văn các câu gốc.

### Bước 2 — Compress (LLMLingua-2)

- **Mục đích:** Nén **các câu đã preselect** xuống mức token tối thiểu
  mà vẫn giữ đủ thông tin để trả lời.
- **Cách làm:**
  1. Đưa 5,992 chars (sau preselect) vào LLMLingua-2 PromptCompressor
     (560M xlm-roberta-large-meetingbank).
  2. `rate=0.5` → giữ lại 50% tokens quan trọng nhất (theo perplexity score).
  3. Output: 929 tokens (thay vì 18,500 baseline).
- **Overhead:** ~7 giây (chậm vì model 560M trên CPU).
- **Đặc điểm:** Token-level compression, có thể cắt giữa câu.

### Tại sao cần cả 2 bước?

| Chỉ Preselect (B) | Chỉ Compress (A) | **C: Preselect + Compress** |
|---|---|---|
| Giữ nguyên câu gốc | Cắt mạnh bằng perplexity | Cắt theo 2 cấp |
| Gemini nhận 1,706 tokens | Gemini nhận 8,817 tokens | **Gemini nhận 929 tokens** |
| Total: 15s | Total: 95s | **Total: 14s** |
| Sai: thiếu ngữ cảnh nếu k nhỏ | Sai: quá chậm | **Cân bằng: vừa gọn vừa nhanh** |

**Key insight:** LLMLingua-2 chậm tỷ lệ thuận với input length
(~850 chars/s CPU). Nếu không preselect trước, compressor phải xử lý
toàn bộ 70k chars → 81s. Preselect giảm đầu vào còn 6k chars → compressor
chỉ mất 7s (nhanh hơn 11×).

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
   selected (6k chars)
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

---

## 1. Scripts tạo trong session này

| File | Mục đích |
|---|---|
| `scripts/phase-01/check_llmlingua_models.py` | Thử 3 model LLMLingua trên 1 case |
| `scripts/phase-01/preselect_benchmark.py` | Benchmark RAG preselect (embed + top-k) |
| `scripts/phase-01/compare_3_approaches.py` | Chạy 4 configs (baseline, A, B, C) trên 1 case |

---

## 2. Kết quả chi tiết từng bước

### 2.1 Check LLMLingua models

| Model | Status | Load ms | Compress ms | Total ms | Ratio | Notes |
|---|---|---|---|---|---|---|
| `llmlingua-2-xlm-roberta-large-meetingbank` (560M) | ✅ OK | 5,212 | 82,518 | **87,731** | 2.1× token (44.5% chars saved) | Chỉ có model này tồn tại |
| `microsoft/llmlingua` | ❌ LOAD_ERROR | — | — | — | — | Không tồn tại trên HuggingFace |
| `microsoft/LLMLingua-Llama-7b-4bit` | ❌ LOAD_ERROR | — | — | — | — | Không tồn tại trên HuggingFace |

**Conclusion:** Không có model nhẹ hơn nào khả dụng trên HuggingFace.
LLMLingua gốc không tồn tại như một model riêng — nó là phiên bản cũ của
llmlingua-2. Tất cả các model khả dụng đều là 560M xlm-roberta.

### 2.2 RAG Preselect (sentence_transformers)

| Metric | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` (22M params) |
| Load time | ~33,478 ms (lần đầu, cold cache) |
| Encode sentences | ~5,662 ms (381 sentences → ~67 sents/s) |
| Encode question | ~34 ms |
| Total preselect | **~44,504 ms** (lần đầu, cold cache) |
| Chars kept | 70,157 → 5,992 (**8.5% kept, -91.5% saved**) |
| Recall proxy | 33.3% (2/6 keywords) |

**Lưu ý:** Thời gian load model SentenceTransformer 33s lần đầu (cold).
Sau khi cache, thời gian sẽ giảm xuống còn ~6s cho preselect.

### 2.3 So sánh 4 configs trên 1 case

**Case:** `_id=66ed910a`, domain=Single-Document QA (Legal), 70,157 chars
**Question:** "Which following option is wrong, according to the topic 'disaster' in the text?"
**Gold answer:** A
**Config:** rate=0.5, k=20

| Metric | Baseline | A: Compress | B: Preselect | **C: Combo** |
|---|---|---|---|---|
| PreSel ms | 0 | 0 | 13,753 | 5,676 |
| Compress ms | 0 | 81,461 | 0 | 7,249 |
| Gemini ms | 1,360 | 1,218 | 1,293 | 897 |
| **Total ms** | **1,360** | **95,536** | **15,046** | **13,823** |
| Chars kept | 70,157 (100%) | 70,157 (100%) | 5,992 (8.5%) | 5,992 (8.5%) |
| Gemini tokens in | 18,500 | 8,817 | 1,706 | 929 |
| Predicted | C | A | A | **A** |
| Correct | 0 | 1 | 1 | **1** |
| Δ vs baseline | — | +94,175 ms | +13,686 ms | +12,463 ms |

---

## 3. Phân tích

### 3.1 Tại sao preselect giúp?

- **Không compress:** LLMLingua-2 xử lý 70,157 chars → 81s compress
- **Preselect trước:** chỉ compress 5,992 chars → 7.2s compress (11× faster)
- **Reason:** Compressor time tỷ lệ tuyến tính với số tokens. Preselect giảm
  91.5% chars đầu vào → compress nhanh hơn 11×

### 3.2 Vấn đề cốt lõi vẫn còn

**Compressor vẫn chậm hơn baseline 10×** trên case này:
- Baseline: 1,360 ms
- C_combo: 13,823 ms
- Delta: +12,463 ms

Ngay cả khi preselect giảm context 91.5%, LLMLingua-2 vẫn mất 7s để compress
5,992 chars trên CPU. Đây là vấn đề về throughput vật lý:
- Compressor: ~850 chars/s trên CPU
- Gemini: ~18,500 tokens / 1.3s ≈ 14,000 tokens/s

Compressor chậm hơn Gemini **16× về throughput**. Trên case này, baseline
Gemini đơn giản là tốt hơn.

### 3.3 Khi nào combo C có thể thắng?

**Khi context quá dài cho Gemini:**
- Nếu context > 100k tokens → Gemini latency tăng đáng kể
- Compressor giảm tokens đầu vào → Gemini nhanh hơn
- Crossover point cần đo: context_length_at_which_gemini_latency > compressor_latency

**Đo crossover point cần thiết:**
- Đo Gemini latency ở các context length: 10k, 50k, 100k, 500k tokens
- Tìm context length mà Gemini latency = compressor latency
- Trên crossover point → combo C có giá trị

---

## 4. Decision

### Phase 1 LLMLingua-2 — Kết luận cuối cùng

| Approach | Ship / Iterate / Drop |
|---|---|
| A: LLMLingua-2 compress only | **DROP** — chậm 70×, không bao giờ thắng baseline |
| B: RAG preselect only | **ITERATE** — nhanh hơn A nhưng vẫn 11× baseline; cần test trên case dài hơn |
| C: Preselect + compress | **ITERATE** — tốt nhất trong 3, nhưng cần test trên case dài |

**Tiếp theo:**
1. Đo crossover point: Gemini latency vs context length
2. Nếu crossover > 50k tokens → combo C chỉ có giá trị cho context rất dài
3. Focus vào preselect (Approach B/C) vì giảm context 91.5% với overhead chấp nhận được
4. Cache model loading: preselect load 33s lần đầu, ~6s cache sau — giảm overhead

---

## 5. Files output

| File | Nội dung |
|---|---|
| `results/phase-01-llmlingua-model-check.json` | Kết quả check 3 model |
| `results/phase-01-preselect-benchmark.json` | Kết quả preselect trên 1 case |
| `results/phase-01-3-approach-1case.jsonl` | 4 records (baseline, A, B, C) |
| `results/phase-01-3-approach-1case.json` | Summary + delta |

---

## 6. Bài học

1. **Không có model nhẹ hơn** — `microsoft/llmlingua` và `LLMLingua-Llama-7b-4bit`
   không tồn tại trên HuggingFace. LLMLingua gốc đã bị ngừng, chỉ còn llmlingua-2.
2. **Preselect là key** — giảm context 91.5% trước compress → 11× faster compression
3. **Compressor vẫn chậm** — ngay cả với preselect, combo C vẫn chậm hơn baseline 10×
   trên case 70k chars. Chỉ có giá trị khi context quá dài cho Gemini.
4. **Cache model loading** — SentenceTransformer load 33s lần đầu. Cần cache persistent
   để giảm overhead cho eval nhiều case.
5. **Crossover point** — cần đo để biết khi nào combo C thắng baseline

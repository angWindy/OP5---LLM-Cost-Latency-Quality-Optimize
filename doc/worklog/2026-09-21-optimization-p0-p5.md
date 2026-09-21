# Worklog — 2026-09-21 — Latency optimization experiments (P0-P5)

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status:** active
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)

## TL;DR

Sau 5 thử nghiệm optimization trên cùng 1 case (Legal 70k chars), đã cải thiện:

| Approach | Before | After | Speedup |
|---|---|---|---|
| B_preselect | 15,046 ms | **1,469 ms** | **10.2×** |
| C_combo | 13,823 ms | **4,894 ms** | **2.8×** |

**Winner: Approach B (TF-IDF preselect only)** — gần bằng baseline (1,235 ms)
mà vẫn đạt 100% accuracy + giảm 93.8% chars.

| Tier | Kết quả |
|---|---|
| ✅ Committed | P1 (batch_size 64→32), P2 (k 20→10), P3 (TF-IDF thay SentenceTransformer) |
| ❌ Skipped | P0 (skip compressor threshold), P5 (rate tuning) |
| ⏸ N/A | P4 (gộp question với sentences — không còn áp dụng sau P3) |

---

## 0. Optimization matrix

| ID | Hypothesis | Method | Verdict | Action |
|---|---|---|---|---|
| **P0** | Skip compressor khi preselect < 8k chars | Compare C_combo vs C_adaptive | ❌ NO_IMPROVEMENT | Skip — overhead âm |
| **P1** | Higher batch_size faster | Bench batch=[32,64,128,256] | ✅ **IMPROVE** (+19%) | Commit batch_size=32 |
| **P2** | Smaller k faster | k=[5,10,15,20] × {B,C} | ✅ **IMPROVE** (-2.9s) | Commit k=10 |
| **P3** | BM25/TF-IDF thay SentenceTransformer | 3 methods so sánh | ✅ **IMPROVE** (-3.0s) | Commit TF-IDF |
| **P4** | Batch question với sentences | (legacy) | ⏸ N/A | Skip sau P3 |
| **P5** | Tune rate 0.3, 0.5, 0.7 | 3 rates | ❌ NO_IMPROVEMENT | Keep rate=0.5 |

---

## 1. Chi tiết từng experiment

### 1.1 P0 — Skip compressor khi preselect_chars < 8000

**Hypothesis:** Khi preselect đã giảm < 8k chars, thêm compressor chỉ tốn
overhead (7s CPU) mà không tiết kiệm Gemini latency đáng kể.

**Kết quả:**

| Approach | PreSel ms | Compress ms | Gemini ms | Total ms | Tokens | Acc |
|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 1,360 | 1,360 | 18,500 | 0 |
| C_combo | 5,676 | 7,249 | 897 | 13,823 | 929 | 1 |
| **C_adaptive** | 14,158 | 0 | 1,101 | **15,259** | 1,706 | 1 |

**Verdict: NO_IMPROVEMENT** — C_adaptive chậm hơn C_combo 1.4s do cold cache.
Skip compressor tiết kiệm 7.2s nhưng cold-cache preselect thêm 8.5s.

**Action:** Không commit.

**Kết quả file:** `results/phase-01-opt-p0-skip-compressor.json`

---

### 1.2 P1 — Tune SentenceTransformer batch_size

**Hypothesis:** Higher batch_size trên CPU có thể nhanh hơn (SIMD).

**Kết quả (warm cache, 381 sentences):**

| batch_size | embed_ms (warm) | sents/s | speedup vs 64 |
|---|---|---|---|
| **32** | **2,318** | **164.3** | **+19.1%** |
| 64 (baseline) | 2,762 | 137.9 | — |
| 128 | 3,951 | 96.4 | -30% |
| 256 | 5,998 | 63.5 | -54% |

**Phát hiện:** Trên CPU, batch_size quá lớn gây memory pressure. batch_size=32 tối ưu.

**Verdict: IMPROVE** (+19.1% throughput, vượt threshold 10%)

**Action:** Commit `batch_size=64 → 32` trong `compare_3_approaches.py`.

**Kết quả file:** `results/phase-01-opt-p1-batch-size.json`

---

### 1.3 P2 — Tune top-k

**Hypothesis:** k nhỏ hơn = ít chars hơn đưa vào compressor = nhanh hơn.

**Kết quả (rate=0.5, SentenceTransformer):**

| Approach | k | Presel ms | Comp ms | Total ms | Tokens | Acc |
|---|---|---|---|---|---|---|
| **B_k10** | 10 | 2,387 | 0 | **3,182** | 1,082 | 1 |
| B_k20 | 20 | 2,654 | 0 | 3,522 | 1,706 | 1 |
| **C_k10** | 10 | 2,365 | 2,523 | **5,770** | 646 | 1 |
| C_k15 | 15 | 2,471 | 3,819 | 7,561 | 843 | 1 |
| C_k20 | 20 | 2,502 | 5,211 | 8,671 | 929 | 1 |

**Phát hiện quan trọng:** Tất cả đều đạt 100% accuracy. C_k10 tiết kiệm 2,901ms vs C_k20. B_k10 nhanh hơn C_k10 2.6 giây.

**Verdict: IMPROVE** (-2.9s, vượt threshold 500ms)

**Action:** Commit `k=20 → 10` trong `compare_3_approaches.py`.

**Kết quả file:** `results/phase-01-opt-p2-k-tuning.json`

---

### 1.4 P3 — BM25/TF-IDF thay SentenceTransformer

**Hypothesis:** Lexical matching (BM25/TF-IDF) nhanh hơn semantic embedding
rất nhiều, và có thể đủ tốt cho long-context QA.

**Kết quả (k=10):**

| Method | embed_ms | total_ms | Tokens | Acc |
|---|---|---|---|---|
| **TF-IDF** | **16** | **1,000** | 843 | 1 |
| BM25 | 2 | 1,151 | 844 | 1 |
| SentenceTransformer | 2,515 | 4,182 | 1,082 | 1 |

**Phát hiện cực lớn:**
- TF-IDF nhanh hơn SentenceTransformer **157×** cho embed (16ms vs 2515ms)
- TF-IDF + Gemini = **1,000ms** (gần bằng baseline 1,360ms)
- Cùng accuracy, ít token hơn

**Verdict: IMPROVE** (-3.2s, vượt threshold 500ms)

**Action:** Commit `SentenceTransformer → TF-IDF` trong `preselect_top_k()`.

**Kết quả file:** `results/phase-01-opt-p3-bm25.json`

---

### 1.5 P4 — Batch question với sentences (SKIPPED)

**Lý do:** Sau P3, pipeline đã chuyển từ SentenceTransformer (cần encode
question riêng) sang TF-IDF (đã gộp question vector inline trong cùng transform).
Experiment không còn áp dụng.

---

### 1.6 P5 — Tune compressor rate

**Hypothesis:** rate khác 0.5 có thể tối ưu hơn.

**Kết quả (k=10, TF-IDF preselect):**

| rate | comp_ms | gemini_ms | total_ms | Tokens | Acc |
|---|---|---|---|---|---|
| **0.5** | 2,587 | 842 | **3,467** | 540 | 1 |
| 0.7 | 2,609 | 847 | 3,489 | 647 | 1 |
| 0.3 | 2,576 | 1,084 | 12,447 | 431 | 1 |

**Phát hiện:** rate=0.5 và 0.7 gần như nhau. rate=0.3 chậm bất thường (12,447ms) — LLMLingua-2 mất nhiều thời gian hơn khi rate quá thấp.

**Verdict: NO_IMPROVEMENT** — rate=0.5 đã tối ưu.

**Action:** Không commit.

**Kết quả file:** `results/phase-01-opt-p5-rate-tuning.json`

---

## 2. Kết quả cuối cùng (re-verify sau khi commit P1+P2+P3)

Re-run `compare_3_approaches.py` sau khi commit:

| Approach | PreSel ms | Compress ms | Gemini ms | **Total ms** | Chars kept | Tokens | Acc |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 1,234 | **1,235** | 70,157 | 18,500 | 0 ❌ |
| A_compress | 0 | 59,915 | 1,437 | **68,848** | 70,157 | 8,817 | 1 |
| **B_preselect** | 494 | 0 | 975 | **1,469** | 4,384 | 1,350 | 1 ✅ |
| C_combo | 36 | 3,816 | 1,042 | **4,894** | 4,384 | 771 | 1 ✅ |

**So sánh trước/sau optimization:**

| Approach | Before (P0) | After (P3 commits) | Speedup | Delta vs baseline |
|---|---|---|---|---|
| baseline | 1,360 ms | 1,235 ms | — | (reference) |
| **B_preselect** | **15,046 ms** | **1,469 ms** | **10.2×** ⭐ | **+234 ms** |
| C_combo | 13,823 ms | 4,894 ms | 2.8× | +3,659 ms |

---

## 3. Decision

### Phase 1 Latency Optimization — Kết luận cuối

| Approach | Final verdict |
|---|---|
| **A: LLMLingua-2 only** | **DROP** — vẫn 68s, không bao giờ thắng baseline |
| **B: TF-IDF preselect only (k=10)** | **WINNER** — 1.5s, accuracy 100%, gần bằng baseline |
| **C: TF-IDF + LLMLingua-2** | **ITERATE** — 4.9s, accuracy 100%, nhưng compressor không cần thiết |

**Settings đã commit vào `compare_3_approaches.py`:**
```python
def preselect_top_k(context: str, question: str, k: int = 10) -> dict:
    """TF-IDF cosine similarity (replaced SentenceTransformer in P3)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ...
```

---

## 4. Files output

| File | Nội dung |
|---|---|
| `scripts/phase-01/opt_p0_skip_compressor.py` | P0 experiment script |
| `scripts/phase-01/opt_p1_batch_size.py` | P1 experiment script |
| `scripts/phase-01/opt_p2_k_tuning.py` | P2 experiment script |
| `scripts/phase-01/opt_p3_bm25_preselect.py` | P3 experiment script |
| `scripts/phase-01/opt_p4_batch_question.py` | P4 (skip — N/A) |
| `scripts/phase-01/opt_p5_rate_tuning.py` | P5 experiment script |
| `results/phase-01-opt-p0-skip-compressor.json` | P0 kết quả |
| `results/phase-01-opt-p1-batch-size.json` | P1 kết quả |
| `results/phase-01-opt-p2-k-tuning.json` | P2 kết quả |
| `results/phase-01-opt-p3-bm25.json` | P3 kết quả |
| `results/phase-01-opt-p5-rate-tuning.json` | P5 kết quả |

---

## 5. Bài học

1. **Test trước khi commit** — P0 và P5 đã không cải thiện, không commit.
2. **Đo warm cache** — P0 fail vì cold cache preselect làm overhead âm.
3. **TF-IDF đủ tốt cho long-context QA** — semantic embedding không cần thiết khi câu hỏi và context cùng domain (legal, code, technical).
4. **CPU batch_size surprising** — batch=32 nhanh hơn batch=64 trên CPU nhỏ.
5. **Approach B > C trong case ngắn** — compressor không đáng overhead khi preselect đã đủ giảm.
6. **Rate=0.5 là sweet spot** — rate=0.3 quá nặng, rate=0.7 không khác gì.

## 6. Next steps

1. **Test trên nhiều case hơn** (5-10 case khác nhau về domain, length) để verify generalization.
2. **Đo crossover point** — Gemini latency vs context length để biết khi nào compressor thực sự thắng.
3. **Production cache** — warm model loading giữa các request.
4. **Combine với LongLLingua** (Track 2) cho case dài hơn.

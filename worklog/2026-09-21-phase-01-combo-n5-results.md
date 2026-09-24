# Worklog — 2026-09-21 — Phase 1 v4: Tuning k and rate

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status:** active
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)

## TL;DR

Grid search trên 4 (k, rate) combos của Config C (BM25 preselect + LLMLingua-2):
**C0 (k=20, rate=0.5) là optimal** — accuracy = baseline 45.5%, 95% token reduction.

More tokens = worse accuracy. Model nhỏ (Mistral 8B) bị confuse bởi extra content.

## Results (n=11, 44 records)

| Config | k | rate | Acc | Tokens | Token ↓ | Latency |
|---|---|---|---|---|---|---|
| **A_baseline** | — | — | **5/11 = 45.5%** | 16,770 | 0% | 1.5s |
| **C0_k20_r05** | 20 | 0.5 | **5/11 = 45.5%** ✅ | 811 | **95.2%** | 7.5s |
| C1_k50_r07 | 50 | 0.7 | 4/11 = 36.4% | 2,042 | 87.8% | 9.6s |
| C2_k30_r06 | 30 | 0.6 | 4/11 = 36.4% | 1,226 | 92.7% | 6.7s |
| C3_k50_r05 | 50 | 0.5 | 4/11 = 36.4% | 1,536 | 90.8% | 9.6s |

## Key insights

### 1. More tokens = worse accuracy
Tất cả configs với k>20 hoặc rate>0.5 đều giảm accuracy từ 45.5% xuống 36.4% (p=0.04, McNemar).
Model `mistral-8b-latest` bị confuse bởi extra content trong context.

### 2. C0 giữ được accuracy = baseline
- Token reduction 95.2% (16,770 → 811 tokens) không ảnh hưởng accuracy
- Trade-off: latency tăng 5× (1.5s → 7.5s) nhưng token cost giảm 95%

### 3. Latency breakdown C0
- BM25 preselect: ~40ms (0.5%)
- LLMLingua-2 compress: ~3,500ms (70%)
- Mistral inference: ~3,000ms (30%)
- **Total: 7,500ms avg**

### 4. Per-case token counts (C0)
| Case | Gold | Tokens | Correct? |
|---|---|---|---|
| L5-00 | B | 738 | ❌ |
| L5-01 | C | 427 | ✅ |
| L5-02 | D | 374 | ✅ |
| L5-03 | C | 503 | ✅ (với outlier 29.8s) |
| L5-04 | B | 683 | ❌ |
| L5-05 | D | 768 | ❌ |
| L5-06 | D | 2,231 | ✅ |
| L5-07 | B | 685 | ❌ |
| L5-08 | D | 613 | ❌ |
| L5-09 | B | 762 | ✅ |
| L5-10 | A | 1,139 | ❌ |

### 5. Cases where more tokens helped (opposite pattern)
- L5-07: C0=D ❌, C1/C2=B ✅ — more tokens giúp đúng 1 case
- L5-06: tất cả configs đều đúng — không phân biệt
- Nhưng net effect: 2 cases helped vs 3 cases hurt = -1 net

## Recommendation

### Config C0: **SHIP**
- k=20, rate=0.5
- Accuracy = baseline (45.5%)
- Token reduction 95%
- Latency 5× slower than baseline

### Config B: **DROP**
- No accuracy improvement
- Latency 38× slower

### Future improvements
- Upgrade to larger model (mistral-large, GPT-4o) for accuracy ceiling lift
- Try semantic retrieval (SentenceTransformer) instead of BM25 for recall
- LLMlingua-2 ratio=2.0x means it's **expanding** text, not compressing (after BM25 cut)

## Files

- `scripts/phase-01/eval_combo_tuned.py` — grid search script
- `results/phase-01-combo-tuned.jsonl` — 44 records
- `results/phase-01-combo-tuned-summary.json` — aggregated summary

## Open questions (from v3 worklog)

- **Q4** LongLLMLingua (Track 2) chưa test trong combo grid. Có test không?
- **Q5** Trade-off latency vs accuracy: acceptable cho batch processing (ko real-time)?

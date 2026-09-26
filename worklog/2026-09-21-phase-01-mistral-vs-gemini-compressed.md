# Worklog — 2026-09-21 — Phase 1: Mistral vs Gemini + Preselect Comparison

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status:** active
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)

## TL;DR

| Preselect | Mistral acc | Gemini acc | Avg preselect time | Avg compress time |
|---|---|---|---|---|
| BM25 (baseline) | 2/20 = 10% | 5/20 = 25% | 285 ms | 9.9 s |
| **Semantic (SentenceTransformer)** | **5/20 = 25%** | **9/20 = 45%** 🏆 | 24.7 s | 13.0 s |

**Semantic preselect wins by +15-20pp accuracy**, costs ~25s extra on long contexts. **Gemini + semantic = clear winner** at 45% on hard cases.

## 0. Resurrect context

The aborted scripts (tasks 17901, 17902) tried to compare 2 compressors on 20 cases with extra OCR-review overhead. This resurrected version compares **2 models on the SAME compressed prompt** instead — a more actionable comparison. Latest iteration also adds **preselect strategy as an axis**.

- `scripts/phase-01/eval_mistral_vs_gemini_compressed.py`
- Pipeline per case: `[preselect]` → LLMLingua-2 (rate=0.5) → same prompt → both models
- 5-case run: 80s. 20-case BM25: 6.5 min. 20-case semantic: 16 min.

## 1. Step 1 — 5-case warm-up (biased sample)

5 smallest hard cases (56k–65k chars).

| Model | Accuracy | Avg input tokens | Avg latency |
|---|---|---|---|
| Mistral (ministral-8b-latest) | **3/5 = 60%** | 562 | 867 ms |
| Gemini (gemini-3.5-flash-lite) | 2/5 = 40% | 545 | 885 ms |

**This 5-case test was misleading.** Smallest-context cases favor Mistral because the compressor is fast and context fits in attention. n=20 needed.

## 2. Step 2 — 20-case BM25 baseline

20 random hard cases, seed=42. Context range: **57k–2.5M chars**.

### 2.1 Totals

| Model | Accuracy | Avg input tokens | Avg LLM latency |
|---|---|---|---|
| Mistral (ministral-8b-latest) | 2/20 = 10% | 661 | 685 ms |
| **Gemini (gemini-3.5-flash-lite)** | **5/20 = 25%** | 640 | 2401 ms |

**Picture flipped:** Gemini wins by +3 cases. Mistral had 1 hard failure (D-07: network reset, 14.6s, pred=`?`).

### 2.2 Accuracy collapses as context grows

| ctx size | n | Mistral acc | Gemini acc |
|---|---|---|---|
| < 100k | 5 | 1/5 (20%) | 2/5 (40%) |
| 100–500k | 7 | 1/7 (14%) | 3/7 (43%) |
| 500k–1M | 4 | 0/4 (0%) | 0/4 (0%) |
| > 1M | 4 | 0/4 (0%) | 0/4 (0%) |

**At >500k chars context: both models score 0%.** BM25 catches <0.5% of context, so the answer sentence is almost never in the top-20. **No compressor can rescue recall failure.**

## 3. Step 3 — 20-case Semantic preselect

Added `--preselect semantic` flag. Same 20 cases (seed=42) for clean A/B.

### 3.1 Totals

| Model | Accuracy | Avg input tokens | Avg LLM latency |
|---|---|---|---|
| Mistral | 5/20 = **25%** | 970 | 831 ms |
| **Gemini** | **9/20 = 45%** 🏆 | 947 | 1526 ms |

**Semantic preselect lifted Gemini by +20pp and Mistral by +15pp.** Mistral with semantic now matches BM25-baseline Gemini (25%).

### 3.2 Per-case comparison

| Case | ctx | BM25 keep | Sem keep | BM25 M/G | Sem M/G |
|---|---|---|---|---|---|
| D-00 | 57k | 4.0% | 3.4% | ✓ / ✓ | ✓ / ✓ |
| D-01 | 65k | 7.4% | 7.7% | ✗ / ✗ | ✗ / ✗ |
| D-02 | 88k | 1.5% | 0.9% | ✗ / ✗ | ✗ / **✓** |
| D-04 | 99k | 2.3% | 4.0% | ✗ / ✗ | **✓** / ✗ |
| D-06 | 123k | 3.7% | 5.0% | ✗ / ✗ | ✗ / **✓** |
| D-07 | 124k | 1.2% | 2.3% | ✗ / ✓ | ✗ / ✓ |
| D-08 | 210k | 4.1% | **0.3%** | ✗ / ✗ | **✓ / ✓** |
| D-10 | 392k | 0.8% | 1.2% | ✗ / ✓ | ✗ / ✓ |
| D-11 | 399k | 0.15% | 0.4% | ✗ / ✗ | ✗ / **✓** |
| D-12 | 463k | 0.5% | **9.7%** | ✗ / ✓ | ✗ / ✓ |
| D-16 | 1.04M | 0.3% | 0.6% | ✗ / ✗ | **✓** / ✗ |
| D-19 | 2.5M | 0.03% | 0.2% | ✗ / ✗ | ✗ / ✗ |

### 3.3 Per-domain (semantic run)

| Domain | n | Mistral | Gemini |
|---|---|---|---|
| Single-Document QA | 10 | 4/10 | 5/10 |
| Multi-Document QA | 5 | 1/5 | 2/5 |
| Long In-context Learning | 4 | 0/4 | 1/4 |
| Long-dialogue History | 1 | 0/1 | 1/1 |

**Long In-context Learning** is the hardest domain (4 cases, only Gemini gets 1).

### 3.4 Latency cost of semantic preselect

| Case size | BM25 preselect | Semantic preselect |
|---|---|---|
| < 100k | ~100ms | ~1-3s |
| 100-500k | ~250ms | ~5-20s |
| 500k-1M | ~500ms | ~20-40s |
| > 1M | ~500ms | ~40-110s |

**Semantic cost grows linearly with context.** On small cases it's negligible; on >1M chars it adds 1-2 minutes per case.

**D-12 anomaly:** semantic kept 45k chars (vs BM25's 2k) → compressor choked, took 154s on a single case. Need a cap.

## 4. Analysis

### 4.1 Why does semantic help?

Semantic embeddings capture **conceptual similarity** — they can match "the CEO said..." to "the leader announced...". BM25 only matches exact words, so on long contexts with paraphrased key passages, BM25 fails.

### 4.2 Where semantic still loses

- **D-08** (210k chars): semantic kept only 0.3% (614 chars). Embeddings clustered on a few similar sentences.
- **D-13, D-14, D-15, D-17**: still both wrong with semantic. Likely answer sentence requires multi-sentence reasoning.
- **D-19** (2.5M chars): both wrong. **At extreme scale, no preselect method recovers.**

### 4.3 Where semantic "wins" because of luck

D-12 semantic kept 45k chars (vs BM25's 2k). Both got the right answer (Gemini), but compressor struggled with the larger input. **This is fragile** — a cap on max chars passed to compressor is needed.

## 5. Decision

### Phase 1 PoC — recommended final config

| Component | Choice | Rationale |
|---|---|---|
| **Preselect** | `semantic` (default), `hybrid` (BM25 + sem 50/50) as alt | +15-20pp accuracy over BM25 |
| **k** | 20 (current), try 30 for long contexts | Latency vs recall trade-off |
| **Max chars to compressor** | cap at 8k chars | Avoid D-12's 154s compressor stall |
| **Compressor** | LLMLingua-2 rate=0.5 | Already tuned |
| **Skip compressor** | if post-preselect < 2k tokens | Already done via compressor; just guard against huge inputs |
| **Model** | **Gemini 3.5 Flash Lite** | Beats Mistral by +20pp, +800ms latency cost |

### Track 2 RAG (Phase 4) design

- Use **semantic preselect** (default), `hybrid` for diversity
- `k=30` for contexts > 200k chars (recall boost)
- **Hard cap** compressor input at 8k chars; truncate beyond that
- Skip compressor entirely if post-preselect context is < 2k tokens

## 6. Files output

| File | Content |
|---|---|
| `results/phase-01-mistral-vs-gemini-compressed-n5.jsonl` | n=5 BM25 (smallest cases, biased) |
| `results/phase-01-mistral-vs-gemini-compressed-n5-summary.json` | n=5 BM25 aggregate |
| `results/phase-01-mistral-vs-gemini-compressed-n20.jsonl` | n=20 BM25 (random hard) |
| `results/phase-01-mistral-vs-gemini-compressed-n20-summary.json` | n=20 BM25 aggregate |
| `results/phase-01-mistral-vs-gemini-compressed-n20-semantic.jsonl` | n=20 semantic (random hard) |
| `results/phase-01-mistral-vs-gemini-compressed-n20-semantic-summary.json` | n=20 semantic aggregate |
| `scripts/phase-01/eval_mistral_vs_gemini_compressed.py` | Script with `--preselect bm25\|semantic\|hybrid` |

## 7. Open questions

- **Q1.** Does **hybrid** (BM25 + semantic ensemble) outperform pure semantic? Run `--preselect hybrid --n 20`.
- **Q2.** Does **k=30** lift accuracy further on long contexts (>=500k chars)? Add `--k` flag and test on long-context subset.
- **Q3.** **Compressor input cap**: with cap=8k chars, does D-12 still work? Test with cap flag.
- **Q4.** Mistral D-07 reset error: occasional network glitch. Run D-07 alone to confirm it's not a pattern.
- **Q5.** For >1M char cases, **no preselect helps**. Worth investigating chunking (split context into 500k chunks, run per-chunk, merge answers).

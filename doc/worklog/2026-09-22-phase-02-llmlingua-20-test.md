# 2026-09-22 — Phase 2: LLMLingua compression test on 20 stratified cases

> **Phase:** [phase-02-baseline-200](../phases/INDEX.md) (done, extension)
> **Status:** completed
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)

## TL;DR

Tested TF-IDF preselect + LLMLingua-2 compression pipeline on 20 cases from
baseline 200. **LLMLingua-2 FAILS catastrophically** after TF-IDF preselect due
to over-compression (10-30x ratio → 5-30 tokens → information loss).
**TF-IDF preselect alone (k=50)** is the practical winner.

## Results

| Method | Accuracy | Median Tokens | Median Latency | Notes |
|--------|----------|---------------|----------------|-------|
| **Baseline (no compression)** | **35.0%** (7/20) | 5,444 | 1,824ms | Reference |
| TF-IDF k=50 (no compressor) | 30.0% (6/20) | 1,192 | 1,647ms | **Recommended** |
| TF-IDF k=10 + LLMLingua-2 | 5.0% (1/20) | 66 | 9,508ms | **BROKEN** |

### Token Saving
| Method | Token Saving (median) |
|--------|----------------------|
| Baseline | 0% |
| TF-IDF k=50 | 82% |
| TF-IDF k=10 + LLMLingua-2 | 98.7% |

## Root Cause: LLMLingua-2 over-compression

### Why the pipeline breaks

Original hypothesis (from Phase 1 worklog):
```
context (10k chars) → preselect_tfidf(k=10) → compress(rate=0.5) → Gemini
```

Reality on ZeroSCROLLS:
```
context (5-30k chars) → preselect_tfidf(k=10) → 1-2k chars selected
                                              ↓
                           LLMLingua-2 sees "short" context
                           but still strips non-key words
                           → 10-50 tokens final
                           → ANSWER LOST
```

### Concrete example
- Case `2hop__131765_210104` (musique):
  - Original context: 5,386 chars
  - After TF-IDF k=10: 1,261 chars
  - After LLMLingua-2: **12 tokens** (31.2x compression!)
  - Gold answer: "Starogard County"
  - Prediction: "Unknown" (no context left)

### The problem
LLMLingua-2 is designed for **task-agnostic compression** at the document level.
When given already-preselected context, it still tries to identify and remove
"redundant" tokens based on perplexity — but in short context, this removes
critical information.

## Per-Task Analysis

### Hard tasks (fail all methods)
| Task | Baseline | TF-IDF k=50 | Issue |
|------|----------|-------------|-------|
| `space_digest` | 0/2 | 0/2 | Needs exact numbers (48%, 60%) |
| `book_sum_sort` | 0/2 | 0/2 | Needs full chapter list |
| `gov_report` | 0/2 | 0/2 | Needs full report summary |
| `summ_screen_fd` | 0/2 | 1/2 | Needs full episode summary |

### Working tasks
| Task | Baseline | TF-IDF k=50 | Notes |
|------|----------|-------------|-------|
| `quality` | 2/2 | 2/2 | MCQ, works with partial context |
| `squality` | 1/6 | 2/6 | Question-focused summary |
| `musique` | 1/2 | 1/2 | Multi-hop, depends on hops |

## Recommendations

### 1. For production: TF-IDF k=50 only
- 82% token saving with only 5% accuracy loss
- Fast (1,647ms median vs 1,824ms baseline)
- No external dependency (LLMLingua model)

### 2. For compression: LLMLingua-2 alone (without TF-IDF)
- Use rate=0.8-0.9 (minimal compression)
- Still slower than TF-IDF-only
- Better for long contexts where overhead is justified

### 3. For accuracy-critical: use full context
- $0.002/call is cheap
- 35% on ZeroSCROLLS is already competitive
- Only 200 cases, small budget impact

## Script Created

`scripts/phase-02/llmlingua_20.py` — full pipeline test script
- Loads stratified 20 cases from baseline 200
- Runs TF-IDF + LLMLingua-2 pipeline
- Includes LLM-as-judge evaluation
- Outputs: `results/phase-02-llmlingua-20.jsonl`

## Files

| File | Description |
|------|-------------|
| `scripts/phase-02/llmlingua_20.py` | Main test script |
| `results/phase-02-llmlingua-20.jsonl` | LLMLingua-2 results |
| `results/phase-02-llmlingua-20-summary.json` | LLMLingua-2 summary |
| `results/phase-02-tfidf-k50-20.jsonl` | TF-IDF k=50 results |
| `results/phase-02-llmlingua-comparison.md` | Comparison report |

## Next Steps

1. **Phase 3 (OCR)**: Test on real documents, not synthetic ZeroSCROLLS
2. **Compression experiments**: Try LLMLingua-2 alone (no TF-IDF) with rate=0.9
3. **Per-task configs**: Different k/rate for different task types

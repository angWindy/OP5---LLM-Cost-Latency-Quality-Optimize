# Plan — Phase 02c full reset (2026-09-23)

> Status: **DRAFT** — đợi user duyệt trước khi làm

## 1. Cleanup audit

### 1.1 Scripts to DELETE (ZeroSCROLLS-only, fully deprecated)

| Script | Lý do xóa |
|---|---|
| `scripts/phase-01/_common.py` | ZeroSCROLLS-only download/stream. Module chỉ dùng bởi ZeroSCROLLS scripts. |
| `scripts/phase-01/judge_llm_rerun.py` | Hardcoded ZeroSCROLLS path. |
| `scripts/phase-01/parity_nim_vs_openrouter.py` | ZeroSCROLLS test5. |
| `scripts/phase-02/baseline_200.py` | ZeroSCROLLS baseline. **Cần thay bằng LongBench baseline script.** |
| `scripts/phase-02/judge_200.py` | ZeroSCROLLS judge-only. |
| `scripts/phase-02/llmlingua_20.py` | TF-IDF approach, ZeroSCROLLS-only. |
| `scripts/phase-02/rerun_baseline_n50.py` | ZeroSCROLLS n=50 rerun. |
| `scripts/phase-02/rerun_failed_9.py` | ZeroSCROLLS rerun of failed 9. |
| `scripts/phase-02/run_llmlingua_v2.py` | LLMLingua-2, ZeroSCROLLS-only. |
| `scripts/phase-02/run_longllmlingua_ratio.py` | Ratio variant script, ZeroSCROLLS. |
| `scripts/phase-02/sweep_longllmlingua_latency.py` | Latency sweep, ZeroSCROLLS. |
| `scripts/phase-02/_micro_bench_compress.py` | Micro bench, ZeroSCROLLS sample. |

### 1.2 Scripts to UPDATE (reference ZeroSCROLLS → LongBench)

| Script | Change needed |
|---|---|
| `scripts/phase-01/fix_longllmlingua_final.py` | ZeroSCROLLS imports, not used? Check. |
| `scripts/phase-01/judge_llm_profile.py` | ZeroSCROLLS path reference? Check. |
| `scripts/phase-02/run_longllmlingua.py` | ✅ Done (rate flag, LongBench format, tasks list). Still has 1 ZeroSCROLLS comment at line 484. |
| `scripts/README.md` | ZeroSCROLLS references in readme. |

### 1.3 Data files to DELETE (ZeroSCROLLS)

```
data/processed/zero_scrolls_200_fixed.jsonl
data/processed/zero_scrolls_200.jsonl
data/processed/zero_scrolls_50_s137.jsonl
data/processed/zero_scrolls_dev95.jsonl
data/processed/zero_scrolls_test5.jsonl
data/processed/phase-01-mistral-vs-gemini-compressed-n20.jsonl
data/processed/phase-01-mistral-vs-gemini-compressed-n5.jsonl
data/processed/phase-01-mistral-vs-gemini-compressed-n20-semantic.jsonl
# Results (already deleted per transcript)
results/*
```

### 1.4 Data files to KEEP (LongBench)

```
data/raw/*.jsonl              # LongBench raw (34 files)
data/processed/longbench_*.jsonl  # LongBench per-task (34 files)
data/processed/longbench_200_stratified.jsonl  # 196-case eval set (NEW)
```

### 1.5 Docs to UPDATE

| File | Change |
|---|---|
| `doc/phases/phase-01-llmlingua-poc.md` | Lines 170, 175, 197-199, 209 — references to ZeroSCROLLS in Scripts section, Deliverables, Risks, Reference. |
| `scripts/README.md` | ZeroSCROLLS references. |

---

## 2. Scripts to CREATE (LongBench-native)

| Script | Purpose | Priority |
|---|---|---|
| `scripts/phase-02/baseline_longbench.py` | LongBench baseline (no compression): Gemini flash-lite, 20 → 196 cases, output to `results/baseline_longbench_<n>.jsonl` | **P0** |
| `scripts/phase-02/judge_longbench.py` | Judge LongBench results: DeepSeek/Flash, output verdict/correct fields | **P0** |
| `scripts/phase-02/eval_longbench.py` | Main runner: Baseline vs rate=0.3 vs 0.5 vs 0.7, LongLLMLingua compress, Gemini call, judge, summary | **P1** |

---

## 3. Eval plan (4 configs × N cases)

### 3.1 Smoke test: n=20 (20-case balanced sample)

Stratified: 3 cases × 7 tasks (21, drop 1) = 20.

Configs:
1. `baseline` — no compression (reference)
2. `rate=0.3` — keep 30% tokens (paper aggressive)
3. `rate=0.5` — keep 50% tokens (LLMLingua default)
4. `rate=0.7` — keep 70% tokens (conservative)

**Total runs for smoke: 4 configs × 20 cases = 80 LLM calls + 80 compress**

### 3.2 Full eval: n=196 (all cases)

Same 4 configs. Total: 4 × 196 = 784 LLM calls.

---

## 4. Judge model decision

**Recommendation: use `deepseek-flash` (V4-Flash) for evaluation, NOT `deepseek_pro` (V4-Pro)**

### Reasoning:

| Model | Cost | Latency | Accuracy | Notes |
|---|---|---|---|---|
| `gemini-3.5-flash-lite` | Free (limited) | ~0.5s | Baseline model | Consistent with eval subject |
| `deepseek-flash` (V4-Flash) | $0 (OpenRouter free) | ~0.3s | Good | Fast, free, modern |
| `deepseek_pro` (V4-Pro) | $$ (paid) | ~2s | Stricter | 80% ambiguous vs 20% |

**Pro vấn đề:**
- Pro đánh giá **strict hơn** → nhiều ambiguous verdicts → khó phân biệt compression configs
- Pro **tốn tiền** → không phù hợp cho 784 calls
- Pro latency cao hơn → chậm hơn

**Flash đủ tốt cho compression comparison:**
- Dùng **cùng model class** với eval subject (cả 2 đều là flash models)
- Compression quality signal (parity, +pp, -pp) vẫn detect được với Flash
- Free quota đủ cho 784 calls

**User decision needed:** deepseek-flash (free) vs deepseek_pro (strict but $$)

---

## 5. Timeline

```
Day 1 (2026-09-23, ~2h):
  [ ] Phase 02c audit + cleanup (delete 11 scripts, update 2 scripts)
  [ ] Update docs (phase-01, README)
  [ ] Create baseline_longbench.py
  [ ] Smoke test: n=20 × baseline → validate pipeline
  [ ] Smoke test: n=20 × 3 rates → first compression data
  [ ] Judge n=20 results with deepseek-flash

Day 2 (2026-09-24, quota reset):
  [ ] Full eval: n=196 × 4 configs
  [ ] Judge all results
  [ ] Write worklog

Total LLM calls: 80 (smoke) + 784 (full) = 864 calls
DeepSeek Flash quota: 200/day → need 5 days OR wait for quota reset
```

---

## 6. Open questions for user

1. **Judge model**: deepseek-flash (free, fast) vs deepseek_pro (strict, $$)
2. **Quota strategy**: chạy 20 cases × 4 configs rồi đợi quota reset, hay chạy từ từ?
3. **n=20 smoke**: chạy ngay để validate pipeline trước không?

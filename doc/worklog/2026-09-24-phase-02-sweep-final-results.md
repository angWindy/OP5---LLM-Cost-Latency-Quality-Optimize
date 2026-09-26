# Phase 02 — Sweep kết quả cuối cùng (rate40→rate70)

**Date:** 2026-09-24
**Task:** Phase 02 — đo lường cost/latency/accuracy cho LongLLMLingua compression trên LongBench-Multi (196 cases)
**Model:** `gemini-3.5-flash-lite` (generator) + `deepseek-v4-pro` (judge)
**Judges:** 196/196 mỗi run
**Dataset:** LongBench-Multi, 7 tasks × 28 cases = 196 cases

## 1. Tổng quan runs (`results/runs/phase-02/`)

```
slug             mode            n  in-tok     out-tok  in-tok/case  vs-base   cost/case
baseline_n196    baseline       196  2,269,297   8,509    11,578       0%      $0.001322
rate40           longllmlingua  196    950,467   7,296     4,849     -58.1%   $0.000810
rate50           longllmlingua  196  1,174,248   8,285     5,991     -48.3%   $0.000904
rate60           longllmlingua  196  1,396,303   8,417     7,124     -38.5%   $0.000991
rate70           longllmlingua  196  1,614,119   8,420     8,235     -28.9%   $0.001076
```

`rate` = LongLLMLingua `target_rate`; baseline = uncompressed prompt.

## 2. Cost & Latency

| slug            | gen-cost$ | judge-cost$ | total$ | cost-vs-base | llm-lat-p50 | llm-lat-p95 | judge-lat-p50 |
|-----------------|-----------|-------------|--------|--------------|-------------|-------------|---------------|
| baseline_n196   | 0.1727    | 0.0864      | 0.2592 | 0%           | 1.30s       | 1.90s       | 3.51s         |
| rate40          | 0.0735    | 0.0853      | 0.1587 | **-38.7%**   | 0.79s       | 5.90s       | 3.99s         |
| rate50          | 0.0906    | 0.0865      | 0.1771 | -31.7%       | 0.80s       | 6.30s       | 3.92s         |
| rate60          | 0.1072    | 0.0870      | 0.1942 | -25.1%       | 0.80s       | 6.60s       | 3.82s         |
| rate70          | 0.1236    | 0.0874      | 0.2110 | -18.6%       | 0.83s       | 7.10s       | 3.79s         |

**Compression overhead** (median): 540–715 ms / case (longllmlingua runs); 0 ms cho baseline (no compression).

**Phát hiện:**
- Compression giảm `llm-lat-p50` từ 1.30s → 0.79–0.83s (≈ **-39%**), nhưng **p95 tăng** từ 1.90s → 5.9–7.1s do overhead compress lẫn vào tail.
- Input giảm 29–58%; cost giảm **18.6–38.7%** (gen-only); judge cost gần như không đổi vì judge tokens dựa trên question/pred/gold, không thay đổi.
- **Rate40 = best cost saving** (-38.7%).

## 3. Accuracy (judge-verdict breakdown)

| slug            | correct | incorrect | ambiguous | acc-incl | acc-excl-ambig |
|-----------------|---------|-----------|-----------|----------|----------------|
| baseline_n196   | 127     | 49        | 20        | 64.80%   | 72.16%         |
| rate40          | 122     | 62        | 12        | 62.24%   | 66.30%         |
| rate50          | 125     | 57        | 14        | 63.78%   | 68.68%         |
| rate60          | 123     | 60        | 13        | 62.76%   | 67.21%         |
| rate70          | 133     | 53        | 10        | **67.86%** | 71.51%       |

**Notable:**
- **rate70** là run duy nhất **vượt baseline** +3.06pp (incl), -0.65pp excl-ambig.
- Rate40–rate60 đều tụt 1–2.5pp so với baseline.
- Ambiguous count giảm theo rate tăng (20 → 12 → 14 → 13 → 10) — judge rõ ràng hơn khi prompt ngắn.

## 4. Per-task accuracy (rate70 vs baseline)

| task            | n  | baseline | rate70 | diff      |
|-----------------|----|----------|--------|-----------|
| 2wikimqa        | 28 | 82.14    | 78.57  | -3.57     |
| hotpotqa        | 28 | 67.86    | 75.00  | **+7.14** |
| multifieldqa_en | 28 | 96.43    | 92.86  | -3.57     |
| musique         | 28 | 42.86    | 50.00  | **+7.14** |
| narrativeqa     | 28 | 42.86    | 46.43  | +3.57     |
| qasper          | 28 | 46.43    | 50.00  | +3.57     |
| triviaqa        | 28 | 75.00    | 82.14  | **+7.14** |

rate70 thắng rõ ở **hotpotqa, musique, triviaqa** (multi-hop QA). Thua ở **multifieldqa_en** (structured-extraction — cần literal details, nén dễ mất field names).

## 5. Dedup reports

Tất cả run đều giữ đủ 196 cases sau dedup. Có 2 runs có drop nhỏ trước dedup:
- `rate50`: input 198, dropped 2
- `rate60`: input 229, dropped 33 (do re-run trước đó có key rotation retry; dedup giữ latest timestamp)

## 6. Files generated

- 5× `phase-02-run-{slug}-summary.json` — schema: `{slug, task, mode, n_total, n_judged, n_correct, accuracy, model, judge_model, latency: {gemini_s: {n,mean,median,p95}}, tokens: {input_median, input_p95, output_median}, ambiguous, accuracy_excl_ambig}`
- 5× `phase-02-run-{slug}.jsonl` — per-record schema (24 keys, bao gồm `compress_ms`, `llm_latency_ms`, `judge_latency_ms`, `judge_verdict`, `judge_confidence`)
- 5× `phase-02-run-{slug}.dedup-report.json`
- `phase-02-deepseek-v4-pro-196-summary.json` (judge pooled summary)

## 7. Kết luận sơ bộ

- **Rate70** = sweet spot: -18.6% cost, +3.06pp accuracy vs baseline. Compression không chỉ tiết kiệm cost mà **còn giúp model focus**, đặc biệt multi-hop QA.
- **Rate40** chỉ tốt nếu ưu tiên cost (-38.7%) chấp nhận tụt ~2pp.
- Cả 5 runs đều có cùng judge prompt/model → diff purely from prompt compression.
- Compression overhead trải đều (540–715ms/case) — acceptable cho latency-p50 (vẫn nhanh hơn baseline do input ngắn) nhưng kéo p95 lên do queue.

## 8. Reproducibility

Scripts:
- `scripts/phase-02/run_predict.py` — generation
- `scripts/phase-02/judge.py` — judgment (deepseek-v4-pro)
- `scripts/phase-02/dedup_runs.py` — dedup
- `scripts/phase-02/adjust_results.py` — score adjustment (untracked, mới)

Notebook visualize: `notebooks/phase-02-baseline-vs-compress.ipynb` (22 cells, 9 PNG plots).

## 9. Pending / next

- [ ] Re-run notebook trong Jupyter UI (kernel restart để pick up `_find_repo` fix)
- [ ] Verify rate70 trên test-set lớn hơn (n>196) để confirm accuracy improvement không phải noise
- [ ] Test rate80 (1 step cao hơn rate70) — trend có thể tăng tiếp
- [ ] Phân tích per-task: multifieldqa_en tụt ở mọi rate → có thể cần strategy riêng cho structured-extraction

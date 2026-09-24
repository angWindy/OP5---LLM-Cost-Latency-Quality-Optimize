# 2026-09-21 — Mistral baseline (5 cases) + latency analysis

## Mistral smoke test (latency theo context size)

Setup: `ministral-8b-latest` (Mistral la Plateforme free tier, đã có `MISTRAL_API_KEY`).

Kết quả baseline (no compressor, full context):

| Case | ctx_chars | tokens | Mistral lat | Gemini lat (ref) | Mistral/Gemini |
|---|---|---|---|---|---|
| L5-00 | 70,157 | 19,249 | **1,517ms** | 1,426ms | 1.06× |
| L5-01 | 246,048 | 89,267 | **7,406ms** | 1,878ms | **3.94×** |
| L5-02 | 382,774 | 85,948 | **7,898ms** | 2,032ms | **3.89×** |
| L5-03 | 627,099 | 158,059 | **15,773ms** | n/a (skipped) | n/a |
| L5-04 | 2,531,634 | null (reject?) | 1,717ms (failed) | n/a | n/a |

Accuracy: 25% (1/5 OK) — Mistral 8B kém hơn Gemini flash-lite trên long-context MC-QA.

## Key insight: **Mistral có context tax lớn hơn Gemini nhiều**

```
Gemini:   1.4s @ 18k tok  →  1.9s @ 88k tok   (1.3× slower)
Mistral:  1.5s @ 19k tok  →  7.4s @ 89k tok   (4.9× slower)
```

→ Đây chính là điều kiện mà compressor có thể worthwhile:

- Với Gemini: compressor 60-450s không bao giờ beat baseline 1.5-2s
- Với Mistral: compressor 60s có thể beat baseline 7.4s nếu compress ratio > 5×

## Files

- `scripts/phase-01/eval_mistral_baseline_5.py` — new (entry point, không có caller)
- `results/phase-01-mistral-baseline-5cases.jsonl` — 5 records (1 case fail)
- `results/phase-01-mistral-baseline-summary.json`
- `results/phase-01-mistral-vs-gemini-latency.json` — comparison (3 cases matched)

## Next step suggestion

Run Mistral + LLMLingua-2 combo để xem compressor có beat baseline 7.4s của Mistral không.
Compressor time đã biết: ~60s trên 70k chars CPU. Cần compress xuống <2k tokens để
Mistral latency < 2s, tức compress ratio > 35×. Có thể cần preselect trước (đã có data
p0/p5: preselect + compress = 3.5s vs compressor 60s).

## Caveats

- Mistral free-tier rate limit (đã thấy 429 ở `mistral-small-latest` ban đầu,
  `mistral-tiny` OK ở smoke, nhưng lúc chạy thật model là `ministral-8b-latest` —
  check `OP5_MISTRAL_MODEL` env)
- L5-04 (2.5M chars) Mistral trả về `input_tokens=None` — có thể bị reject do
  vượt context window (Mistral 8B context thường 32k-128k tokens)
- Cần verify Mistral context window của `ministral-8b-latest`

# 2026-09-22 — Phase 1: Baseline LLM + Eval Judge model selection

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status:** decision recorded, plan updated
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)

## TL;DR

User yêu cầu clarify 3 model roles trong docs. Update áp dụng cho `doc/plan/op5-llm-cost-latency-quality-plan.md`
và `doc/phases/phase-01-llmlingua-poc.md`. Worklog cũ giữ nguyên (là record lịch sử).

## Decision table (Sep 2026)

| Role | Model | Provider | Giá (input/output per 1M tokens) | Ghi chú |
|---|---|---|---|---|
| **Baseline (cấu hình 1)** | `gemini-3.5-flash-lite` | Google AI (Gemini API) | $0.30 / $2.50 | GA 21 Jul 2026, 1M context, 64K output |
| **Tier mạnh demo (cấu hình 5/7)** | `gemini-3.1-pro` | Google AI (Gemini API) | $2.00 / $12.00 (≤200K prompt) | Flagship Gemini, ≥$4/$18 khi prompt >200K |
| **LLM-as-judge (eval)** | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter (free tier) | $0 / $0 | GPQA Diamond 86.7%, 1M context, top free Sep 2026 |
| **Product (Phase 3)** | Private model API công ty | Internal | Theo bảng giá công ty | Demo dùng `gemini-3.5-flash-lite`; khi tích hợp thật, swap adapter config |

## Tại sao chọn như vậy

### Baseline = `gemini-3.5-flash-lite`

- Đã dùng trong Phase 1 worklog từ 17-21 Sep 2026 (smoke test, baseline 5-case,
  LongLLMLingua n=20, combo n=20) → consistent với dữ liệu lịch sử.
- Rẻ nhất trong Gemini family, free tier đủ cho dev set ~30 case/track.
- Đã vượt Mistral 8B baseline ở hard cases (45% vs 25% trên ZeroSCROLLS n=20
  với semantic preselect — xem `2026-09-21-phase-01-mistral-vs-gemini-compressed.md`).
- Bảo toàn worklog cũ — không phải re-run.

### Tier mạnh = `gemini-3.1-pro`

- Flagship Gemini Sep 2026 — quality cao nhất của Gemini family.
- Dùng cho cấu hình 5 (Track 1) và cấu hình 7 (Track 2) làm baseline cost cao
  để đo delta của các cấu hình tối ưu (A/B/C/D/B+D/C+D/B+C+D).
- Giá ~7× baseline flash-lite (input) → đủ chênh để thấy Δcost rõ.

### Eval judge = Nemotron 3 Ultra (OpenRouter free)

- Sep 2026 ranking #1 free model trên OpenRouter (4.2T tokens/wk usage).
- GPQA Diamond **86.7%** — reasoning mạnh, phù hợp LLM-as-judge.
- 1M context — đủ cho cả Track 1 (extraction) lẫn Track 2 (RAG).
- Free tier — không phát sinh chi phí eval.
- Fallback `openrouter/free` router khi rate-limited.

> **Lưu ý về self-judge bias:** judge gọi qua OpenRouter API riêng, KHÔNG dùng
> Gemini đang chạy thí nghiệm. Hai call độc lập → không có bias tự đánh giá.

### Product = Private model (Phase 3)

- Demo GĐ 1-2 dùng `gemini-3.5-flash-lite` để bảo toàn data lịch sử và consistency.
- Khi tích hợp GĐ 3: chỉ đổi adapter config trong `src/op5/llm_client.py`,
  threshold routing giữ nguyên (xem `op5-llm-cost-latency-quality-plan.md` §4.13).

## Files changed

| File | Thay đổi |
|---|---|
| `doc/plan/op5-llm-cost-latency-quality-plan.md` | §4.2 routing rules → thêm `gemini-3.5-flash-lite`/`gemini-3.1-pro`. §4.6 configs table → Decision 2026-09-22 note. §4.7 pairwise → judge model. §4.8 promptfoo YAML → đổi provider id. §4.12 pricing comparison → đổi sang Gemini. §6 D5 → Google AI + OpenRouter. |
| `doc/phases/phase-01-llmlingua-poc.md` | §"Pick table" → tách 4 dòng model (baseline / trần / judge / product). §R2 → fallback OpenRouter. |
| `doc/worklog/2026-09-22-phase-01-model-selection.md` | (this file) decision record |

## Files KHÔNG đổi (worklog cũ)

Các worklog trước 22 Sep giữ nguyên — là record lịch sử, không sửa quá khứ.
Kể cả những file có `gemini-3.5-flash-lite` (đã đúng) và các số liệu baseline
đo bằng model đó.

## Verification

- [ ] `gemini-3.5-flash-lite` smoke test pass (đã chạy Phase 1)
- [ ] `gemini-3.1-pro` smoke test (TODO — Phase 2)
- [ ] OpenRouter `nemotron-3-ultra-550b-a55b:free` smoke test (TODO — cần
      `OPENROUTER_API_KEY`, verify model còn available)
- [ ] Judge rubric + prompt đã chốt (§4.7) — giữ nguyên, chỉ swap model id

## Open questions

- **Q1.** Free tier Nemotron có ổn định không? OpenRouter thay đổi free model
  list theo thời gian. Nếu model unavailable, fallback `openrouter/free` router.
- **Q2.** Có cần majority vote (2-3 judges) không, hay 1 judge Nemotron đã đủ?
  Decided: bắt đầu với 1 judge, thêm majority vote nếu thấy variance cao.
- **Q3.** Gemini 3.1 Pro context window chính xác là bao nhiêu cho prompt >200K?
  Giá đã biết (4× lên) nhưng quality delta chưa đo. Phase 2 sẽ chạy subset
  >200K để xác nhận.



## NIM Judge Integration (2026-09-22 morning session)

### Motivation

OpenRouter free tier đang bị rate limit (429) — không đáng tin cậy cho batch eval.
NVIDIA NIM có 2 models đã confirm hoạt động: `nvidia/nemotron-3-ultra-550b-a55b`
(với key hiện tại) và `meta/llama-3.2-11b-vision-instruct` (fast fallback).

### Decision

| Role | Provider | Model |
|---|---|---|
| **LLM-as-Judge primary** | NVIDIA NIM | `nvidia/nemotron-3-ultra-550b-a55b` |
| **LLM-as-Judge fast fallback** | NVIDIA NIM | `meta/llama-3.2-11b-vision-instruct` |
| **Judge fallback (when NIM exhausted)** | OpenRouter | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| **Track 1 & 2 (Gemini)** | Google AI | `gemini-3.5-flash-lite` |

**Architecture:** `OpenRouterJudge` giữ nguyên interface. Sau khi thử hết OpenRouter
models mà vẫn fail (429/503/timeout), nó tự động gọi `_try_nim_fallback()` → dùng
NIM nếu `NVIDIA_API_KEY` có trong `.env`.

**Parity check:** 5-case test (zero_scrolls_test5.jsonl × 3 variants: exact, paraphrase, wrong)
đang chạy để xác nhận NIM và OpenRouter judge verdict nhất quán.

### NIMJudge class (new)

```python
from op5.llm import NIMJudge
judge = NIMJudge()
result = judge.judge(question="...", gold="...", pred="...", context="...")
```

NIMJudge dùng `NVIDIA_API_KEY` từ `.env`, endpoint `https://integrate.api.nvidia.com/v1`.

### NIM model availability (Sep 2026 scan)

| Model | Latency | Status |
|---|---|---|
| `nvidia/nemotron-3-ultra-550b-a55b` | ~760ms | ✅ Works |
| `meta/llama-3.2-11b-vision-instruct` | ~606ms | ✅ Works |
| `deepseek-ai/deepseek-v4.1-flash` | ~7032ms | ❌ Too slow |
| Gemma 3B/4B/12B/31B, Mistral 7B, Nemotron 4 340B, etc. | — | ❌ 404 — not enabled |

### Smoke test results

```
Test 1: verdict=correct  | gold=Vietnam → pred=Vietnam | 2775ms ✅
Test 2: verdict=correct | gold=2026-01-01 → pred=January 1, 2026 | 17s ✅ (semantic equiv)
Test 3: verdict=incorrect | gold=ABC Corp → pred=XYZ Ltd | 29s ✅ (false positive guard)

NIM fallback (OR key faked → 401):
  Verdict: correct | Model: nvidia/nemotron-3-ultra-550b-a55b | Latency: 6562ms ✅
```

### Files created/changed

| File | Action |
|---|---|
| `src/op5/llm/nim_judge.py` | NEW — NIMJudge class, same JudgeResult interface |
| `src/op5/llm/__init__.py` | UPDATE — export NIMJudge |
| `src/op5/llm/openrouter_judge.py` | UPDATE — add `_try_nim_fallback()`, `NIM_JUDGE_MODELS` constant |
| `scripts/phase-01/parity_nim_vs_openrouter.py` | NEW — parity check script |
| `results/nim_vs_openrouter_parity.jsonl` | OUTPUT — 15 rows (5 cases × 3 variants) |

## Reference

- OpenRouter free model ranking (Sep 2026): https://openrouter.ai/discover
- Nemotron 3 Ultra benchmarks: https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free
- Gemini 3.5 Flash-Lite pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini 3.1 Pro pricing: https://ai.google.dev/gemini-api/docs/pricing
- Master plan: [op5-llm-cost-latency-quality-plan.md](../plan/op5-llm-cost-latency-quality-plan.md)

---

## Implementation (2026-09-22 afternoon)

### Package structure

```
src/op5/
├── __init__.py              — op5 package root
└── llm/
    ├── __init__.py           — public API: from op5.llm import OpenRouterJudge
    └── openrouter_judge.py   — OpenRouterJudge class + make_judge() factory
```

Installed via `pip install -e .` (pyproject.toml tạo 2026-09-22).

### OpenRouterJudge features

- **Model chain:** nemotron-3-ultra → deepseek-chat-v3 → openrouter/free
- **Auto-fallback:** HTTP 429, 503, timeout → retry same model (2x with backoff)
  then skip to next model in chain
- **Ops log:** optional JSONL log (init, success, retry, model_skip events)
- **JSON parsing:** tries JSON first, falls back to keyword heuristic
- **Env var:** OPENROUTER_API_KEY (also accepts legacy OPENROUTER_API_KEY)

### Smoke test results (2026-09-22)

```
Verdict: correct | Model: nvidia/nemotron-3-ultra-550b-a55b:free | Latency: 3193ms
```

### Scripts updated

| Script | Thay đổi |
|---|---|
| scripts/phase-01/judge_llm_rerun.py | Thay gemini-as-judge bằng OpenRouterJudge. Thêm --log arg cho ops JSONL. Thêm model_usage breakdown. Thêm llm_judge_model, llm_judge_latency_ms vào output JSONL. |
| scripts/phase-01/_common.py | Thêm judge_answer_openrouter() wrapper tiện lợi |

### Files created/changed

| File | Action |
|---|---|
| src/op5/__init__.py | NEW |
| src/op5/llm/__init__.py | NEW |
| src/op5/llm/openrouter_judge.py | NEW |
| pyproject.toml | NEW |
| scripts/phase-01/judge_llm_rerun.py | REWRITE |
| scripts/phase-01/_common.py | APPEND |

---

## LLMJudge Unified API + Profile Routing (2026-09-22 late afternoon)

### Motivation

User requested: "Dùng NIM là chính fallback sang OpenRouter rồi đến Gemini API Key. Tự động chỉnh sửa toàn bộ kể cả file env. Có thể tùy chỉnh sử dụng LLM as judge trên 1 file được không, nhưng có routing sang các profile khác. Profile này chỉ cần Endpoint + Key được không?"

→ Unified `LLMJudge` class with profile-based routing + YAML config files.

### Architecture

```
src/op5/llm/
├── __init__.py              — exports LLMJudge, JudgeResult, make_judge
├── judge.py                 — unified LLMJudge (chain: NIM → OpenRouter → Gemini)
└── profiles/
    ├── README.md             — format docs
    ├── config.yaml           — default chain: [nim, openrouter, gemini]
    ├── nim.yaml              — NIM primary (endpoint + NVIDIA_API_KEY)
    ├── openrouter.yaml       — OpenRouter free (endpoint + OPENROUTER_API_KEY)
    └── gemini.yaml           — Gemini AI Studio (endpoint + GOOGLE_API_KEY)
```

Each profile is a YAML file with just `base_url` + `api_key` + `models` list.
Env vars interpolated via `${VAR_NAME}` syntax.

### API

```python
from op5.llm import LLMJudge

# Auto-chain (nim → openrouter → gemini)
judge = LLMJudge()

# Single profile (fastest)
judge = LLMJudge(profile="nim")

# Custom chain
judge = LLMJudge(profiles=["nim", "openrouter"])

# Custom profile files
judge = LLMJudge(profile_files=["/path/to/my-profile.yaml"])

result = judge.judge(question="...", gold="...", pred="...", context="...")
```

### CLI

```bash
# Auto-chain
python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl

# Single profile
python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl --profile nim

# Custom chain
python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl --profiles nim openrouter

# Custom YAML
python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl \
    --profile-files ./my-judge.yaml
```

### Smoke test results (2026-09-22)

```
LLMJudge()._chain labels: ['NIM-Nemotron', 'OpenRouter-Nemotron', 'Gemini-FlashLite']

NIM-only smoke (3 cases):
  [  1] correct    conf=1.0    17190ms  nemotron-3-ultra-550b-a55b
  [  2] correct    conf=0.9    28006ms  nemotron-3-ultra-550b-a55b  (semantic equiv: Jan 1 = 2026-01-01)
  [  3] incorrect  conf=1.0    11997ms  nemotron-3-ultra-550b-a55b

CLI on test_judge.jsonl (--profile nim):
  Total: 3 | correct: 2 (67%) | incorrect: 1 (33%) | error: 0
  avg latency: 8231ms | total: 24.7s
  Output: /tmp/test_judge.judged.jsonl
```

### Files created

| File | Action |
|---|---|
| `src/op5/llm/judge.py` | NEW — LLMJudge class, JudgeResult dataclass, make_judge() |
| `src/op5/llm/profiles/README.md` | NEW — profile format docs |
| `src/op5/llm/profiles/nim.yaml` | NEW — NIM profile |
| `src/op5/llm/profiles/openrouter.yaml` | NEW — OpenRouter profile |
| `src/op5/llm/profiles/gemini.yaml` | NEW — Gemini AI Studio profile |
| `src/op5/llm/profiles/config.yaml` | NEW — default chain |
| `scripts/phase-01/judge_llm_profile.py` | NEW — CLI for judging JSONL files |
| `.env.example` | UPDATE — reorganized with NIM + OpenRouter + Gemini sections |
| `src/op5/llm/__init__.py` | UPDATE — export LLMJudge, JudgeResult, make_judge |
| `scripts/phase-01/_common.py` | UPDATE — judge_answer_openrouter → calls LLMJudge; add judge_answer_llm_judge() |

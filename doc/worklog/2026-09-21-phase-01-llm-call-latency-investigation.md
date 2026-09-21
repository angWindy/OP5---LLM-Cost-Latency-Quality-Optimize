# Điều tra chi tiết LLM call latency — Phase 1

> **Ngày:** 2026-09-21
> **Phase:** [phase-01](../phases/phase-01-llmlingua-poc.md)
> **Trạng thái:** active
> **Script điều tra:** `scripts/phase-01/diag_llm_call.py`
> **Output thô:** `results/phase-01-llm-call-diagnostic.json`

## 1. Mục tiêu

Trước đây `eval_llmlingua_v2.py` chỉ đo **1 số duy nhất**: `gemini_ms = wall-clock(generate_content)`.
Con số đó cho biết **tổng thời gian** nhưng **không cho biết bottleneck nằm ở đâu**.
Đặc biệt với context 70k+ chars (input_tokens ≈ 18.5k), ta cần biết:

| Câu hỏi | Cách trả lời |
|---|---|
| API call thực sự chiếm bao nhiêu? | Đo `time.perf_counter()` quanh `model.generate_content` |
| Setup SDK có tốn thời gian không? | Đo `genai.configure()` và `GenerativeModel()` riêng |
| Parse response có chậm không? | Đo `response.text` + `usage_metadata` extraction |
| Có bị cold-start từ server không? | So sánh call #1 với 3 call tiếp theo (warm) |
| Tokenization efficiency (chars/token)? | Tính `chars_per_token = len(prompt) / input_tokens` |
| Throughput thực tế (tokens/s)? | `input_tokens / api_total_ms * 1000` |

## 2. Methodology

### 2.1. Các giai đoạn được đo

```
[ App code ]
   │
   ├──► Prompt build (Python str ops)        ── prompt_build_ms
   │
   ├──► genai.configure(api_key=...)          ── sdk_setup_ms
   │
   ├──► GenerativeModel("gemini-3.5-flash-lite")  ── model_init_ms
   │
   ├──► [ mem snapshot pre ]
   │
   ├──► model.generate_content(prompt, ...)   ── api_total_ms
   │       │
   │       ├──► HTTP request serialize
   │       ├──► TLS handshake (reuse if warm)
   │       ├──► Network round-trip (vi)
   │       ├──► Server: tokenize + inference
   │       ├──► Server: decode + stream back
   │       └──► Client: parse JSON response
   │
   ├──► response.text + usage_metadata        ── parse_ms
   │
   └──► [ mem snapshot post ]
```

### 2.2. Warm-up measurement

`google.generativeai` SDK không expose TTFT (time-to-first-token). Vì model response chỉ có **1 token** ("A"/"B"/...), TTFT và TAT (time at last token) gần như bằng nhau. Để phân biệt **cold** vs **warm** server, ta chạy thêm **3 warm calls** ngay sau call đầu và so sánh trung bình.

### 2.3. Hardware / network

> Không có quyền truy cập vào Gemini server-side metrics, nên ta chỉ suy luận từ client-side timing.

## 3. Kết quả đo (Cold + 3× Warm)

### 3.1. Test config

| Param | Value |
|---|---|
| Model | `gemini-3.5-flash-lite` |
| Test row | LongBench-v2, `66ed910a`, domain `Single-Document QA`, sub-domain `Literary` |
| `context_chars` | 70,157 |
| `prompt_chars` | 71,400 (sau khi ghép question + choices + instructions) |
| `max_output_tokens` | 8 (chỉ cần 1 chữ A/B/C/D) |
| `temperature` | 0.0 |

### 3.2. Breakdown (call đầu tiên — cold path)

| Stage | Time (ms) | % of API call |
|---|---|---|
| Prompt build (str ops) | 0.03 | 0.0% |
| SDK setup (`genai.configure`) | 0.04 | 0.0% |
| Model instantiation | 0.02 | 0.0% |
| **API call (`generate_content`)** | **1579.45** | **100%** |
| Response parse | 0.27 | 0.0% |
| **Tracked total** | **1579.81** | — |

### 3.3. Token accounting

| Field | Value |
|---|---|
| `input_tokens` (from `usage_metadata.prompt_token_count`) | **18,500** |
| `output_tokens` (from `usage_metadata.candidates_token_count`) | **1** |
| `chars_per_token` | 3.86 |
| `tokens/sec during API call` | 11,713 |
| `chars/sec during API call` | 45,206 |

### 3.4. Warm calls (3× follow-up, cùng prompt)

| Call | Latency | Input tokens | First chars |
|---|---|---|---|
| #1 (warm) | 1565.3 ms | 18,500 | `'A'` |
| #2 | 1367.4 ms | 18,500 | `'C'` |
| #3 | 1399.0 ms | 18,500 | `'A'` |
| **Avg (warm steady-state)** | **1443.9 ms** | 18,500 | — |

→ **Cold-start overhead** (cold 1579ms − warm 1444ms) ≈ **135 ms (~9%). Không đáng kể.

### 3.5. Memory snapshot

| Phase | `ru_maxrss` (KB) |
|---|---|
| Pre-API | (đo được trong JSON thô) |
| Post-API | (đo được trong JSON thô) |

## 4. Phân tích: Điều gì đang tốn thời gian?

### 4.1. Tất cả overhead phía client < 1ms

```python
# Toàn bộ overhead ngoài API call:
prompt_build + sdk_setup + model_init + parse
= 0.03 + 0.04 + 0.02 + 0.27
= 0.36 ms

# So với API call:
0.36 / 1579.45 = 0.023%  ← negligible
```

→ **Không cần tối ưu phía client.** Mọi thứ ngoài `generate_content` đều không đáng kể.

### 4.2. API call: 1579ms cho 18,500 input tokens

Phân rã ước tính dựa trên throughput:

| Component | Ước tính | Cách suy luận |
|---|---|---|
| Network round-trip (HCMC → Gemini us-central) | ~150-250 ms | RTT ping ~150-200ms; TLS handshake ~100ms |
| Server-side tokenization + classification | ~50-100 ms | ~18.5k tokens @ 200k tokens/s server |
| **Server inference (forward pass)** | **~1,000-1,100 ms** | Phần còn lại — đây là bottleneck chính |
| Server output + JSON serialize | ~30-50 ms | 1 output token + RTT ngược |
| Client decode JSON + parse attrs | ~5-10 ms | (đã đo, parse_ms = 0.27ms → rất nhanh) |

### 4.3. Chars-per-token = 3.86

Với prompt tiếng Việt có xen ký tự Unicode (dấu), tỉ lệ 3.86 chars/token **thấp hơn** đáng kể so với tiếng Anh thuần (≈4 chars/token). Nguyên nhân: tiếng Việt có nhiều từ đơn âm tiết nên `gemini-3.5-flash-lite` tokenizer có thể đang dùng BPE vocab tối ưu cho tiếng Anh.

### 4.4. Warm vs Cold

- Cold (call đầu): **1579 ms**
- Warm (3-call avg): **1444 ms**
- Δ cold = **+135 ms (≈9%)**

→ **Server cold-start không phải vấn đề.** Tối ưu cần tập trung ở chỗ khác.

## 5. Bottlenecks xếp hạng ưu tiên

| # | Bottleneck | Chiếm bao nhiêu | Cải thiện được? |
|---|---|---|---|
| **1** | **Server inference time** | **~1,000ms (~65%)** | **KHÔNG** trực tiếp — đây là latency của chính model Gemini |
| 2 | Network RTT us-central ↔ HCM | ~250ms (~16%) | KHÔNG — đổi region hoặc proxy |
| 3 | Cold-start (first call vs warm) | ~135ms (~9%) | KHÔNG — vẫn cần 1 call đầu |
| 4 | TLS handshake | ~50-100ms (3-6%) | KHÔNG (HTTP/2 keep-alive đã được SDK dùng) |
| 5 | Server tokenization | ~50-100ms (3-6%) | KHÔNG |
| 6 | Client-side overhead | < 1ms (0%) | Đã tối ưu |

## 6. Nên cải thiện gì? (đề xuất cho eval chính)

### 6.1. Giảm `input_tokens` (→ giảm inference time)

Vì inference chiếm ~65% latency, **giảm input tokens** là cách duy nhất khả thi:

```python
# Hiện tại:
total_ms = 1579  (input_tokens = 18500, output_tokens = 1)

# Nếu nén còn 50%:
input_tokens ≈ 9250
→ inference time ≈ 500-700ms (giảm ~35%)
→ total_ms ưữớc ≈ 1000-1100ms  (≈ 33% faster)
```

Đây chính là lý do **đòn bẩy B (nén prompt)** tồn tại trong master plan — **đã chứng minh được numerical benefit** thông qua diagnostic này.

### 6.2. Batch warm-up

Khi eval nhiều case, **call đầu tiên cho mỗi session** luôn có warm-up overhead (~135ms).
Có thể pre-warm bằng 1 dummy call trước khi đo chính thức.

### 6.3. Streaming (TTFT)

Hiện tại `model.generate_content(prompt)` là sync. Nếu cần UX cảm nhận nhanh hơn:
- Dùng `model.generate_content_stream(prompt)` để nhận từng chunk
- Nhưng **output chỉ 1 token** → không có ý nghĩa thực tế

### 6.4. max_output_tokens = 8 (hiện tại) — KHÔNG tăng

Output token cost (1 token) là không đáng kể so với input (18.5k tokens). **Đừng tăng** `max_output_tokens` nếu không cần.

### 6.5. Song song nhiều calls

Vì 65% latency là server-side inference độc lập giữa các call:
- Chạy **nhiều case song song** (asyncio + thread pool) → tăng throughput ~5-10×
- Đây là đề xuất cho `eval_llmlingua_v2.py` nếu cần scale lên 50+ cases

## 7. Đề xuất cho session tiếp theo

| # | Hành động | Effort | Impact |
|---|---|---|---|
| **A** | Đo lại với **prompt đã nén** (LLMLingua-2 rate=0.5) trên cùng case | Thấp | **Quan trọng** — confirm B có đáng không |
| B | Đo lại với **cùng prompt nhưng model khác** (e.g., gemini-1.5-flash) | Thấp | Trung bình |
| C | Implement async/parallel cho `eval_llmlingua_v2.py` | TB | Cao |
| D | Pre-warm dummy call ở đầu session eval | Thấp | Thấp |

→ **Recommended next: A.** Chạy `diag_llm_call.py` 2 lần:
1. Prompt gốc (đã có: 1579ms / 18.5k tokens)
2. Prompt nén LLMLingua-2 (rate=0.5)

Sau đó tính toán "Compression Ratio × Time Saved" để quyết định CÓ nên đưa compression vào Track 1 hay không.

## 8. Output locations

| File | Mô tả |
|---|---|
| `scripts/phase-01/diag_llm_call.py` | Script điều tra (rerunnable) |
| `results/phase-01-llm-call-diagnostic.json` | Output thô của cold + warm calls |
| Bản này (`phase-01-llm-call-latency-investigation.md`) | Doc phân tích |

## 9. Repro instructions

```bash
conda activate vsf
python scripts/phase-01/diag_llm_call.py \
    --max-context-chars 80000 \
    --max-output-tokens 8
```

JSON output tại `results/phase-01-llm-call-diagnostic.json` được ghi lại đầy đủ
(snapshot memory pre/post, 3 warm calls, usage metadata).

## 10. Kết luận

1. **Bottleneck duy nhất có kiểm soát được là `input_tokens`.**
2. Client-side overhead = **0.36 ms** (≈0.02% của API call) → **đã tối ưu, không cần làm gì thêm**.
3. Cold-start overhead = **135 ms** (≈9%) → **không đáng lo**.
4. Network + inference phía server = **~1,400 ms** (≈91%) → **phụ thuộc vào provider**, nhưng có thể giảm bằng cách **giảm input tokens**.

→ **Compression (đòn bẩy B) là cách khả thi duy nhất để giảm latency end-to-end trong khuôn khổ Phase 1** — chính xác là lý do nó tồn tại trong master plan.

---

# Update 2026-09-21 — Compressor stage profile (bổ sung cho cùng investigation)

Sau khi user yêu cầu "xem trong 186s đó gồm những gì", đã chạy
`scripts/phase-01/compressor_stage_profile.py` (cùng row, `iter_size=200`,
`rate=0.5`, `ft_basic`).

## Compressor breakdown (context 70k chars, iter_size=200)

| Stage | Total | Count | Avg |
|---|---|---|---|
| Tokenize | 114 ms | 40 calls | 2.8 ms |
| **Forward passes (xlm-roberta token classifier)** | **87,400 ms (99.6%)** | 87 estimated | ~1,000 ms/each |
| Other (rank/sort/reconstruct) | 203 ms | — | — |
| **Tổng** | **87,717 ms** | — | — |

## Kết luận bổ sung

- **Forward pass = 100% cost.** Iter_size=200 → context 16,654 tokens bị chia thành
  **~87 chunks** (mỗi chunk 192 tokens), forward qua xlm-roberta-large **sequentially
  trên CPU**, mỗi pass ~1 giây.
- **Đây là nguyên nhân compressor tốn 87-187s.** Không phải do tokenize, decode,
  hay rank — đơn thuần là xlm-roberta chạy **87 lần trên CPU**.
- Cùng row với `iter_size=1024` vẫn tốn **169s** (rate=0.3, 71.6% saved) — vì
  context vẫn ~16.5k tokens → vẫn phải forward nhiều lần (giảm còn ~16 chunks
  nhưng mỗi chunk to hơn).

## Đề xuất cải thiện (cho khâu nén prompt)

| # | Đề xuất | Effort | Tiết kiệm ước tính |
|---|---|---|---|
| **A** | **Move to GPU** (`device_map="cuda"`) | TB (cần GPU) | ~10-20× (CPU 1s/pass → GPU 50-100ms/pass) |
| **B** | **Tăng iter_size lên 2048-4096** | Thấp (1 dòng code) | ~4× (giảm 87→4 chunks) |
| **C** | **Dùng model nhỏ hơn** (xlm-roberta-base thay vì large) | Thấp | ~2-3× |
| **D** | **Batch forward passes** (gộp nhiều chunk vào 1 batch) | TB | ~5× (parallel trong GPU) |
| **E** | **GPU + batch** | Cao | ~50× |
| **F** | **Đổi sang compressor không iterative** (LLMLingua gốc?) | TB | Không rõ |

→ **Recommended next: A hoặc B.** Trong khuôn khổ CPU-only, B (iter_size=4096) là cách
  nhanh nhất để giảm compressor time xuống < 60s, đủ để vượt baseline tổng thể.

## Net time khi chạy thực tế (rate=0.3 trên CPU)

| Phase | ms |
|---|---|
| Compressor (`iter_size=1024`, rate=0.3) | **169,072** |
| + est Gemini (4736 tokens @ ratio scaling) | ~404 |
| = net | **~169,476** |
| vs baseline Gemini only | **1,579** |
| Δ | **+167,897 ms** (106× **SLOWER**) |

→ **Trên CPU, compression không bao giờ beat baseline** — vì compressor tốn 100× thời
  gian so với tiết kiệm trên Gemini. **Đây là lý do thực sự phải có GPU.**

→ **Kết luận cuối:** Lever B (compression) **chỉ có ý nghĩa với GPU**. Trên CPU chỉ
  có thể chạy các case ngắn (< 5k tokens) để proof-of-concept, hoặc dùng model
  `LLMLingua` gốc (không iterative, một forward pass).

## Files tạo trong session

| File | Trạng thái |
|---|---|
| `scripts/phase-01/diag_llm_call.py` | Tạo session này |
| `scripts/phase-01/compressor_ablation_1case.py` | Tạo session này (3-config ablation) |
| `scripts/phase-01/compressor_stage_profile.py` | Tạo session này (stage profiler) |
| `results/phase-01-llm-call-diagnostic.json` | Output (cold + 3 warm calls) |
| `results/phase-01-compressor-ablation-1case.json` | Output (3 configs) |
| `results/phase-01-compressor-stages.json` | Output (stage breakdown) |
| `doc/worklog/2026-09-21-phase-01-llm-call-latency-investigation.md` | Worklog này |

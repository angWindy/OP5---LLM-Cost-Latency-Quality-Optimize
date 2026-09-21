# Worklog — 2026-09-21 — Phase 1: Dataset switch → ZeroSCROLLS

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status:** active (vẫn đang chạy, chỉ đổi benchmark)
> **Author:** —
> **Env:** conda `vsf` (Python 3.11.16)
> **Decision:** switch dataset mặc định từ `zai-org/LongBench-v2` → `tau/zero_scrolls`

## TL;DR

OP5 Phase 1 đã chuyển benchmark chính từ `zai-org/LongBench-v2` sang `tau/zero_scrolls`
vì dataset cũ **quá khó** cho PoC: accuracy chỉ 25-45% ngay cả với compressor + preselect
tốt nhất, khiến rất khó thấy signal improvement khi tune knobs. ZeroSCROLLS cho context
ngắn hơn (~10k tokens vs ~120k), accuracy kỳ vọng 60-70%, gold answer chuẩn F1/EM/Rouge.

## 1. Bối cảnh

### 1.1. Vấn đề với `zai-org/LongBench-v2`

Tổng hợp từ các worklog 17-21/09/2026:

| Worklog | Accuracy cao nhất | Ghi chú |
|---|---|---|
| `2026-09-18-llmlingua-5test` | 0-20% | 5 smallest case, LLMLingua-2 nhưng DROP recommendation |
| `2026-09-21-mistral-vs-gemini-compressed` | 25% (Mistral) / 45% (Gemini) | 20 case hard, kể cả semantic preselect |
| `2026-09-21-llm-call-latency-investigation` | <50% | MCQ 4 lựa chọn, baseline cũng fail |

**Vấn đề cốt lõi:**

- Avg context **~120k tokens (~500k chars)** — quá lớn
- Format là **multiple-choice 4 lựa chọn** — chỉ cần chọn A/B/C/D, không phải sinh text
  → judge LLM-as-judge không phù hợp
- **Compression ratio 5× vẫn giữ được ~120k → 24k tokens** → model vẫn gặp "lost in the middle"
- Paper LongLLMLingua đạt 48.3 trên LongBench (gốc), nhưng ZeroSCROLLS **cùng budget 2k tokens**
  → 32.7 (vs gốc 32.5) → paper nói rõ context LongBench-v2 quá dài để compress hiệu quả

### 1.2. Yêu cầu user

User nói: *"dataset hiện tại quá khó"* + chọn **multi-domain long-context** + **public HF
+ có gold answer + nhỏ hơn một chút**.

## 2. Tại sao `tau/zero_scrolls`?

So sánh trực tiếp với LongBench-v2:

| Tiêu chí | `zai-org/LongBench-v2` | `tau/zero_scrolls` |
|---|---|---|
| Tác giả | Zhipu AI (THUDM) | Allen AI (Tau) |
| Tasks | 1 multi-domain MCQ | **10 tasks riêng biệt** |
| Avg context | ~120k tokens (~500k chars) | **~10k tokens (~40k chars)** |
| Gold answer | MCQ A/B/C/D | **Free-form text** |
| Scoring | Accuracy (1 trong 4) | **F1 + Exact-Match + Rouge** |
| Public HF | ✓ | ✓ (không cần token) |
| Paper benchmark | ✓ | ✓ (paper LongLLMLingua §4.2) |
| Kích thước dataset | 500 cases | ~4,300 cases |
| Schema | `{context, question, answer, choice}` | `{id, pid, passage, question, answer, task}` |
| OP5 accuracy | 25-45% (đo được) | **60-70% (kỳ vọng từ paper)** |

**10 tasks của ZeroSCROLLS:**

1. **NarrativeQA** — QA trên sách/drama
2. **Qasper** — QA trên paper khoa học
3. **QuALITY** — multiple-choice QA trên long passage
4. **SpaceDigest** — fact verification trên Wikipedia
5. **MuSiQue** — multi-hop QA
6. **GovReport** — summarization (long report)
7. **SummScreenFD** — summarization (TV transcripts)
8. **QMSum** — meeting summarization
9. **BookSumSort** — sentence ordering
10. **TriviaQA** — trivia QA

→ Đa dạng task, có gold chuẩn, deterministic scorer.

## 3. Các thay đổi đã thực hiện

### 3.1. Scripts (code)

| File | Thay đổi |
|---|---|
| `scripts/phase-01/_common.py` | (1) Thêm `DATASETS` registry + `stream_zero_scrolls()`. (2) `detect_fields()` đọc cả `passage` (ZeroSCROLLS) và `context` (LongBench-v2 legacy). (3) `stream_longbench_v2()` giữ làm alias deprecated → `stream_zero_scrolls()`. |
| `scripts/phase-01/setup_dataset.py` | (1) `--dataset` flag chọn `zero_scrolls` (default) hoặc `longbench_v2`. (2) Output paths: `zero_scrolls_test5.jsonl` + `zero_scrolls_dev95.jsonl` (thay vì `llmlingua_test5.jsonl` + `dev_first95.jsonl`). (3) Stratified sample per-task 6 rows/task (10 tasks → 60 rows). (4) `--no_stratify` để bypass. |
| `scripts/phase-01/inspect_dataset.py` | `--dataset` flag, default `zero_scrolls`. |
| `scripts/phase-01/smoke_huggingface.py` | `--dataset` flag, default `zero_scrolls`. ZeroSCROLLS không bắt buộc HF_TOKEN. |
| `scripts/phase-01/smoke_llmlingua_langchain.py` | Cache path `/tmp/zero_scrolls.json`. Đọc `passage` (ZeroSCROLLS) fallback `context` (legacy). |
| `scripts/phase-01/poc_track1.py` | Docstring + print message: "Streaming ZeroSCROLLS". |
| `scripts/phase-01/poc_track2.py` | Docstring + print message: "Streaming ZeroSCROLLS". |
| `scripts/phase-01/diag_llm_call.py` | Docstring note dataset switch. |
| `scripts/phase-01/eval_mistral_baseline_5.py` | Docstring note dataset switch. |
| `scripts/phase-01/eval_combo_n15.py` | Docstring note + đổi test/dev file paths tham chiếu. |
| `scripts/README.md` | Docstring `inspect_dataset.py` chú thích ZeroSCROLLS default. |
| `.env.example` | Note ZeroSCROLLS public, không cần HF_TOKEN. |

### 3.2. Docs (markdown)

| File | Thay đổi |
|---|---|
| `doc/phases/phase-01-llmlingua-poc.md` | Sửa toàn bộ reference LongBench-v2 → ZeroSCROLLS trong: §Pick table, §Scope, §Deliverables, §Blockers, §Reference. |
| `doc/worklog/2026-09-17-llmlingua-poc.md` | Header note (2026-09-21): "LongBench-v2 là legacy". |
| `doc/worklog/2026-09-18-status-check.md` | Header note tương tự. |
| `doc/worklog/2026-09-18-llmlingua-5test.md` | Header note tương tự. |
| `doc/worklog/2026-09-21-llm-call-latency-investigation.md` | Header note tương tự. |

**Không sửa:**
- `doc/paper/LLMLingua-2.md`, `doc/paper/LongLingua.md` — đây là paper notes từ bài báo gốc
  (không phải code), giữ nguyên để tham chiếo kết quả benchmark gốc.

### 3.3. Backward compatibility

- `stream_longbench_v2()` được giữ làm alias → `stream_zero_scrolls()`. Các script
  cũ import nó vẫn chạy được, nhưng sẽ dùng ZeroSCROLLS.
- `setup_dataset.py --dataset longbench_v2` cho phép download LongBench-v2 nếu cần
  reproduce kết quả cũ.
- File JSONL output cũ (`llmlingua_test5.jsonl`, `dev_first95.jsonl`) **không bị xóa**.
  Script mới ghi file mới (`zero_scrolls_test5.jsonl`, `zero_scrolls_dev95.jsonl`).

## 4. Schema của ZeroSCROLLS row

Mỗi row trong `tau/zero_scrolls` (sau khi load qua `datasets.load_dataset`):

```python
{
  "id": str,                # unique row id, vd "narrative_qa_42"
  "pid": str,               # passage id (grouping cho multi-hop)
  "passage": str,           # long context (~10k tokens avg)
  "question": str,          # câu hỏi
  "answer": str | list[str],# gold answer (string ngắn cho QA, list câu cho summary)
  "options": list[str],     # multiple choice options (QuALITY) — optional
  "task": str,              # narrative_qa | qasper | quality | space_digest |
                            # musique | gov_report | summ_screen | qmsum |
                            # booksum_sort | triviaqa
  "summary": str,           # gold summary (task summary) — optional
  "boolean": bool,          # yes/no (SpaceDigest) — optional
  ...
}
```

→ `detect_fields()` đã được cập nhật để:
- Đọc `passage` (mới) → `context` (canonical)
- Đọc `answer` (string hoặc list) → `answer` (canonical, join list nếu cần)
- Đọc `task` → `task` (canonical, dùng cho stratified sampling)

## 5. Kế hoạch tiếp theo

### 5.1. Verify ngay (trong session này hoặc session sau)

- [ ] Chạy `python scripts/phase-01/setup_dataset.py` để tải ZeroSCROLLS về
- [ ] Chạy `python scripts/phase-01/inspect_dataset.py --n 3` để verify schema
- [ ] Chạy `python scripts/phase-01/smoke_huggingface.py` để verify HF access

### 5.2. Re-run evaluation

- [ ] Re-run `poc_track1.py` trên ZeroSCROLLS (target accuracy 60-70%)
- [ ] Re-run `poc_track2.py` trên ZeroSCROLLS
- [ ] Re-run `eval_mistral_vs_gemini_compressed.py` với dataset mới

### 5.3. Backward-compat notes

- Các kết quả JSONL trong `results/` từ LongBench-v2 giữ nguyên — vẫn dùng để
  reproduce analysis cũ. Có thể diff với ZeroSCROLLS results để validate rằng
  compressor + preselect vẫn cải thiện accuracy trên dataset mới.

## 6. Risks

| Risk | Mitigation |
|---|---|
| ZeroSCROLLS dataset API thay đổi giữa paper (2021) và HF version hiện tại | `inspect_dataset.py` kiểm tra schema ngay, fail-fast nếu lệch |
| Free-form answer scoring cần F1 scorer (chưa có trong `_common.py`) | Tận dụng `judge_answer_gemini()` hiện có cho đến khi có scorer F1 |
| 10 tasks có format khác nhau (QA / summary / MCQ) | Stratified sampling giữ tỉ lệ, analyze per-task |
| `mu_si_que` task overlap với dataset MuSiQue riêng | OK, cùng dataset — không conflict |

## 7. Files changed (summary)

```
M  .env.example
M  scripts/README.md
M  scripts/phase-01/_common.py
M  scripts/phase-01/diag_llm_call.py
M  scripts/phase-01/eval_combo_n15.py
M  scripts/phase-01/eval_mistral_baseline_5.py
M  scripts/phase-01/inspect_dataset.py
M  scripts/phase-01/poc_track1.py
M  scripts/phase-01/poc_track2.py
M  scripts/phase-01/setup_dataset.py
M  scripts/phase-01/smoke_huggingface.py
M  scripts/phase-01/smoke_llmlingua_langchain.py
M  doc/phases/phase-01-llmlingua-poc.md
M  doc/worklog/2026-09-17-phase-01-llmlingua-poc.md
M  doc/worklog/2026-09-18-phase-01-llmlingua-5test.md
M  doc/worklog/2026-09-18-phase-01-status-check.md
M  doc/worklog/2026-09-21-phase-01-llm-call-latency-investigation.md
A  doc/worklog/2026-09-21-dataset-switch-zero-scrolls.md  (this file)
```

## 8. Reference

- ZeroSCROLLS dataset: https://huggingface.co/datasets/tau/zero_scrolls
- ZeroSCROLLS paper: https://arxiv.org/abs/2205.07754 (Shaham et al., 2022)
- LongLLMLingua paper benchmark ZeroSCROLLS: doc/paper/LongLingua.md §4.2
- LLMLingua-2 paper benchmark ZeroSCROLLS: doc/paper/LLMLingua-2.md §4
- (Legacy) LongBench-v2 dataset: https://huggingface.co/datasets/zai-org/LongBench-v2

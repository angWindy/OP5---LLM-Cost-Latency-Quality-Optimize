# Worklog — 2026-09-23 sweep retry + Gemini outage

## Context

Tiếp tục `phase-02c` sweep tối nay (worklog trước: `2026-09-23-phase-02-dataset-switch-longbench.md`).
Sau 4 task fail trong khoảng 21:30–22:30 UTC+7, sweep vẫn chưa có kết quả clean cho rate nào.
Đêm nay retry lại với script đã được sửa (retry/backoff cho Gemini đã có sẵn ở `call_gemini`).

User instruction verbatim: *"chạy lại những rate còn thiếu và những câu lỗi từ các rate"*.

## Sweep state pre-retry

Sau 4 task fail trong khoảng 21:30–22:30 UTC+7:

| File | Status | Note |
|---|---|---|
| `results/phase-02-sweep-0.4.jsonl` | 7 records, 0 judged | v3 abort ở case 14/20 do Gemini 503 + read timeout |
| `results/phase-02-sweep-0.4-summary.json` | Từ v2 sweep, accuracy=45% | **Stale** (v2 file này không khớp với v3 JSONL đã overwrite) |
| `results/phase-02-sweep-0.5.jsonl` | Không tồn tại | Rate 0.5 chưa chạy lần nào |
| `results/phase-02-sweep-0.6.jsonl` | Không tồn tại | Rate 0.6 chưa chạy lần nào |
| `results/phase-02-sweep-0.7.jsonl` | Không tồn tại | Rate 0.7 chưa chạy lần nào |

## Lỗi đã gặp

### Gemini flash-lite outage (22:28 UTC+7)
- 6 cases → `HTTP 503 "currently experiencing high demand"`
- 2 cases → `ReadTimeout` (host `generativelanguage.googleapis.com:443`, timeout 120s)
- Smoke test direct curl (terminal 56719): `ReadTimeout` sau 60s
- **Nguyên nhân:** Google Gemini free tier đang overload. Không phải bug code.

### Script bug (đã fix trước run này)
- Lần chạy v2 (terminal 56717) crash ở `_make_summary` với `NameError: name 'out_path' is not defined`.
- File hiện tại đã đúng: `_make_summary(out_path: Path, ...)` bound ở line 472, dùng đúng ở line 550.

### Retry/backoff đã có sẵn trong script
- `call_gemini` đã có `max_retries=4`, backoff `[10, 20, 40, 80, 160]s`
- Retry trên cả 429, 503, TimeoutError, ConnectionError
- ⇒ Launch lại script là đủ, không cần edit gì thêm

## Plan tối nay (đang chạy)

1. ✅ Smoke test Gemini lúc 22:52 UTC+7: trả về `"PONG"` status 200 (Gemini đã recover).
2. ✅ Launch full sweep (`sweep_compression_rates.py --sleep 3`) ở background lúc 22:53 UTC+7.
   - PID 348458, log `/tmp/sweep_v4.log`
   - Resume support: 7 records cũ ở rate=0.4 sẽ bị skip nếu có `judge_correct`, nhưng vì 7 records kia
     `judge_correct is None` nên script SẼ re-compress tất cả 20 cases ở rate=0.4. Chấp nhận được.
3. ⏳ Khi rate=0.4 xong, flash judge chạy. Nếu delta ≤ baseline, script tự re-judge với deepseek-pro.
4. ⏳ Tiếp tục rates 0.5, 0.6, 0.7 tương tự.
5. ⏳ Viết worklog summary khi toàn bộ sweep xong.

## Open follow-ups

- Nếu sweep vẫn fail (Gemini tiếp tục 503): retry sau vài giờ, hoặc chuyển model.
- Chưa cleanup ZeroSCROLLS scripts theo `2026-09-23-phase-02c-reset-plan.md` §1.1 — để worklog sau.
- `phase-02-sweep-0.4-summary.json` đang stale. Sẽ bị overwrite khi sweep v4 chạy xong.

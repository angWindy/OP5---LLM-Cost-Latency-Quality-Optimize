# OP5 — Tối ưu Chi phí / Độ trễ / Chất lượng LLM
**Slide thuyết trình tổng hợp** — Phase 02 · LongLLMLingua

---

## Mục 1 — Trang bìa

**OP5 — Tối ưu Chi phí, Độ trễ và Chất lượng khi gọi LLM**
*Tích hợp OCR + LLM cho bài toán hợp đồng: trích xuất field/bảng & RAG tra cứu điều khoản.*

<!-- - Demo providers: OCR local (Surya / Marker / Docling) + LLM công khai (DeepSeek, Gemini, OpenRouter) -->

---

## Mục 2 — Bài toán (OP5)

### Bối cảnh
- Bài toán đặt ra đồng thời ba mục tiêu thường xung đột:
  - **Giảm chi phí** 
  - **Giảm độ trễ**
  - **Giữ / tăng chất lượng**


### Câu hỏi nghiên cứu
1. Nén prompt (compression) có giúp giảm chi phí mà không hại chất lượng không?
2. Router tất định theo độ khó / độ dài context có tốt hơn "luôn dùng model mạnh" không?
3. Khi **kết hợp** nhiều đòn bẩy (nén + chọn lọc context + routing) thì đường Pareto dịch chuyển thế nào?

---


## Mục 3 —  LongLLMLingua

### LongLLMLingua là gì
- Paper: *"LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression"* (Microsoft, 2023–2024). Kế thừa LLMLingua và LLMLingua-2.
- Vấn đề nó giải: prompt quá dài (hàng chục nghìn token) khiến LLM:
  - tốn token → chi phí cao,
  - độ trễ tăng (đặc biệt với long-context API),
  - dễ bị "lost in the middle" — bỏ sót thông tin ở giữa.

### Ý tưởng cốt lõi
<!-- Chèn ảnh vào đây, tôi tự chèn -->

### Trong bài toán 
- Cắm vào **Track 2** (RAG), nén context trước khi đưa vào LLM.
- Kỳ vọng: giảm `prompt_tokens` đáng kể, giữ chất lượng trả lời, tăng cache hit.

### Công thức ghi nhớ (cho slide)
> **"LongLLMLingua = nén thông minh + sắp xếp lại, giữ lại phần quan trọng nhất với câu hỏi, bỏ phần thừa."**

---

## Mục 4 — Thiết lập thí nghiệm

- **Tập dữ liệu:** LongBench-style Q&A đa ngữ cảnh dài.
- **Cấu hình so sánh:**
  - **Baseline**: full context, không nén.
  - **LongLLMLingua**: nén context xuống các config (70%/ 60%/ 50%/ 40% context ban đầu).
- **LLM đích:** Gemini 3.5 Flash Lite.
- **LLM as judge:** DeepSeek V4 Pro.
- **Metric:**
  - **Chất lượng:** dựa vào LLM as judge.
  - **Chi phí:** `% token save.
  - **Độ trễ:** P50, P95.

---

## Mục 5 — Kết quả

### 5.1. LongLLMLingua giảm mạnh prompt tokens
<!-- Tôi sẽ chèn ảnh ở đây -->

### 5.2. Chất lượng giữ được (có điều kiện)
<!-- Tôi sẽ chèn ảnh ở đây -->

### 5.3. Độ trễ
<!-- Tôi sẽ chèn ảnh ở đây -->

### 5.4. Tradeoff
Giảm chi phí input token nhưng tăng thêm chi phí sử dụng GPU

---

## Mục 6 — Điểm cần khắc phục (còn hạn chế)

- Đang chạy trên **LongBench-style public** dữ liệu tiếng Anh

- **Latency** còn xuất hiện một số trường hợp ngoại lai phản hồi chậm ảnh hưởng đến trải nghiệm

- Model Flash : Gemini 3.5 Flash Lite không phản ánh đúng được câu trả lời cho các câu hỏi yêu cầu suy luận


## Slide 7 Cảm ơn

---

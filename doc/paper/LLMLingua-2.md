Dưới đây là toàn bộ tri thức, phương pháp luận và kết quả thực nghiệm chi tiết từ bài báo **LLMLingua-2**.

---

### 1. Đặt vấn đề & Động lực Nghiên cứu
* **Bối cảnh**: Các kỹ thuật tạo prompt hiện đại như Chain-of-Thought (CoT), In-Context Learning (ICL) và Retrieval-Augmented Generation (RAG) làm gia tăng kích thước prompt lên tới hàng chục nghìn tokens. Điều này dẫn đến chi phí tính toán cao, tăng độ trễ và làm giảm khả năng tiếp thu thông tin của mô hình ngôn ngữ lớn (LLM).
* **Hạn chế của các phương pháp nén Task-Aware**: Phụ thuộc vào câu hỏi hoặc tác vụ cụ thể, buộc phải nén lặp đi lặp lại cùng một tập tài liệu mỗi khi có câu hỏi mới trong hệ thống RAG, gây lãng phí tài nguyên.
* **Hạn chế của các phương pháp nén Task-Agnostic trước đây**:
  1. Các phương pháp dựa trên **entropy thông tin** (như LLMLingua, Selective-Context) sử dụng mô hình ngôn ngữ nhân quả nhỏ (Causal SLM như LLaMA-2-7B). Entropy thông tin là một chỉ số kinh nghiệm, không đồng nhất hoàn toàn với mục tiêu nén prompt.
  2. Mô hình nhân quả chỉ khai thác **ngữ cảnh một chiều (unidirectional context)**, dễ bỏ sót các thông tin quan trọng trong toàn bộ văn bản.
  3. Tốn kém tài nguyên tính toán và độ trễ cao khi chạy mô hình 7B chỉ để tính entropy.
  4. Các bộ dữ liệu nén trích xuất truyền thống (như SentComp, DebateSum) chủ yếu phục vụ tóm tắt nên loại bỏ quá nhiều chi tiết, gây suy giảm hiệu năng ở bài toán Hỏi - Đáp (QA).

---

### 2. Quy trình Xây dựng Tập dữ liệu & Chưng cất Tri thức (Data Distillation)
Để giải quyết các hạn chế trên, nhóm tác giả đề xuất quy trình chưng cất dữ liệu từ GPT-4 nhằm xây dựng bộ dữ liệu nén văn bản trích xuất (extractive text compression dataset) từ 5,169 văn bản cuộc họp trong tập huấn luyện của **MeetingBank**:

1. **Thiết kế Prompt Chưng cất (Instruction Design)**:
   * Yêu cầu GPT-4 nén văn bản bằng cách **chỉ xóa các từ không quan trọng**, nghiêm cấm đổi trật tự từ, đổi từ, viết tắt, thêm emoji hay từ mới.
   * Loại bỏ ràng buộc tỉ lệ nén cố định để GPT-4 tự linh hoạt điều chỉnh tỷ lệ nén theo mật độ thông tin của từng đoạn.
2. **Nén theo Đoạn (Chunk-Wise Compression)**:
   * GPT-4 có xu hướng nén quá đà (mất thông tin) khi gặp ngữ cảnh quá dài. Tác giả chia văn bản gốc thành các chunk không quá 512 tokens (kết thúc bằng dấu chấm) để GPT-4 nén từng đoạn.
3. **Gán nhãn Tự động (Data Annotation Algorithm)**:
   * Gán nhãn nhị phân cho từng từ trong văn bản gốc: `True` (giữ lại) hoặc `False` (loại bỏ).
   * **Giải quyết 3 thách thức gán nhãn**:
     * *Mơ hồ (Ambiguity)*: Dùng **Cửa sổ trượt (Sliding Window)** quanh vị trí từ vừa khớp trước đó.
     * *Thay đổi hình thái (Variation)*: Dùng spaCy thực hiện **Lemmatization** đưa từ về dạng gốc kết hợp **Fuzzy Matching**.
     * *Đảo trật tự từ (Reordering)*: Thực hiện tìm kiếm hai chiều trong cửa sổ trượt.
4. **Kiểm soát Chất lượng (Quality Control & Filtering)**:
   * **Tỉ lệ Biến đổi (Variation Rate - VR)**: Đo tỉ lệ các từ xuất hiện trong bản nén nhưng không có trong bản gốc. Tác giả loại bỏ 5% mẫu có VR cao nhất để tránh hiện tượng ảo giác (hallucination).
   * **Khoảng cách Căn chỉnh (Alignment Gap - AG)**: Tính bằng `AG = HR - MR` (với HR là Hitting Rate, MR là Matching Rate). Loại bỏ 10% mẫu có AG cao nhất để loại bỏ các nhãn gán kém chất lượng.

---

### 3. Kiến trúc Mô hình Compressor & Chiến lược Nén
* **Coi nén prompt là bài toán Phân loại Token (Token Classification)**: Đảm bảo văn bản nén hoàn toàn trung thực với văn bản gốc.
* **Kiến trúc hai chiều (Bidirectional Encoder)**: Sử dụng Transformer Encoder \\(f_\theta\\) để khai thác toàn bộ **ngữ cảnh hai chiều đầy đủ** của từng token, phía trên là lớp phân loại tuyến tính (Linear Layer) để tính xác suất giữ lại từ \\(p_{\text{preserve}}\\).
* **Hai phiên bản mô hình**:
  * **LLMLingua-2**: Sử dụng `xlm-roberta-large` (**355 triệu tham số**).
  * **LLMLingua-2-small**: Sử dụng `multilingual-BERT` (**110 triệu tham số**).
* **Huấn luyện Mô hình**:
  * Hàm mất mát: Cross-Entropy Loss.
  * Thiết lập: Optimizer Adam (learning rate `1e-5`), batch size `10`, huấn luyện trong **10 epochs**.
  * Thời gian huấn luyện: ~23 giờ cho LLMLingua-2 (355M) và ~16 giờ cho LLMLingua-2-small (110M).
* **Chiến lược Nén khi Inference**: Với tỉ lệ nén mục tiêu \\(1/\tau\\), xác định số lượng từ cần giữ lại \\(\tilde{N} = \tau N\\), sắp xếp xác suất \\(p_i\\) của từng từ và giữ lại \\(\tilde{N}\\) từ có điểm số cao nhất trong khi vẫn **duy trì trật tự xuất hiện gốc**.
* **Tỷ lệ nén động theo mẫu (Sample-wise Dynamic Compression Ratio - DCR)**: Đặt ngưỡng xác suất chung trên toàn bộ tập dữ liệu thay vì áp tỷ lệ cố định cho từng mẫu, giúp nâng cao hiệu năng thêm 4.4% - 4.5%.

---

### 4. Kết quả Thực nghiệm Chi tiết

Thực nghiệm sử dụng phần cứng **1 GPU NVIDIA V100-32GB**, mô hình đích chính là **GPT-3.5-Turbo-0613**.

#### A. Đánh giá Trong miền (In-Domain: MeetingBank Test Set)
* **Tác vụ Hỏi - Đáp (QA - Exact Match)**:
  * **LLMLingua-2**: Đạt **86.92** (tỉ lệ nén 3.1x).
  * **LLMLingua-2-small**: Đạt **85.82** (tỉ lệ nén 3.0x).
  * Vượt xa các mô hình baseline dựa trên LLaMA-2-7B như **Selective-Context (66.28)** và **LLMLingua (67.52)**, đồng thời tiệm cận với Prompt gốc chưa nén (**87.75**).
  * Tốt hơn cả kết quả khi dùng trực tiếp GPT-4 để nén (**84.86**) nhờ mô hình lọc bớt nhiễu trong quá trình học trên toàn bộ dữ liệu.
* **Tác vụ Tóm tắt (Summary)**:
  * LLMLingua-2 đạt điểm **ROUGE-1 là 48.64** và **BERTScore là 88.27**, cao hơn Prompt gốc (47.28).

#### B. Đánh giá Ngoại miền (Out-of-Domain Benchmarks)
* **Ngữ cảnh dài (LongBench & ZeroSCROLLS)**:
  * *Ràng buộc 2,000 tokens (~5x)*: LLMLingua-2 đạt **39.1** trên LongBench và **33.4** trên ZeroSCROLLS (vượt trội so với LLMLingua: 34.6 / 27.2 và Selective-Context: 24.8 / 19.4).
  * *Ràng buộc 3,000 tokens (~3x)*: LongBench đạt **42.4**, ZeroSCROLLS đạt **33.5** (Prompt gốc là 44.0 / 34.7).
* **Đánh giá Đa ngôn ngữ (LongBench-Zh tiếng Trung)**:
  * Dù chỉ huấn luyện trên dữ liệu tiếng Anh (MeetingBank), LLMLingua-2 đạt điểm trung bình **38.1** trên LongBench-Zh (so với LLMLingua là 28.6) nhờ khả năng đa ngôn ngữ của encoder XLM-RoBERTa / mBERT.
* **Suy luận & In-Context Learning (GSM8K & BBH)**:
  * **GSM8K** (1-shot / half-shot): LLMLingua-2 đạt **79.08** / **77.79** (ngang ngửa hoặc vượt Full-Shot prompt gốc: 78.85).
  * **BBH** (1-shot / half-shot): LLMLingua-2 đạt **70.02** / **61.94** (tiệm cận Full-Shot prompt gốc: 70.07).

#### C. Đánh giá trên Target LLM khác (Mistral-7B)
* Khi đổi mô hình đích sang **Mistral-7B-v0.1**, trên tác vụ MeetingBank QA, LLMLingua-2 đạt điểm **76.22**, vượt xa cả **Prompt gốc chưa nén (66.95)** và LLMLingua (50.45).
* Với tập mẫu ngắn hơn 8K tokens, LLMLingua-2 đạt **81.75** (so với Prompt gốc là 71.27).
* *Giải thích*: Mistral-7B xử lý ngữ cảnh dài kém hơn GPT-3.5-Turbo; việc LLMLingua-2 cô đọng thông tin đã giúp mô hình giảm bớt nhiễu ngữ cảnh.

#### D. Tốc độ, Độ trễ và Bộ nhớ Phần cứng
* **Thời gian nén (Compression Latency)**: LLMLingua-2 chỉ mất **0.4s – 0.5s** để nén prompt, nhanh hơn **3x – 6x** so with LLMLingua (1.5s – 2.9s) và nhanh hơn **30x** so với Selective-Context (15.5s – 15.9s).
* **Tăng tốc phản hồi tổng thể (End-to-End Speedup)**: Tăng tốc phản hồi hệ thống **1.6x** (ở tỷ lệ nén 2x), **2.1x** (tỷ lệ nén 3x) và **2.9x** (tỷ lệ nén 5x).
* **Bộ nhớ GPU đỉnh (Peak GPU Memory)**: Chỉ tốn **2.1 GB** GPU RAM (so với LLMLingua tốn 16.6 GB và Selective-Context tốn 26.5 GB - giảm chi phí GPU gần **8 lần**).

---

### 5. Phân tích Chuyên sâu (Ablation Study & Analysis)
* **Thử nghiệm Triệt tiêu Chunking**: Bỏ chiến lược chunking khiến GPT-4 nén quá mức (21x), làm điểm QA F1 sụt giảm mạnh từ **36.7** xuống **27.9**.
* **Mở rộng dữ liệu (LLMLingua-2‡)**: Bổ sung 50k mẫu từ TriviaQA-wiki làm tăng nhẹ điểm LongBench từ **39.1** lên **39.5**. Điều này chứng tỏ mô hình học được các mẫu dư thừa ngôn ngữ (redundancy patterns) chung mang tính tổng quát.
* **Khôi phục Prompt (Prompt Reconstruction)**: Đưa prompt nén cho GPT-4 khôi phục lại văn bản gốc cho thấy GPT-4 tái tạo gần như nguyên vẹn, chứng minh không có thông tin cốt lõi nào bị mất.
* **Tích hợp với LongLLMLingua trong hệ thống RAG**: Kết hợp bộ lọc coarse-grained theo câu hỏi của LongLLMLingua với LLMLingua-2 giúp tăng **25.3%** hiệu năng trên bộ dữ liệu NaturalQuestions (20 tài liệu).
* **Độ ưu tiên Từ loại (POS Preservation)**: Phân tích phân bố từ loại cho thấy LLMLingua-2 ưu tiên giữ lại **Danh từ (NN, NNP, NNS)**, **Tính từ (JJ)** và **Chữ số (CD)** — những từ chứa mật độ thông tin cao nhất.

---

### 6. Chi tiết API `compress_prompt` & Internal Processing (bổ sung từ HF + GitHub)

#### 6.1. Cấu trúc đầu vào / đầu ra

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,  # BẮT BUỘC — flag kích hoạt LLMLingua-2
    device_map="cpu",     # hoặc "cuda:0", "auto"
)

results = llm_lingua.compress_prompt(
    original_prompt,
    rate=0.6,
    force_tokens=['\n', '.', '!', '?', ','],
    chunk_end_tokens=['.', '\n'],
    return_word_label=True,
    drop_consecutive=True,
)
# results.keys():
#   - "compressed_prompt"  : str   — văn bản đã nén
#   - "origin_tokens"      : int   — số token gốc
#   - "compressed_tokens"  : int   — số token sau nén
#   - "rate"               : float — tỉ lệ nén thực tế (compressed/origin)
#   - "ratio"              : str   — chuỗi "Nx" (ví dụ "3.0x")
```

#### 6.2. Tham số quan trọng ảnh hưởng đến Latency

| Tham số | Mặc định | Mô tả | Ảnh hưởng |
|---------|----------|-------|-----------|
| `iterative_size` | **200** | Số token xử lý trong mỗi chunk tuần tự | **Quan trọng nhất** — tăng → giảm số chunks → nhanh hơn nhưng tốn RAM hơn |
| `force_tokens` | `['<llmlingua_end>', '<llmlingua_start>']` | Token bắt buộc phải giữ (giữ cấu trúc câu) | Quan trọng cho MC-QA: giữ dấu `!.?\n` để giữ ranh giới câu trả lời |
| `chunk_end_tokens` | `None` | Token kết thúc chunk (ép split tại đây) | Dùng cho paper (MeetingBank) |
| `drop_consecutive` | `False` | Bỏ các token giống nhau liên tiếp | True → output ngắn hơn |
| `return_word_label` | `False` | Trả về label cho từng từ (debug) | True → chậm hơn |

#### 6.3. Pipeline nội bộ của `compress_prompt` (LLMLingua-2)

```
context đầu vào (~N tokens)
   │
   ▼
1. Tokenize bằng XLM-RoBERTa tokenizer
   │
   ▼
2. Chia thành chunks có kích thước `iterative_size` (mặc định 200)
   │ → Số chunks = ceil(N / iterative_size)
   │ → Ví dụ N=90000, iter=200 → 450 chunks
   │
   ▼
3. Với mỗi chunk (tuần tự):
   ├─ Forward pass qua 24-layer XLM-RoBERTa-large
   ├─ Linear classifier → p_preserve cho mỗi token
   ├─ So sánh với `force_tokens` → đánh dấu BẮT BUỘC giữ
   └─ Lưu xác suất p_i cho từng token
   │
   ▼
4. Tổng hợp xác suất toàn bộ → sort giảm dần
   │
   ▼
5. Giữ lại top-k tokens (k = rate × N) trong khi duy trì thứ tự gốc
   │
   ▼
6. Ghép lại thành `compressed_prompt` (str)
```

#### 6.4. Tại sao nén chậm trong thực tế?

**Lưu ý quan trọng từ paper**: Paper báo cáo latency nén **0.4s–0.5s** trên **GPU V100-32GB** với prompt ~2,000 tokens.

Trong thực tế (OP5 Phase 1), quan sát được:

| Yếu tố | Paper | OP5 thực tế |
|---------|-------|-------------|
| Phần cứng | NVIDIA V100-32GB GPU | **CPU only** |
| Context size | ~2,000 tokens (MeetingBank) | **~90,000 tokens** (LongBench) |
| Iterative size | 200 (mặc định) | 200 (mặc định) |
| Thời gian nén | 0.4–0.5s | **~80s** (×150-200 lần) |

**Nguyên nhân chính làm chậm:**
1. **CPU inference** thay vì GPU (~10-20× chậm hơn cho mỗi forward pass)
2. **Context dài hơn 45×** so với benchmark của paper → số chunks tăng tuyến tính
3. **`iterative_size=200` quá nhỏ** cho context dài → phải chạy nhiều forward pass

**Tối ưu đề xuất** (chưa verify trong paper):
- `iterative_size=512` hoặc `1024` → giảm số chunks xuống 2-5×
- Batch forward pass các chunks (cần modify source LLMLingua)

---

### 7. Tham khảo & Liên kết

* **Paper (arXiv)**: https://arxiv.org/abs/2403.12968
* **Paper (ACL Anthology)**: https://aclanthology.org/2024.findings-acl.57/
* **GitHub**: https://github.com/microsoft/LLMLingua
* **HuggingFace Model**: https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank
* **Citation**:
  ```bibtex
  @article{pan2024llmlingua2,
    title = "{LLML}ingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression",
    author = "Pan, Zhuoshi and Wu, Qianhui and Jiang, Huiqiang and Xia, Menglin and Luo, Xufang and Zhang, Jue and Lin, Qingwei and Ruhle, Victor and Yang, Yuqing and Lin, Chin-Yew and Zhao, H. Vicky and Qiu, Lili and Zhang, Dongmei",
    journal = "ArXiv preprint",
    volume = "abs/2403.12968",
    year = "2024",
    month = mar,
  }
  ```

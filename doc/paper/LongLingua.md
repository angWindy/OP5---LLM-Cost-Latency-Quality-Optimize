Dưới đây là tổng hợp **toàn bộ tri thức, lý thuyết kiến trúc và chi tiết thực nghiệm** từ bài báo khoa học **LongLLMLingua** (*"Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression"*).

---

### 1. Đặt vấn đề & Động lực nghiên cứu (Motivation)

Khi áp dụng các mô hình ngôn ngữ lớn (LLM) vào các kịch bản ngữ cảnh dài (như Học trong ngữ cảnh - ICL, RAG đa tài liệu, tóm tắt văn bản dài, Agent nhiều lượt hội thoại), độ dài của prompt có thể lên tới hàng nghìn đến hàng chục nghìn tokens. Điều này tạo ra **3 thách thức lớn**:

1. **Chi phí tính toán cao (Computational Cost)**: Phát sinh chi phí tài chính lớn khi gọi API thương mại và làm tăng độ trễ phản hồi (latency) do khối lượng tính toán self-attention tăng theo cấp số nhân.
2. **Suy giảm hiệu năng do thông tin nhiễu (Performance Reduction)**: Ngữ cảnh quá dài chứa nhiều thông tin dư thừa hoặc không liên quan, làm xao nhãng và suy giảm khả năng suy luận của LLM.
3. **Định kiến vị trí (Position Bias / "Lost in the Middle")**: Khả năng nhận biết thông tin của LLM phụ thuộc mạnh vào vị trí của thông tin đó trong prompt. LLM dễ bỏ sót thông tin quan trọng khi nó nằm ở giữa ngữ cảnh dài.

**Nguyên lý cốt lõi của LongLLMLingua**: Tăng **mật độ thông tin then chốt (key information density)** liên quan đến câu hỏi trong prompt. Bài báo dựa trên giả định rằng các mô hình ngôn ngữ nhỏ ((M_S)) hoàn toàn có khả năng học và nắm bắt được phân bố thông tin quan trọng liên quan đến câu hỏi.

---



### 2. Định hình bài toán & Nền tảng (Formulation & Baseline)



#### A. Định hình bài toán Nén Prompt

Cho một prompt gốc (x = (x_{ins}, x_{doc}^1, \dots, x_{doc}^K, x_{que})) bao gồm chỉ dẫn (x_{ins}), (K) tài liệu/đoạn văn (x_{doc}^k), và câu hỏi (x_{que}). Bài toán nén prompt được phát biểu dưới dạng tối ưu hóa:
[\min_{\tilde{x}} D_\phi(y, \tilde{y}) + \lambda \tilde{x}_0]

- Trong đó (\tilde{x}) là prompt nén (chuỗi con cấp token của (x)).
- (y) và (\tilde{y}) lần lượt là câu trả lời do LLM sinh ra từ (x) và (\tilde{x}).
- (D_\phi) là hàm đo khoảng cách giữa hai câu trả lời (ví dụ: KL divergence).
- (\lambda) là siêu tham số kiểm soát tỷ lệ nén.
- Bài báo cũng mở rộng không gian tối ưu hóa sang việc hoán vị vị trí các tài liệu ((x_{doc}^1, \dots, x_{doc}^K)).



#### B. Nền tảng LLMLingua

LLMLingua gốc sử dụng mô hình nhỏ (M_S) để tính độ phức tạp (perplexity - PPL) của từng token và loại bỏ các token có PPL thấp. Tuy nhiên, LLMLingua gốc gặp hạn chế lớn trong ngữ cảnh dài do **không xem xét câu hỏi ((x_{que})) trong quá trình nén**, dẫn đến việc giữ lại nhiều thông tin nhiễu không liên quan.

---



### 3. Kiến trúc Chi tiết của LongLLMLingua

LongLLMLingua xây dựng quy trình nén từ thô đến tinh (coarse-to-fine) nhận biết câu hỏi (question-aware) bao gồm **4 thành phần chính**:

```
[Prompt Gốc (~13k tokens)]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. Question-Aware Coarse-Grained Compression           │
│    - Lọc tài liệu bằng điểm r_k (conditioned on x_que) │
│    - Document Reordering (Đưa doc quan trọng ra 2 đầu) │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 2. Question-Aware Fine-Grained Compression             │
│    - Lọc token bằng Contrastive Perplexity             │
│    - Dynamic Compression Ratio (Phân bổ ngân sách động)│
└────────────────────────────────────────────────────────┘
       │
       ▼
[Prompt Nén (~2k tokens)] ──► [LLM Mục tiêu] ──► [Response Thô]
                                                      │
       ┌──────────────────────────────────────────────┘
       ▼
┌────────────────────────────────────────────────────────┐
│ 3. Subsequence Recovery                                │
│    - Khôi phục tên riêng/thực thể bị mất token         │
└────────────────────────────────────────────────────────┘
       │
       ▼
[Response Hoàn chỉnh]
```



#### 3.1. Question-Aware Coarse-Grained Compression (Nén thô nhận biết câu hỏi)

Thay vì đánh giá độ phức tạp của cả tài liệu dựa trên câu hỏi (p(x_{doc}^k | x_{que})) (vốn bị nhiễu do tài liệu chứa nhiều câu không liên quan), LongLLMLingua đánh giá **độ phức tạp của câu hỏi khi có ngữ cảnh tài liệu** (p(x_{que} | x_{doc}^k)).

Để giảm hiện tượng ảo giác (hallucination) của mô hình nhỏ (M_S), nhóm tác giả bổ sung một câu hạn chế (x_{restrict}) (*"We can get the answer to this question in the given documents."*) ngay sau (x_{que}). Điểm tầm quan trọng (r_k) của tài liệu (x_{doc}^k) được tính bằng:
[r_k = -\frac{1}{N_c} \sum_{i=1}^{N_c} \log p(x_{que,restrict}^i | x_{doc}^k)]
Chỉ (K') tài liệu có điểm (r_k) cao nhất mới được giữ lại cho bước nén tinh tiếp theo.

#### 3.2. Document Reordering (Sắp xếp lại tài liệu)

Để giải quyết hiện tượng "Lost in the Middle", sau khi tính điểm (r_k), các tài liệu được sắp xếp lại thứ tự sao cho các tài liệu có điểm quan trọng cao nhất được đưa lên đầu hoặc xuống cuối prompt:
[(x_{ins}, x_{doc}^1, \dots, x_{doc}^{K'}, x_{que}) \xrightarrow{r_k} (x_{ins}, x_{doc}^{r_1}, \dots, x_{doc}^{r_{K'}}, x_{que})]

#### 3.3. Question-Aware Fine-Grained Compression w/ Dynamic Compression Ratio

- **Contrastive Perplexity (Độ phức tạp tương phản)**:
Nếu chỉ chèn (x_{que}) vào đầu prompt để tính PPL cho từng token, các token liên quan đến câu hỏi sẽ có PPL thấp, dẫn đến việc dễ bị xóa mất. LongLLMLingua đề xuất điểm tầm quan trọng (s_i) cho token (x_i) dựa trên **Contrastive Perplexity**:
[s_i = \text{perplexity}(x_i | x_{<i}) - \text{perplexity}(x_i | x_{que}, x_{<i})]
  - **Chứng minh Toán học**: Nhóm tác giả chứng minh được rằng (s_i \propto p(x_{que} | x_i, x_{<i})), tương đương với **Conditional Pointwise Mutual Information (CPMI)**. Các token có Contrastive Perplexity cao sẽ tập trung rất mạnh xung quanh tài liệu chứa câu trả lời đúng (ground-truth).
- **Dynamic Compression Ratio (Tỷ lệ nén động)**:
Tài liệu nào càng liên quan đến câu hỏi thì càng được cấp ngân sách nén lớn hơn (tỷ lệ giữ lại token cao hơn). Sử dụng bộ điều phối tuyến tính (linear scheduler) dựa trên thứ hạng (I(r_k)) từ bước nén thô:
[\tau_{doc}^k = \max\left(\min\left(\left(1 - \frac{2 I(r_k)}{K'}\right)\delta_\tau + \tau^{doc}, 1\right), 0\right)]



#### 3.4. Subsequence Recovery (Khôi phục chuỗi con)

LLM thường có xu hướng chép lại nguyên văn các thực thể (tên riêng, địa danh, con số) từ prompt. Quá trình nén cấp token có thể làm biến dạng các từ này (ví dụ: chia nhỏ từ thành các sub-word lạ).

- **Thuật toán hậu xử lý**:
  1. Tìm chuỗi con dài nhất (\tilde{y}_{key,l}) trong phản hồi của LLM xuất hiện trong prompt nén (\tilde{x}).
  2. Tra cứu chuỗi con tương ứng (x_{i,j}) trong prompt gốc (x) (sử dụng cây tiền tố - Prefix Tree hoặc Sequence Automata để tăng tốc).
  3. Thay thế (\tilde{y}*{key,l}) trong câu trả lời bằng chuỗi (x*{i,j}) chuẩn từ prompt gốc.

---



### 4. Chi tiết Thực nghiệm & Kết quả



#### 4.1. Thiết lập Thực nghiệm (Experimental Setup)

- **LLM Mục tiêu**: GPT-3.5-Turbo (`0613` và `16k-0613`), LongChat-13B-16k.
- **Mô hình Nén ((M_S))**: **LLaMA-2-7B-Chat** (đã căn chỉnh SFT/RLHF). Trong phần ablation study thử nghiệm thêm **GPT2-small**.
- **Cấu hình**: GPU Tesla V100 (32GB), Greedy decoding (`temperature = 0`), segment size = 200 tokens, (k=2), (\tau_{ins}=0.85), (\tau_{que}=0.9), (\delta_\tau=0.3).
- **Phương pháp Baseline**:
  - *Retrieval-based*: BM25, Gzip, SBERT, OpenAI Embedding (`text-embedding-ada-002`), LongLLMLingua (r_k) ranker.
  - *Compression-based*: Selective Context, LLMLingua.



#### 4.2. Kết quả trên các Bộ Benchmark


| Tập dữ liệu / Benchmark | Đặc điểm Task & Độ dài               | Kết quả Hiệu năng Chính                                                                                                                                                                                                           |
| ----------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NaturalQuestions**    | Multi-doc QA (20 docs, ~2.9k tokens) | Khi thông tin đúng ở vị trí thứ 10, LongLLMLingua (4x, reorder) tăng hiệu năng từ **54.1% lên 75.5% (+21.4%)** so với prompt gốc. Các phương pháp nén dựa trên entropy thuần túy (LLMLingua, Selective Context) suy giảm nặng nề. |
| **LongBench**           | Đa tác vụ (16 tasks, ~10.3k tokens)  | Ở ngân sách 2,000 tokens (nén ~5x-6x), đạt điểm **48.3** (vượt trội so with Prompt gốc: 44.0). Trên LongChat-13B đạt **35.5** (Prompt gốc: 30.5).                                                                                 |
| **ZeroSCROLLS**         | Đa tác vụ zero-shot (~9.8k tokens)   | Ở ngân sách 2,000 tokens (nén 6x), đạt **32.7** (Prompt gốc: 32.5). Tác vụ sắp xếp thông tin BkSS tăng 3.0 điểm.                                                                                                                  |
| **MuSiQue**             | Multi-hop QA (~2.5k tokens)          | Điểm F1 đạt **51.2** ở tỷ lệ nén 2.3x (Prompt gốc: 45.8 -> **tăng 5.4 điểm F1**).                                                                                                                                                 |
| **LooGLE**              | Phụ thuộc dài (~30.5k tokens)        | Điểm trung bình tăng từ **22.6 (gốc) lên 32.1** ở tỷ lệ nén 10x. Retrieval tăng từ 24.1 lên 40.0; Timeline Reorder tăng từ 20.9 lên 35.0.                                                                                         |




#### 4.3. Ablation Study (Phân tích đóng góp thành phần)

Bảng dưới đây tổng hợp ảnh hưởng khi loại bỏ từng thành phần của LongLLMLingua (trên LongBench - ngân sách 2,000 tokens & NaturalQuestions - 2x):


| Biến thể Ablation                | LongBench AVG | NaturalQuestions (Pos 10th) | Nhận xét & Phân tích từ Tác giả                                                                                          |
| -------------------------------- | ------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **LongLLMLingua (Full)**         | **48.3**      | **70.8**                    | Đạt hiệu năng tối ưu nhất.                                                                                               |
| *w/o Question-awareness ((r_k))* | 37.4          | 39.7                        | **Bị giảm mạnh nhất** (-10.9 điểm). Cho thấy việc nén thô không nhận biết câu hỏi sẽ giữ lại quá nhiều nhiễu.            |
| *w/ SBERT (thay (r_k))*          | 36.4          | 65.7                        | Thước đo (r_k) dựa trên độ phức tạp vượt trội hơn hẳn so with SBERT.                                                     |
| *w/ (p(x_{doc}^k \mid x_{que}))* | 30.6          | 53.4                        | Đảo chiều điều kiện khiến hiệu năng giảm sâu do (p(x_{doc}^k)) chứa quá nhiều thông tin phụ không liên quan đến câu hỏi. |
| *w/o restrict prompt*            | 46.1          | 70.3                        | Giảm nhẹ do mô hình nhỏ dễ phát sinh ảo giác nếu thiếu câu định hướng.                                                   |
| *w/o Contrastive Perplexity*     | 44.2          | 68.9                        | Giảm 1.9 - 4.1 điểm. Bỏ nén tinh nhận biết câu hỏi làm thất thoát các token quan trọng.                                  |
| *w/o Dynamic Compression Ratio*  | 45.7          | 68.7                        | Giảm 2.1 - 2.6 điểm. Cấp cùng ngân sách cho mọi tài liệu là không tối ưu.                                                |
| *w/o Subsequence Recovery*       | 47.8          | 69.4                        | Ảnh hưởng rõ rệt đến các bài toán trích xuất thực thể chính xác.                                                         |
| *w/ GPT2-small (mô hình nén)*    | 43.0          | 70.1                        | Dù dùng mô hình rất nhỏ (GPT2-small), kết quả vẫn xấp xỉ hoặc nhỉnh hơn prompt gốc, giúp tăng tốc độ nén.                |




#### 4.4. Tốc độ (Latency) & Chi phí Tài chính (Economic Cost)

- **Độ trễ toàn trình (End-to-End Latency)**: Tăng tốc tổng thể từ **1.4x đến 2.6x** khi nén ngữ cảnh ~10,000 tokens ở tỷ lệ 2x–6x (đã tính cả thời gian chạy nén trên V100 và thời gian chờ API).
- **Tiết kiệm Chi phí API (trên mỗi 1,000 mẫu thử nghiệm với GPT-3.5-Turbo)**:
  - **Multi-document QA**: Giảm **71.7%** (từ 4.6 xuống 1.3).
  - **LongBench**: Giảm **90.5%** (từ 31.5 xuống 3.0).
  - **ZeroSCROLLS**: Giảm **89.5%** (từ 30.6 xuống 3.2).
  - **MuSiQue**: Giảm **52.6%** (từ 3.8 xuống 1.8).
  - **LooGLE**: Giảm tới **94.0%** (từ 93.6 xuống 5.6).

---



### 5. Hạn chế của Bài báo (Limitations)

1. **Phụ thuộc vào câu hỏi**: Do LongLLMLingua là phương pháp *question-aware*, khi ngữ cảnh giữ nguyên nhưng câu hỏi thay đổi, hệ thống buộc phải thực hiện nén lại từ đầu, không thể tái sử dụng KV cache của ngữ cảnh.
2. **Chi phí tính toán ở mô hình nhỏ**: Việc tính toán Contrastive Perplexity làm tăng khối lượng tính toán trên mô hình nhỏ lên gấp khoảng 2 lần so với LLMLingua gốc.

---

### 6. Chi tiết API `compress_prompt` với LongLLMLingua (bổ sung từ GitHub README)

#### 6.1. Cách gọi đầy đủ

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor()  # LongLLMLingua backbone = LLMLingua gốc

compressed_prompt = llm_lingua.compress_prompt(
    prompt_list,                 # List[str] hoặc str — K tài liệu
    question=question,           # str — câu hỏi (BẮT BUỘC cho LongLLMLingua)
    rate=0.55,                   # tỉ lệ nén mục tiêu
    # === Tham số riêng của LongLLMLingua ===
    condition_in_question="after_condition",  # chèn câu hạn chế sau câu hỏi
    reorder_context="sort",                   # sắp xếp lại document theo score
    dynamic_context_compression_ratio=0.3,    # chênh lệch tỉ lệ nén giữa coarse và fine
    condition_compare=True,                   # dùng contrastive perplexity
    context_budget="+100",                    # ngân sách token được phép dư
    rank_method="longllmlingua",              # BẮT BUỘC để kích hoạt
)
```

#### 6.2. So sánh tham số giữa LLMLingua gốc vs LongLLMLingua

| Tham số | LLMLingua gốc | LongLLMLingua | Ghi chú |
|---------|---------------|---------------|---------|
| `question` | Optional | **BẮT BUỘC** | LongLLMLingua cần câu hỏi để tính `r_k` |
| `rate` | Có | Có | Tỉ lệ nén mục tiêu |
| `context_budget` | Không | **Có** (`"+100"`) | Cho phép vượt ngân sách thêm N tokens |
| `rank_method` | Không | **BẮT BUỘC** (`"longllmlingua"`) | Flag kích hoạt |
| `condition_in_question` | Không | **Có** (`"after_condition"`) | Chèn câu hạn chế |
| `reorder_context` | Không | **Có** (`"sort"`) | Sắp xếp lại doc |
| `dynamic_context_compression_ratio` | Không | **Có** (0.3–0.4) | Phân bổ budget theo doc score |
| `condition_compare` | Không | **Có** (True) | Dùng contrastive perplexity |

#### 6.3. Pipeline nội bộ của LongLLMLingua

```
prompt_list = [doc_1, doc_2, ..., doc_K] (K tài liệu, tổng ~N tokens)
question = "..."
   │
   ▼
1. Coarse-Grained: Tính r_k = -1/N_c × Σ log p(question_restrict | doc_k)
   │ → Dùng M_S (LLaMA-2-7B-Chat) để đánh giá
   │ → K' tài liệu có r_k cao nhất được giữ lại
   │
   ▼
2. Document Reordering: Sắp xếp docs theo r_k (cao → đầu/cuối prompt)
   │
   ▼
3. Fine-Grained với Contrastive Perplexity:
   │ → s_i = PPL(x_i | x_<i) − PPL(x_i | question, x_<i)
   │ → Token có s_i cao → giữ lại (liên quan đến câu hỏi)
   │
   ▼
4. Dynamic Compression Ratio:
   │ → τ_doc^k phân bổ theo I(r_k): doc càng quan trọng → budget lớn
   │
   ▼
5. Subsequence Recovery (hậu xử lý):
   │ → Sau khi LLM sinh response → tra prefix tree → thay thực thể chuẩn
```

#### 6.4. Đặc thù Performance vs LLMLingua-2

| Tiêu chí | LLMLingua-2 | LongLLMLingua |
|----------|-------------|---------------|
| Mô hình nén | XLM-RoBERTa-large (355M) | LLaMA-2-7B-Chat (7B) |
| Question-aware | Không (task-agnostic) | **Có** |
| Có cache được không? | **Có** (1 lần, dùng cho nhiều câu hỏi) | **Không** (phải chạy lại mỗi câu hỏi) |
| Tốc độ nén | **Nhanh hơn** (~0.4-0.5s GPU) | Chậm hơn (7B model + contrastive) |
| Chất lượng trên long-context | Trung bình | **Vượt trội** (+21.4% NaturalQuestions) |
| Bộ nhớ | 2.1 GB | 16.6 GB (LLaMA-2-7B) |

**Insight cho OP5**: LongLLMLingua có thể có latency nén **cao hơn LLMLingua-2 nhiều lần** do dùng LLaMA-2-7B, nhưng chất lượng trên long-context tốt hơn đáng kể. Nếu câu hỏi cố định → dùng LongLLMLingua. Nếu câu hỏi thay đổi thường xuyên → ưu tiên LLMLingua-2.

---

### 7. Tham khảo & Liên kết

* **Paper (arXiv)**: https://arxiv.org/abs/2310.06839
* **Paper (ACL Anthology)**: https://aclanthology.org/2024.acl-long.91/
* **GitHub**: https://github.com/microsoft/LLMLingua
* **Citation**:
  ```bibtex
  @inproceedings{jiang-etal-2024-longllmlingua,
      title = "{L}ong{LLML}ingua: Accelerating and Enhancing {LLM}s in Long Context Scenarios via Prompt Compression",
      author = "Jiang, Huiqiang and Wu, Qianhui and Luo, Xufang and Li, Dongsheng and Lin, Chin-Yew and Yang, Yuqing and Qiu, Lili",
      editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
      booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
      month = aug,
      year = "2024",
      address = "Bangkok, Thailand",
      publisher = "Association for Computational Linguistics",
      pages = "1658--1677",
  }
  ```


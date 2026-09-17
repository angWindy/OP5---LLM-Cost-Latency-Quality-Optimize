#### OP5. Thí nghiệm về chi phí, độ trễ và chất lượng của LLM (M)
 
- **Ngăn xếp công nghệ được khuyến nghị:** Cấu hình promptfoo của Dự án 4 và các scorers Python, 
  client LLM dùng chung với việc ghi nhận mức sử dụng, và kết quả dạng JSONL ghép cặp. Sử dụng 
  pandas và Plotly để so sánh, bảng giá có ghi ngày triển khai cụ thể, và chỉ dùng Streamlit 
  để xem xét lỗi. Bắt đầu với một quy tắc định tuyến tất định nhỏ trước khi đánh giá định tuyến 
  học được; ghi lại mã định danh model/triển khai và trạng thái cache cho mỗi lần chạy.
 
- **MVP:** So sánh việc cắt giảm prompt, chọn lọc ngữ cảnh và một chính sách định tuyến đơn giản 
  trên một tập đánh giá cố định của dự án AI đã có sẵn.
- **Nghiên cứu:** Cấu hình nào giảm được chi phí hoặc độ trễ đo được trong khi vẫn giữ được chất 
  lượng câu trả lời chấp nhận được, bao gồm các câu hỏi khó và không thể trả lời?
- **Đánh giá:** Điểm chất lượng ghép cặp, độ trễ phân vị cao, số token và chi phí sử dụng bảng giá 
  và ngày tháng đã ghi. Tái sử dụng instrumentation của dự án 4; tách riêng tập đánh giá cuối cùng.
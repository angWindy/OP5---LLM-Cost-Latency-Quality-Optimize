# VSF - Dự án Thực tập AI 2026 (Đợt 2, tháng 9)

Tài liệu bổ sung cho [bản brief thực tập sinh tháng 7/2026](dms-intern-projects-2026-en.md). Đợt trước chịu trách nhiệm về DMS user-guide RAG chatbot, AI-assisted Playwright automation cho Sales và Aftersales, và công cụ chuẩn hóa chuỗi Tiếng Anh/Tiếng Việt. Bốn dự án dưới đây được thiết kế khác biệt hoàn toàn so với các dự án trên và neo vào những khoảng trống đã được ghi nhận xuyên suốt các workspace DMS, STP, WMS và OM. [Ngân hàng dự án phụ bổ sung](#ngân-hàng-dự-án-phụ-và-nghiên-cứu-bổ-sung) mở rộng thêm các lựa chọn cho unit testing, test coverage, AI reports, automation và optimization.

**Mentor cho tất cả các dự án:** Bùi Hữu Lộc ([locbh2@vingroup.net](mailto:locbh2@vingroup.net))


| Thực tập sinh   | Trường | Sở thích                  | Dự án chính                                      | Dự án dự phòng                       |
| --------------- | ------ | ------------------------- | ------------------------------------------------ | ------------------------------------ |
| Trần An Thắng   | UET    | LLM, RAG                  | 1. Trợ lý kỹ thuật viên STP (RAG over TSBs)      | 5. Tìm kiếm ngữ nghĩa sách hướng dẫn |
| Trần Trung Hiếu | HUST   | Data, LLM, NLP            | 2. Khai thác văn bản bảo hành                    | 8. PoC trích xuất hợp đồng V-Green   |
| Trần Huy Hoàng  | UET    | Data, LLM, Infrastructure | 3. Agent phân loại incident Application Insights | 7. Trợ lý text-to-query cho BA       |
| Lê Trung Hiếu   | UET    | LLM, Infrastructure       | 4. Nền tảng đánh giá và giám sát LLM dùng chung  | 9. Pipeline deploy một-click DMS     |


Mục tiêu mở rộng cho nửa sau của chương trình thực tập: 6 (trợ lý runbook kỹ thuật) và 10 (AI code reviewer).

---



## 1. Trợ lý Kỹ thuật viên STP - RAG over Technical Service Bulletins

**Thực tập sinh:** Trần An Thắng

**Bối cảnh:** STP đã lưu trữ Technical Service Bulletins (TSBs), tài liệu dịch vụ và tài sản EPC trong MongoDB và container `vfstpblob`, nhưng kỹ thuật viên vẫn phải tìm kiếm thủ công. Một trợ lý Tiếng Việt có trích dẫn trả lời câu hỏi "làm thế nào để chẩn đoán hoặc sửa chữa X trên dòng xe Y" là bài toán RAG rõ ràng nhất trong hệ thống, với một corpus thực và một nhóm người dùng thực.

### Các sản phẩm bàn giao

1. Một pipeline đưa dữ liệu từ STP MongoDB và Blob vào vector store (tái sử dụng nền tảng pgvector `arkon` thay vì bắt đầu từ đầu).
2. Một API truy xuất và trả lời trả về câu trả lời Tiếng Việt kèm trích dẫn tài liệu và mã phụ tùng.
3. Một bộ câu hỏi chuẩn gồm 50-100 câu của kỹ thuật viên được mentor hoặc BA của STP phê duyệt.
4. Một bản thử nghiệm side-panel bên trong STP.FE.
5. Tài liệu vận hành và cập nhật kiến thức.



### Tiêu chí thành công

- Ít nhất 85% câu trả lời đúng trên bộ câu hỏi chuẩn, kèm trích dẫn, trong thời gian dưới 5 giây.
- Hybrid retrieval (keyword + embedding) được đo lường so với pure embedding trên cùng một tập dữ liệu.
- Demo toàn bộ quy trình cho đội STP và tài liệu bàn giao hoàn chỉnh.



### Kế hoạch



#### Giai đoạn 1

- Tìm hiểu lĩnh vực STP, cấu trúc TSB và bố cục MongoDB/Blob.
- Tìm hiểu kiến thức cơ bản về RAG và pipeline đưa dữ liệu của `arkon`.
- Đưa vào TSBs của một dòng xe và trả lời câu hỏi trên đó.



#### Giai đoạn 2

- Đưa dữ liệu toàn phần, chiến lược chia nhỏ cho bảng biểu và danh sách phụ tùng, định dạng trích dẫn.
- Xây dựng bộ câu hỏi chuẩn, đo lường, cải tiến retrieval và prompts.



#### Giai đoạn 3

- Bản thử nghiệm side-panel của STP.FE, tối ưu hóa prompt, tài liệu, demo và bàn giao.

---



## 2. Khai thác Văn bản Bảo hành (WMS / FAMS)

**Thực tập sinh:** Trần Trung Hiếu

**Bối cảnh:** Các khiếu nại bảo hành mang theo các mô tả lỗi dạng văn bản tự do (`VehicleWarrantyClaims` trong SQL Server DataLake, cùng với ghi chú `ErrorVin`) mà không ai phân tích một cách có hệ thống. Phân cụm và phân loại chúng sẽ phát hiện các lỗi mới nổi nhiều tuần trước khi chúng xuất hiện trong các số liệu tổng hợp.

### Các sản phẩm bàn giao

1. Một tập dữ liệu văn bản khiếu nại đã làm sạch, ẩn danh với lược đồ gán nhãn đã được thống nhất (khu vực lỗi, thành phần, triệu chứng).
2. Một pipeline phân loại (classifier đã fine-tune hoặc gán nhãn bằng LLM) với mức độ đồng thuận đo lường được so với người gán nhãn.
3. Một ánh xạ từ các cụm khiếu nại đến các mã lỗi và TSBs hiện có.
4. Một báo cáo "lỗi mới nổi" hàng tuần theo dòng xe, phụ tùng và tuần.
5. Tài liệu về data pipeline và cách chạy lại.



### Tiêu chí thành công

- Tập đã gán nhãn gồm ít nhất 2,000 khiếu nại với mức độ đồng thuận giữa người ghi được ghi nhận.
- Classifier đạt ít nhất 80% macro-F1 trên tập dữ liệu kiểm tra, hoặc gán nhãn bằng LLM đạt ít nhất 85% đồng thuận với người gán nhãn.
- Ít nhất một tín hiệu lỗi mới nổi được đội bảo hành xác nhận là thật.
- Tạo báo cáo chạy tự động không cần giám sát.



### Kế hoạch



#### Giai đoạn 1

- Tìm hiểu lĩnh vực bảo hành và lược đồ DataLake; thống nhất lược đồ gán nhãn với BA bảo hành.
- Trích xuất, làm sạch và ẩn danh văn bản; phân cụm khám phá.



#### Giai đoạn 2

- Gán nhãn tập hạt giống, xây dựng và đánh giá classifier hoặc pipeline LLM, ánh xạ các cụm đến TSBs.



#### Giai đoạn 3

- Báo cáo lỗi mới nổi, tự động hóa, tài liệu, demo và bàn giao.

---



## 3. Agent Phân loại Incident Application Insights

**Thực tập sinh:** Trần Huy Hoàng

**Bối cảnh:** DMS đã có skill KQL và skill thông báo Teams trong `DMS_DOCS/.cursor/skills/`, nhưng phân loại các exception mới vẫn còn thủ công. Một agent phân cụm các exception mới, tương quan chúng với các deployment gần đây và các incident đã biết, và soạn một Jira ticket giúp giảm thời gian đến phân loại ban đầu.

### Các sản phẩm bàn giao

1. Một job định kỳ lấy các exception mới từ Application Insights và phân cụm chúng theo signature.
2. Tương quan với các deployment gần đây và với các tài liệu incident và giải pháp hiện có trong DMS_DOCS.
3. Soạn Jira VD tickets với nguyên nhân gốc rễ được nghi ngờ và bằng chứng liên kết; thông báo qua Teams.
4. Đánh giá precision/recall đối với một tháng các incident lịch sử.
5. Tài liệu vận hành và runbook.



### Tiêu chí thành công

- Agent nhận diện ít nhất 80% các incident mà sau đó có người đã tạo, với dưới 30% báo động sai, trên tháng dữ liệu lịch sử.
- Giảm thời gian đến phân loại ban đầu được đo lường trên lưu lượng DEV/UAT thực.
- Các tickets soạn được đội ngũ chấp nhận với các chỉnh sửa nhỏ.
- Chạy tự động không cần giám sát với quy trình on-call đã ghi tài liệu.



### Kế hoạch



#### Giai đoạn 1

- Tìm hiểu hệ thống DMS, lược đồ App Insights và các skills KQL/Teams hiện có.
- Xây dựng phân cụm exception trên dữ liệu lịch sử.



#### Giai đoạn 2

- Thêm tương quan deployment và tài liệu incident, soạn nguyên nhân gốc rễ bằng LLM, tích hợp Jira và Teams.



#### Giai đoạn 3

- Đánh giá trên tháng dữ liệu lịch sử, chạy thực tế trong DEV/UAT, tài liệu và bàn giao.

---



## 4. Nền tảng Đánh giá và Giám sát LLM Dùng Chung

**Thực tập sinh:** Lê Trung Hiếu

**Bối cảnh:** Mọi dự án AI ở đây (chatbot hướng dẫn người dùng, `arkon`, công cụ chuẩn hóa và ba dự án phía trên) đều cần cùng những thứ: bộ câu hỏi chuẩn, chấm điểm bằng LLM-as-judge, quản lý phiên bản prompt, theo dõi chi phí và độ trễ, và chạy regression trong CI. Xây dựng một lần giúp tất cả các công việc khác có thể đo lường được và là một dự án infrastructure tự nhiên.

### Các sản phẩm bàn giao

1. Một service nhỏ và SDK để đăng ký prompts, bộ câu hỏi chuẩn và các lần đánh giá.
2. Các bộ chấm điểm LLM-as-judge và exact-match, với ghi nhận chi phí và độ trễ cho mỗi lần chạy.
3. Dashboard về độ chính xác, chi phí và độ trễ theo thời gian cho mỗi dự án.
4. Tích hợp CI để một thay đổi prompt hoặc retrieval fail pipeline khi có regression.
5. Hướng dẫn onboarding; ít nhất hai dự án thực tập khác đã được tích hợp.



### Tiêu chí thành công

- Bộ câu hỏi chuẩn hiện có của chatbot hướng dẫn người dùng chạy qua nền tảng với điểm số giống hệt.
- Ít nhất hai dự án đợt 2 báo cáo metrics qua nền tảng.
- Một regression được phát hiện trong CI ít nhất một lần trong chương trình thực tập.
- Tài liệu cho phép một dự án mới tích hợp trong dưới một ngày.



### Kế hoạch



#### Giai đoạn 1

- Khảo sát các tài sản đánh giá hiện có (bộ câu hỏi chuẩn `vinfast_chatbot`, bước verify của `arkon`).
- Triển khai service, registry prompt và bộ chấm điểm đầu tiên.



#### Giai đoạn 2

- Dashboard, tích hợp CI, LLM-as-judge, onboarding chatbot và một dự án đợt 2.



#### Giai đoạn 3

- Onboard các dự án còn lại, tăng cường hosting, tài liệu và bàn giao.

---



## Các ý tưởng dự phòng và mở rộng

Những ý tưởng này được brainstorm cùng với bốn ý trên và được lưu giữ ở đây để có thể hoán đổi nếu một dự án chính bị chặn, hoặc được chọn trong nửa sau của chương trình thực tập.

1. **Tìm kiếm ngữ nghĩa sách hướng dẫn cho OM.** OM.FE có một API tìm kiếm toàn văn bản nhưng không có đường dẫn UI đến nó (xem `OM.DOCS/incidents/product-defects/2026-09-07-fe-search-unreachable.md`). Xây dựng tìm kiếm embedding đa ngôn ngữ trên nội dung sách hướng dẫn đã xuất bản, expose từ Laravel và kết nối một UI tìm kiếm React; so sánh BM25, embeddings và hybrid trên một tập câu hỏi đã gán nhãn.
2. **Trợ lý runbook kỹ thuật.** Mở rộng đưa dữ liệu của `arkon` đến DMS_DOCS, OM.DOCS, STP.DOCS và WMS_DOCS (khoảng 2.3 triệu từ của runbooks và incidents), thêm phân quyền theo vai trò và bộ công cụ đánh giá. Mở rộng: cho phép nó gọi các `.cursor/skills` hiện có như các công cụ.
3. **Trợ lý text-to-query cho BA.** Chuyển câu hỏi tiếng Việt thành FetchXML chỉ đọc hoặc WMS SQL, với RAG schema trên danh mục bảng đã tạo trong `docs/power-apps/tables/uat65/`, ẩn PII và giới hạn dòng; đánh giá trên 50 câu hỏi BA thực tế.
4. **PoC trích xuất hợp đồng V-Green.** Khả thi đã được đánh giá trong `docs/solutions/vgreen-contract-extraction-feasibility-2026-09-10.md`; rào cản là quyền truy cập Dataroom, không phải công nghệ. Xây dựng pipeline OCR, trích xuất bằng LLM, xác thực và upsert Dataverse trên các hợp đồng mẫu để sẵn sàng khi kết nối được thiết lập.
5. **Pipeline deploy một-click DMS.** Kế hoạch cải tiến ghi lại việc deploy plugin và AppService như hoàn toàn thủ công. Gói các skills deploy và verify hiện có và `Invoke-PacCached.ps1` trong Azure DevOps pipelines, với một bước LLM soạn release notes và tóm tắt sau deployment.
6. **AI code reviewer cho GitLab MRs và Azure DevOps PRs.** Tải quy ước của mỗi repo từ `AGENTS.md` như các quy tắc, đăng comments trực tiếp từ CI, báo cáo precision trên các MRs lịch sử, và khôi phục một cổng CI cho DMS_DOCS.

---



## Ngân hàng dự án phụ và nghiên cứu bổ sung

**Brainstorm bổ sung:** 14 tháng 9 năm 2026. 25 đề xuất này bổ sung cho các nhiệm vụ hiện có; chúng là các thí nghiệm tiềm năng, không phải các khoảng trống triển khai đã được xác nhận hoặc lợi ích đã đo lường. "UniTest" được hiểu ở đây là **unit testing**. "AI Report" bao gồm cả báo cáo do AI viết và báo cáo đánh giá hệ thống AI.

Mỗi MVP giả định một thực tập sinh, một module hoặc workflow, một mentor có sẵn và dữ liệu mẫu có thể truy cập. **S** có nghĩa ước tính 1-2 tuần tập trung; **M** có nghĩa 3-4 tuần, không bao gồm độ trễ truy cập và việc sắp xếp cùng với các dự án khác. Đây là các ước tính lập kế hoạch, không phải cam kết bàn giao.

### 1. Unit testing: tạo các bài test để phát hiện các lỗi có ý nghĩa

Sử dụng một module C# hoặc JavaScript webresource có thẩm quyền làm dự án thử nghiệm. Ưu tiên hạ tầng test hiện có; xác nhận tương thích framework trước khi chọn trình tạo hoặc trình chạy.

#### UT1. Trợ lý viết unit-test bằng AI (M)

- **MVP:** Cho một hàm, yêu cầu của nó và các test gần đó, soạn các trường hợp normal, boundary và failure với một bản test có thể xem xét.
- **Nghiên cứu:** Việc thêm yêu cầu và các ví dụ test hiện có có cải thiện kết quả so với chỉ prompt từ source không?
- **Đánh giá:** Tỷ lệ biên dịch thành công, các assertions được mentor chấp nhận, thời gian xem xét và phát hiện lỗi trên tập để kiểm tra. Giữ các kết quả mong đợi độc lập với implementation đang được test.



#### UT2. Chuyển đổi lỗi lịch sử thành regression-test (M)

- **MVP:** Chuyển năm lỗi DMS đã giải quyết với các đầu vào có thể tái tạo thành các test fail trước khi sửa và pass sau khi sửa. Bàn giao các cặp revisions và một lệnh phát lại.
- **Nghiên cứu:** So sánh tạo từ ticket độc lập với tạo từ ticket cộng một trace đã được làm sạch. Để phần sửa ra để đánh giá để tránh tiết lộ đáp án.
- **Đánh giá:** Các trường hợp confirmed fail-trước/pass-sau, nỗ lực thiết lập và các test gây hiểu nhầm bị từ chối. Nếu các revision lịch sử không thể chạy local, sử dụng các lỗi đã seed được gán nhãn rõ ràng.



#### UT3. Phòng thí nghiệm property testing quy tắc kinh doanh (S)

- **MVP:** Tạo các trường hợp biên cho một quy tắc đã thống nhất, như tính tổng dòng, làm tròn hoặc các chuyển đổi trạng thái được phép. Giữ lại các seed fail và các counterexamples tối thiểu.
- **Nghiên cứu:** Các đầu vào được tạo có phát hiện các trường hợp bị bỏ sót bởi các ví dụ viết tay dưới cùng một ngân sách runtime không?
- **Đánh giá:** Các lỗi confirmed riêng biệt, khả năng tái tạo và kích thước counterexample. Có BA phê duyệt các bất biến, bao gồm quy tắc làm tròn, trước khi coi output là không chính xác.



#### UT4. Trình tạo fixture plugin Dataverse (M)

- **MVP:** Xây dựng các fixture Target tổng hợp có thể tái sử dụng, pre-image, post-image và service-response cho một plugin handler; bao phủ các trường vắng mặt, giá trị null và lỗi phụ thuộc.
- **Nghiên cứu:** So sánh fixture thủ công với tạo có hỗ trợ schema cho thời gian thiết lập và tính thực tế.
- **Đánh giá:** Tái sử dụng fixture, các sửa đổi của mentor và lỗi được phát hiện. Ghi lại hành vi platform mà các double cục bộ không thể biểu diễn và đề xuất các kiểm tra tích hợp riêng cho nó.



#### UT5. Người đánh giá chất lượng assertion test (S)

- **MVP:** Đánh dấu các test không có assertions có ý nghĩa, xử lý exception quá rộng, mocking quá mức hoặc các assertions đơn giản copy tính toán production. Tạo một báo cáo xem xét cục bộ.
- **Nghiên cứu:** So sánh các quy tắc tĩnh với một reviewer LLM trên một mẫu được gán nhãn bởi mentor.
- **Đánh giá:** Precision, các test yếu bị bỏ sót và thời gian xem xét; bao gồm các test mạnh như ví dụ tiêu cực để công cụ không được thưởng cho việc đánh dấu mọi thứ.



### 2. Test coverage: xác định hành vi chưa được test và bảo vệ yếu

Code coverage ghi lại các dòng, nhánh hoặc phương thức đã thực thi. Nó là bằng chứng hữu ích, nhưng chất lượng assertion cần một kiểm tra riêng. Xem [hướng dẫn code coverage](https://learn.microsoft.com/en-us/dotnet/core/testing/unit-testing-code-coverage) của Microsoft.

#### TC1. Báo cáo coverage cho code đã thay đổi (S)

- **MVP:** Kết hợp một export coverage hiện có với một Git diff để liệt kê các dòng và nhánh thực thi đã thay đổi mà các test không thực thi. Bắt đầu với một C# project.
- **Nghiên cứu:** Báo cáo changed-code có giúp reviewers tìm các khoảng trống liên quan nhanh hơn một phần trăm toàn repo không?
- **Đánh giá:** Độ chính xác ánh xạ trên các diff mẫu và thời gian của reviewer. Hiển thị instrumentation thiếu như unknown, với các loại trừ rõ ràng và mẫu số.



#### TC2. Thử nghiệm mutation testing (M)

- **MVP:** Thêm các mutation code nhỏ trong một module quy tắc kinh doanh và kiểm tra các mutation sống sót; thêm test cho một vài khoảng trống được mentor xác nhận.
- **Nghiên cứu:** Test nào thực thi code nhưng fail phát hiện hành vi đã thay đổi? Đây là thí nghiệm cốt lõi được mô tả trong [giới thiệu Stryker](https://stryker-mutator.io/docs/).
- **Đánh giá:** Mutation được phát hiện trước và sau, runtime và các mutation có thể hành động được. Báo cáo mutation tương đương, mutation không hợp lệ và timeout riêng biệt.



#### TC3. Bản đồ coverage quy tắc kinh doanh (M)

- **MVP:** Ánh xạ yêu cầu của một workflow đến các test thực thi và kết quả cuối cùng của chúng, ví dụ các quy tắc xác thực Work Order xuyên suốt các kết hợp trạng thái, vai trò và đầu vào.
- **Nghiên cứu:** Một LLM có thể đề xuất các liên kết requirement-to-test chính xác sử dụng tên và nội dung test không?
- **Đánh giá:** Precision và recall của liên kết được mentor xác nhận, cộng với các yêu cầu không có kiểm tra đã xác minh. Một test gần đó pass không tự động cover một yêu cầu.



#### TC4. Bộ ưu tiên khoảng trống test (M)

- **MVP:** Xếp hạng mười ứng viên test sử dụng các nhánh chưa cover, tần suất thay đổi gần đây và lịch sử incident đã biết, với các lý do hiển thị cho mỗi xếp hạng.
- **Nghiên cứu:** Xếp hạng này có tìm được các test hữu ích hơn so với chỉ sắp xếp theo coverage thấp nhất không?
- **Đánh giá:** Các đánh giá liên quan của mentor và lỗi confirmed được tìm thấy cho cùng một nỗ lực kỹ thuật. Sử dụng các incident cũ hơn để xây dựng xếp hạng và các incident sau để đánh giá nó.



#### TC5. Ma trận coverage lỗi tích hợp (M)

- **MVP:** Thực thi một callback hoặc queue handler cục bộ với các message trùng lặp, trễ, sai định dạng và không đúng thứ tự, timeout và lỗi phụ thuộc một phần.
- **Nghiên cứu:** Tổ hợp lỗi nào phát hiện khoảng trống mà các test đường thành công thông thường bỏ sót?
- **Đánh giá:** Kết quả đúng đối với một contract đã thống nhất, tác dụng phụ trùng lặp và các kiểm tra khôi phục thiếu. Báo cáo coverage kịch bản riêng biệt với coverage dòng.



### 3. AI reports: chuyển bằng chứng thành các giải thích hữu ích, có thể kiểm chứng

Cho các dự án này, tính toán các con số bằng code xác định và cung cấp cho AI các bảng kết quả cộng với các định danh bằng chứng. So sánh output với một mẫu báo cáo đơn giản. Tái sử dụng nền tảng đánh giá của dự án 4 nếu có; một tập dữ liệu và trình chạy cục bộ là đủ để bắt đầu.

#### AR1. Tóm tắt chất lượng AI và kết quả test (S)

- **MVP:** Chuyển một lần chạy test, export coverage và danh sách thay đổi thành một báo cáo ngắn: cái gì thất bại, cái gì thay đổi, cái gì chưa được test và các bước điều tra tiếp theo.
- **Nghiên cứu:** Giải thích bằng AI có giúp developers chẩn đoán kết quả nhanh hơn một mẫu không?
- **Đánh giá:** Độ chính xác thực tế, tính đúng đắn của liên kết bằng chứng, các tuyên bố không được hỗ trợ và thời gian hoàn thành tác vụ của người đọc. Xác định revision nguồn và lần chạy test trên mọi báo cáo.



#### AR2. Trợ lý viết narrative KPI kinh doanh (M)

- **MVP:** Giải thích một tập dữ liệu hàng tuần cố định cho một chủ đề, như các Work Orders đang treo hoặc thời gian xử lý bảo hành, với các tham chiếu drill-down và một bảng thuật ngữ định nghĩa các chỉ số.
- **Nghiên cứu:** So sánh một mẫu với một narrative bằng LLM về tính hữu ích và độ chính xác số liệu.
- **Đánh giá:** Sự đồng ý của số được tính lại, điểm hữu ích từ BA và các tuyên bố nhân quả không được hỗ trợ. Bắt đầu với một export đã được ẩn danh; một thay đổi KPI đơn thuần không thiết lập nguyên nhân của nó.



#### AR3. Báo cáo bằng chứng xác minh release (M)

- **MVP:** Chuyển các so sánh nguồn đã lưu, manifest gói, kết quả deployment và output smoke-test thành một tóm tắt release có liên kết bằng chứng với các trạng thái pass, fail và unknown.
- **Nghiên cứu:** Báo cáo có thể phân biệt chính xác thay đổi nguồn, thành phần đã deploy và hành vi runtime khi một số bằng chứng cố ý bị bỏ qua không?
- **Đánh giá:** Độ chính xác tuyên bố-đến-bằng chứng và các kết luận sai-sẵn-sàng trên các bundle được gán nhãn bởi mentor. Bổ sung cho dự án 9 bằng cách tiêu thụ artifacts thay vì thực hiện deployment.



#### AR4. Người kiểm tra thực tế báo cáo AI (M)

- **MVP:** Kiểm tra các báo cáo được tạo đối với các bảng đầu vào và artifacts của chúng; đánh dấu tổng sai, so sánh không được hỗ trợ, trích dẫn thiếu và so sánh qua các cửa sổ thời gian không tương thích.
- **Nghiên cứu:** So sánh xác thực xác định, một critic LLM và sự kết hợp của chúng.
- **Đánh giá:** Precision và recall trên các báo cáo chứa lỗi được cố ý đặt, cộng với báo động sai trên các báo cáo đúng. Giữ các ví dụ đánh giá riêng biệt khỏi các ví dụ điều chỉnh prompt.



#### AR5. Báo cáo so sánh thí nghiệm AI (S)

- **MVP:** Tạo một so sánh của hai phiên bản prompt hoặc retrieval: chất lượng, thời gian phản hồi, chi phí đo lường được và các lỗi đại diện trên cùng các câu hỏi.
- **Nghiên cứu:** Việc trình bày các ví dụ cặp có thay đổi phiên bản nào reviewers chọn so với chỉ hiển thị trung bình không?
- **Đánh giá:** Các tính toán có thể tái tạo, sự đồng thuận của reviewers và nhận dạng đúng các đánh đổi. Mở rộng báo cáo của dự án 4 thay vì tạo thêm một nền tảng đánh giá khác.



### 4. Automation: loại bỏ một công việc kỹ thuật lặp đi lặp lại

Những ý tưởng này hỗ trợ công việc Playwright của đợt tháng 7 thông qua fixtures, chẩn đoán và công cụ; chúng không tạo ra một dự án browser-test Sales/Aftersales thứ hai.

#### AU1. Nhà máy dữ liệu test tổng hợp (M)

- **MVP:** Tạo một bundle fixture cục bộ xác định cho một workflow, bao gồm các bản ghi liên quan hợp lệ và các trường hợp cố ý không hợp lệ. Bao gồm một seed, manifest và lệnh phát lại.
- **Nghiên cứu:** So sánh tạo chỉ từ schema với schema cộng các ràng buộc kinh doanh do BA định nghĩa.
- **Đánh giá:** Tính hợp lệ của ràng buộc, đa dạng kịch bản và thời gian thiết lập của developer. Bất kỳ việc seed DEV/UAT nào sau đó cần một môi trường rõ ràng và kế hoạch dọn dẹp dựa trên quyền sở hữu.



#### AU2. Bộ phân loại lỗi CI (S)

- **MVP:** Phân loại các log job thất bại đã lưu thành lỗi biên dịch, assertion test, phụ thuộc, xác thực hoặc hạ tầng và liên kết đến runbook liên quan.
- **Nghiên cứu:** So sánh quy tắc từ khóa, similarity văn bản và một LLM trên các log lịch sử đã gán nhãn.
- **Đánh giá:** Precision/recall theo danh mục, các trường hợp từ chối và thời gian phân loại. Giữ cái này tập trung vào các lỗi CI để bổ sung cho việc phân loại application-incident của dự án 3.



#### AU3. Detector trôi dạt tài liệu và contract (M)

- **MVP:** So sánh một contract API hoặc schema cấu hình với tài liệu của nó và soạn một báo cáo về các trường đã bị xóa, yêu cầu đã thay đổi hoặc ví dụ thiếu.
- **Nghiên cứu:** So sánh các diff có cấu trúc với diễn giải LLM cho việc phát hiện trôi dạt có thể hành động.
- **Đánh giá:** Precision trên các thay đổi lịch sử, các bất tương thích bị bỏ sót và nỗ lực xem xét. Bắt đầu với một loại contract và chỉ tạo patch khi bằng chứng đủ.



#### AU4. Bác sĩ thiết lập developer (S)

- **MVP:** Kiểm tra SDK, dependencies và các điều kiện tiên quyết service cục bộ của một repo và xuất các bước khắc phục chính xác với một kết quả có thể đọc bằng máy.
- **Nghiên cứu:** Chẩn đoán nhận biết dependency có giải thích các lỗi thiết lập tốt hơn một checklist phẳng không?
- **Đánh giá:** Độ chính xác chẩn đoán cho các vấn đề thiết lập đã seed và thời gian đến một local build thành công. Làm cho các kiểm tra có thể lặp lại mà không thay đổi cấu hình developer một cách âm thầm.



#### AU5. Trình tái tạo test không ổn định (M)

- **MVP:** Chạy các test nghi ngờ với các seed, thứ tự và parallelism được kiểm soát; gói lỗi có thể tái tạo nhỏ nhất với logs và chi tiết môi trường của nó.
- **Nghiên cứu:** Các nhiễu loạn nào phát hiện các phụ thuộc thứ tự, vấn đề thời gian hoặc trạng thái chia sẻ?
- **Đánh giá:** Tỷ lệ tái tạo trên mỗi ngân sách runtime và các nguyên nhân được xác nhận. Các lần chạy lại nên giữ lại lỗi đầu tiên như bằng chứng thay vì biến một lần pass sau đó thành kết quả sạch.



### 5. Optimization: đo một nút thắt cổ chai và test một cải tiến giới hạn

Chọn một workload cục bộ có thể tái tạo trước. Ghi lại revision, kích thước dữ liệu, số lần lặp, trạng thái warm/cold, trung vị và tail latency, lỗi và tính đúng đắn của output. Kết quả cục bộ mô tả workload đó; chúng không tự chúng thiết lập một cải tiến production.

#### OP1. Trợ lý thí nghiệm hiệu năng FetchXML (M)

- **MVP:** Mở rộng workflow bisect entity-list hiện có với một manifest thí nghiệm và báo cáo cho một view chậm. Thay đổi một yếu tố tại một thời điểm: các cột, sắp xếp, joins hoặc tổng số.
- **Nghiên cứu:** Yếu tố nào giải thích chi phí, và các biến thể nhanh hơn nào bảo toàn các bản ghi yêu cầu, thứ tự và hành vi người dùng thấy được?
- **Đánh giá:** Thời gian lặp lại và tương đương ngữ nghĩa. Gán nhãn các biến thể thay đổi hành vi như chẩn đoán; tuân theo [hướng dẫn hiệu năng FetchXML](https://learn.microsoft.com/en-us/power-apps/developer/data-platform/fetchxml/optimize-performance) của Microsoft, bao gồm các hạn chế trên query hints.



#### OP2. Detector cuộc gọi API trùng lặp (S)

- **MVP:** Phân tích một HAR đã lưu hoặc luồng trang cục bộ đã được đo lường, nhóm các request có khả năng dư thừa và hiển thị người khởi tạo và thời gian của chúng trong một báo cáo waterfall.
- **Nghiên cứu:** Signature request cộng thời gian có thể phân biệt các bản sao tình cờ khỏi các refresh, retry và phân trang hợp lệ không?
- **Đánh giá:** Precision trùng lặp được mentor xác nhận; sau đó so sánh số lượng request, bytes và thời gian hoàn thành tác vụ sau một sửa đổi trong khi kiểm tra tính tươi mới và hành vi lỗi.



#### OP3. Benchmark điểm nóng Plugin và API (M)

- **MVP:** Lập hồ sơ một handler có thể tái tạo cục bộ và so sánh một cải tiến được đề xuất, như truy xuất ít trường hơn hoặc tránh các tra cứu lặp lại, trong harness cục bộ hiện có.
- **Nghiên cứu:** Thời gian có tập trung vào tính toán, serialization hoặc các cuộc gọi phụ thuộc không?
- **Đánh giá:** Latency, phân bổ bộ nhớ where measurable, số lượng cuộc gọi phụ thuộc và kết quả kinh doanh giống hệt. Giữ thời gian phụ thuộc thực riêng biệt với các cái mô phỏng.



#### OP4. Trình mô phỏng chiến lược cache (M)

- **MVP:** Phát lại một trace request/update tổng hợp hoặc đã được ẩn danh đối với các chiến lược no-cache, fixed-expiry và explicit-invalidation cho một workload tra cứu.
- **Nghiên cứu:** Hit rate, tải backend và các phản hồi cũ thay đổi như thế nào khi tần suất cập nhật tăng?
- **Đánh giá:** Hit rate, latency, bộ nhớ và thời lượng phản hồi cũ, bao gồm cold starts và các refresh thất bại. Xác thực hành vi đối với một yêu cầu tính tươi mới được định nghĩa.



#### OP5. Thí nghiệm chi phí, độ trễ và chất lượng LLM (M)

- **MVP:** So sánh cắt tỉa prompt, lựa chọn ngữ cảnh và một chính sách routing đơn giản trên một tập đánh giá cố định của một dự án AI hiện có.
- **Nghiên cứu:** Cấu hình nào giảm chi phí hoặc latency đo lường được trong khi giữ chất lượng câu trả lời chấp nhận được, bao gồm các câu hỏi khó và không thể trả lời?
- **Đánh giá:** Các điểm chất lượng theo cặp, tail latency, tokens và chi phí sử dụng giá và ngày đã ghi. Tái sử dụng instrumentation của dự án 4; để ra tập đánh giá cuối cùng.



### Danh sách khởi đầu được đề xuất

Danh sách này là một phán đoán lập kế hoạch dựa trên phạm vi MVP và khả năng sẵn có của đầu vào. Xác nhận module đã chọn, fixtures có sẵn và dung lượng mentor trước khi giao việc.


| Lĩnh vực      | Lựa chọn đầu tiên                    | Tại sao bắt đầu ở đây                                       | Thí nghiệm thay thế                   |
| ------------- | ------------------------------------ | ----------------------------------------------------------- | ------------------------------------- |
| Unit testing  | UT2: Lỗi thành regression test       | Demo fail-trước/pass-sau cụ thể                             | UT3: Property testing                 |
| Test coverage | TC1: Coverage cho code đã thay đổi   | Output nhỏ reviewers có thể dùng ngay                       | TC2: Mutation testing                 |
| AI reports    | AR1: Tóm tắt chất lượng và test      | Bắt đầu từ artifacts đã lưu với các thực tế có thể xác minh | AR4: Kiểm tra thực tế báo cáo         |
| Automation    | AU2: Bộ phân loại lỗi CI             | Baseline offline và đánh giá có nhãn đơn giản               | AU5: Trình tái tạo test không ổn định |
| Optimization  | OP2: Detector cuộc gọi API trùng lặp | Một workflow đã ghi cho một điều tra giới hạn               | OP1: Thí nghiệm FetchXML              |


Các ghép nối tùy chọn với sở thích hiện tại: Trần An Thắng có thể khám phá AR4 hoặc OP5; Trần Trung Hiếu có thể khám phá AR2 hoặc TC4; Trần Huy Hoàng có thể khám phá AU2 hoặc OP3; Lê Trung Hiếu có thể khám phá TC1 hoặc AU4. Đây là các tùy chọn thảo luận, không phải thay đổi cho các nhiệm vụ chính.

Cho một demo chia sẻ, kết nối **UT2 → TC1 → AR1**: tái tạo một lỗi lịch sử, hiển thị test cover gì, sau đó giải thích lần chạy với bằng chứng liên kết. Giữ mỗi component có thể sử dụng độc lập.

### Định dạng thí nghiệm và bàn giao phổ biến

1. Đặt tên một người dùng, một workflow, vấn đề và baseline để đánh bại. Xác nhận quyền truy cập trong phiên làm việc đầu tiên; sử dụng đầu vào cục bộ hoặc tổng hợp khi không cần quyền truy cập thực.
2. Bàn giao một MVP có thể tái tạo trước khi thêm UI hoặc điều phối agent. Lưu trữ revision, cấu hình, manifest đầu vào và lệnh đánh giá cùng với kết quả.
3. So sánh với một baseline đơn giản trên cùng các trường hợp để kiểm tra và ngân sách nỗ lực/runtime. Báo cáo kích thước mẫu, các lỗi và hạn chế; một thí nghiệm không tìm thấy cải tiến là hữu ích.
4. Demo kết quả và bàn giao các bước thiết lập, các phát hiện đã xem xét và một gia tăng tiếp theo giới hạn. Giữ các ghi chú DMS lâu bền trong DMS_DOCS và bằng chứng được tạo hàng loạt trong DMS_FILES.



### Tác động workspace của brainstorm này

`DMS.code-workspace` hiện tại đã được kiểm tra membership. Phiên này chỉ mở rộng tài liệu này; các nhà đề xuất triển khai bên dưới không phải kiểm toán nguồn hoặc phân công triển khai.


| Repository workspace       | Phân loại cho phiên này | Relevance cho triển khai sau này                                                          |
| -------------------------- | ----------------------- | ----------------------------------------------------------------------------------------- |
| DMS_DOCS                   | Bị ảnh hưởng            | Brief dự án, mô tả thí nghiệm và các bàn giao bền vững trong tương lai                    |
| DMS_PROD                   | Không áp dụng           | Ứng viên cho các test hoặc sửa lỗi plugin và webresource sau khi chọn dự án               |
| VinFast.DMS.Api            | Không áp dụng           | Ứng viên cho các thí nghiệm API, callback và queue                                        |
| VF.TNS.Integration         | Không áp dụng           | Ứng viên cho các test tích hợp legacy; xác minh khả năng build cục bộ trước               |
| VinFast.AppService.NetCore | Không áp dụng           | Ứng viên cho các thí nghiệm contract API và hiệu năng inbound                             |
| Technosoft.Yana.Vinfast    | Không áp dụng           | Ứng viên cho các test quy tắc legacy; xác minh quyền sở hữu nguồn và khả năng build trước |
| DMS_LOCAL_HOST             | Không áp dụng           | Ứng viên cho việc tái sử dụng harness phát lại và benchmark cục bộ                        |
| bu-setup-automation        | Không áp dụng           | Không có triển khai provisioning nào được yêu cầu trong brainstorm này                    |


Không có thay đổi repo anh em nào cần thiết để định nghĩa các tùy chọn này. STP, WMS và OM vẫn là các ví dụ ngữ cảnh từ brief gốc, ngoài membership workspace hiện tại. Các repo mẫu của Microsoft vẫn chỉ để tham khảo, và archive `technosoft-dms` đã decompile vẫn chỉ để đọc.
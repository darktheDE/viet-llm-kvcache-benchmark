# RESEARCH PROPOSAL (GROUP 1)
## Benchmarking TurboQuant and KV Cache Compression Methods on Vietnamese Large Language Models

---

## 1. Tóm tắt (Abstract)
Việc triển khai các mô hình ngôn ngữ lớn (Large Language Models – LLMs) cho các ứng dụng tiếng Việt đang ngày càng mở rộng, đặc biệt trong các bài toán xử lý văn bản dài, hỏi đáp theo ngữ cảnh, tóm tắt tài liệu, và trợ lý hội thoại. Tuy nhiên, trong quá trình suy luận tự hồi quy, bộ nhớ đệm khóa–giá trị (Key-Value Cache – KV Cache) thường trở thành nút thắt chính về tài nguyên, do dung lượng bộ nhớ tăng tuyến tính theo độ dài ngữ cảnh. Điều này đặc biệt nghiêm trọng trong các môi trường phần cứng có VRAM giới hạn, nơi mà khả năng xử lý long-context inference bị suy giảm đáng kể.

Các phương pháp nén KV Cache gần đây như TurboQuant, RaBit-Q, PolarQuant và HQQ đã cho thấy tiềm năng lớn trong việc giảm bộ nhớ và tăng tốc suy luận. Tuy nhiên, phần lớn các nghiên cứu hiện có tập trung vào ngữ cảnh tiếng Anh hoặc các bộ benchmark tổng quát, trong khi hiệu quả của các phương pháp này trên mô hình ngôn ngữ lớn tiếng Việt vẫn chưa được khảo sát một cách hệ thống. Nghiên cứu này đề xuất một giao thức benchmark để so sánh TurboQuant với các phương pháp nén KV Cache tiêu biểu trên các mô hình ngôn ngữ lớn tiếng Việt trong bối cảnh suy luận dài ngữ cảnh. Phạm vi đánh giá tập trung vào ba khía cạnh chính: mức giảm bộ nhớ, tốc độ suy luận và chất lượng đầu ra của mô hình, trong đó chất lượng được đo chủ yếu bằng perplexity và một số chỉ số bổ trợ theo tác vụ nếu cần.

---

## 2. Giới thiệu (Introduction)
Các mô hình ngôn ngữ lớn đã trở thành thành phần cốt lõi trong nhiều hệ thống xử lý ngôn ngữ tự nhiên hiện đại, bao gồm chatbot, hệ thống tóm tắt, hỏi đáp, và phân tích tài liệu. Đối với tiếng Việt, nhu cầu sử dụng LLM đang tăng nhanh, nhưng việc triển khai thực tế vẫn gặp nhiều hạn chế do chi phí suy luận cao, độ trễ lớn, và yêu cầu bộ nhớ cao. Những vấn đề này càng trở nên nghiêm trọng hơn khi mô hình phải xử lý văn bản dài, vì KV Cache tăng tuyến tính theo độ dài chuỗi đầu vào và thường chiếm phần lớn bộ nhớ trong quá trình sinh tự hồi quy.

Mặc dù đã có nhiều phương pháp nén KV Cache được đề xuất gần đây, phần lớn các nghiên cứu hiện tại vẫn tập trung vào benchmark tiếng Anh hoặc các tác vụ long-context tổng quát. Do đó, vẫn chưa rõ các phương pháp này hoạt động như thế nào trong bối cảnh tiếng Việt, nơi đặc tính token hóa và cấu trúc câu có thể khác đáng kể. Sự thiếu vắng các benchmark chuyên biệt cho tiếng Việt tạo ra một khoảng trống nghiên cứu rõ ràng. Vì vậy, việc xây dựng một benchmark thực nghiệm cho KV Cache compression trên Vietnamese LLM là cần thiết để đánh giá tính phù hợp của các kỹ thuật hiện đại trong bối cảnh ngôn ngữ này.

Nghiên cứu này giải quyết khoảng trống đó bằng cách benchmark TurboQuant và một số phương pháp nén KV Cache tiêu biểu trên các mô hình ngôn ngữ lớn tiếng Việt. Thay vì đề xuất một thuật toán mới, nghiên cứu tập trung vào đánh giá thực nghiệm, khả năng tái lập, và phân tích so sánh. Cách tiếp cận này phù hợp để xây dựng một bài báo mang tính độc lập, có giá trị ứng dụng thực tiễn cho hệ sinh thái AI tiếng Việt.

---

## 3. Bài toán nghiên cứu (Research Problem)
Bài toán trọng tâm của nghiên cứu là hiệu quả suy luận dài ngữ cảnh của các mô hình ngôn ngữ lớn tiếng Việt trong điều kiện hạn chế tài nguyên bộ nhớ. Trong các kịch bản triển khai thực tế, việc lưu trữ KV Cache đầy đủ có thể nhanh chóng vượt quá dung lượng VRAM khả dụng, đặc biệt khi xử lý tài liệu dài, hội thoại nhiều lượt, hoặc các prompt có ngữ cảnh lớn. Vì vậy, hệ thống phải đối mặt với bài toán đánh đổi giữa chất lượng sinh văn bản và khả năng vận hành trên phần cứng hạn chế.

Nghiên cứu đặt ra câu hỏi liệu nén KV Cache có thể giảm đáng kể mức sử dụng bộ nhớ và cải thiện thông lượng suy luận mà không làm suy giảm quá mức chất lượng của mô hình hay không. Cụ thể hơn, nghiên cứu xem xét TurboQuant có thể so sánh như thế nào với các phương pháp nén khác như RaBit-Q, PolarQuant, và HQQ khi được đánh giá trên các mô hình ngôn ngữ lớn tiếng Việt. Bài toán không chỉ là phương pháp nào nén tốt hơn, mà còn là phương pháp nào giữ được hiệu năng tốt nhất khi độ dài ngữ cảnh tăng cao.

---

## 4. Mục tiêu nghiên cứu (Research Objectives)
*   **Mục tiêu thứ nhất:** Định lượng mức tiết kiệm bộ nhớ đạt được bởi TurboQuant và các phương pháp baseline so với cấu hình Full KV Cache.
*   **Mục tiêu thứ hai:** Đo lường tác động của từng phương pháp nén lên độ trễ suy luận và thông lượng trong các điều kiện phần cứng được kiểm soát.
*   **Mục tiêu thứ ba:** Đánh giá chất lượng đầu ra của mô hình bằng perplexity và các chỉ số đánh giá theo tác vụ khi cần thiết.
*   **Mục tiêu thứ tư:** Phân tích đánh đổi giữa hiệu quả và chất lượng theo từng độ dài ngữ cảnh và loại tác vụ.

Tổng thể các mục tiêu hướng tới việc mô tả toàn diện tính phù hợp của KV Cache compression đối với triển khai LLM tiếng Việt.

---

## 5. Phạm vi nghiên cứu (Scope of Study)
Nghiên cứu này chỉ tập trung vào các mô hình ngôn ngữ lớn tiếng Việt ở chế độ suy luận, không bao gồm huấn luyện từ đầu hay đề xuất một thuật toán nén mới. 

*   **Các mốc đối sánh:** Full KV Cache làm baseline và bốn phương pháp nén: TurboQuant, RaBit-Q, PolarQuant, và HQQ.
*   **Mô hình thử nghiệm:** Các mô hình thuần Việt (như PhoGPT, URA-LLaMa) và các biến thể adapted tiếng Việt khác (như Qwen, Llama).
*   **Dữ liệu đánh giá:** Các benchmark tiếng Việt và các prompt dài phản ánh các tình huống xử lý ngôn ngữ thực tế.
*   **Giới hạn:** Nghiên cứu không mở rộng sang bài toán đa phương thức, không đi sâu vào fine-tuning quy mô lớn, và cũng không xem xét toàn bộ pipeline triển khai hệ thống. Trọng tâm của bài báo là đánh giá thực nghiệm các phương pháp nén KV Cache trong bối cảnh tiếng Việt.

---

## 6. Câu hỏi nghiên cứu (Research Questions)
Nghiên cứu được dẫn dắt bởi bốn câu hỏi chính:
*   **RQ1:** TurboQuant có thể giảm đáng kể dung lượng KV Cache so với Full KV Cache trên các mô hình tiếng Việt hay không?
*   **RQ2:** Việc giảm bộ nhớ đó có mang lại cải thiện đáng kể về tốc độ suy luận hay không?
*   **RQ3:** Chất lượng đầu ra của mô hình thay đổi như thế nào dưới các phương pháp nén khác nhau, được đo bằng perplexity và cá
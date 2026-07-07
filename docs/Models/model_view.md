# Thông tin Tổng quan các Mô hình (Model View)

Dự án đánh giá kỹ thuật nén KV Cache tập trung vào 5 đại diện xuất sắc (bao gồm các hệ ngôn ngữ và kích thước từ 3B đến 8B), với sự chú trọng đặc biệt vào khả năng xử lý **Tiếng Việt** trên các luồng ngữ cảnh siêu lớn (Long-context).

Dưới đây là thông tin tổng quan của 5 mô hình được chọn để làm Benchmark:

## 1. `gemma4:e4b`
*   **Hệ / Nguồn gốc:** Dòng mô hình đa ngôn ngữ (Multilingual), nền tảng Qwen2.5.
*   **Vai trò trong Benchmark:** Baseline đa ngôn ngữ. Đây là mô hình tối ưu đặc biệt cho các ngôn ngữ Đông Nam Á (SEA-optimized), thay thế cho VinaLLaMA-7B cũ (do model cũ bị kẹt ở ngữ cảnh 4K).
*   **Ngữ cảnh hỗ trợ (Native Context):** 32K ~ 128K tokens.
*   **Điểm mạnh:** Giữ văn phong mượt mà khi sinh chữ ở chế độ Full KV Cache, đại diện cho nhóm mô hình có trọng số tham số khoảng 7B-8B.

## 2. `qwen3:8b`
*   **Hệ / Nguồn gốc:** Qwen Team (Alibaba Cloud).
*   **Vai trò trong Benchmark:** Mô hình đo lường State-Of-The-Art (SOTA) quốc tế mới nhất trong phân khúc dưới 10B.
*   **Ngữ cảnh hỗ trợ (Native Context):** 128K tokens (nhờ công nghệ RoPE scale cải tiến).
*   **Điểm mạnh:** Đạt điểm rất cao trong khả năng giải luận, coding và hỗ trợ đa ngôn ngữ bao gồm tiếng Việt. Dùng để đối chiếu xem phương pháp nén (TurboQuant/PolarQuant) ảnh hưởng thế nào đến các kiến trúc SOTA nhất.

## 3. `llama3.2:3b`
*   **Hệ / Nguồn gốc:** Meta Llama.
*   **Vai trò trong Benchmark:** Baseline nhỏ gọn (Lightweight Compact Baseline).
*   **Ngữ cảnh hỗ trợ (Native Context):** 128K tokens.
*   **Điểm mạnh:** Với kích thước chỉ 3B tham số, nó tiêu thụ cực ít tài nguyên. Mô hình này giúp trả lời câu hỏi: *Việc nén KV Cache trên một mô hình vốn dĩ đã có trọng số (weights) rất nhỏ thì có khiến chất lượng bị phân rã (degradation) thảm hại hơn so với các mô hình 7B-8B hay không?*

## 4. `arcee-ai/Arcee-VyLinh`
*   **Hệ / Nguồn gốc:** Phát triển bởi nhóm nghiên cứu URA (Đại học Bách Khoa TP.HCM - HCMUT), kế thừa kiến trúc Llama-3.
*   **Vai trò trong Benchmark:** Đại diện mô hình "Thuần Việt" xuất sắc (Vietnamese LLM Baseline).
*   **Ngữ cảnh hỗ trợ (Native Context):** 128K tokens.
*   **Điểm mạnh:** Tối ưu hóa sâu cho việc hỏi đáp, đọc hiểu văn bản tiếng Việt. Việc sử dụng Arcee-VyLinh giúp đề tài có tính thực tiễn cao cho các bài toán doanh nghiệp địa phương ở Việt Nam.

## 5. `Qwen/Qwen2.5-7B-Instruct-1M`
*   **Hệ / Nguồn gốc:** Qwen Team (Alibaba Cloud).
*   **Vai trò trong Benchmark:** Giới hạn trên của ngữ cảnh (Long-context Upper-bound Baseline).
*   **Ngữ cảnh hỗ trợ (Native Context):** 1 Triệu (1M) tokens.
*   **Điểm mạnh:** Khả năng nhồi nhét tài liệu khổng lồ (hàng chục cuốn sách) vào bộ nhớ. Đây là "khúc xương khó gặm" nhất cho mọi kỹ thuật nén, vì khi context lên tới 1M, kích thước KV Cache sẽ vượt cả kích thước Weights của chính mô hình. Thử nghiệm trên mô hình này sẽ cho thấy rõ ràng nhất hiệu quả tiết kiệm bộ nhớ (Peak VRAM) và tốc độ (ITL).

# SPRINT 03

### TASK 1: TỐI ƯU HẠ TẦNG & SỬA LỖI (TECHNICAL & EXPERIMENT)

*   **Title:** `[TECH] Tối ưu hóa VRAM, Xử lý lỗi Corner Cases/OOM & Chạy bù các mốc thử nghiệm còn thiếu`
*   **Module:** Infrastructure & Bug-fixing
*   **Sprint:** Sprint 3 (Week 5-6)
*   **Team / Role:** Technical & Experiment (Nguyen Van Quang Duy - Lead, Ho Viet Anh)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Đảm nhận vai trò "Help Desk" hỗ trợ kỹ thuật cho toàn nhóm. Giải quyết triệt để các trường hợp thử nghiệm bị sập do lỗi tràn bộ nhớ GPU (CUDA Out of Memory) ở các mốc ngữ cảnh lớn (16k hoặc 32k). Cấu hình tối ưu hóa tham số bộ nhớ trong vLLM và phối hợp chạy bù các cấu hình bị thiếu sót hoặc bị lỗi trong Sprint trước.

#### 2. Tài liệu đọc tham khảo (References)
*   *Tối ưu bộ nhớ vLLM:* [vLLM Memory Management & Optimization](https://docs.vllm.ai/en/latest/models/engine_args.html) - Hướng dẫn tinh chỉnh các tham số `gpu_memory_utilization` và `max_model_len`.
*   *Tài liệu kỹ thuật:* Khắc phục lỗi phân mảnh bộ nhớ khi nén KV Cache dưới dạng 4-bit (vLLM Issue Tracker 2026).

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Thu thập thông tin từ các tệp log chạy thực nghiệm của Sprint 2, lọc ra các mốc cấu hình (Model, Method, Context) bị dính lỗi CUDA OOM.
*   **Bước 2:** Cấu hình tinh chỉnh tham số bộ nhớ trong script khởi tạo vLLM:
    *   Tăng giá trị `--gpu-memory-utilization` lên `0.95` hoặc `0.98` để tận dụng tối đa VRAM khả dụng.
    *   Cấu hình tham số `--max-num-seqs` nhỏ lại (ví dụ bằng 1 hoặc 2) để giảm tải cho pha Decode khi benchmark ngữ cảnh dài.
*   **Bước 3:** Đối với các mốc mô hình kích thước lớn (như Qwen2.5-7B) ở ngữ cảnh 32k tokens bị OOM trên card 24GB, tiến hành cấu hình chạy bù sử dụng phương pháp **FlashAttention-2** hoặc giảm thiểu kích thước `block_size` trong vLLM từ 16 xuống 8.
*   **Bước 4:** Ghi nhận lại các cấu hình bắt buộc phải hy sinh (không thể chạy được dù đã tối ưu) và đánh dấu rõ mốc đó là "OOM" trong tệp dữ liệu chung thay vì bỏ trống.

#### 4. Kết quả đầu ra (Expected Output)
*   Bản vá mã nguồn (patch hoặc config cập nhật) giải quyết các lỗi OOM.
*   Dữ liệu chạy bù đầy đủ cho các mốc thử nghiệm còn thiếu được nạp vào thư mục `results/`.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] 100% các mốc ngữ cảnh lớn được thử nghiệm tối đa, các mốc không thể chạy được đã được gắn nhãn "OOM" rõ ràng.
*   [ ] Không còn hiện tượng script bị treo hoặc crash ngầm giữa chừng không rõ nguyên nhân trong quá trình benchmark.

---

### TASK 2: TỔNG HỢP & TRỰC QUAN HÓA SỐ LIỆU (DATA & ANALYSIS)

*   **Title:** `[ANALYSIS] Tổng hợp dữ liệu CSV, Tính toán Thống kê & Vẽ Biểu đồ Pareto Trade-off bằng Python`
*   **Module:** Phase 3: Phân tích & xử lý số liệu
*   **Sprint:** Sprint 3 (Week 5-6)
*   **Team / Role:** Data & Analysis (Huynh Ngoc Thach - Lead, Nguyen Ho Phat, Huynh Huu Huy)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Viết script Python tự động gom cụm tất cả các file CSV số liệu riêng lẻ từ các đợt chạy thực nghiệm của team Tech thành một bảng dữ liệu hợp nhất. Thực hiện tính toán các chỉ số thống kê trung bình (mean) và độ lệch chuẩn (std). Sau đó, trực quan hóa kết quả bằng cách vẽ các biểu đồ phân tích đánh đổi (Trade-off curves) giữa hiệu năng phần cứng và chất lượng ngôn ngữ.

#### 2. Tài liệu đọc tham khảo (References)
*   *Vẽ biểu đồ Pareto:* Hướng dẫn vẽ đường cong Pareto Frontier và Trade-off curves trong nghiên cứu hệ thống LLM (Towards Data Science / Towards AI 2026).
*   *Tài liệu thư viện:* Tài liệu sử dụng các thư viện Python: `pandas`, `matplotlib`, và `seaborn` / `plotly`.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Viết script `scripts/plot_results.py` sử dụng thư viện `pandas` để tự động quét thư mục `results/`, đọc toàn bộ các file CSV thô và gộp (merge) chúng lại theo cấu trúc thống nhất.
*   **Bước 2:** Tính toán các chỉ số thống kê trung bình và độ lệch chuẩn của Peak VRAM, Latency per token, Throughput và Perplexity cho từng điều kiện thử nghiệm (Model $\times$ Method $\times$ Context).
*   **Bước 3:** Sử dụng `matplotlib` hoặc `seaborn` để thiết kế và vẽ 3 loại đồ thị chuẩn học thuật lưu vào `results/plots/`:
    *   *Đồ thị 1:* Memory tiêu thụ (trục Y) biểu diễn theo độ dài ngữ cảnh (trục X), phân tách màu sắc theo từng phương pháp nén.
    *   *Đồ thị 2:* Latency / Throughput sinh từ (trục Y) biểu diễn theo độ dài ngữ cảnh (trục X).
    *   *Đồ thị 3 (Đồ thị Pareto quan trọng nhất):* Sự đánh đổi giữa Perplexity (chất lượng ngôn ngữ - trục Y) và Peak VRAM tiêu thụ (hiệu năng phần cứng - trục X).
*   **Bước 4:** Định dạng các đồ thị theo đúng chuẩn hiển thị của bài báo khoa học: phông chữ rõ ràng, phân biệt màu sắc tốt cho cả chế độ in trắng đen, có chú giải (legend) đầy đủ.

#### 4. Kết quả đầu ra (Expected Output)
*   File gộp kết quả cuối cùng `results/all_results_compiled.csv`.
*   Tối thiểu 3 tệp tin đồ thị định dạng ảnh chất lượng cao (PNG hoặc PDF vector) trong thư mục `results/plots/`.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Script `plot_results.py` thực thi thành công không lỗi, tự động xuất ra toàn bộ các biểu đồ khi chạy.
*   [ ] Các biểu đồ hiển thị rõ ràng, ghi chú đầy đủ nhãn trục (axis labels), đơn vị đo (MB, ms/token, tokens/s, PPL) và các chú thích phân biệt rõ ràng giữa các phương pháp nén.

---

### TASK 3: PHÂN TÍCH KẾT QUẢ & LUẬN GIẢI (RESEARCH & SCOPE)

*   **Title:** `[RESEARCH] Biên soạn Báo cáo Phân tích Kết quả (Results & Discussion) & Nhận xét Mô hình`
*   **Module:** Phase 3: Phân tích & xử lý số liệu
*   **Sprint:** Sprint 3 (Week 5-6)
*   **Team / Role:** Research & Scope (Nguyen Dang Quoc Anh - Lead, Phan Trong Phu)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Dựa trên bảng số liệu tổng hợp và các biểu đồ phân tích đánh đổi thu được từ Task 2, tiến hành phân tích sâu về mặt khoa học. Luận giải các xu hướng biến thiên của chỉ số, so sánh hiệu quả thực tế của **TurboQuant** so với các baseline nén khác trên tiếng Việt. Viết bản thảo chương **Results & Discussion** và chương **Conclusion** bằng tiếng Anh để đưa vào Overleaf.

#### 2. Tài liệu đọc tham khảo (References)
*   *Mẫu phân tích khoa học:* Các bài nghiên cứu hệ thống (System Papers) về lượng tử hóa LLM tại các hội thảo lớn (NeurIPS, ICLR, MLSys) để học cách viết luận giải số liệu.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Đọc và phân tích kỹ các biểu đồ trade-off từ Task 2. Xác định xu hướng:
    *   TurboQuant giảm bao nhiêu % bộ nhớ VRAM so với Full KV cache? Có đúng trong khoảng 50–60% lý thuyết không?
    *   Mức tăng Perplexity tương ứng là bao nhiêu điểm?
    *   Phương pháp nén nào duy trì được chất lượng ổn định nhất khi độ dài ngữ cảnh tăng từ 4k lên 16k?
*   **Bước 2:** Viết các đoạn văn bản phân tích (1–2 đoạn ngắn cho mỗi biểu đồ) mô tả chính xác các quy luật thu được. Sử dụng các số liệu định lượng cụ thể từ bảng dữ liệu thay vì nhận xét chung chung.
*   **Bước 3:** Viết phần **Discussion** nhằm lý giải nguyên nhân vật lý/kiến trúc: Tại sao TurboQuant lại tối ưu hơn FP8? Tại sao tiếng Việt lại nhạy cảm với việc nén KV cache sâu?
*   **Bước 4:** Soạn thảo chương **Conclusion** và tóm tắt lại các đóng góp thực tiễn của nghiên cứu đối với cộng đồng triển khai LLM tiếng Việt. Đẩy toàn bộ văn bản thảo lên dự án Overleaf chung.

#### 4. Kết quả đầu ra (Expected Output)
*   Bản thảo chương **Results & Discussion** và **Conclusion** bằng tiếng Anh được cập nhật trực tiếp trên dự án Overleaf.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Toàn bộ kết quả thử nghiệm định lượng của nhóm được phản ánh chính xác bằng văn bản học thuật tiếng Anh.
*   [ ] Không còn các đoạn văn phân tích mang tính cảm tính; 100% các nhận định rút ra đều được đối chiếu bằng số liệu thực tế từ biểu đồ và bảng kết quả.

---

### TASK 4: TỔNG HỢP BẢN THẢO BÁO CÁO TIẾNG VIỆT (WRITING & COORDINATION)

*   **Title:** `[COORD] Tổng hợp Bản thảo Báo cáo Tiếng Việt (Word Draft v1) & Review Tiến độ Dự án`
*   **Module:** Phase 4: Viết bài + chỉnh sửa
*   **Sprint:** Sprint 3 (Week 5-6)
*   **Team / Role:** Writing & Coordination (Do Kien Hung - Lead, Phan Trong Qui)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Với vai trò PM điều phối tiến độ, thực hiện thu thập toàn bộ nội dung từ các nhóm chuyên môn để biên dịch và tổng hợp thành bản thảo đầu tiên của **Báo cáo chi tiết tiếng Việt (Word Document $\ge 30$ trang)** đáp ứng tiêu chí đánh giá môn học BDML. Đồng thời, rà soát tiến độ của toàn bộ các task trên `plane.so` để chuẩn bị cho Sprint cuối cùng.

#### 2. Tài liệu đọc tham khảo (References)
*   *Quy chế môn học:* Đề cương chi tiết môn học DBML434077 (HCM-UTE) - Phần yêu cầu định dạng và cấu trúc nội dung của báo cáo Word cuối kỳ.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Khởi tạo tệp văn bản Word chung trên Google Docs hoặc OneDrive để các thành viên có thể đồng chỉnh sửa. Thiết lập lề, mục lục tự động, và định dạng phông chữ chuẩn theo quy định của môn học.
*   **Bước 2:** Thu thập nội dung từ các team chuyên môn để đưa vào báo cáo tiếng Việt:
    *   Lấy phần Đề cương, Câu hỏi nghiên cứu và Lý thuyết nền tảng từ team Research.
    *   Lấy sơ đồ kiến trúc hệ thống, mã nguồn và quy trình đo đạc từ team Tech.
    *   Lấy bộ dữ liệu, schema prompt, biểu đồ phân tích và số liệu bảng từ team Data.
*   **Bước 3:** Biên dịch chọn lọc các phần đã viết bằng tiếng Anh của team Research sang tiếng Việt mượt mà để hoàn thiện cấu trúc báo cáo.
*   **Bước 4:** Rà soát tiến độ của toàn nhóm trên `plane.so`. Phát hiện các thành viên hoặc các mốc công việc bị chậm tiến độ (bottlenecks) để điều phối hỗ trợ kịp thời.

#### 4. Kết quả đầu ra (Expected Output)
*   Link tệp tin bản thảo Báo cáo Word tiếng Việt (Draft v1) có độ dài sơ bộ đạt tối thiểu 20-25 trang văn bản.
*   Báo cáo tổng kết trạng thái công việc (Task Status Report) của nhóm trên `plane.so` trước khi bước vào Sprint 4.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Bản thảo báo cáo Word được định dạng nhất quán, có mục lục tự động hiển thị đầy đủ các chương mục chính của đề tài.
*   [ ] Toàn bộ số liệu thực nghiệm đo được và các hình vẽ đồ thị phân tích đã được chèn thành công vào văn bản với chú thích hình/bảng rõ ràng.
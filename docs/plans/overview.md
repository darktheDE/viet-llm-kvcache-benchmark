# Overview: Benchmark KV Cache Compression trên Vietnamese LLMs

Tài liệu này cung cấp cái nhìn tổng quan (High-level Overview) về luồng hoạt động, cấu trúc mô hình và các thành tựu hệ thống sau khi triển khai thành công script `run_baseline.py` theo đúng kế hoạch.

---

## 1. Luồng dữ liệu tổng quan (Pipeline Data Flow)

Hệ thống Benchmark hoạt động theo một luồng Pipeline một chiều tuyến tính, đi từ bước nạp dữ liệu đến bước xuất kết quả:

**`[1. Nạp Dữ Liệu]`** -> **`[2. Tải Mô hình (16-bit)]`** -> **`[3. Kích hoạt Lõi Nén KV Cache]`** -> **`[4. Khai thác & Đo lường]`** -> **`[5. Ghi Log CSV]`**

1. **Nạp Dữ liệu:** Đọc các mẫu văn bản tiếng Việt dài (4k, 8k, 16k) từ file `.json`.
2. **Tải Mô hình:** Đưa toàn bộ trọng số (weights) của mô hình ngôn ngữ vào bộ nhớ GPU (Luôn giữ nguyên trọng số ở định dạng 16-bit nguyên bản, không nén trọng số).
3. **Kích hoạt Lõi Nén:** Khởi động Engine `vLLM` cùng với các thuật toán nén (như TurboQuant).
4. **Khai thác & Đo lường:** Trong quá trình Engine sinh chữ, script sẽ dùng thư viện `pynvml` đứng ở ngoài để giám sát và bắt lấy đỉnh VRAM cao nhất (Peak VRAM) và độ trễ (Latency).
5. **Ghi Log:** Định dạng các chỉ số và lưu nối tiếp (append) vào tệp `results/template_log.csv`.

---

## 2. Thông tin về 4 Mô hình được sử dụng

Dự án đánh giá hiệu suất nén trên 4 mô hình ngôn ngữ (tập trung vào năng lực tiếng Việt):

> **Ghi chú phạm vi:** `gemma4:e4b-it-bf16` đã được loại khỏi benchmark chính thức vì nhóm đánh giá model này đã có mức tối ưu/nén sẵn cao, không còn phù hợp để làm đối chứng công bằng cho các kỹ thuật KV cache compression.

1. **`qwen3:8b-fp16`**: SOTA model thế giới ở mức 8B tham số, chạy FP16. Hỗ trợ 128K context nhờ RoPE scale cải tiến. Được dùng làm thước đo chuẩn quốc tế.
2. **`llama3.1:8b-instruct-fp16`**: Mô hình Llama 3.1 8B Instruct từ Meta AI, chạy FP16. Baseline phổ biến nhất thị trường mã nguồn mở. Hỗ trợ 128K context.
3. **`mistral:7b-instruct-v0.3-fp16`**: Mô hình Mistral 7B Instruct v0.3 từ Mistral AI (Pháp), chạy FP16. Kiến trúc GQA (Grouped Query Attention) giúp KV Cache tự nhiên nhỏ hơn. Hỗ trợ 32K context.
4. **`qwen2.5:7b-instruct-fp16`**: Mô hình Qwen2.5 7B Instruct (Alibaba), chạy FP16. Đại diện cho dòng Qwen2.5, hỗ trợ 128K context. Rất mạnh về tiếng Việt.

---

## 3. Step-by-Step Mô Phỏng Luồng Chạy Thực Tế

Để xác nhận xem Plan có đi đúng hướng không, hãy tưởng tượng bạn gõ dòng lệnh sau vào Terminal:
`python scripts/run_baseline.py --model qwen3:8b-fp16 --context_length 8000 --kv_cache_type TurboQuant`

Đây là chuyện gì sẽ diễn ra bên trong hệ thống:

*   **Bước 1 (Nhận lệnh & Data):** Hệ thống nhận tham số `--context_length 8000`. Nó vào file `datasets/test_set_small.json`, bốc ra vài văn bản tiếng Việt dài đúng 8000 chữ.
*   **Bước 2 (Tải Model):** Hệ thống tải `qwen3:8b-fp16` (chạy FP16 ~16GB) nhét vào con Card RTX 4090.
*   **Bước 3 (Đọc & Nén - Trọng tâm):** Bắt đầu quá trình Inference. Mô hình đọc 8000 chữ đó. Lúc này, quá trình đọc sinh ra cực kỳ nhiều "bộ nhớ tạm" (KV Cache). Ngay lập tức, lệnh `--kv_cache_type TurboQuant` kích hoạt lõi CUDA của TurboQuant bên dưới vLLM, **bóp nghẹt cái đống "bộ nhớ tạm" đó từ 16-bit xuống còn 4-bit theo thời gian thực (real-time)** để tiết kiệm chỗ chứa.
*   **Bước 4 (Giám sát):** Mã nguồn `run_baseline.py` của chúng ta (dùng `pynvml`) đứng ngoài quan sát và ghi nhận: *"À, nhờ có TurboQuant nén bộ nhớ tạm lại, nên lúc đọc 8000 chữ, GPU chỉ tốn tổng cộng 16GB VRAM (chứ không bị lố lên 30GB gây nổ card như bình thường). Thời gian nhả chữ là 30ms/token"*.
*   **Bước 5 (Xuất Kết quả):** Các con số *(qwen3:8b-fp16, TurboQuant, 8000, 16GB, 30ms)* được xuất thành 1 dòng trong file `results/template_log.csv`. Hoàn tất một vòng lặp đo đạc!

---

## 4. Những Điều Nhóm Đã Đạt Được (Thành quả sau khi có Plan)

Sau khi thiết kế và làm theo bản Plan này (kết tinh ở tệp code `scripts/run_baseline.py`), nhóm đã đạt được 3 thành quả cực lớn để đưa vào báo cáo:

1.  **Sở hữu Hệ thống Benchmark Tự Động (Automated Pipeline):** Không cần phải ngồi canh và sửa code thủ công từng model nữa. Giờ đây chỉ cần 1 câu lệnh truyền tham số `argparse`, script tự động thay model, tự động đổi thuật toán nén, và tự động lưu CSV.
2.  **Khắc phục Triệt Để Lỗi OOM (Out-of-Memory):** Thay vì để máy chủ treo cứng khi nhét văn bản 32k tokens vào gây sập GPU, code đã được tích hợp khối `try...except torch.cuda.OutOfMemoryError`. Nó sẽ tự động dán nhãn "OOM" vào file kết quả CSV một cách "Graceful" (thanh lịch) và chạy tiếp sang cấu hình khác.
3.  **Tích hợp Chế Độ Mock (Giả lập):** Điểm sáng tạo giúp các thành viên không có điều kiện thuê GPU 24GB vẫn có thể chạy luồng phần mềm. Chế độ Mock giúp sinh ra dữ liệu mẫu (Dummy data) tự động, giúp nhóm song song lấy kết quả đó để đi vẽ biểu đồ Pareto mà không cần phải chờ đợi ai cả.

# Thông tin Tổng quan các Mô hình (Model View)

Dự án đánh giá kỹ thuật nén KV Cache tập trung vào 4 đại diện xuất sắc (bao gồm các hệ ngôn ngữ và kích thước từ 7B đến 8B), với sự chú trọng đặc biệt vào khả năng xử lý **Tiếng Việt** trên các luồng ngữ cảnh siêu lớn (Long-context).

Tất cả 4 model đều chạy qua **Ollama** ở chế độ full precision (FP16/BF16) trên máy thuê GPU.

> **Ghi chú phạm vi:** `gemma4:e4b-it-bf16` không còn nằm trong benchmark chính thức vì nhóm xác định model này đã được tối ưu nén sẵn, dễ làm lệch so sánh giữa các phương pháp KV cache compression.

| # | Ollama Tag | HuggingFace Repo | Precision | Size |
|---|---|---|---|---|
| 1 | `qwen3:8b-fp16` | Qwen/Qwen3-8B | FP16 | ~16 GB |
| 2 | `llama3.1:8b-instruct-fp16` | meta-llama/Llama-3.1-8B-Instruct | FP16 | ~16 GB |
| 3 | `mistral:7b-instruct-v0.3-fp16` | mistralai/Mistral-7B-Instruct-v0.3 | FP16 | ~14 GB |
| 4 | `qwen2.5:7b-instruct-fp16` | Qwen/Qwen2.5-7B-Instruct | FP16 | ~15 GB |

---

## 1. `qwen3:8b-fp16`
*   **Hệ / Nguồn gốc:** Qwen Team (Alibaba Cloud).
*   **Vai trò trong Benchmark:** Mô hình đo lường State-Of-The-Art (SOTA) quốc tế mới nhất trong phân khúc dưới 10B.
*   **Ngữ cảnh hỗ trợ (Native Context):** 128K tokens (RoPE scale cải tiến).
*   **Điểm mạnh:** Đạt điểm rất cao trong khả năng giải luận, coding và hỗ trợ đa ngôn ngữ bao gồm tiếng Việt. Dùng để đối chiếu xem phương pháp nén (TurboQuant/PolarQuant) ảnh hưởng thế nào đến kiến trúc SOTA.

## 2. `llama3.1:8b-instruct-fp16`
*   **Hệ / Nguồn gốc:** Meta AI, kiến trúc Llama 3.1.
*   **Vai trò trong Benchmark:** Baseline phổ biến nhất thị trường mã nguồn mở.
*   **Ngữ cảnh hỗ trợ (Native Context):** 128K tokens.
*   **Điểm mạnh:** Được fine-tune cho instruction-following, hỗ trợ tiếng Việt ở mức khá. Là model "chuẩn mực" để so sánh với các kỹ thuật nén vì được cộng đồng nghiên cứu sử dụng rộng rãi.

## 3. `mistral:7b-instruct-v0.3-fp16`
*   **Hệ / Nguồn gốc:** Mistral AI (Pháp).
*   **Vai trò trong Benchmark:** Baseline kiến trúc Mistral với Sliding Window Attention (SWA).
*   **Ngữ cảnh hỗ trợ (Native Context):** 32K tokens.
*   **Điểm mạnh:** Kiến trúc GQA (Grouped Query Attention) giúp KV Cache tự nhiên nhỏ hơn. Benchmark sẽ cho thấy hiệu quả nén KV Cache trên kiến trúc đã được tối ưu GQA so với kiến trúc MHA truyền thống.

## 4. `qwen2.5:7b-instruct-fp16`
*   **Hệ / Nguồn gốc:** Qwen Team (Alibaba Cloud), phiên bản Qwen2.5.
*   **Vai trò trong Benchmark:** Đại diện cho dòng Qwen2.5 7B instruction-tuned chuẩn.
*   **Ngữ cảnh hỗ trợ (Native Context):** 128K tokens.
*   **Điểm mạnh:** Rất mạnh về tiếng Việt nhờ được train trên tập dữ liệu đa ngôn ngữ khổng lồ. Là model Qwen thế hệ trước để đối chiếu với Qwen3 8B trong cùng benchmark.



# KIẾN TRÚC HỆ THỐNG TỔNG THỂ (OVERALL SYSTEM ARCHITECTURE)

Hệ thống benchmark được chia làm **4 tầng chức năng chính** hoạt động theo mô hình luồng dữ liệu một chiều (Pipeline-driven):

```text
+-----------------------------------------------------------------------------------+
|                        TẦNG 1: TIỀN XỬ LÝ & ĐÓNG GÓI DỮ LIỆU                      |
|  [Raw Vietnamese Datasets] -> [NVIDIA NeMo Curator] -> [Context Bucketizer]       |
|                                                              | (4k, 8k, 16k, 32k) |
|                                                              v                    |
|                                                    [JSON/JSONL Test Suite]        |
+-----------------------------------------------------------------------------------+
                                                               |
                                                               v
+-----------------------------------------------------------------------------------+
|                     TẦNG 2: THỰC THI SUY LUẬN & NÉN (LLM SERVING)                 |
|   +---------------------------------------------------------------------------+   |
|   |  Inference Engine (vLLM / SGLang with PagedAttention)                     |   |
|   |  - Model Loader: [qwen3:8b-fp16] / [llama3.1:8b-instruct-fp16] / [mistral:7b-instruct-v0.3-fp16] / [qwen2.5:7b-instruct-fp16] (FP16/BF16) |   |
|   +---------------------------------------------------------------------------+   |
|                                                              |                    |
|   +----------------------------------------------------------v----------------+   |
|   |  Quantization & Compression Controller                                    |   |
|   |  - Weight Compressor: [HQQ / Marlin Kernels]                              |   |
|   |  - KV Cache Quantizer: [Full BF16] vs [FP8] vs [PolarQuant] vs [TurboQuant]   |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
                                                               |
                                                               v
+-----------------------------------------------------------------------------------+
|                        TẦNG 3: GIÁM SÁT & ĐO ĐẠC CHỈ SỐ                           |
|   +------------------------+  +--------------------------+  +-----------------+   |
|   |   Hardware Profiler    |  |  Performance Latency     |  | Quality Monitor |   |
|   |  (PyNVML/Max VRAM Hook)|  |  (TTFT & ITL Time Hooks) |  | (Perplexity API)|   |
|   +------------------------+  +--------------------------+  +-----------------+   |
+-----------------------------------------------------------------------------------+
                                                               |
                                                               v
+-----------------------------------------------------------------------------------+
|                       TẦNG 4: LƯU TRỮ, PHÂN TÍCH & TRỰC QUAN                      |
|  [CSV Log Aggregator] -> [Statistical Analyzer] -> [Pareto Frontier Plotter]      |
+-----------------------------------------------------------------------------------+
```

---

## CHI TIẾT CÁC TẦNG CHỨC NĂNG (COMPONENT DETAILS)

### 1. Tầng 1: Tiền xử lý & Đóng gói dữ liệu (Data Ingestion & Preparation Layer)
Nhiệm vụ chính là lọc sạch và phân mảnh dữ liệu ngữ cảnh dài để đảm bảo việc đo đạc công bằng:
*   **NVIDIA NeMo Curator Pipeline:** Thực hiện lọc trùng lặp (deduplication), chuẩn hóa bảng mã Unicode tiếng Việt, loại bỏ nhiễu và văn bản lỗi từ các nguồn dữ liệu thô (VMLU, VTSNLP, ViNews).
*   **Context Bucketizer (Phân nhóm độ dài):** Phân bổ dữ liệu đầu vào thành các nhóm ngữ cảnh (Buckets) mục tiêu: **4,000, 8,000, 16,000, và 32,000 tokens** để kiểm tra giới hạn nén.
*   **Standardized Formatter:** Đóng gói dữ liệu thành tệp tin cấu trúc `datasets/test_set_small.json` chứa thông tin định tuyến (QA, Summarization, Retrieval).

### 2. Tầng 2: Thực thi suy luận & Lõi nén (LLM Serving & Compression Layer)
Đây là trái tim của hệ thống, điều phối luồng nén và tính toán Attention trên phần cứng:
*   **LLM Serving Engine (vLLM):** Sử dụng vLLM (phiên bản 0.25) làm công cụ phục vụ chính nhờ cơ chế quản lý bộ nhớ **PagedAttention**. PagedAttention sẽ chia nhỏ KV Cache thành các khối bộ nhớ (Blocks). Khi kích hoạt các thuật toán nén, vLLM sẽ cấp phát các khối này dưới dạng định dạng nén tương ứng.
*   **Model Loader:** Nạp các mô hình tiếng Việt đã được kiểm chứng ở định dạng gốc 16-bit (BF16/FP16). Trọng tâm chính của nghiên cứu này là lượng tử hóa và nén KV Cache khi thực thi suy luận, trọng số (weights) của mô hình gốc vẫn được giữ nguyên ở định dạng 16-bit (không nén weight).
*   **Lõi lượng tử hóa KV Cache (KV Cache Quantizer):**
    *   **Full KV Cache (BF16):** Đóng vai trò là mốc so sánh (Baseline chuẩn, mapping `auto` trong vLLM).
    *   **FP8 (8-bit Float):** Nén tuyến tính chuẩn công nghiệp (mapping `fp8` trong vLLM).
    *   **HQQ (Half-Quadratic Quantization):** Mapped sang `int4_per_token_head` (giải pháp thay thế gần nhất của nhân Marlin trong vLLM 0.25).
    *   **PolarQuant (via PolarEngine/Triton):** Mapped sang `fp8` làm fallback trong vLLM 0.25.
    *   **TurboQuant:** Mapped sang `turboquant_4bit_nc` (4-bit no compensation) và `turboquant_3bit_nc` (3-bit no compensation) tích hợp sẵn trong nhân vLLM.
    *   *Cơ chế hoạt động:* Các thuật toán nén KV Cache được tích hợp trực tiếp vào inference engine qua các đối số khởi dựng để đo đạc và đánh giá hiệu năng thực tế. Nghiên cứu tập trung vào việc benchmark đánh giá và tích hợp hệ thống, không yêu cầu huấn luyện lại mô hình hoặc viết lại các kernel Triton/CUDA mới từ đầu.

### 3. Tầng 3: Giám sát & Đo đạc chỉ số (Instrumentation & Monitoring Layer)
Tầng này hoạt động song song với quá trình sinh từ (generation process) của mô hình để ghi nhận các thông số hệ thống một cách chính xác nhất:
*   **Hardware Profiler:**
    *   Sử dụng thư viện quản lý GPU của NVIDIA (`pynvml`) kết hợp với luồng đo chạy nền song song (**`VRAMMonitor` background thread**).
    *   Bộ giám sát chạy nền thực hiện lấy mẫu (polling) liên tục mỗi 50ms trong suốt pha Prefill và Decode để bắt được chính xác lượng dung lượng bộ nhớ đỉnh Peak VRAM thực tế (`peak_memory_mb`) thay vì chỉ đọc thụ động sau sinh từ.
*   **Performance Tracer (Trình đo độ trễ):**
    *   Sử dụng các mốc thời gian hệ thống (High-resolution timing hooks) để ghi nhận:
        *   **Latency (ms/token):** Thời gian sinh trung bình của một token.
        *   **Throughput (Tokens/s):** Tổng số token sinh ra chia cho tổng thời gian decode.
*   **Quality Monitor (Trình đánh giá chất lượng):**
    *   **Perplexity (PPL) Offline Evaluator:** Được thực hiện **offline** (sau khi kết thúc benchmark) bằng cách dùng script `compute_all_ppl.py` tải mô hình tham chiếu gốc không nén (BF16/FP16) để đo Perplexity khách quan trên các chuỗi văn bản sinh ra nhằm tránh sai số hoặc bias.
    *   **Linguistic Quality Auditor:** Đo đạc chỉ số lặp từ n-grams (`repeated_ngram_ratio`) và tỷ lệ ký tự rác (`special_char_ratio`) để tự động gán cờ cảnh báo chất lượng (`repetition_flag`, `gibberish_flag`, `quality_warning`).

### 4. Tầng 4: Lưu trữ, Phân tích & Trực quan (Storage, Analysis & Reporting Layer)
Tầng cuối cùng chịu trách nhiệm tổng hợp dữ liệu thô thành tri thức khoa học:
*   **CSV Log Aggregator:** Tập hợp kết quả thử nghiệm từ tất cả các cặp chạy mô hình vào file thống nhất (ví dụ: `template_log_real_run_all.csv`, `template_log_real_run_mistral_final_all.csv`) theo định dạng cấu trúc định nghĩa trước.
*   **Statistical Analyzer:** Sử dụng thư viện `pandas` để tự động tổng hợp kết quả (`all_results_compiled.csv`) và tính toán giá trị trung bình (mean) và độ lệch chuẩn (standard deviation) cho từng kịch bản thử nghiệm (`all_results_summary.csv`).
*   **Pareto Frontier Plotter:** Sử dụng `matplotlib` để tự động hóa việc vẽ biểu đồ phân tích đánh đổi (Trade-off curves) lưu vào `results/plots/`, bao gồm cả đồ thị Pareto Frontier giữa chất lượng ngôn ngữ Perplexity và Peak VRAM tiêu thụ.

---

## Ý NGHĨA CỦA KIẾN TRÚC

1.  **Tính khoa học và tách biệt (Separation of Concerns):** Cho thấy hệ thống đo đạc (Tầng 3) không can thiệp và làm ảnh hưởng đến hiệu năng thực tế của lõi suy luận (Tầng 2).
2.  **Tính thực tiễn (Industrial Relevance):** Việc phân tách đo đạc giữa pha *Prefill* và *Decode* là tối quan trọng trong các hệ thống LLM serving hiện đại vào năm 2026.
3.  **Khả năng tái lập (Reproducibility):** Cấu trúc này mô tả rõ ràng luồng đi của dữ liệu từ khâu xử lý thô đến khâu vẽ đồ thị, giúp các nghiên cứu sau dễ dàng lặp lại thực nghiệm của nhóm.

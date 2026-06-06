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
|   |  - Model Loader: [PhoGPT-7B5] / [Qwen2.5-7B] / [Llama-3.1-8B] (BF16)      |   |
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
*   **LLM Serving Engine (vLLM):** Sử dụng vLLM làm công cụ phục vụ chính nhờ cơ chế quản lý bộ nhớ **PagedAttention**. PagedAttention sẽ chia nhỏ KV Cache thành các khối bộ nhớ (Blocks). Khi kích hoạt các thuật toán nén, vLLM sẽ cấp phát các khối này dưới dạng định dạng nén tương ứng.
*   **Model Loader:** Nạp các mô hình tiếng Việt đã được kiểm chứng ở định dạng gốc 16-bit (BF16/FP16). Trọng tâm chính của nghiên cứu này là lượng tử hóa và nén KV Cache khi thực thi suy luận, trọng số (weights) của mô hình gốc vẫn được giữ nguyên ở định dạng 16-bit (không nén weight).
*   **Lõi lượng tử hóa KV Cache (KV Cache Quantizer):**
    *   **Full KV Cache (BF16):** Đóng vai trò là mốc so sánh (Baseline chuẩn).
    *   **FP8 (8-bit Float):** Nén tuyến tính chuẩn công nghiệp.
    *   **PolarQuant (via PolarEngine/Triton):** Chuyển đổi dữ liệu KV Cache sang hệ tọa độ cực thông qua phép quay trực giao ngẫu nhiên kết hợp tối ưu hóa Lloyd-Max.
    *   **TurboQuant:** Lấy PolarQuant làm nền tảng, áp dụng thêm một bit sửa sai **QJL (Quantized Johnson-Lindenstrauss)** trên mỗi vector để bù đắp sai số tính toán Attention.
    *   *Cơ chế hoạt động:* Các thuật toán nén KV Cache (FP8, PolarQuant, TurboQuant) được tích hợp trực tiếp vào inference engine qua các thư viện và nhân (kernels) CUDA/Triton sẵn có để đo đạc và đánh giá hiệu năng thực tế. Nghiên cứu tập trung vào việc benchmark đánh giá và tích hợp hệ thống, không yêu cầu huấn luyện lại mô hình hoặc viết lại các kernel Triton/CUDA mới từ đầu.

### 3. Tầng 3: Giám sát & Đo đạc chỉ số (Instrumentation & Monitoring Layer)
Tầng này hoạt động song song với quá trình sinh từ (generation process) của mô hình để ghi nhận các thông số hệ thống một cách chính xác nhất:
*   **Hardware Profiler:**
    *   Sử dụng thư viện quản lý GPU của NVIDIA (`pynvml`) kết hợp với `torch.cuda.max_memory_allocated()`.
    *   Bộ giám sát sẽ bắt đầu ghi nhận từ lúc mô hình được nạp (Base Model Memory), sau đó đo lượng VRAM tăng lên trong pha xử lý prompt đầu vào (**Prefill Peak VRAM**) và lượng VRAM đỉnh khi sinh toàn bộ chuỗi văn bản (**Decode Peak VRAM**).
*   **Performance Tracer (Trình đo độ trễ):**
    *   Sử dụng các mốc thời gian hệ thống (High-resolution timing hooks) của Python để ghi nhận:
        *   **Time to First Token (TTFT):** Thời gian tính từ lúc gửi prompt đến khi token đầu tiên xuất hiện.
        *   **Inter-Token Latency (ITL):** Thời gian trung bình để sinh mỗi token tiếp theo.
        *   **Throughput (Tokens/s):** Tổng số token sinh ra chia cho tổng thời gian decode.
*   **Quality Monitor (Trình đánh giá chất lượng):**
    *   **Perplexity (PPL) Evaluator:** Tính toán giá trị lũy thừa của hàm mất mát Cross-Entropy trung bình trên tập dữ liệu thử nghiệm. Chỉ số PPL càng thấp chứng tỏ khả năng hiểu tiếng Việt của mô hình càng ít bị suy giảm sau khi nén.
    *   **Downstream Task Evaluator:** Đo đạc chất lượng thực tế thông qua các hàm so khớp chuỗi (Exact Match, F1 Score cho Hỏi đáp, ROUGE-L cho Tóm tắt).

### 4. Tầng 4: Lưu trữ, Phân tích & Trực quan (Storage, Analysis & Reporting Layer)
Tầng cuối cùng chịu trách nhiệm tổng hợp dữ liệu thô thành tri thức khoa học:
*   **CSV Log Aggregator:** Tập hợp kết quả thử nghiệm từ tất cả các cặp chạy mô hình vào file thống nhất `results/template_log.csv` theo định dạng cấu trúc định nghĩa trước.
*   **Statistical Analyzer:** Sử dụng thư viện `pandas` để tính toán giá trị trung bình (mean) và độ lệch chuẩn (standard deviation) cho từng kịch bản thử nghiệm nhằm đảm bảo tính lặp lại (reproducibility) của kết quả.
*   **Pareto Frontier Plotter:** Sử dụng `matplotlib` và `plotly` để tự động hóa việc vẽ biểu đồ phân tích đánh đổi (Trade-off curves). Biểu đồ này sẽ chỉ ra điểm **Pareto-optimal** (điểm mà tại đó dung lượng bộ nhớ được tiết kiệm nhiều nhất nhưng chất lượng ngôn ngữ suy giảm ít nhất).

---

## Ý NGHĨA CỦA KIẾN TRÚC NÀY TRONG PAPER

Khi bạn trình bày kiến trúc này trong phần **Section 5: Methodology** hoặc **Section 6: System Architecture** của bài báo tiếng Anh (Paper EN), cấu trúc 4 tầng rõ ràng này sẽ giúp các phản biện (reviewers) đánh giá cao vì:
1.  **Tính khoa học và tách biệt (Separation of Concerns):** Cho thấy hệ thống đo đạc (Tầng 3) không can thiệp và làm ảnh hưởng đến hiệu năng thực tế của lõi suy luận (Tầng 2).
2.  **Tính thực tiễn (Industrial Relevance):** Việc phân tách đo đạc giữa pha *Prefill* và *Decode* là tối quan trọng trong các hệ thống LLM serving hiện đại vào năm 2026.
3.  **Khả năng tái lập (Reproducibility):** Cấu trúc này mô tả rõ ràng luồng đi của dữ liệu từ khâu xử lý thô đến khâu vẽ đồ thị, giúp các nghiên cứu sau dễ dàng lặp lại thực nghiệm của nhóm.
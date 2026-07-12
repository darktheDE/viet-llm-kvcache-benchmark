# Results Plots (Biểu đồ Kết quả)

Thư mục này chứa các biểu đồ trực quan hóa và các biểu đồ phân tích đánh đổi (trade-off) (ví dụ: Peak VRAM vs Perplexity, Latency vs Context Length) được sinh từ file log kết quả đo đạc.

## File Metadata

| Tên File | Người tạo | Vai trò / Mục đích |
| :--- | :--- | :--- |
| **[latency_vs_context.png](latency_vs_context.png)** | HuynhThach1606 \<23133072@student.hcmute.edu.vn\> | Biểu đồ so sánh độ trễ (TTFT & ITL) theo độ dài ngữ cảnh ngữ cảnh đầu vào (4k, 8k, 16k, 32k). |
| **[pareto_ppl_vs_vram.png](pareto_ppl_vs_vram.png)** | HuynhThach1606 \<23133072@student.hcmute.edu.vn\> | Biểu đồ Pareto Frontier tối ưu hóa giữa chất lượng mô hình (Perplexity) và bộ nhớ tiêu thụ (Peak VRAM). |
| **[throughput_vs_context.png](throughput_vs_context.png)** | HuynhThach1606 \<23133072@student.hcmute.edu.vn\> | Biểu đồ so sánh thông lượng sinh từ (tokens/sec) theo độ dài ngữ cảnh tăng dần. |
| **[vram_vs_context.png](vram_vs_context.png)** | HuynhThach1606 \<23133072@student.hcmute.edu.vn\> | Biểu đồ so sánh lượng bộ nhớ VRAM đỉnh (Peak VRAM) tiêu thụ thực tế theo chiều dài ngữ cảnh. |

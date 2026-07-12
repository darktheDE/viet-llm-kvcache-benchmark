# Results (Thư mục Kết quả Thực nghiệm)

Thư mục này chứa toàn bộ các kết quả thô, file log nén, số liệu thống kê được biên dịch, các Jupyter Notebook phân tích và biểu đồ trực quan hóa biểu diễn sự đánh đổi (trade-off) giữa hiệu năng phần cứng và chất lượng ngôn ngữ của các mô hình LLM tiếng Việt khi áp dụng các kỹ thuật nén KV Cache.

---

## 1. Cấu trúc thư mục & Metadata chi tiết

| Tên File / Thư mục | Người tạo | Vai trò / Mục đích chi tiết |
| :--- | :--- | :--- |
| **[plots/](plots/)** | HuynhThach1606 | Thư mục chứa các biểu đồ PNG so sánh hiệu năng được tự động vẽ bởi script `plot_results.py`. |
| **[template_log.csv](template_log.csv)** | Quan-min211 | File log mẫu định nghĩa cấu trúc cột tiêu chuẩn (Schema) để ghi nhận kết quả đo đạc. |
| **[template_log_real_run.csv](template_log_real_run.csv)** | Quan-min211 | Số liệu benchmark thô thu thập trực tiếp từ GPU A100 cho Qwen3 và Qwen2.5. |
| **[template_log_real_run_all.csv](template_log_real_run_all.csv)** | QUOC ANH \<quocanh0815@gmail.com\> | Kết quả đã backfill Perplexity offline và cờ cảnh báo chất lượng tiếng Việt cho Qwen3, Qwen2.5, Phi-4, Gemma-3. |
| **[template_log_real_run_extra.csv](template_log_real_run_extra.csv)** | Quan-min211 | Số liệu benchmark thô cho các mô hình bổ sung (Phi-4, Gemma-3). |
| **[template_log_real_run_mistral_final.csv](template_log_real_run_mistral_final.csv)** | Quan-min211 | Số liệu benchmark thô từ đợt chạy GPU của Mistral 7B. |
| **[template_log_real_run_mistral_final_all.csv](template_log_real_run_mistral_final_all.csv)** | QUOC ANH \<quocanh0815@gmail.com\> | Kết quả đã backfill Perplexity offline và cờ cảnh báo chất lượng tiếng Việt cho Mistral 7B. |
| **[all_results_compiled.csv](all_results_compiled.csv)** | HuynhThach1606 | File CSV tổng hợp toàn bộ các kết quả riêng lẻ được script `plot_results.py` tự động quét và gộp lại. |
| **[all_results_summary.csv](all_results_summary.csv)** | HuynhThach1606 | Bản tóm tắt tính toán giá trị trung bình (mean) và độ lệch chuẩn (std) của các metrics theo từng cấu hình (Model x Method x Context). |
| **[benchmark_analysis.ipynb](benchmark_analysis.ipynb)** | Quan-min211 | Jupyter Notebook phân tích và trực quan hóa số liệu của pha chạy giả lập (Mock). |
| **[real_benchmark_analysis.ipynb](real_benchmark_analysis.ipynb)** | Quan-min211 | Jupyter Notebook phân tích chi tiết dữ liệu thực nghiệm thật thu được trên GPU Cloud. |

---

## 2. Đặc tả các cột dữ liệu (Metrics Schema)

Các file CSV kết quả cuối cùng (có đuôi `_all.csv`) bao gồm các trường thông tin sau:

### A. Thông số cấu hình thực nghiệm
*   `model`: Tên định danh của mô hình LLM (ví dụ: `qwen3:8b-fp16`, `mistral:7b-instruct-v0.3-fp16`).
*   `kv_cache_type`: Phương pháp nén KV Cache được áp dụng (`FP16`, `FP8`, `HQQ`, `PolarQuant`, `TurboQuant`).
*   `context_length`: Chiều dài ngữ cảnh tối đa cấu hình cho lượt chạy (4k, 8k, 16k).
*   `status`: Trạng thái của lượt chạy (`OK` nếu thành công, `OOM` nếu tràn bộ nhớ GPU, `RUN_ERROR` nếu lỗi tokenizer/hệ thống).

### B. Chỉ số Hiệu năng phần cứng (Hardware Performance)
*   `peak_memory_mb`: Dung lượng đỉnh VRAM của GPU ghi nhận được trong suốt pha Prefill và Decode (được đo động bằng background thread lấy mẫu mỗi 50ms qua `pynvml`).
*   `latency_ms_per_token`: Thời gian sinh trung bình của một token (ms/token).
*   `throughput_tokens_per_s`: Tốc độ giải mã trung bình (tokens/second).

### C. Chỉ số Chất lượng Sinh & Ngôn ngữ (Text Quality & Perplexity)
*   `perplexity` (PPL): Chỉ số Perplexity đo offline sử dụng mô hình tham chiếu gốc không nén (FP16/BF16) đối chiếu với văn bản sinh ra (Giá trị thấp hơn thể hiện chất lượng ngôn ngữ tốt hơn).
*   `ppl_loss`: Giá trị Cross-Entropy Loss trung bình tính toán được trong quá trình sinh văn bản.
*   `ppl_tokens`: Số lượng tokens thực tế được dùng để tính toán Perplexity.
*   `repetition_flag`: Cờ cảnh báo khi phát hiện hiện tượng mô hình bị lặp từ tuần hoàn vô hạn (`True`/`False`).
*   `gibberish_flag`: Cờ cảnh báo khi phát hiện mô hình sinh các ký tự đặc biệt vô nghĩa hoặc rác (`True`/`False`).
*   `repeated_ngram_ratio`: Tỷ lệ lặp lại của các cụm n-grams trong văn bản sinh ra, dùng để đánh giá định lượng lỗi lặp từ.
*   `special_char_ratio`: Tỷ lệ xuất hiện của các ký tự đặc biệt lạ, đánh giá lỗi sinh rác văn bản.
*   `output_length`: Chiều dài (số lượng token) của văn bản được mô hình sinh ra.
*   `quality_warning`: Nhãn phân loại cảnh báo lỗi ngôn ngữ sinh ra (như `repetition`, `gibberish`, hoặc để trống nếu văn bản mượt mà ổn định).

---

## 3. Thư mục Biểu đồ [plots/](plots/)

Sau khi chạy lệnh `python scripts/plot_results.py`, thư mục này sẽ sinh ra các biểu đồ trực quan hóa chính:
1.  **`vram_vs_context.png`**: Biểu diễn VRAM tiêu thụ tăng dần theo chiều dài ngữ cảnh, phân biệt theo các phương pháp nén.
2.  **`latency_vs_context.png`**: Biểu diễn độ trễ tăng dần của pha decode theo ngữ cảnh.
3.  **`throughput_vs_context.png`**: Thể hiện tốc độ sinh từ giảm dần khi prompt phình to.
4.  **`pareto_ppl_vs_vram.png`**: **Biểu đồ Pareto Frontier cốt lõi**, phân tích sự đánh đổi giữa mức độ tiết kiệm VRAM vật lý (trục X) và độ suy giảm Perplexity ngôn ngữ (trục Y). Giúp xác định phương pháp tối ưu nhất cho deploy thực tế.

# Nhật ký tiến trình và Review trạng thái triển khai ngày 13/07/2026

## 1. Các hạng mục đã triển khai và hoàn thành
So với đợt kiểm kê ngày 12/07/2026, nhóm Technical & Experiment đã đạt được bước tiến lớn, giải quyết hầu hết các lỗi nghiêm trọng và hoàn thành phần lớn kết quả benchmark thực tế:

### A. Hạ tầng & Môi trường chạy
*   **Cập nhật `requirements.txt`:** Đã khai báo đầy đủ các thư viện phụ thuộc cốt lõi cho môi trường GPU gồm `torch`, `vllm`, `pynvml`, và `psutil`. Môi trường ảo conda `viet-llm` đã cài đặt và biên dịch thành công.

### B. Kịch bản chạy Benchmark GPU thực tế
*   **Chạy thành công Grid Search:** Đã chạy và ghi nhận số liệu GPU thật trên card A100 80GB cho Qwen3, Qwen2.5, Phi-4, Gemma-3 và Mistral 7B. Dữ liệu thực tế được lưu tại:
    *   [template_log_real_run.csv](../../results/template_log_real_run.csv) (Qwen3, Qwen2.5)
    *   [template_log_real_run_extra.csv](../../results/template_log_real_run_extra.csv) (Phi-4, Gemma-3)
    *   [template_log_real_run_mistral_final.csv](../../results/template_log_real_run_mistral_final.csv) (Mistral 7B)
*   **Đo đạc Peak VRAM nền chính xác:** Thay thế việc đọc VRAM thụ động sau khi chạy bằng luồng chạy nền `VRAMMonitor` (sử dụng thư viện `pynvml`). Sampler nền này lấy mẫu liên tiếp mỗi 50ms trong suốt pha Prefill và Decode để ghi nhận Peak VRAM thực sự cao nhất của GPU.
*   **Tối ưu hóa chạy Mistral:** Xây dựng script [run_mistral_optimized.py](../test/run_mistral_optimized.py) và [run_mistral_single_method.py](../test/run_mistral_single_method.py) theo cơ chế **"Load Once Per Method"**. Model chỉ load vào GPU đúng 1 lần cho mỗi phương pháp nén (FP16, FP8, HQQ, PolarQuant, TurboQuant) rồi chạy liên tiếp 3 context lengths (4k, 8k, 16k) giúp tiết kiệm 70% thời gian chạy.
*   **Bổ sung tham số đầu ra:** Cập nhật script optimized của Mistral hỗ trợ tham số `--output` và `--dataset` để ghi dữ liệu ra các file riêng biệt, tránh ghi đè làm mất dữ liệu của Qwen3/Qwen2.5.

### C. Cơ chế tính Perplexity (PPL) Offline & Backfill
*   Triển khai bộ script [compute_all_ppl.py](../compute_all_ppl.py) và `compute_ppl_offline.py` hoạt động ổn định.
*   Script tự động ánh xạ alias sang HuggingFace Repo, đọc đường dẫn file `.jsonl` từ cột `output_path` của file CSV đầu vào, sau đó tải model tham chiếu gốc để đo PPL khách quan.
*   Đã backfill thành công Perplexity và các chỉ số cảnh báo lặp từ (`repetition_flag`, `quality_warning`, `repeated_ngram_ratio`) cho Qwen3, Qwen2.5, Phi-4 và Mistral 7B (Lưu trong các file kết quả đuôi `_all.csv`).

---

## 2. Nhật ký xử lý sự cố kỹ thuật (Troubleshooting)

### A. Sự cố tràn ngữ cảnh 16k của Mistral 7B
*   **Hiện tượng:** Khi chạy Mistral ở context 16k, vLLM báo lỗi `RUN_ERROR` do prompt thực tế dài **30,177 tokens** (vượt quá `max_model_len = 30176` được tính từ công thức cũ với `MISTRAL_RATIO = 1.75` và `buf = 2048`).
*   **Khắc phục:** Tăng hệ số phình to token `MISTRAL_RATIO` lên **`1.9`** và giảm buffer an toàn xuống **`1024`**, đồng thời ràng buộc `min(..., 32768)` để không vượt quá giới hạn phần cứng của Mistral. Lượng ngữ cảnh cấp phát mới sẽ là **`31,552`** tokens, vừa đủ dung lượng chạy vừa an toàn cho GPU.

### B. Thiếu chỉ số Perplexity của Gemma 3 4B
*   **Nguyên nhân:** Gặp lỗi xác thực Hugging Face khi tải model gated `google/gemma-3-4b-it` trên môi trường tính toán offline.
*   **Khắc phục:** Cần sử dụng Token Hugging Face đã được xác minh quyền truy cập để tải và tính toán.

### C. Sự không tương thích của Phi-4 Mini Reasoning
*   **Nguyên nhân:** Các phương pháp nén KV Cache nâng cao như `HQQ` và `TurboQuant` không hỗ trợ kiến trúc của Phi-4 trên phiên bản vLLM 0.25 hiện tại.
*   **Kết luận:** Nhóm ghi nhận đây là hạn chế phần cứng/thư viện và loại trừ kết quả HQQ/TurboQuant của Phi-4 khỏi bảng so sánh (chỉ đánh giá FP16, FP8 và PolarQuant).

---

## 3. Trạng thái hoàn thành DoD và Kế hoạch tiếp theo

### A. Đánh giá mức độ hoàn thiện
*   **DoD kịch bản phục dựng mã nguồn:** Đạt **80%**. Các kịch bản chạy thật, sampler đo VRAM, tối ưu hóa load model và bộ tính PPL đã hoạt động tốt.
*   **Dữ liệu thực nghiệm:** Đạt **70%**. Đã có dữ liệu thật của 4 trên 5 model (Qwen3, Qwen2.5, Phi-4, Mistral, một phần Gemma-3).
*   **Báo cáo & Bài báo:** Đạt **25%** (Đang trong giai đoạn viết Word và thiết kế Slide).

### B. Các bước tiếp theo cần thực hiện
1.  Kích hoạt chạy benchmark cho model chính cuối cùng: `llama3.1:8b-instruct-fp16`.
2.  Chạy lại benchmark Mistral 16k sau khi cập nhật hệ số `MISTRAL_RATIO = 1.9` để lấy đủ dữ liệu.
3.  Xin quyền truy cập cho Gemma-3 để tính toán nốt PPL offline.
4.  Chạy script `python scripts/plot_results.py` để gom tất cả các file CSV và vẽ biểu đồ Pareto Frontier cuối cùng.
5.  Hoàn thiện file báo cáo Word 30 trang và Slide bảo vệ slide-deck.

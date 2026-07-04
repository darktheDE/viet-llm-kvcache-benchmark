# Kế hoạch Tái cấu trúc luồng Đo đạc PPL & Ghi Log (PPL & Quality Logging Refactor)

Tài liệu này tổng hợp các vấn đề về hệ thống đo đạc hiện tại do team phát hiện, và đưa ra giải pháp xử lý (Action Plan) thống nhất để giải quyết triệt để.

---

## 1. Các vấn đề hiện tại (Problems)

Nhận xét của team về kiến trúc đo đạc hiện tại là **hoàn toàn chính xác**:
1. **Tính toán PPL bị sai nguyên lý:** Hàm `compute_perplexity()` trong `run_real_benchmark.py` đang dùng chính logprobs của model bị nén để tính PPL. Để so sánh chất lượng, PPL cần được tính bằng model tham chiếu (Reference Model - BF16).
2. **Thiếu Quality Flags:** Dù có hàm trong `utils_text.py`, nhưng benchmark hiện tại không ghi nhận các cờ quan trọng như: `repetition_flag`, `gibberish_flag`, `repeated_ngram_ratio`, `special_char_ratio`, `output_length`, `quality_warning`.
3. **Schema CSV lỏng lẻo:** `results/template_log.csv` chỉ có 8 cột, chưa đủ chỗ chứa PPL và Quality Flags.
4. **Lỗi Parse Dữ liệu (Downstream Parsing):** Các cột số (VRAM, Latency) đang bị chèn chuỗi như `"OOM"`, `"ERROR"`, `"N/A"`, khiến thư viện (như Pandas) nhận diện sai kiểu dữ liệu thành String.
5. **Rủi ro khi Gom file:** Lệnh `glob("*.csv")` trong `plot_results.py` sẽ vô tình đọc cả các file kết quả tổng hợp (`all_results_compiled.csv`), dẫn đến lặp dữ liệu.

---

## 2. Giải pháp Kỹ thuật (Action Plan)

Dưới đây là các bước cần thực hiện để sửa toàn bộ các lỗi trên:

### Bước 2.1: Áp dụng cơ chế Backfill cho Perplexity (PPL)
*   **Bỏ tính PPL trực tiếp (Online):** Xóa hàm tính PPL hiện tại trong các file `run_baseline.py` và `run_real_benchmark.py`. Cột `perplexity` tạm thời để trống (`None` / rỗng).
*   **Lưu trữ văn bản sinh ra (Generated Text):** Trong quá trình chạy benchmark, mọi văn bản sinh ra (`generated_text`) sẽ được lưu vào một kho chung là `results/generated_texts.jsonl`. Mỗi dòng chứa đủ Metadata: `model`, `kv_cache_type`, `context_length`, và `text`.
*   **Tính PPL Offline:** Sau khi Benchmark hoàn thành, team sẽ tạo một script mới (vd: `scripts/compute_ppl_offline.py`). Script này sẽ load model BF16 nguyên bản, đọc file JSONL, tính toán PPL chuẩn cho từng text, và "backfill" (cập nhật) lại cột PPL trong file CSV.

### Bước 2.2: Nâng cấp và Tích hợp Quality Flags
*   Cập nhật file `scripts/utils_text.py`: Bổ sung các logic tính N-gram để tìm tỷ lệ lặp (`repeated_ngram_ratio`), tỷ lệ ký tự lạ (`special_char_ratio`), từ đó tự động dán nhãn `repetition_flag` và `gibberish_flag`.
*   Gọi hàm này ở bước Decode của script Benchmark và map kết quả trực tiếp vào các cột CSV mới.

### Bước 2.3: Chuẩn hóa lại Schema CSV
*   Thêm các cột mới: `repetition_flag`, `gibberish_flag`, `repeated_ngram_ratio`, `special_char_ratio`, `output_length`, `quality_warning`. (Sẽ cập nhật tài liệu `results/README.md`).
*   **Quy tắc điền lỗi:** Nếu bị OOM hoặc Error, các cột dạng số (Peak VRAM, Latency) sẽ bị bỏ trống (`None`), và nguyên nhân lỗi (OOM, CUDA_ERROR...) sẽ được ghi tường minh vào cột `status`. Pandas sẽ tự hiểu cột trống là `NaN` và giữ nguyên tính toán số học.

### Bước 2.4: Sửa lỗi Globbing trong hàm Vẽ Biểu đồ
*   Cập nhật `scripts/plot_results.py`: Không dùng `glob("*.csv")` bừa bãi. Script sẽ chỉ tìm và đọc những file có tiền tố rõ ràng (như `template_log_real_run.csv` hoặc `template_log_demo_run.csv`), hoặc chủ động ignore các file `all_results_*`.

---

## 3. Các việc cần làm tiếp theo (Next Steps)

*   [ ] Review tài liệu này để đảm bảo mọi bên hiểu rõ thiết kế mới.
*   [ ] Triển khai sửa file `utils_text.py`.
*   [ ] Triển khai sửa file `run_real_benchmark.py` và `run_baseline.py`.
*   [ ] Triển khai sửa file `plot_results.py`.
*   [ ] Viết script `compute_ppl_offline.py` (Có thể làm sau).

*(Ghi chú: Nếu team đồng ý với Action Plan này, AI Agent có thể tiến hành sửa đổi tự động hàng loạt các mã nguồn trong dự án ngay lập tức.)*

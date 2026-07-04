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

### ✅ Bước 2.1: Cơ chế Backfill PPL (ĐÃ TRIỂN KHAI)

**Kiến trúc lưu trữ 2 lớp:**
- **CSV** (nhẹ, cho biểu đồ): metadata + `sample_id` + `output_path`. Cột `perplexity` để trống, sẽ backfill sau.
- **JSONL** (nặng, cho PPL offline): mỗi dòng chứa đầy đủ thông tin cần thiết.

**Schema JSONL (mỗi dòng):**
```json
{
  "sample_id": "sail/Sailor2-8B-Chat__FP16__8000__s0",
  "prompt_text": "Văn bản đầu vào gốc...",
  "generated_text": "Văn bản model sinh ra...",
  "generated_tokens": 128,
  "model": "sail/Sailor2-8B-Chat",
  "dataset": "datasets/test_set_small.json",
  "context_length": 8000,
  "kv_cache_type": "FP16",
  "kv_cache_dtype": "auto",
  "max_new_tokens": 128,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "seed": null,
  "status": "OK",
  "error_message": null
}
```

**Schema CSV mở rộng (10 cột):**
```
model, kv_cache_type, context_length, peak_memory_mb, latency_ms_per_token, throughput_tokens_per_s, perplexity, status, sample_id, output_path
```

**File đã sửa:**
- `scripts/run_baseline.py` — Thêm `persist_generated_texts()`, mở rộng CSV header, sửa sentinel strings.
- `scripts/test/run_real_benchmark.py` — Xóa `compute_perplexity()` sai logic, thêm persist JSONL, sửa sentinel strings.

**Quy tắc điền lỗi (Sentinel Strings → Empty):**
- Nếu OOM hoặc Error: cột số (VRAM, Latency, Throughput, PPL) → để trống `""`. Pandas tự nhận diện thành `NaN`.
- Nguyên nhân lỗi ghi tường minh vào cột `status` (vd: `"OOM"`, `"ERROR: CUDA out of memory"`).

### Bước 2.2: Tính PPL Offline (CẦN LÀM)
*   Viết script mới `scripts/compute_ppl_offline.py`.
*   Script đọc file JSONL, load model BF16 gốc, tính PPL chuẩn cho từng `prompt_text + generated_text`.
*   Backfill cột `perplexity` trong file CSV theo `sample_id`.

### Bước 2.3: Nâng cấp Quality Flags (CẦN LÀM)
*   Cập nhật `scripts/utils_text.py`: bổ sung `repeated_ngram_ratio`, `gibberish_flag`, `quality_warning`.
*   Tích hợp gọi hàm này ở bước Decode và ghi vào CSV.

### Bước 2.4: Sửa lỗi Globbing (CẦN LÀM)
*   Cập nhật `scripts/plot_results.py`: loại trừ `all_results_compiled.csv` và `all_results_summary.csv` khỏi glob.

---

## 3. Các việc cần làm tiếp theo (Next Steps)

- [x] Xóa hàm `compute_perplexity()` sai logic trong `run_real_benchmark.py`.
- [x] Thêm hàm `persist_generated_texts()` lưu JSONL cho cả 2 file script.
- [x] Mở rộng CSV schema thêm `sample_id`, `output_path`.
- [x] Sửa sentinel strings (`"OOM"`, `"ERROR"`, `"N/A"` → `""`) trong các cột số.
- [ ] Viết script `compute_ppl_offline.py`.
- [ ] Nâng cấp `utils_text.py` với quality flags mới.
- [ ] Sửa globbing trong `plot_results.py`.

*(Ghi chú: Các mục đánh dấu [x] đã được triển khai xong trong code. Các mục [ ] cần team tiếp tục phát triển.)*

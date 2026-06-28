# Process Log – Analytics & Plotting

## Metadata

- Người thực hiện: Huỳnh Ngọc Thạch
- Team: Data & Analysis
- Vai trò: Analytics & Plotting
- Plane task: [ANALYSIS] Tổng hợp dữ liệu CSV, Tính toán Thống kê & Vẽ Biểu đồ Pareto Trade-off bằng Python
- Trạng thái hiện tại: Đã hoàn thành pipeline phân tích với mock data, chờ real benchmark data từ team Technical.

## Contribution

### 1. Benchmark result plotting script

- File tạo mới: `scripts/plot_results.py`
- Mục đích: Tự động đọc dữ liệu benchmark dạng CSV, gộp dữ liệu, tính thống kê và vẽ biểu đồ phân tích hiệu năng/chất lượng.
- Input:
  - `results/template_log.csv` dùng để test với mock data.
  - `results/real_benchmark_log.csv` sẽ dùng khi team Technical có dữ liệu thật.
- Output:
  - `results/all_results_compiled.csv`
  - `results/all_results_summary.csv`
  - `results/plots/vram_vs_context.png`
  - `results/plots/latency_vs_context.png`
  - `results/plots/throughput_vs_context.png`
  - `results/plots/pareto_ppl_vs_vram.png`

## Các bước đã thực hiện

1. Kiểm tra cấu trúc dữ liệu hiện có trong `results/template_log.csv`.
2. Xác định các cột cần thiết cho phân tích:
   - `model`
   - `kv_cache_type`
   - `context_length`
   - `peak_memory_mb`
   - `latency_ms_per_token`
   - `throughput_tokens_per_s`
   - `perplexity`
   - `status`
3. Viết script `scripts/plot_results.py`.
4. Chạy thử script bằng lệnh:

```bash
python3 scripts/plot_results.py --input-file results/template_log.csv
```

5. Kiểm tra các file output trong `results/` và `results/plots/`.
6. Ghi nhận rằng dữ liệu hiện tại là `MOCK_OK`, chỉ dùng để kiểm thử pipeline, chưa dùng để kết luận khoa học.

## Blocker hiện tại

- Chưa có file `results/real_benchmark_log.csv`.
- Cần team Technical chạy benchmark thật trên GPU Cloud để có dữ liệu thực nghiệm.

## Việc tiếp theo

1. Cập nhật Plane task với trạng thái hiện tại.
2. Theo dõi tiến độ team Technical để xác nhận timeline sinh `real_benchmark_log.csv`.
3. Khi có real data, chạy lại:

```bash
python3 scripts/plot_results.py --input-file results/real_benchmark_log.csv
```

4. Phân tích biểu đồ thật và viết phần nhận xét cho báo cáo/slide/paper.

# Process Log — Analytics & Plotting

## Metadata

- Người thực hiện: Huỳnh Ngọc Thạch
- Team: Data & Analysis
- Vai trò: Analytics & Plotting
- Plane task: [ANALYSIS] Tổng hợp dữ liệu CSV, Tính toán Thống kê & Vẽ Biểu đồ Pareto Trade-off bằng Python
- Trạng thái hiện tại: **Đã hoàn thành** — đã chạy với real benchmark data từ GPU A100 80 GB.

## Contribution

### 1. Script `scripts/plot_results.py`

- **Mục đích:** Tự động đọc dữ liệu benchmark dạng CSV (`*_all.csv` — đã backfill PPL), gộp dữ liệu, tính thống kê và vẽ 5 biểu đồ chuẩn học thuật (PDF + PNG) cho paper.
- **Input real:** `results/template_log_real_run_all.csv` (Qwen3, Qwen2.5, Phi-4, Gemma-3 — 42 rows) + `results/template_log_real_run_mistral_final_all.csv` (Mistral 7B — 15 rows).
- **Output:**
  - `results/all_results_compiled.csv` — dữ liệu gộp
  - `results/all_results_summary.csv` — thống kê trung bình/median

### 2. Bộ 5 biểu đồ (4 slot docs + 1 bonus)

| Slot | File | Mô tả |
|------|------|-------|
| Slot 1 | `vram_vs_context.pdf/png` | Peak VRAM vs context — pool-dominated, spread <2% |
| Slot 2 | `scaling_grid.pdf/png` | 3 panel: Latency / Throughput / PPL (median) vs context |
| Slot 3 | `pareto_ppl_vs_kvsize.pdf/png` | Pareto: PPL vs KV size lý thuyết (% FP16) @ 16k |
| Slot 4 | `compression_efficiency.pdf/png` | Bar ratio lý thuyết + effective throughput (quality-agnostic) |
| Bonus | `pareto_ppl_vs_speed.pdf/png` | Speed-quality trade-off thực tế |

## Các bước đã thực hiện

1. Kiểm tra cấu trúc dữ liệu hiện có trong `results/`.
2. Viết script `scripts/plot_results.py` phiên bản đầu (4 biểu đồ, mock data).
3. Khi team Technical có dữ liệu thật, nâng cấp script lên phiên bản data-honest (5 biểu đồ).
4. Sửa 3 lỗi từ peer review: trục Y VRAM từ 0 + GPU reference line; de-collision nhãn Pareto (FP8/PolarQuant); bar thay dashed cho compression ratio.
5. Xoá biểu đồ cũ (`latency_vs_context.png`, `throughput_vs_context.png`, etc.) không còn tương thích.
6. Cập nhật `results/plots/README.md` với caption tiếng Anh sẵn sàng cho paper.

## Data-integrity notes (embedded trong caption từng hình)

- **VRAM không phản ánh nén:** vLLM pre-allocate pool `gpu_memory_utilization ≈ 0.9` → peak VRAM ~70-71 GB như nhau. Hình Slot 1 chỉ là documented finding.
- **Pareto dùng KV size lý thuyết** (bit-width / 16) thay vì VRAM đo được — vì VRAM đo không phân biệt được method.
- **PolarQuant = FP8 fallback** trong setup này → 2 điểm trùng ở Pareto và bar compression.
- **Effective throughput** (Slot 4) là quality-agnostic — HQQ 4× ratio nhưng chất lượng kém (xem quality figure).
- **Không có OOM heatmap:** vì không có OOM thật, chỉ là context-window limit (32k).
- **Perplexity:** median cho scaling lines (robust), mean cho Pareto (phản ánh tail HQQ).

## Lệnh chạy

```bash
python3 scripts/plot_results.py
```

Kết quả: 5 file PDF + 5 file PNG trong `results/plots/`.

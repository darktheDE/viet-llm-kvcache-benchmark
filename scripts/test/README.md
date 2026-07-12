# GPU Benchmarking Scripts (Kịch bản Đo đạc Thực tế trên GPU)

Thư mục này chứa các kịch bản chính thức chạy trên máy chủ Cloud GPU (chúng tôi sử dụng **A100 80GB** làm server chạy chính và **RTX 3090/4090** để chạy thử nghiệm) nhằm đo đạc hiệu năng phần cứng và chất lượng sinh chữ thực tế của các mô hình LLM tiếng Việt.

---

## 1. Cơ chế cô lập bộ nhớ GPU (Subprocess Isolation Pattern)

Một trong những thiết kế quan trọng nhất của hệ thống grid search ở đây là **sử dụng Subprocess để cô lập CUDA memory**:

```
+------------------------------------------------------+
|               run_real_grid.py (Quản đốc)            |
+------------------------------------------------------+
                           |
            (Khởi chạy qua subprocess.Popen)
                           v
+------------------------------------------------------+
|           run_real_benchmark.py (Công nhân)          |
+------------------------------------------------------+
| - Nạp model vào GPU                                 |
| - Kích hoạt lõi nén (TurboQuant/Polar/HQQ/FP8)      |
| - Đo đạc Peak VRAM & sinh văn bản thô               |
| - Bắt lỗi CUDA OOM (Ghi nhãn OOM)                   |
| - Giải phóng VRAM khi kết thúc tiến trình           |
+------------------------------------------------------+
                           |
             (Kết thúc -> Trả quyền cho Quản đốc)
                           v
          (Tiến hành cấu hình tiếp theo...)
```

> [!IMPORTANT]
> **Tại sao phải dùng Subprocess?**
> PyTorch và vLLM không giải phóng hoàn toàn bộ nhớ VRAM của GPU khi ta delete đối tượng mô hình trong cùng một session Python (do cơ chế lưu cache của CUDA). Bằng việc gọi mỗi cấu hình chạy dưới dạng một tiến trình con (subprocess) độc lập, hệ điều hành sẽ tự động thu hồi và dọn dẹp sạch sẽ 100% VRAM giải phóng từ GPU khi tiến trình con kết thúc, giúp ngăn ngừa hoàn toàn hiện tượng rò rỉ bộ nhớ (memory leaks) khi lặp qua 60 cấu hình liên tiếp.

---

## 2. Danh mục kịch bản & Metadata chi tiết

| Tên File | Người tạo | Vai trò / Mục đích chi tiết |
| :--- | :--- | :--- |
| **[run_real_grid.py](run_real_grid.py)** | Quan-min211 | Trình quản đốc Grid Search cho các mô hình chính (Qwen3, Qwen2.5, Mistral, Llama 3.1). Lặp qua 60 cấu hình kết hợp giữa Mô hình x Phương pháp nén x Độ dài ngữ cảnh. |
| **[run_real_benchmark.py](run_real_benchmark.py)** | Quan-min211 | Công nhân thực thi đo đạc cho đúng 1 cấu hình đơn lẻ, tích hợp lõi nén vLLM, background thread `VRAMMonitor` và lưu log thô ra CSV/JSONL. |
| **[run_real_grid_extra.py](run_real_grid_extra.py)** | Quan-min211 | Trình quản đốc Grid Search dành riêng cho các mô hình phụ/thử nghiệm (Phi-4, Gemma-3). |
| **[run_real_benchmark_extra.py](run_real_benchmark_extra.py)** | Quan-min211 | Công nhân thực thi đo đạc cho 1 cấu hình đơn lẻ của các mô hình phụ. |
| **[run_mistral_optimized.py](run_mistral_optimized.py)** | Kien Hung \<kienhung.do1105@gmail.com\> & QUOC ANH \<quocanh0815@gmail.com\> | Trình điều khiển tối ưu hóa riêng cho Mistral 7B. Áp dụng chiến lược **"Load Once Per Method"** để nạp mô hình vào GPU duy nhất 1 lần cho mỗi phương pháp nén, chạy lần lượt 3 context lengths trong cùng session. |
| **[run_mistral_single_method.py](run_mistral_single_method.py)** | Kien Hung \<kienhung.do1105@gmail.com\> & QUOC ANH \<quocanh0815@gmail.com\> | Công nhân chạy tối ưu cho Mistral, nhận diện hệ số phình to token tiếng Việt mới (`MISTRAL_RATIO=1.9`) để tránh lỗi tràn context 16k. |
| **[generate_real_analysis.py](generate_real_analysis.py)** | Quan-min211 | Tự động quét dữ liệu log thật và sinh ra file Jupyter Notebook `real_benchmark_analysis.ipynb` để nhóm phân tích. |

---

## 3. Hướng dẫn chạy các kịch bản chính trên Server

### A. Chạy Grid Search mô hình tiêu chuẩn
```bash
# Đảm bảo môi trường conda đã được kích hoạt
conda activate viet-llm

# Khởi chạy tự động quét 60 cấu hình
python scripts/test/run_real_grid.py --hf_token "your_huggingface_token"
```

### B. Chạy Grid Search mô hình bổ sung (Phi-4, Gemma-3)
```bash
python scripts/test/run_real_grid_extra.py --hf_token "your_huggingface_token"
```

### C. Chạy tối ưu hóa riêng cho Mistral 7B
Để chạy 15 cấu hình của Mistral một cách nhanh nhất (tiết kiệm 70% thời gian nạp trọng số nặng của Mistral):
```bash
python scripts/test/run_mistral_optimized.py --hf_token "your_huggingface_token"
```
Số liệu sinh ra sẽ được ghi nhận vào file kết quả riêng biệt `results/template_log_real_run_mistral_final.csv`.

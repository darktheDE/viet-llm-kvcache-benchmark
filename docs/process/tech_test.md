# Báo cáo Kỹ thuật: Hệ thống Benchmark Thực tế trên GPU Cloud

**Ngày tạo:** 2026-06-28
**Vị trí file:** `scripts/test/`
**Kết quả CSV:** `results/real_benchmark_log.csv`
**Notebook phân tích:** `results/real_benchmark_analysis.ipynb`

---

## 1. Tổng quan: Những gì đã được tạo

Bộ công cụ benchmark thực tế (Real Mode) gồm **3 file Python** và **1 file Notebook** nằm trong thư mục `scripts/test/`, được thiết kế để chạy trên máy chủ GPU Cloud (RunPod hoặc Vast.ai) với card đồ họa tối thiểu 24GB VRAM.

| File | Vai trò | Mô tả |
|:---|:---|:---|
| `scripts/test/run_real_benchmark.py` | Công nhân thực thi | Chạy đo đạc cho **1 cấu hình** (1 Model + 1 Method + 1 Context). Gọi trực tiếp vLLM để tải model thật, chạy inference thật, đo VRAM thật. |
| `scripts/test/run_real_grid.py` | Quản đốc phân việc | Chạy vòng lặp Grid Search tự động gọi `run_real_benchmark.py` cho **75 cấu hình** (5×5×3). Có timeout 10 phút/cấu hình. |
| `scripts/test/generate_real_analysis.py` | Sinh Notebook | Tạo file `results/real_benchmark_analysis.ipynb` chứa code vẽ 6 biểu đồ phân tích. |
| `results/real_benchmark_analysis.ipynb` | Notebook phân tích | Đọc `real_benchmark_log.csv` và vẽ biểu đồ so sánh Peak VRAM, Latency, Throughput, PPL. |

---

## 2. Sự khác biệt so với hệ thống Mock trước đó

| Tiêu chí | Mock Mode (`scripts/run_mock_grid.py`) | Real Mode (`scripts/test/run_real_grid.py`) |
|:---|:---|:---|
| **Tải model thật?** | Không. Dùng `time.sleep()` giả vờ. | Có. Tải trọng số từ HuggingFace vào GPU. |
| **Đo VRAM thật?** | Không. Dùng công thức `14000 + ctx * 0.x`. | Có. Dùng `pynvml` (luồng nền 50ms) + `torch.cuda.max_memory_allocated()`. |
| **Đo Latency thật?** | Không. Dùng công thức `30.5 + ctx/1000`. | Có. Dùng `time.time()` bấm giờ thực tế khi inference. |
| **Đo PPL thật?** | Không. Dùng `random.uniform(5, 8)`. | Có. Dùng `prompt_logprobs` của vLLM để tính PPL theo công thức $\exp(-\frac{1}{N}\sum\log P)$. |
| **Yêu cầu phần cứng** | Laptop văn phòng bình thường. | GPU >= 24GB VRAM (RTX 3090/4090/L4). |
| **Thời gian chạy** | ~3 phút cho 75 cấu hình. | ~5-10 tiếng cho 75 cấu hình (tùy model). |
| **File CSV output** | `results/template_log.csv` (status: MOCK_OK) | `results/real_benchmark_log.csv` (status: OK/OOM) |

---

## 3. Chi tiết kỹ thuật từng file

### A. `run_real_benchmark.py` — Công nhân thực thi

**Pipeline 5 bước:**

1. **Nạp Dataset:** Đọc `datasets/test_set_small.json`, lấy `num_samples` mẫu văn bản tiếng Việt dài.
2. **Khởi tạo VRAM Monitor:** Tạo một `threading.Thread` chạy nền, cứ mỗi 50ms lại gọi `pynvml.nvmlDeviceGetMemoryInfo()` để ghi nhận mức VRAM cao nhất.
3. **Tải Model qua vLLM:** Gọi `vllm.LLM()` với các tham số tối ưu:
   - `gpu_memory_utilization=0.98`: Tận dụng tối đa 98% VRAM.
   - `max_num_seqs=2`: Giới hạn batch nhỏ để tránh OOM ở context dài.
   - `kv_cache_dtype`: Ánh xạ từ tên phương pháp nén sang tham số vLLM (FP16→`auto`, FP8→`fp8`, TurboQuant→`turboquant_4bit_nc`).
4. **Chạy Inference & Đo đạc:**
   - Bật VRAM Monitor → Chạy `llm.generate()` → Dừng VRAM Monitor.
   - Tính Latency = `(total_time / total_tokens) * 1000`.
   - Tính Throughput = `total_tokens / total_time`.
   - Cross-check Peak VRAM = `max(pynvml_peak, torch_peak)`.
5. **Tính PPL:** Dùng `prompt_logprobs` của vLLM để lấy log-probability của từng token, rồi áp công thức Perplexity chuẩn.

**Xử lý lỗi:**
- `torch.cuda.OutOfMemoryError` → Ghi `OOM` vào CSV, tiếp tục chạy cấu hình khác.
- Lỗi khác → Ghi `ERROR: <chi_tiet>` vào CSV.
- Sau mỗi cấu hình → Gọi `del llm; torch.cuda.empty_cache()` giải phóng VRAM.

### B. `run_real_grid.py` — Quản đốc

- Quét qua ma trận: **5 Models × 5 Methods × 3 Contexts = 75 cấu hình**.
- Gọi `run_real_benchmark.py` qua `subprocess.run()` với `timeout=600s` (10 phút).
- In báo cáo tổng kết cuối cùng: số cấu hình thành công / OOM / lỗi khác.

### C. `generate_real_analysis.py` — Sinh Notebook

Tạo file `results/real_benchmark_analysis.ipynb` gồm **7 phần**:
1. Công thức đo lường (VRAM, Latency, Throughput, PPL).
2. Bar chart: Peak VRAM @ 16K.
3. Line chart: VRAM Scaling theo Context Length.
4. Scatter plot: Pareto Frontier (Memory vs Latency).
5. Bar chart: Throughput @ 8K.
6. Box plot: Phân phối PPL theo phương pháp nén.
7. Bảng tổng hợp: % Tiết kiệm VRAM so với FP16.

---

## 4. Hướng dẫn Chạy trên GPU Cloud

### Bước 1: Thuê máy chủ GPU
Thuê instance trên RunPod hoặc Vast.ai với cấu hình:
- GPU: RTX 4090 / RTX 3090 / L4 (>= 24GB VRAM)
- OS: Ubuntu + CUDA 12.x
- RAM: >= 32GB

### Bước 2: Cài đặt môi trường
```bash
conda create -n dbml_benchmark python=3.10 -y
conda activate dbml_benchmark
pip install vllm pynvml transformers pandas matplotlib seaborn
```

### Bước 3: Clone repo và chạy Grid Search
```bash
git clone <repo_url>
cd viet-llm-kvcache-benchmark
python scripts/test/run_real_grid.py
```

### Bước 4: Sinh Notebook và phân tích
```bash
python scripts/test/generate_real_analysis.py
# Mở results/real_benchmark_analysis.ipynb trong Jupyter/VSCode
```

---

## 5. Cấu trúc file CSV đầu ra

File `results/real_benchmark_log.csv` có cùng cấu trúc header với `results/template_log.csv` để tương thích ngược:

| Cột | Kiểu | Mô tả |
|:---|:---|:---|
| `model` | string | Tên model trên HuggingFace |
| `kv_cache_type` | string | FP16 / FP8 / HQQ / PolarQuant / TurboQuant |
| `context_length` | int | 4000 / 8000 / 16000 |
| `peak_memory_mb` | float | Peak VRAM thực tế (MB) hoặc "OOM" |
| `latency_ms_per_token` | float | Độ trễ thực tế (ms/token) hoặc "OOM" |
| `throughput_tokens_per_s` | float | Thông lượng thực tế (tokens/s) hoặc "OOM" |
| `perplexity` | float | PPL thực tế hoặc "N/A" |
| `status` | string | OK / OOM / ERROR |

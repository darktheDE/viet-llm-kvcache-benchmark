# Hướng dẫn Cấu hình & Chạy Model trên Cloud (Rented Machines)

## 1. Vấn đề giới hạn Context Window (Tránh Truncation)
Khi nạp văn bản lớn hơn 4K token vào Ollama hoặc các nền tảng chạy model GGUF (như llama.cpp), hệ thống thường **tự động cắt (truncate)** văn bản về giới hạn mặc định của hệ thống (thường là 2048 hoặc 4096 token) để bảo vệ RAM của các máy cá nhân.
Điều này khiến các phép thử nghiệm mốc ngữ cảnh lớn (8K, 16K, 32K) trở nên vô nghĩa nếu không cấu hình lại, vì mô hình không thực sự "đọc" được lượng token mà bạn muốn đo đạc.

### Khắc phục trên Ollama (Dùng Modelfile)
Với các model alias của Ollama (`gemma4:e4b`, `qwen3:8b`, `llama3.2:3b`), bạn phải định nghĩa lại giới hạn bằng **Modelfile**:
```dockerfile
# Ví dụ Modelfile.gemma4
FROM gemma4:e4b

# Buộc hệ thống cấp phát đủ context memory
PARAMETER num_ctx 32768
SYSTEM ""
```
Các file cấu hình Modelfile mẫu đã được tạo sẵn trong thư mục `docs/Models/modelfiles`. Bạn có thể sử dụng lệnh `ollama create <ten_moi> -f <duong_dan_modelfile>` để build và chạy mô hình.

### Khắc phục trên vLLM (Real GPU Benchmark)
Khi chạy trực tiếp với Engine vLLM (trong file `scripts/test/run_real_benchmark.py`), bạn thiết lập giới hạn thông qua code thay vì Modelfile. Lưu ý quan trọng:
1. `max_model_len`: Mở rộng khả năng xử lý ngữ cảnh của mô hình.
2. `max_num_batched_tokens`: Kích thước lô xử lý. Giá trị này bắt buộc phải **lớn hơn hoặc bằng `max_model_len`**, nếu không hệ thống sẽ ném lỗi khi nạp prompt dài hơn `max_num_batched_tokens`. Script hiện tại đã được cấu hình tự động: `max_num_batched_tokens=max(args.context_length, 4096)`.

---

## 2. Thiết lập chạy trên Máy Thuê (Vast.ai, RunPod, Cloud GPUs)

### 2.1 Truyền Access Token (Hugging Face / Gated Models)
Một số model như của Mistral, LLaMa hay các repository riêng tư yêu cầu quyền truy cập (Access Token). Hệ thống chạy benchmark (file `run_real_grid.py`) đã tích hợp sẵn cơ chế nạp Access Token tự động.

**Cách an toàn & chuẩn xác nhất (Thiết lập biến môi trường)**
Trước khi chạy script, bạn export token của tài khoản HF (lấy tại https://huggingface.co/settings/tokens) vào môi trường Ubuntu/Linux của máy thuê:
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
# Chạy script grid search tự động
python scripts/test/run_real_grid.py
```
*(Hệ thống sẽ tự động quét biến này, thực hiện hàm `hf_login()` và cấp phát quyền tải model về cho thư viện vLLM)*

**Truyền trực tiếp qua tham số dòng lệnh**
Nếu bạn chạy các script con lẻ tẻ, bạn cũng có thể gán nó vào flag `--hf_token`:
```bash
python scripts/test/run_real_benchmark.py --model llama3.2:3b --context_length 16000 --hf_token "hf_xxxxxxxxxxx"
```

### 2.2 Tải trước (Pre-pull) Model cho Ollama
Trong môi trường Cloud thuần (bare-metal) mới khởi tạo, các model của hệ Ollama chưa hề có sẵn. Nếu chạy grid search, đôi khi quá trình pull model tự động qua vLLM có thể mất thời gian hoặc bị lỗi mạng. 

Script đã bổ sung chức năng `--pull_ollama` để tự động kéo model về trước khi thực thi đo đạc:
```bash
python scripts/test/run_real_benchmark.py --model gemma4:e4b --pull_ollama
```
Tính năng này sẽ gọi tiến trình phụ `ollama pull gemma4:e4b` ngầm dưới background để chuẩn bị môi trường sẵn sàng trước khi nạp model vào vLLM.

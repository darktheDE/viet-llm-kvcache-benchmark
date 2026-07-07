\[TECH\] Khởi tạo Môi trường Cloud GPU, Cấu hình vLLM Engine & Đo thử nghiệm mốc BF16 Baseline

#### **1\. Mô tả chi tiết Task (Description)**

Khởi động hạ tầng tính toán của dự án. Thiết lập môi trường chạy thử nghiệm trên máy chủ GPU đám mây (Vast.ai hoặc RunPod) sử dụng card đồ họa tối thiểu 24GB VRAM (như RTX 3090/4090 hoặc L4). Cấu hình bộ thư viện **vLLM** hỗ trợ nhân tính toán tối ưu TurboQuant và chạy thử thành công kịch bản đo đạc baseline ban đầu (Full KV Cache BF16).

#### **2\. Tài liệu đọc tham khảo (References)**

* *Hướng dẫn vLLM & TurboQuant:* [vLLM Documentation \- Supported KV Cache Dtypes](https://docs.vllm.ai/en/latest/models/engine_args.html) (Cập nhật 2026 hỗ trợ `--kv-cache-dtype turboquant_4bit_nc`).  
* *Plugin TurboQuant:* [turboquant-vllm GitHub Repository](https://github.com/).  
* *Thư viện đo bộ nhớ:* NVIDIA Management Library (`pynvml`) Python API.

#### **3\. Từng bước thực hiện chi tiết (Step-by-Step)**

* **Bước 1:** Khởi tạo một phiên máy ảo (instance) trên RunPod hoặc Vast.ai với cấu hình tối thiểu: 1 GPU RTX 4090/3090 (24GB VRAM), hệ điều hành Ubuntu, đã cài sẵn PyTorch và CUDA 12.x.

**Bước 2:** Cài đặt môi trường Conda và cài các gói phụ thuộc:  
conda create \-n dbml\_benchmark python=3.10 \-y  
conda activate dbml\_benchmark

* pip install vllm pynvml transformers pandas  
* **Bước 3:** Viết cấu trúc nền tảng cho file đo đạc `scripts/run_baseline.py`. Script phải tự động ghi nhận Peak VRAM của GPU thông qua thư viện `pynvml` hoặc `torch.cuda.max_memory_allocated()` trong suốt pha Prefill và Decode.  
* **Bước 4:** Chạy kiểm thử đo đạc thực tế mốc không nén (Full KV Cache BF16) với mô hình `gemma4:e4b` sử dụng bộ dữ liệu `datasets/test_set_small.json`. Ghi nhận thử nghiệm kết quả đầu ra vào file CSV cục bộ.

#### **4\. Kết quả đầu ra (Expected Output)**

* Môi trường máy chủ ảo GPU đã thiết lập xong, có thể truy cập qua SSH.  
* File mã nguồn ban đầu của script đo đạc `scripts/run_baseline.py` đẩy lên Git.  
* Một dòng kết quả đo đạc mẫu của mốc BF16 ghi nhận thành công trong file CSV cục bộ.

#### **5\. Tiêu chuẩn hoàn thành (Definition of Done \- DoD)**

* Thao tác cài đặt và import thư viện `vllm` hoạt động ổn định trên GPU đám mây mà không báo lỗi Driver hay CUDA.  
* Script đo đạc đo được chính xác dung lượng bộ nhớ đỉnh Peak VRAM (tính bằng MB/GB) và độ trễ sinh từ (ITL) của mô hình chạy thử nghiệm.  
* Đã cấu hình và thử nghiệm thành công mốc chạy baseline không lỗi bộ nhớ (OOM) ở độ dài ngữ cảnh tối thiểu 8k tokens.

\[TECH\] Hoàn thiện Script Đo đạc tự động (run\_baseline.py) tích hợp vLLM TurboQuant, PolarQuant, HQQ & FP8  
**1\. Mô tả chi tiết Task (Description)**

Xây dựng và hoàn thiện tệp mã nguồn Python `scripts/run_baseline.py` để tự động hóa toàn bộ quy trình đo đạc hiệu năng phần cứng và lưu vết chỉ số chất lượng ngôn ngữ. Script phải nhận tham số đầu vào qua dòng lệnh (`argparse`) để cấu hình linh hoạt cho từng kịch bản (Grid Search) gồm: Mô hình, Phương pháp nén, Độ dài ngữ cảnh.

#### **2\. Tài liệu đọc tham khảo (References)**

* *Tài liệu API vLLM:* [vLLM Offline Inference API](https://docs.vllm.ai/en/latest/quantization/auto_awq.html) \- Hướng dẫn sử dụng lớp `vllm.LLM` để truyền các đối số nén KV Cache.  
* *Mã nguồn tham khảo:* Các repo mã nguồn mở tích hợp TurboQuant trong vLLM (`--kv-cache-dtype turboquant_4bit_nc`).

#### **3\. Từng bước thực hiện chi tiết (Step-by-Step)**

* **Bước 1:** Khai báo thư viện `argparse` trong `run_baseline.py` để nhận các tham số:  
  * `--model`: Đường dẫn hoặc ID mô hình trên Hugging Face.  
  * `--kv_cache_type`: FP16, FP8, HQQ, PolarQuant, TurboQuant, TurboQuant-NoQJL.  
  * `--context_length`: 4000, 8000, 16000, 32000\.

**Bước 2:** Viết hàm cấu hình công cụ `vllm.LLM`. Đối với TurboQuant và PolarQuant, sử dụng cấu hình nén thông qua biến môi trường hoặc tùy chọn khởi dựng:  
from vllm import LLM, SamplingParams  
\# Cấu hình tham số động  
llm \= LLM(  
    model=args.model,  
    kv\_cache\_dtype="turboquant\_4bit\_nc" if args.kv\_cache\_type \== "TurboQuant" else "auto",  
    max\_model\_len=args.context\_length,  
    max\_num\_batched\_tokens=4096, \# Tránh lỗi phân mảnh bộ nhớ trên Qwen  
    trust\_remote\_code=True

* )  
* **Bước 3:** Tích hợp trình đo đạc phần cứng `pynvml`:  
  * Trước khi chạy sinh từ: Gọi `pynvml.nvmlDeviceGetMemoryInfo` để lưu mốc VRAM cơ bản.  
  * Trong quá trình Prefill & Decode: Thiết lập luồng đo song song (background thread) để lấy Peak VRAM cao nhất đạt được.  
* **Bước 4:** Tích hợp đo đạc thời gian:  
  * Sử dụng callback hoặc trigger của vLLM để ghi nhận chính xác thời điểm xuất hiện token đầu tiên (TTFT) và khoảng cách giữa các token tiếp theo (ITL).  
* **Bước 5:** Đảm bảo dữ liệu kết quả đo đạc được định dạng đúng và ghi đè/nối tiếp (append) vào tệp CSV cục bộ theo đúng cấu trúc `results/template_log.csv`.

#### **4\. Kết quả đầu ra (Expected Output)**

* Mã nguồn `scripts/run_baseline.py` hoàn chỉnh, hoạt động không lỗi, chấp nhận tất cả các tham số truyền vào từ dòng lệnh.

#### **5\. Tiêu chuẩn hoàn thành (Definition of Done \- DoD)**

* Chạy thử thành công lệnh đo đạc tự động với một mẫu thử ngắn từ dòng lệnh mà không gặp lỗi cú pháp.  
* File CSV ghi nhận chính xác các chỉ số đo được (`peak_memory_mb`, `latency_ms_per_token`, `throughput_tokens_per_s`).

\[TECH\] Tối ưu hóa VRAM, Xử lý lỗi Corner Cases/OOM & Chạy bù các mốc thử nghiệm còn thiếu  
**1\. Mô tả chi tiết Task (Description)**

Đảm nhận vai trò "Help Desk" hỗ trợ kỹ thuật cho toàn nhóm. Giải quyết triệt để các trường hợp thử nghiệm bị sập do lỗi tràn bộ nhớ GPU (CUDA Out of Memory) ở các mốc ngữ cảnh lớn (16k hoặc 32k). Cấu hình tối ưu hóa tham số bộ nhớ trong vLLM và phối hợp chạy bù các cấu hình bị thiếu sót hoặc bị lỗi trong Sprint trước.

#### **2\. Tài liệu đọc tham khảo (References)**

* *Tối ưu bộ nhớ vLLM:* [vLLM Memory Management & Optimization](https://docs.vllm.ai/en/latest/models/engine_args.html) \- Hướng dẫn tinh chỉnh các tham số `gpu_memory_utilization` và `max_model_len`.  
* *Tài liệu kỹ thuật:* Khắc phục lỗi phân mảnh bộ nhớ khi nén KV Cache dưới dạng 4-bit (vLLM Issue Tracker 2026).

#### **3\. Từng bước thực hiện chi tiết (Step-by-Step)**

* **Bước 1:** Thu thập thông tin từ các tệp log chạy thực nghiệm của Sprint 2, lọc ra các mốc cấu hình (Model, Method, Context) bị dính lỗi CUDA OOM.  
* **Bước 2:** Cấu hình tinh chỉnh tham số bộ nhớ trong script khởi tạo vLLM:  
  * Tăng giá trị `--gpu-memory-utilization` lên `0.95` hoặc `0.98` để tận dụng tối đa VRAM khả dụng.  
  * Cấu hình tham số `--max-num-seqs` nhỏ lại (ví dụ bằng 1 hoặc 2\) để giảm tải cho pha Decode khi benchmark ngữ cảnh dài.  
* **Bước 3:** Đối với các mốc mô hình kích thước lớn (như Qwen2.5-7B) ở ngữ cảnh 32k tokens bị OOM trên card 24GB, tiến hành cấu hình chạy bù sử dụng phương pháp **FlashAttention-2** hoặc giảm thiểu kích thước `block_size` trong vLLM từ 16 xuống 8\.  
* **Bước 4:** Ghi nhận lại các cấu hình bắt buộc phải hy sinh (không thể chạy được dù đã tối ưu) và đánh dấu rõ mốc đó là "OOM" trong tệp dữ liệu chung thay vì bỏ trống.

#### **4\. Kết quả đầu ra (Expected Output)**

* Bản vá mã nguồn (patch hoặc config cập nhật) giải quyết các lỗi OOM.  
* Dữ liệu chạy bù đầy đủ cho các mốc thử nghiệm còn thiếu được nạp vào thư mục `results/`.

#### **5\. Tiêu chuẩn hoàn thành (Definition of Done \- DoD)**

* 100% các mốc ngữ cảnh lớn được thử nghiệm tối đa, các mốc không thể chạy được đã được gắn nhãn "OOM" rõ ràng.  
* Không còn hiện tượng script bị treo hoặc crash ngầm giữa chừng không rõ nguyên nhân trong quá trình benchmark.


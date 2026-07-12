# SPRINT 02

### TASK 1: PHÁT TRIỂN MÃ NGUỒN (TECHNICAL & EXPERIMENT)

*   **Title:** `[TECH] Hoàn thiện Script Đo đạc tự động (run_baseline.py) tích hợp vLLM TurboQuant, PolarQuant, HQQ & FP8`
*   **Module:** Infrastructure & Inference Engine Setup
*   **Sprint:** Sprint 2 (Week 3-4)
*   **Team / Role:** Technical & Experiment (Pham Minh Quan - Lead) / Writing & Coordination (Nguyen Van Quang Duy - Technical Liaison)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Xây dựng và hoàn thiện tệp mã nguồn Python `scripts/run_baseline.py` để tự động hóa toàn bộ quy trình đo đạc hiệu năng phần cứng, thu thập các chỉ số nén nâng cao và lưu vết chỉ số chất lượng ngôn ngữ. Script phải nhận tham số đầu vào qua dòng lệnh (`argparse`) để cấu hình linh hoạt cho từng kịch bản (Grid Search) gồm: Mô hình, Phương pháp nén, Độ dài ngữ cảnh.

#### 2. Tài liệu đọc tham khảo (References)
*   *Tài liệu API vLLM:* [vLLM Offline Inference API](https://docs.vllm.ai/en/latest/quantization/auto_awq.html) - Hướng dẫn sử dụng lớp `vllm.LLM` để truyền các đối số nén KV Cache.
*   *Mã nguồn tham khảo:* Các repo mã nguồn mở tích hợp TurboQuant trong vLLM (`--kv-cache-dtype turboquant_4bit_nc`).

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Khai báo thư viện `argparse` trong `run_baseline.py` để nhận các tham số:
    *   `--model`: Đường dẫn hoặc ID mô hình trên Hugging Face.
    *   `--kv_cache_type`: FP16, FP8, HQQ, PolarQuant, TurboQuant, TurboQuant-NoQJL.
    *   `--context_length`: 4000, 8000, 16000, 32000.
*   **Bước 2:** Viết hàm cấu hình công cụ `vllm.LLM`. Đối với TurboQuant và PolarQuant, sử dụng cấu hình nén thông qua biến môi trường hoặc tùy chọn khởi dựng:
    ```python
    from vllm import LLM, SamplingParams
    # Cấu hình tham số động
    llm = LLM(
        model=args.model,
        kv_cache_dtype="turboquant_4bit_nc" if args.kv_cache_type == "TurboQuant" else "auto",
        max_model_len=args.context_length,
        max_num_batched_tokens=4096, # Tránh lỗi phân mảnh bộ nhớ trên Qwen
        trust_remote_code=True
    )
    ```
*   **Bước 3:** Tích hợp trình đo đạc phần cứng `pynvml`:
    *   Trước khi chạy sinh từ: Gọi `pynvml.nvmlDeviceGetMemoryInfo` để lưu mốc VRAM cơ bản (Model Base VRAM).
    *   Trong quá trình Prefill & Decode: Thiết lập luồng đo song song (background thread) để lấy Peak VRAM cao nhất đạt được (giúp phân biệt Base VRAM vs Dynamic VRAM).
*   **Bước 4:** Tích hợp đo đạc thời gian:
    *   Sử dụng callback hoặc trigger của vLLM để ghi nhận chính xác thời điểm xuất hiện token đầu tiên (TTFT) và khoảng cách giữa các token tiếp theo (ITL).
*   **Bước 5:** Tích hợp tính toán các chỉ số bổ sung:
    *   **KV Cache Compression Ratio:** Đo dung lượng KV cache thực tế và so sánh với dung lượng gốc ở dạng BF16.
    *   **GPU Memory Efficiency Index (Tokens/s/MB):** Tính bằng throughput decode chia cho dung lượng bộ nhớ động VRAM đã sử dụng.
*   **Bước 6:** Đảm bảo dữ liệu kết quả đo đạc được định dạng đúng và ghi đè/nối tiếp (append) vào tệp CSV cục bộ theo đúng cấu trúc của `results/template_log.csv` (có bổ sung thêm cột cho các chỉ số mới).

#### 4. Kết quả đầu ra (Expected Output)
*   Mã nguồn `scripts/run_baseline.py` hoàn chỉnh, hoạt động không lỗi, chấp nhận tất cả các tham số truyền vào từ dòng lệnh và thu thập đầy đủ chỉ số mới.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [x] Chạy thử thành công lệnh đo đạc tự động với một mẫu thử ngắn từ dòng lệnh mà không gặp lỗi cú pháp.
*   [ ] File CSV ghi nhận chính xác các chỉ số đo được, bao gồm cả `compression_ratio`, `gpu_efficiency_index`, `base_vram_mb` và `dynamic_vram_mb`.

---

### TASK 2: THỰC THI CHẠY THỬ NGHIỆM (TECHNICAL & EXPERIMENT)

*   **Title:** `[EXP] Thực thi Benchmark đa cấu hình trên bộ 4 mô hình benchmark chính thức`
*   **Module:** Phase 2: Thực nghiệm – chạy benchmark trên nhiều model
*   **Sprint:** Sprint 2 (Week 3-4)
*   **Team / Role:** Technical & Experiment (Tran Minh Khanh) / Writing & Coordination (Nguyen Van Quang Duy - Technical Liaison)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Nhận bàn giao script từ Task 1 và bộ dữ liệu từ team Data. Thực hiện chạy thử nghiệm lưới (Grid Search) lặp lại trên máy chủ Cloud GPU. Đo đạc các mốc so sánh gồm: Full KV Cache (Baseline), FP8, HQQ, PolarQuant, và TurboQuant trên bộ 4 mô hình mục tiêu chính thức là `qwen3:8b-fp16`, `llama3.1:8b-instruct-fp16`, `mistral:7b-instruct-v0.3-fp16`, và `qwen2.5:7b-instruct-fp16` ở các mốc ngữ cảnh 4k, 8k, 16k.

#### 2. Tài liệu đọc tham khảo (References)
*   *Mô hình sử dụng:* Hugging Face Repos của [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), và [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Chuẩn bị sẵn không gian đĩa cứng trên máy chủ để tải tự động các trọng số của bốn mô hình mục tiêu (mỗi mô hình chiếm khoảng 14GB - 16GB).
*   **Bước 2:** Viết một script shell (`scripts/run_grid_experiments.sh`) để tự động hóa việc lặp qua tất cả cấu hình nhằm tránh thao tác thủ công:
    ```bash
    # Ví dụ vòng lặp chạy thực nghiệm
    for model in "qwen3:8b-fp16" "llama3.1:8b-instruct-fp16" "mistral:7b-instruct-v0.3-fp16" "qwen2.5:7b-instruct-fp16"; do
      for kv_type in "BF16" "FP8" "HQQ" "PolarQuant" "TurboQuant"; do
        for ctx in 4000 8000 16000; do
          python scripts/run_baseline.py --model $model --kv_cache_type $kv_type --context_length $ctx
        done
      done
    done
    ```
*   **Bước 3:** Giám sát liên tục các phiên chạy (sử dụng công cụ `screen` hoặc `tmux` trên Linux) để kịp thời phát hiện lỗi Out-of-Memory (OOM). Nếu xảy ra lỗi OOM, ghi chú lại mốc ngữ cảnh bị sập vào file CSV kết quả.
*   **Bước 4:** Thu thập toàn bộ file log kết quả sinh ra của các đợt chạy và chuyển giao cho team Analysis.

#### 4. Kết quả đầu ra (Expected Output)
*   Các tệp tin log kết quả dạng CSV thô cho từng mô hình được lưu trữ tại thư mục `results/`.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Hoàn thành 100% các lượt chạy thành công cho cả 4 dòng mô hình mục tiêu ở các mốc ngữ cảnh trước khi xảy ra OOM (nếu có).
*   [ ] Số liệu trong file CSV không bị trống (null) ở các trường thông số phần cứng.

---

### TASK 3: QUẢN LÝ DỮ LIỆU & ĐÁNH GIÁ CHẤT LƯỢNG (DATA & ANALYSIS)

*   **Title:** `[DATA] Quản lý cấu trúc Log CSV, kiểm định chất lượng tạo văn bản & tính toán Perplexity (PPL)`
*   **Module:** Dataset Engineering & Preprocessing
*   **Sprint:** Sprint 2 (Week 3-4)
*   **Team / Role:** Data & Analysis (Nguyen Ho Phat - Lead, Huynh Huu Huy, Huynh Ngoc Thach)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Kiểm soát chất lượng đầu ra của các thử nghiệm trên tập dữ liệu benchmark và dữ liệu thời sự cào mới. Tích hợp phương thức tính toán độ suy giảm ngôn ngữ **Perplexity (PPL)** thông qua việc tính toán tổn thất entropy chéo (Cross-Entropy Loss) của văn bản sinh ra đối chiếu ngược lại với mô hình gốc BF16. Rà soát các tệp CSV kết quả để đảm bảo số liệu được định dạng nhất quán.

#### 2. Tài liệu đọc tham khảo (References)
*   *Lý thuyết Perplexity:* [Hugging Face Documentation - Perplexity of Language Models](https://huggingface.co/docs/transformers/perplexity) - Cách tính toán PPL chi tiết trên một chuỗi văn bản [1.2.2].

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Viết một hàm Python bổ trợ trong `run_baseline.py` để lấy được giá trị loss (logits) của mô hình khi sinh text, sau đó tính toán PPL:
    $$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i})\right)$$
*   **Bước 2:** Viết bộ lọc rà soát văn bản sinh ra (output generation inspection) để phát hiện các lỗi lặp từ vô hạn (repetition loops) hoặc sinh ký tự rác (gibberish tokens) khi nén KV cache sâu, đặc biệt trên tập dữ liệu thời sự nóng và mạng xã hội cào bổ sung.
*   **Bước 3:** Đo đạc các chỉ số Exact Match (EM) và F1 của mô hình trên tập dữ liệu QA/Retrieval thời sự bổ sung.
*   **Bước 4:** Phối hợp chặt chẽ với team Tech để kiểm tra tính hợp lệ của file `results/template_log.csv`, đảm bảo có đầy đủ các cột metrics mới bổ sung (compression ratio, gpu efficiency index, base/dynamic VRAM) mà không bị lệch cột.

#### 4. Kết quả đầu ra (Expected Output)
*   Hàm tính toán PPL và hệ thống kiểm định EM/F1 hoạt động chính xác tích hợp sẵn vào script đo đạc chung.
*   Báo cáo đánh giá sơ bộ về hiện tượng lỗi sinh từ (sinh rác, lặp từ) của các mô hình khi nén ở mốc 16k tokens trên tập dữ liệu thời sự bổ sung.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Chỉ số Perplexity, Exact Match và F1 được ghi nhận đầy đủ cho các mốc chạy thử nghiệm mà không gây lỗi crash runtime.
*   [ ] File CSV dữ liệu kết quả đo đạc sạch sẽ, không bị lệch cột, ghi nhận đầy đủ các cột chỉ số phần cứng mới.

---

### TASK 4: KHUNG TÀI LIỆU KHOA HỌC & METHODOLOGY (RESEARCH & SCOPE)

*   **Title:** `[RESEARCH] Thiết lập khung tài liệu LaTeX (Overleaf) & Soạn thảo chương Phương pháp luận (Methodology)`
*   **Module:** Phase 4: Viết bài + chỉnh sửa
*   **Sprint:** Sprint 2 (Week 3-4)
*   **Team / Role:** Research & Scope (Nguyen Dang Quoc Anh - Lead) / Writing & Coordination (Phan Trong Phu - Paper Draft Support)
*   **Priority:** Medium

#### 1. Mô tả chi tiết Task (Description)
Khởi tạo và cấu hình dự án Overleaf dùng chung cho nhóm sử dụng mẫu chuẩn của IEEE hoặc Springer. Tiến hành soạn thảo chương cốt lõi của bài báo khoa học bằng tiếng Anh: **Section III - Methodology & Experimental Setup** nhằm mô tả chi tiết kiến trúc thử nghiệm, các mốc nén KV cache và các tham số đo đạc phần cứng/chất lượng để chuẩn bị cho giai đoạn viết kết quả ở Phase 3.

#### 2. Tài liệu đọc tham khảo (References)
*   *LaTeX Template:* [IEEE Manuscript Templates for Conference Proceedings](https://www.ieee.org/conferences/publishing/templates.html) - Biểu mẫu chuẩn để viết bài báo nghiên cứu khoa học.
*   *Mẫu cấu trúc bài nghiên cứu:* Các bài báo hệ thống về KV Cache trên thư viện arXiv để tham khảo cách hành văn khoa học.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Tạo một Project trên Overleaf bằng tài khoản học viên, nhập mẫu chuẩn LaTeX của IEEE Conference. Add cộng tác viên cho các thành viên thuộc team Writing.
*   **Bước 2:** Chia nhỏ cấu trúc thư mục Overleaf:
    *   `/sections` (chứa các file `.tex` riêng biệt: `introduction.tex`, `methodology.tex`, `results.tex`, `discussion.tex`).
    *   `/figures` (chứa sơ đồ hệ thống dạng vector hoặc PNG sắc nét).
*   **Bước 3:** Viết chi tiết phần **Methodology**:
    *   Mô tả toán học về cách thức TurboQuant thực hiện lượng tử hóa và nén KV cache.
    *   Mô tả chi tiết môi trường phần cứng đo đạc (GPU mã hiệu gì, bao nhiêu VRAM, OS nào, CUDA version nào).
    *   Định nghĩa rõ công thức tính toán các chỉ số Performance (TTFT, ITL, Throughput) và Quality (Perplexity).
*   **Bước 4:** Thiết lập danh mục tài liệu tham khảo bằng file `references.bib`, nạp sẵn các trích dẫn chuẩn của các bài nghiên cứu TurboQuant, PolarQuant, vLLM.

#### 4. Kết quả đầu ra (Expected Output)
*   Link dự án Overleaf hoạt động ổn định.
*   Bản thảo chương **Methodology** bằng tiếng Anh hoàn chỉnh, được biên dịch thử không lỗi hiển thị trên Overleaf.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Phần chương Methodology viết bằng tiếng Anh học thuật hoàn chỉnh, không có lỗi ngữ pháp lớn, dài tối thiểu 1.5 trang LaTeX hai cột.
*   [ ] Toàn bộ các tài liệu tham khảo chính yếu về TurboQuant, PolarQuant đã được liên kết chính xác trong tệp `.bib` và trích dẫn thành công trong văn bản.

# SPRINT 02

### TASK 1: PHÁT TRIỂN MÃ NGUỒN (TECHNICAL & EXPERIMENT)

*   **Title:** `[TECH] Hoàn thiện Script Đo đạc tự động (run_baseline.py) tích hợp vLLM TurboQuant, PolarQuant, HQQ & FP8`
*   **Module:** Infrastructure & Inference Engine Setup [2]
*   **Sprint:** Sprint 2 (Week 3-4) [2]
*   **Team / Role:** Technical & Experiment (Pham Minh Quan - Lead, Nguyen Van Quang Duy) [1, 2]
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Xây dựng và hoàn thiện tệp mã nguồn Python `scripts/run_baseline.py` để tự động hóa toàn bộ quy trình đo đạc hiệu năng phần cứng và lưu vết chỉ số chất lượng ngôn ngữ [1, 2]. Script phải nhận tham số đầu vào qua dòng lệnh (`argparse`) để cấu hình linh hoạt cho từng kịch bản (Grid Search) gồm: Mô hình, Phương pháp nén, Độ dài ngữ cảnh [1, 2].

#### 2. Tài liệu đọc tham khảo (References)
*   *Tài liệu API vLLM:* [vLLM Offline Inference API](https://docs.vllm.ai/en/latest/quantization/auto_awq.html) - Hướng dẫn sử dụng lớp `vllm.LLM` để truyền các đối số nén KV Cache [2].
*   *Mã nguồn tham khảo:* Các repo mã nguồn mở tích hợp TurboQuant trong vLLM (`--kv-cache-dtype turboquant_4bit_nc`) [2].

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Khai báo thư viện `argparse` trong `run_baseline.py` để nhận các tham số [1, 2]:
    *   `--model`: Đường dẫn hoặc ID mô hình trên Hugging Face [2].
    *   `--kv_cache_type`: FP16, FP8, HQQ, PolarQuant, TurboQuant, TurboQuant-NoQJL [1, 2].
    *   `--context_length`: 4000, 8000, 16000, 32000 [1, 2].
*   **Bước 2:** Viết hàm cấu hình công cụ `vllm.LLM` [2]. Đối với TurboQuant và PolarQuant, sử dụng cấu hình nén thông qua biến môi trường hoặc tùy chọn khởi dựng [2]:
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
*   **Bước 3:** Tích hợp trình đo đạc phần cứng `pynvml` [1, 2]:
    *   Trước khi chạy sinh từ: Gọi `pynvml.nvmlDeviceGetMemoryInfo` để lưu mốc VRAM cơ bản [1, 2].
    *   Trong quá trình Prefill & Decode: Thiết lập luồng đo song song (background thread) để lấy Peak VRAM cao nhất đạt được [1, 2].
*   **Bước 4:** Tích hợp đo đạc thời gian:
    *   Sử dụng callback hoặc trigger của vLLM để ghi nhận chính xác thời điểm xuất hiện token đầu tiên (TTFT) và khoảng cách giữa các token tiếp theo (ITL).
*   **Bước 5:** Đảm bảo dữ liệu kết quả đo đạc được định dạng đúng và ghi đè/nối tiếp (append) vào tệp CSV cục bộ theo đúng cấu trúc `results/template_log.csv` [1, 2].

#### 4. Kết quả đầu ra (Expected Output)
*   Mã nguồn `scripts/run_baseline.py` hoàn chỉnh, hoạt động không lỗi, chấp nhận tất cả các tham số truyền vào từ dòng lệnh [1, 2].

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Chạy thử thành công lệnh đo đạc tự động với một mẫu thử ngắn từ dòng lệnh mà không gặp lỗi cú pháp.
*   [ ] File CSV ghi nhận chính xác các chỉ số đo được (`peak_memory_mb`, `latency_ms_per_token`, `throughput_tokens_per_s`) [1, 2].

---

### TASK 2: THỰC THI CHẠY THỬ NGHIỆM (TECHNICAL & EXPERIMENT)

*   **Title:** `[EXP] Thực thi Benchmark đa cấu hình trên mô hình PhoGPT-7B5 và Qwen2.5-7B`
*   **Module:** Phase 2: Thực nghiệm – chạy benchmark trên nhiều model [2]
*   **Sprint:** Sprint 2 (Week 3-4) [2]
*   **Team / Role:** Technical & Experiment (Tran Minh Khanh, Nguyen Van Quang Duy - Support) [2]
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Nhận bàn giao script từ Task 1 và bộ dữ liệu từ team Data [1, 2]. Thực hiện chạy thử nghiệm lưới (Grid Search) lặp lại trên máy chủ Cloud GPU [2]. Đo đạc các mốc so sánh gồm: Full KV Cache (Baseline) [1, 2], FP8 [2], HQQ [1, 2], PolarQuant [1, 2], và TurboQuant [1, 2] trên hai dòng mô hình mục tiêu chính là PhoGPT-7B5 và Qwen2.5-7B [2] ở các mốc ngữ cảnh 4k, 8k, 16k [1, 2].

#### 2. Tài liệu đọc tham khảo (References)
*   *Mô hình sử dụng:* Hugging Face Repos của [VinAI/PhoGPT-7B5-Instruct](https://huggingface.co/vinai/PhoGPT-7B5-Instruct) và [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) [2].

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Chuẩn bị sẵn không gian đĩa cứng trên máy chủ để tải tự động các trọng số của hai mô hình mục tiêu (mỗi mô hình chiếm khoảng 14GB - 16GB).
*   **Bước 2:** Viết một script shell (`scripts/run_grid_experiments.sh`) để tự động hóa việc lặp qua tất cả cấu hình nhằm tránh thao tác thủ công [1]:
    ```bash
    # Ví dụ vòng lặp chạy thực nghiệm
    for model in "vinai/PhoGPT-7B5-Instruct" "Qwen/Qwen2.5-7B-Instruct"; do
      for kv_type in "BF16" "FP8" "HQQ" "PolarQuant" "TurboQuant"; do
        for ctx in 4000 8000 16000; do
          python scripts/run_baseline.py --model $model --kv_cache_type $kv_type --context_length $ctx
        done
      done
    done
    ```
*   **Bước 3:** Giám sát liên tục các phiên chạy (sử dụng công cụ `screen` hoặc `tmux` trên Linux) để kịp thời phát hiện lỗi Out-of-Memory (OOM) [2]. Nếu xảy ra lỗi OOM, ghi chú lại mốc ngữ cảnh bị sập vào file CSV kết quả [2].
*   **Bước 4:** Thu thập toàn bộ file log kết quả sinh ra của các đợt chạy và chuyển giao cho team Analysis [2].

#### 4. Kết quả đầu ra (Expected Output)
*   Các tệp tin log kết quả dạng CSV thô cho từng mô hình được lưu trữ tại thư mục `results/` [1, 2].

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Hoàn thành 100% các lượt chạy thành công cho cả 2 dòng mô hình mục tiêu ở các mốc ngữ cảnh trước khi xảy ra OOM (nếu có) [2].
*   [ ] Số liệu trong file CSV không bị trống (null) ở các trường thông số phần cứng [2].

---

### TASK 3: QUẢN LÝ DỮ LIỆU & ĐÁNH GIÁ CHẤT LƯỢNG (DATA & ANALYSIS)

*   **Title:** `[DATA] Quản lý cấu trúc Log CSV, kiểm định chất lượng tạo văn bản & tính toán Perplexity (PPL)`
*   **Module:** Dataset Engineering & Preprocessing [2]
*   **Sprint:** Sprint 2 (Week 3-4) [2]
*   **Team / Role:** Data & Analysis (Nguyen Ho Phat - Lead, Huynh Huu Huy, Huynh Ngoc Thach) [1, 2]
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Kiểm soát chất lượng đầu ra của các thử nghiệm [1, 2]. Tích hợp phương thức tính toán độ suy giảm ngôn ngữ **Perplexity (PPL)** thông qua việc tính toán tổn thất entropy chéo (Cross-Entropy Loss) của văn bản sinh ra đối chiếu ngược lại với mô hình gốc BF16 [1, 2]. Rà soát các tệp CSV kết quả để đảm bảo số liệu được định dạng nhất quán [1, 2].

#### 2. Tài liệu đọc tham khảo (References)
*   *Lý thuyết Perplexity:* [Hugging Face Documentation - Perplexity of Language Models](https://huggingface.co/docs/transformers/perplexity) - Cách tính toán PPL chi tiết trên một chuỗi văn bản [1.2.2].

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Viết một hàm Python bổ trợ trong `run_baseline.py` để lấy được giá trị loss (logits) của mô hình khi sinh text, sau đó tính toán PPL [1, 2]:
    $$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i})\right)$$
*   **Bước 2:** Viết bộ lọc rà soát văn bản sinh ra (output generation inspection) để phát hiện các lỗi lặp từ vô hạn (repetition loops) hoặc sinh ký tự rác (gibberish tokens) - hiện tượng thường gặp khi nén KV cache quá sâu mà không có cơ chế bù lỗi tốt [2].
*   **Bước 3:** Phối hợp chặt chẽ với team Tech để kiểm tra tính hợp lệ của file `results/template_log.csv` [1, 2], đảm bảo không có sự sai lệch về thứ tự các trường hoặc định dạng dữ liệu giữa các mốc chạy khác nhau [2].

#### 4. Kết quả đầu ra (Expected Output)
*   Hàm tính toán PPL hoạt động chính xác tích hợp sẵn vào script đo đạc chung [1, 2].
*   Báo cáo đánh giá sơ bộ về hiện tượng lỗi sinh từ (sinh rác, lặp từ) của các mô hình khi nén ở mốc 16k tokens [2].

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Chỉ số Perplexity được ghi nhận đầy đủ cho các mốc chạy thử nghiệm mà không gây lỗi crash runtime [1, 2].
*   [ ] File CSV dữ liệu kết quả đo đạc sạch sẽ, không bị lệch cột hoặc sai định dạng kiểu dữ liệu [1, 2].

---

### TASK 4: KHUNG TÀI LIỆU KHOA HỌC & METHODOLOGY (RESEARCH & SCOPE)

*   **Title:** `[RESEARCH] Thiết lập khung tài liệu LaTeX (Overleaf) & Soạn thảo chương Phương pháp luận (Methodology)`
*   **Module:** Phase 4: Viết bài + chỉnh sửa [2]
*   **Sprint:** Sprint 2 (Week 3-4) [2]
*   **Team / Role:** Research & Scope (Nguyen Dang Quoc Anh - Lead, Phan Trong Phu) [1, 2]
*   **Priority:** Medium

#### 1. Mô tả chi tiết Task (Description)
Khởi tạo và cấu hình dự án Overleaf dùng chung cho nhóm sử dụng mẫu chuẩn của IEEE hoặc Springer [2, 3]. Tiến hành soạn thảo chương cốt lõi của bài báo khoa học bằng tiếng Anh: **Section III - Methodology & Experimental Setup** [2] nhằm mô tả chi tiết kiến trúc thử nghiệm, các mốc nén KV cache và các tham số đo đạc phần cứng/chất lượng để chuẩn bị cho giai đoạn viết kết quả ở Phase 3 [2].

#### 2. Tài liệu đọc tham khảo (References)
*   *LaTeX Template:* [IEEE Manuscript Templates for Conference Proceedings](https://www.ieee.org/conferences/publishing/templates.html) - Biểu mẫu chuẩn để viết bài báo nghiên cứu khoa học.
*   *Mẫu cấu trúc bài nghiên cứu:* Các bài báo hệ thống về KV Cache trên thư viện arXiv để tham khảo cách hành văn khoa học [2].

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Tạo một Project trên Overleaf bằng tài khoản học viên, nhập mẫu chuẩn LaTeX của IEEE Conference [3]. Add cộng tác viên cho các thành viên thuộc team Writing [2].
*   **Bước 2:** Chia nhỏ cấu trúc thư mục Overleaf [2]:
    *   `/sections` (chứa các file `.tex` riêng biệt: `introduction.tex`, `methodology.tex`, `results.tex`, `discussion.tex`) [2].
    *   `/figures` (chứa sơ đồ hệ thống dạng vector hoặc PNG sắc nét).
*   **Bước 3:** Viết chi tiết phần **Methodology** [2]:
    *   Mô tả toán học về cách thức TurboQuant thực hiện lượng tử hóa và nén KV cache [1, 2].
    *   Mô tả chi tiết môi trường phần cứng đo đạc (GPU mã hiệu gì, bao nhiêu VRAM, OS nào, CUDA version nào) [1, 2].
    *   Định nghĩa rõ công thức tính toán các chỉ số Performance (TTFT, ITL, Throughput) và Quality (Perplexity) [1, 2].
*   **Bước 4:** Thiết lập danh mục tài liệu tham khảo bằng file `references.bib`, nạp sẵn các trích dẫn chuẩn của các bài nghiên cứu TurboQuant, PolarQuant, vLLM [1, 2].

#### 4. Kết quả đầu ra (Expected Output)
*   Link dự án Overleaf hoạt động ổn định [3].
*   Bản thảo chương **Methodology** bằng tiếng Anh hoàn chỉnh, được biên dịch thử không lỗi hiển thị trên Overleaf [2, 3].

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Phần chương Methodology viết bằng tiếng Anh học thuật hoàn chỉnh, không có lỗi ngữ pháp lớn, dài tối thiểu 1.5 trang LaTeX hai cột [2, 3].
*   [ ] Toàn bộ các tài liệu tham khảo chính yếu về TurboQuant, PolarQuant đã được liên kết chính xác trong tệp `.bib` và trích dẫn thành công trong văn bản [1, 2].
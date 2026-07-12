# SPRINT 01

### TASK 1: QUẢN LÝ DỰ ÁN & ĐIỀU PHỐI (WRITING & COORDINATION)

*   **Title:** `[COORD] Thiết lập Không gian làm việc chung, Agile Backlog trên Plane.so & Soạn thảo Tài liệu Khởi động dự án`
*   **Module:** Project Management & Coordination
*   **Sprint:** Sprint 1 (Week 1-2)
*   **Team / Role:** Writing & Coordination (Do Kien Hung - Lead, Phan Trong Qui)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Thiết lập toàn bộ hạ tầng cộng tác số cho dự án nhóm (Drive, Git, Plane.so). Đồng thời, chuẩn bị sẵn sàng các tài liệu điều phối cốt lõi bao gồm cấu trúc thư mục lưu trữ, template biên bản họp để đảm bảo luồng thông tin trong nhóm xuyên suốt.

#### 2. Tài liệu đọc tham khảo (References)
*   **Tài liệu Agile/Scrum:** Hướng dẫn quản trị tiến độ dự án bằng Agile/Scrum (Atlassian Agile Coach).
*   **Quy ước Repository:** Hướng dẫn chuẩn hóa thư mục nghiên cứu khoa học của nhóm (nhóm sử dụng cấu trúc thư mục quy định tại file README.md gốc).

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Khởi tạo một Shared Google Drive hoặc OneDrive dùng chung cho cả nhóm. Thiết lập cấu trúc thư mục:
    *   `01_Proposals_and_Papers/`
    *   `02_Datasets/`
    *   `03_Presentation_Slides/`
    *   `04_Meeting_Minutes/`
*   **Bước 2:** Khởi tạo Repository trên GitHub và add cộng tác viên cho tất cả thành viên trong nhóm. Cấu hình phân quyền nhánh `main` cần được review trước khi merge.
*   **Bước 3:** Tạo và cấu hình các cột Kanban trên `plane.so` đại diện cho vòng đời Task: `Backlog` $\rightarrow$ `To Do` $\rightarrow$ `In Progress` $\rightarrow$ `Under Review` $\rightarrow$ `Done`. Gán nhãn (labels) cho từng team chuyên môn.
*   **Bước 4:** Tạo tệp tin `kickoff/minutes_template.md` và phân phối lịch họp Tuần 1 (Kickoff) và Tuần 2 (Check-in giữa tuần) qua Google Calendar cho toàn nhóm.

#### 4. Kết quả đầu ra (Expected Output)
*   Link Shared Google Drive chung của dự án.
*   Link Github Repository đã cấu hình xong phân quyền và sơ đồ thư mục mẫu.
*   Backlog Plane.so đã được nạp đầy đủ task của Sprint 1.
*   File biên bản họp kickoff `minutes_template.md` lưu trữ trên Git.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [x] 100% thành viên nhóm đã truy cập được Shared Drive và Github Repo.
*   [x] Toàn bộ task Sprint 1 đã được gán đúng người chịu trách nhiệm và deadline trên plane.so.
*   [x] Biên bản họp Kickoff Tuần 1 đã được ghi nhận đầy đủ chữ ký xác nhận vai trò của các thành viên.

---

### TASK 2: NGHIÊN CỨU LÝ THUYẾT & ĐỀ CƯƠNG (RESEARCH & SCOPE)

*   **Title:** `[RESEARCH] Hoàn thiện Đề cương chi tiết (Proposal v2) & Đọc hiểu lý thuyết TurboQuant vs. PolarQuant`
*   **Module:** Theoretical Framework & Literature Review
*   **Sprint:** Sprint 1 (Week 1-2)
*   **Team / Role:** Research & Scope (Nguyen Dang Quoc Anh - Lead) / Writing & Coordination (Phan Trong Phu - Literature Review Support)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Xây dựng nền tảng lý thuyết vững chắc cho bài báo tiếng Anh (Paper EN). Tập trung làm rõ kiến trúc hoạt động của **TurboQuant** (cơ chế quay ngẫu nhiên kết hợp lượng tử hóa tọa độ cực Lloyd-Max và cơ chế sửa lỗi sai số bằng một bit QJL) so với phương pháp gốc **PolarQuant**. Đọc hiểu mức độ nhạy cảm của tiếng Việt khi nén sâu KV Cache để chuẩn bị viết phần Introduction.

#### 2. Tài liệu đọc tham khảo (References)
*   *Bài báo gốc TurboQuant:* Zandieh, A., et al. (2025). [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) [1.1.2].
*   *Bài báo PolarQuant:* Han, I., et al. (2025). [PolarQuant: Quantizing KV Caches with Polar Transformation (arXiv:2502.02617)](https://arxiv.org/abs/2502.02617) [1.3.1].
*   *Bài báo đánh giá độ nhạy đa ngôn ngữ:* [KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs (arXiv:2602.05929)](https://arxiv.org/abs/2602.05929).

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Đọc hiểu chi tiết 3 bài báo khoa học trong danh mục tham chiếu nêu trên.
*   **Bước 2:** Viết bản thảo mô tả thuật toán của TurboQuant dưới dạng toán học hóa (Sơ đồ khối và giả mã thuật toán nén/giải nén) lưu trữ tại `proposal/problem_definition.md`.
*   **Bước 3:** Hoàn thiện Đề cương nghiên cứu chi tiết bằng tiếng Anh (Research Proposal v2) bao gồm các phần: *Introduction*, *Problem Formulation*, *Research Questions (RQ1 - RQ4)*, và *Expected Contributions*.
*   **Bước 4:** Tạo một tài liệu ghi nhận sự khác biệt lý thuyết giữa **TurboQuant (Full)** và **TurboQuant (tắt QJL)** để định hướng cho team Tech thử nghiệm so sánh.

#### 4. Kết quả đầu ra (Expected Output)
*   Tệp tin `proposal/proposal_1_2_pages_v1.md` hoàn chỉnh bằng tiếng Anh gửi cho thầy hướng dẫn duyệt.
*   Tệp tin `proposal/problem_definition.md` chứa định nghĩa toán học của bài toán nén KV Cache.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [x] Đề cương Proposal v2 đã được viết bằng tiếng Anh chuẩn học thuật và lưu trên Overleaf/Git.
*   [x] Được giảng viên hướng dẫn (TS. Lê Ngọc Hiếu) duyệt thông qua định hướng lý thuyết và các câu hỏi nghiên cứu (RQ1-RQ4).

---

### TASK 3: KỸ THUẬT DỮ LIỆU & TIỀN XỬ LÝ (DATA & ANALYSIS)

*   **Title:** `[DATA] Xây dựng Pipeline tiền xử lý bằng NVIDIA NeMo Curator & Đóng gói Test-set ngữ cảnh dài tiếng Việt`
*   **Module:** Dataset Engineering & Preprocessing
*   **Sprint:** Sprint 1 (Week 1-2)
*   **Team / Role:** Data & Analysis (Nguyen Ho Phat - Lead, Huynh Huu Huy)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Xây dựng pipeline tự động tải dữ liệu, làm sạch tiếng Việt bằng bộ công cụ **NVIDIA NeMo Curator** để lọc trùng lặp và loại bỏ văn bản lỗi. Đồng thời, thiết kế bộ mẫu dữ liệu thử nghiệm dài ngữ cảnh (mốc 4k, 8k, 16k tokens) đóng gói chuẩn định dạng JSON phục vụ cho quá trình đo đạc hiệu năng và chất lượng.

#### 2. Tài liệu đọc tham khảo (References)
*   *Công cụ làm sạch:* [NVIDIA NeMo Curator Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/curator/index.html) - Hướng dẫn sử dụng công cụ lọc dữ liệu quy mô lớn.
*   *Tập dữ liệu nguồn:* [Hugging Face VTSNLP Curated Dataset](https://huggingface.co/datasets/VTSNLP/vietnamese_curated_dataset).
*   *Đánh giá Benchmark:* [VMLU Dataset GitHub/HF Repo](https://huggingface.co/datasets/5760/vmlu) [1.1.2].

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Tải dữ liệu thô từ các tập dữ liệu `VMLU`, `V-Bench`, `VTSNLP/vietnamese_curated_dataset`.
*   **Bước 2:** Viết script sử dụng bộ thư viện **NVIDIA NeMo Curator** để lọc bỏ các câu lỗi Unicode tiếng Việt, lọc trùng lặp cấp ký tự và lọc các chuỗi văn bản quá ngắn hoặc chứa ký tự đặc biệt rác.
*   **Bước 3:** Lọc và gom cụm dữ liệu đã làm sạch thành 3 nhóm ngữ cảnh mục tiêu: 3-5 mẫu ngắn (~4k tokens), 3-5 mẫu trung bình (~8k tokens), và 3-5 mẫu dài (~16k tokens).
*   **Bước 4:** Thiết kế định dạng JSON thống nhất cho bộ dữ liệu thử nghiệm tại tệp `datasets/test_set_small.json` theo schema cấu trúc quy định.
*   **Bước 5:** Viết file mô tả dữ liệu ngắn `datasets/dataset_brief.md` để bàn giao cho team kỹ thuật.

#### 4. Kết quả đầu ra (Expected Output)
*   Tệp tin `datasets/test_set_small.json` chứa 10-20 mẫu thử nghiệm dài ngữ cảnh đã làm sạch và tokenize chuẩn.
*   Tệp tin tài liệu hướng dẫn `datasets/dataset_brief.md` lưu trữ trên Git.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [x] Bộ test-set nhỏ được đóng gói thành công ở định dạng JSON chuẩn không bị lỗi cú pháp.
*   [x] Detokenize thử nghiệm 100% các mẫu trong test-set đảm bảo hiển thị đúng phông chữ tiếng Việt có dấu.
*   [x] Team Technical xác nhận cấu trúc dữ liệu tương thích hoàn toàn với script benchmark đầu vào.

---

### TASK 4: HẠ TẦNG KỸ THUẬT & TRÌNH PHỤC VỤ (TECHNICAL & EXPERIMENT)

*   **Title:** `[TECH] Khởi tạo Môi trường Cloud GPU, Cấu hình vLLM Engine & Đo thử nghiệm mốc BF16 Baseline`
*   **Module:** Infrastructure & Inference Engine Setup
*   **Sprint:** Sprint 1 (Week 1-2)
*   **Team / Role:** Technical & Experiment (Pham Minh Quan - Lead, Tran Minh Khanh) / Writing & Coordination (Nguyen Van Quang Duy - Technical Liaison)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Khởi động hạ tầng tính toán của dự án. Thiết lập môi trường chạy thử nghiệm trên máy chủ GPU đám mây (Vast.ai hoặc RunPod) sử dụng card đồ họa tối thiểu 24GB VRAM (như RTX 3090/4090 hoặc L4). Cấu hình bộ thư viện **vLLM** hỗ trợ nhân tính toán tối ưu TurboQuant và chạy thử thành công kịch bản đo đạc baseline ban đầu (Full KV Cache BF16).

#### 2. Tài liệu đọc tham khảo (References)
*   *Hướng dẫn vLLM & TurboQuant:* [vLLM Documentation - Supported KV Cache Dtypes](https://docs.vllm.ai/en/latest/models/engine_args.html) (Cập nhật 2026 hỗ trợ `--kv-cache-dtype turboquant_4bit_nc`).
*   *Plugin TurboQuant:* [turboquant-vllm GitHub Repository](https://github.com/).
*   *Thư viện đo bộ nhớ:* NVIDIA Management Library (`pynvml`) Python API.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Khởi tạo một phiên máy ảo (instance) trên RunPod hoặc Vast.ai với cấu hình tối thiểu: 1 GPU RTX 4090/3090 (24GB VRAM), hệ điều hành Ubuntu, đã cài sẵn PyTorch và CUDA 12.x.
*   **Bước 2:** Cài đặt môi trường Conda và cài các gói phụ thuộc:
    ```bash
    conda create -n dbml_benchmark python=3.10 -y
    conda activate dbml_benchmark
    pip install vllm pynvml transformers pandas
    ```
*   **Bước 3:** Viết cấu trúc nền tảng cho file đo đạc `scripts/run_baseline.py`. Script phải tự động ghi nhận Peak VRAM của GPU thông qua thư viện `pynvml` hoặc `torch.cuda.max_memory_allocated()` trong suốt pha Prefill và Decode.
*   **Bước 4:** Chạy kiểm thử đo đạc thực tế mốc không nén (Full KV Cache FP16/BF16) với một mô hình trong bộ benchmark chính thức 4-model suite (ưu tiên `qwen3:8b-fp16`) sử dụng bộ dữ liệu `datasets/test_set_small.json`. Ghi nhận thử nghiệm kết quả đầu ra vào file CSV cục bộ.

#### 4. Kết quả đầu ra (Expected Output)
*   Môi trường máy chủ ảo GPU đã thiết lập xong, có thể truy cập qua SSH.
*   File mã nguồn ban đầu của script đo đạc `scripts/run_baseline.py` đẩy lên Git.
*   Một dòng kết quả đo đạc mẫu của mốc BF16 ghi nhận thành công trong file CSV cục bộ.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Thao tác cài đặt và import thư viện `vllm` hoạt động ổn định trên GPU đám mây mà không báo lỗi Driver hay CUDA.
*   [ ] Script đo đạc đo được chính xác dung lượng bộ nhớ đỉnh Peak VRAM (tính bằng MB/GB) và độ trễ sinh từ (ITL) của mô hình chạy thử nghiệm.
*   [ ] Đã cấu hình và thử nghiệm thành công mốc chạy baseline không lỗi bộ nhớ (OOM) ở độ dài ngữ cảnh tối thiểu 8k tokens.

---

### TASK 5: CÀO BỔ SUNG DỮ LIỆU THỜI SỰ (DATA & ANALYSIS)

*   **Title:** `[DATA-ADD] Cào và tiền xử lý dữ liệu sách/báo/mạng xã hội có tính thời sự nóng bằng NeMo Curator`
*   **Module:** Dataset Engineering & Preprocessing
*   **Sprint:** Sprint 1 (Week 1-2)
*   **Team / Role:** Data & Analysis (Nguyen Ho Phat - Lead, Huynh Huu Huy)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Thu thập dữ liệu sách, bài báo thời sự và các thảo luận mạng xã hội tiếng Việt có tính thời sự cao để bổ sung vào bộ dữ liệu thử nghiệm. Dữ liệu này sẽ được tiền xử lý và nhúng vào làm nhiễu/ngữ cảnh dài để kiểm tra độ nhạy cảm của tiếng Việt khi nén KV Cache.

#### 2. Tài liệu đọc tham khảo (References)
*   *Scraper mẫu:* Thư mục `scripts/scrape_news_sample.py` (Script cào tin tức VnExpress mẫu).
*   *Dữ liệu HF:* `vietnews` dataset và các corpus báo chí tiếng Việt sẵn có.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Sử dụng scraper mẫu hoặc các bộ thư viện python (`requests`, `beautifulsoup4`) để viết script cào bổ sung 50-100 bài báo thời sự nóng hoặc bài thảo luận mạng xã hội tiếng Việt mới nhất.
*   **Bước 2:** Kết hợp với dữ liệu đóng gói sẵn như `vietnews` hoặc các corpus mạng xã hội tiếng Việt để lấy thêm văn bản đa dạng ngữ cảnh.
*   **Bước 3:** Sử dụng **NVIDIA NeMo Curator** để làm sạch bảng mã Unicode NFC, chuẩn hóa văn bản và lọc trùng lặp.
*   **Bước 4:** Trộn các tài liệu mới cào này vào bộ dữ liệu `test_set_small.json` để mở rộng ngữ cảnh thử nghiệm.

#### 4. Kết quả đầu ra (Expected Output)
*   Kho dữ liệu cào mới được chuẩn hóa và lưu trữ tại `datasets/scraped_news_social.json`.
*   Bộ dữ liệu `datasets/test_set_small.json` được cập nhật thêm các mẫu thử có chứa tin tức thời sự nóng.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [x] Cào và làm sạch thành công ít nhất 100 văn bản tiếng Việt thời sự mới.
*   [ ] Tích hợp thành công dữ liệu này vào tệp `test_set_small.json` mà không làm thay đổi cấu trúc schema cũ.

---

### TASK 6: TÍCH HỢP SỚM THUẬT TOÁN (TECHNICAL & EXPERIMENT)

*   **Title:** `[TECH-ADD] Nghiên cứu cấu hình và tích hợp sớm nhân lượng tử hóa (TurboQuant/PolarQuant) vào vLLM`
*   **Module:** Infrastructure & Inference Engine Setup
*   **Sprint:** Sprint 1 (Week 1-2)
*   **Team / Role:** Technical & Experiment (Pham Minh Quan) / Writing & Coordination (Nguyen Van Quang Duy - Technical Liaison)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Tiến hành cài đặt, biên dịch các nhân CUDA/Triton cho TurboQuant/PolarQuant trên Cloud GPU đã thuê. Chạy thử nghiệm nén KV Cache cơ bản để phát hiện sớm các lỗi xung đột compiler hoặc driver CUDA trước khi bước vào phase chạy thực nghiệm hàng loạt.

#### 2. Tài liệu đọc tham khảo (References)
*   *Kernel vLLM:* [turboquant-vllm GitHub Repository](https://github.com/) & vLLM quantization manuals.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** SSH vào Cloud GPU (RTX 3090/4090) đã thuê.
*   **Bước 2:** Tải code và tiến hành biên dịch (compile from source) các nhân Triton/CUDA của repo `turboquant-vllm`.
*   **Bước 3:** Chạy một script kiểm thử nén KV Cache ngắn (context length ~2k) để kiểm tra xem vLLM có nhận diện được đối số `--kv-cache-dtype turboquant_4bit_nc` và `--kv-cache-dtype polarquant` mà không crash hay không.
*   **Bước 4:** Ghi nhận lại các lỗi phát sinh (nếu có) và thảo luận phương án xử lý cùng PM.

#### 4. Kết quả đầu ra (Expected Output)
*   Môi trường vLLM tích hợp thành công nhân TurboQuant/PolarQuant trên Cloud GPU.
*   Log chạy thử nén thành công không crash.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Biên dịch thành công các nhân CUDA/Triton của TurboQuant trên Cloud GPU đã thuê.
*   [ ] Chạy thử nghiệm nén KV Cache ở mức 4-bit thành công với một mô hình thử nghiệm nhỏ.

---

### TASK 7: BÁO CÁO TIẾN ĐỘ ĐỊNH KỲ (WRITING & COORDINATION)

*   **Title:** `[COORD-ADD] Tổng hợp tiến độ Sprint 1 và soạn tài liệu báo cáo gửi giảng viên hướng dẫn (Thứ 4 - 24/06/2026)`
*   **Module:** Project Management & Coordination
*   **Sprint:** Sprint 1 (Week 1-2)
*   **Team / Role:** Writing & Coordination (Do Kien Hung - Lead, Phan Trong Qui)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Tổng hợp các kết quả sơ bộ đạt được trong Sprint 1 của tất cả các team (Data, Tech, Research) và chuẩn bị slide/tài liệu tóm tắt ngắn để báo cáo tiến độ trực tiếp cho TS. Lê Ngọc Hiếu vào Thứ 4 ngày 24/06/2026.

#### 2. Tài liệu đọc tham khảo (References)
*   *Kế hoạch tổng thể:* `docs/master-plan.md` [2.1.2].

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Họp nhanh các nhóm trưởng để lấy thông tin: Team Data (số lượng mẫu dữ liệu thời sự đã cào), Team Tech (trạng thái setup Cloud GPU và chạy thử baseline BF16), Team Research (tiến độ Proposal trên Overleaf).
*   **Bước 2:** Biên soạn slide hoặc file báo cáo tiến độ dài 2-3 trang dạng tóm tắt.
*   **Bước 3:** Trình bày báo cáo cho giảng viên vào ngày họp nhóm Thứ 4, tiếp thu các góp ý mới của thầy để cập nhật vào backlog.

#### 4. Kết quả đầu ra (Expected Output)
*   Slide hoặc tài liệu báo cáo tiến độ `results/Sprint1_Progress_Report_24_06.pdf`.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Báo cáo tiến độ được gửi trước cho giảng viên và thảo luận thành công trong buổi họp Thứ 4.
*   [ ] Ghi nhận đầy đủ feedback của giảng viên vào biên bản họp nhóm tuần 2.

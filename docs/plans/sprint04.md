# SPRINT 04
### TASK 1: HOÀN THIỆN BÁO CÁO & SLIDES (WRITING & COORDINATION)

*   **Title:** `[COORD] Hoàn thiện Báo cáo Word tiếng Việt (>=30 trang), Thiết kế Slide thuyết trình bảo vệ (15-25 slides) & Chuẩn bị kịch bản phản biện (Mock Q&A)`
*   **Module:** Phase 4: Viết bài + chỉnh sửa
*   **Sprint:** Sprint 4 (Week 7)
*   **Team / Role:** Writing & Coordination (Do Kien Hung - Lead, Phan Trong Qui)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Tổng hợp toàn bộ nội dung bản thảo thu được từ Sprint 3 để biên tập, hiệu chỉnh và định dạng chuẩn hóa **Báo cáo chi tiết tiếng Việt (Word Document $\ge 30$ trang)** bao gồm dữ liệu cào thời sự và các metrics nâng cao. Đồng thời, thiết kế **Slide thuyết trình (15-25 slides)** bao quát toàn bộ nội dung cốt lõi của nghiên cứu. Chuẩn bị kịch bản câu hỏi phản biện (Mock Q&A) dự kiến từ Hội đồng chấm thi và TS. Lê Ngọc Hiếu.

#### 2. Tài liệu đọc tham khảo (References)
*   *Quy chế môn học:* Đề cương chi tiết môn học DBML434077 (HCM-UTE) - Yêu cầu cấu trúc nội dung và định dạng Báo cáo Word và Slide bảo vệ.
*   *Mẫu Slide thiết kế:* Các bài thuyết trình khoa học tại hội thảo MLSys / ICLR để học cách bố trí thông tin trực quan.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Rà soát và hoàn thiện toàn bộ các chương trong tệp Báo cáo tiếng Việt (Word), đảm bảo dung lượng tối thiểu đạt 30 trang:
    *   *Chương 1:* Tóm tắt & Giới thiệu đề tài (Research Proposal gốc).
    *   *Chương 2:* Cơ sở lý thuyết sơ bộ (Cơ chế hoạt động của TurboQuant, PolarQuant, HQQ, FP8 KV Cache).
    *   *Chương 3:* Thiết kế hệ thống & Phương pháp thực nghiệm (Kiến trúc hệ thống, Test-set dữ liệu thời sự cào bổ sung, các metrics bổ sung nâng cao, script đo đạc).
    *   *Chương 4:* Kết quả thực nghiệm, Luận giải khoa học & Các biểu đồ Pareto Frontier thu được (bao gồm cả phân tích chỉ số nén nâng cao).
    *   *Chương 5:* Kết luận & Đóng góp thực tiễn.
*   **Bước 2:** Định dạng văn bản: Tạo mục lục tự động, danh mục hình vẽ, danh mục bảng biểu; căn lề chuẩn, sửa các lỗi chính tả, phông chữ tiếng Việt.
*   **Bước 3:** Biên soạn bộ Slide bảo vệ (15-25 slides) bằng PowerPoint hoặc Google Slides. Cấu trúc slide đề xuất:
    *   *Slide 1-3:* Tiêu đề, Thành viên nhóm, Bối cảnh & Lý do thực hiện đề tài.
    *   *Slide 4-6:* Vấn nghiệp nghiên cứu, Câu hỏi nghiên cứu (RQ1-RQ4) và Điểm mới (Novelty).
    *   *Slide 7-9:* Kiến trúc hệ thống, Dữ liệu thời sự bổ sung và Phương pháp thực nghiệm.
    *   *Slide 10-15:* Kết quả thực nghiệm trực quan (Trình bày các biểu đồ VRAM, Latency, Pareto Perplexity và biểu đồ hiệu suất bổ sung).
    *   *Slide 16-18:* Thảo luận, Đóng góp thực tiễn và Kết luận.
*   **Bước 4:** Xây dựng tệp tài liệu `Q&A_Preparation.md` dự đoán trước các câu hỏi phản biện của thầy hướng dẫn và hội đồng chấm thi (ví dụ: *"Tại sao không so sánh với các kỹ thuật nén lượng tử hóa trọng số khác?", "Ý nghĩa thực tiễn của biểu đồ Pareto đối với việc deploy mô hình?", "Sự nhạy cảm của tiếng Việt khi nén sâu thể hiện qua các chỉ số Compression Ratio và GPU Efficiency thế nào?"*).

#### 4. Kết quả đầu ra (Expected Output)
*   Tệp tin `results/Bao_cao_cuoi_ky_BDML_Group1.docx` hoàn chỉnh ($\ge 30$ trang).
*   Tệp tin Slide thuyết trình `results/Slide_bao_ve_cuoi_ky_BDML_Group1.pptx`.
*   Tài liệu kịch bản phản biện `kickoff/Q&A_Preparation.md`.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Báo cáo Word được định dạng nhất quán, không lỗi phông chữ, độ dài tối thiểu đạt 30 trang và đã xuất bản dạng PDF, phản ánh đầy đủ dữ liệu và metrics mới.
*   [ ] Slide thuyết trình bảo vệ được hoàn thiện, thiết kế trực quan, súc tích và có thời lượng trình bày thử nghiệm tối ưu trong khoảng 15-20 phút.
*   [ ] Bộ kịch bản Q&A chuẩn bị tối thiểu 5 câu hỏi tình huống kỹ thuật cốt lõi kèm câu trả lời định lượng chi tiết (có lồng ghép chỉ số mới).

---

### TASK 2: BIÊN TẬP BÀI BÁO TIẾNG ANH (RESEARCH & SCOPE)

*   **Title:** `[PAPER] Biên tập & Đóng gói Bài báo tiếng Anh (Overleaf LaTeX >=6 trang) chuẩn IEEE`
*   **Module:** Phase 4: Viết bài + chỉnh sửa
*   **Sprint:** Sprint 4 (Week 7)
*   **Team / Role:** Research & Scope (Nguyen Dang Quoc Anh - Lead, Phan Trong Phu)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Biên tập chi tiết, hiệu chỉnh học thuật và đóng gói hoàn thiện **Bài báo khoa học bằng tiếng Anh (Research Paper $\ge 6$ trang)** trên hệ thống Overleaf sử dụng cấu trúc định dạng chuẩn của IEEE Conference. Tập trung tối ưu hóa ngôn ngữ học thuật, kiểm tra tính logic từ đặt vấn đề đến kết quả thực nghiệm với các chỉ số nén nâng cao và dữ liệu thời sự cào mới, đồng thời chuẩn hóa danh mục tài liệu tham khảo.

#### 2. Tài liệu đọc tham khảo (References)
*   *LaTeX Style Guide:* [IEEE Editorial Style Manual](https://ieeeauthorcenter.ieee.org/create-your-ieee-article/use-authoring-tools-and-templates/ieee-article-templates/) - Hướng dẫn chi tiết về cấu trúc câu, trích dẫn bảng biểu và định dạng tài liệu trong văn bản khoa học.
*   *Mẫu cấu trúc bài nghiên cứu:* Các bài viết đạt giải Best Paper tại các hội thảo lớn để học cách căn chỉnh lề, dòng và vị trí đặt biểu đồ khoa học.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Rà soát và hoàn thiện toàn bộ các chương mục trên Overleaf:
    *   *Abstract:* Viết súc tích (150-200 từ) tóm tắt bối cảnh, phương pháp benchmark, kết quả định lượng chính (tiết kiệm VRAM, độ tăng Perplexity và các chỉ số nén/hiệu suất mới).
    *   *Section I - Introduction & Section II - Related Work:* Đảm bảo các trích dẫn tài liệu tham khảo (citations) hoạt động đúng định dạng IEEE.
    *   *Section III - Methodology:* Kiểm tra các công thức toán học mô tả thuật toán TurboQuant và PolarQuant, các metrics mới được định nghĩa rõ ràng.
    *   *Section IV - Experimental Evaluation:* Chèn các biểu đồ trade-off (bao gồm cả biểu đồ chỉ số nén bổ sung) dạng ảnh vector hoặc PDF chất lượng cao thu được từ team Analysis. Viết luận giải chi tiết dựa trên dữ liệu cào mới.
    *   *Section V - Conclusion & Future Work:* Tóm tắt ngắn gọn đóng góp thực tiễn.
*   **Bước 2:** Chuẩn hóa danh mục tài liệu tham khảo trong tệp `references.bib`. Đảm bảo tất cả các bài báo tham chiếu đều có đầy đủ thông tin: Tác giả, Tên bài báo, Tên hội thảo/tạp chí, Năm xuất bản, Số trang, Mã DOI.
*   **Bước 3:** Biên dịch dự án Overleaf, sửa toàn bộ các lỗi cảnh báo (Warnings) và lỗi biên dịch (Compilation Errors) của LaTeX.
*   **Bước 4:** Xuất bản bài báo khoa học dưới dạng file PDF chuẩn hóa.

#### 4. Kết quả đầu ra (Expected Output)
*   Dự án Overleaf được đóng gói sạch sẽ, không còn lỗi biên dịch.
*   Tệp tin bài báo khoa học tiếng Anh `paper/vietnamese_llm_kv_cache_benchmark_paper.pdf` ($\ge 6$ trang).

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Bài báo được viết 100% bằng tiếng Anh học thuật chuẩn mực, độ dài tối thiểu đạt 6 trang hai cột theo mẫu IEEE, trình bày rõ các chỉ số bổ sung.
*   [ ] Đã loại bỏ hoàn toàn các lỗi biên dịch của LaTeX trên hệ thống Overleaf.
*   [ ] 100% tài liệu tham khảo được trích dẫn chính xác trong văn bản thông qua lệnh `\cite{...}` và xuất hiện đầy đủ trong danh mục tham chiếu cuối bài.

---

### TASK 3: ĐÓNG GÓI MÃ NGUỒN & HƯỚNG DẪN TÁI LẬP (TECHNICAL & EXPERIMENT)

*   **Title:** `[CODE] Đóng gói mã nguồn GitHub Repo, Viết tài liệu hướng dẫn tái lập (README.md) & Refactor Code`
*   **Module:** Team 2: Technical & Experiment (4 người)
*   **Sprint:** Sprint 4 (Week 7)
*   **Team / Role:** Technical & Experiment (Pham Minh Quan - Lead, Tran Minh Khanh, Nguyen Van Quang Duy, Huynh Ngoc Thach)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Thực hiện tối ưu và dọn dẹp mã nguồn (code refactoring) để đảm bảo tính sạch sẽ, dễ đọc. Đóng gói toàn bộ mã nguồn của dự án lên GitHub Repository. Biên soạn tài liệu hướng dẫn sử dụng và tái lập thực nghiệm chi tiết (`README.md` hoàn chỉnh) và cấu hình tệp tin phụ thuộc (`requirements.txt`).

#### 2. Tài liệu đọc tham khảo (References)
*   *Tài liệu GitHub:* [How to write a Great README](https://github.com/dbader/readme-template) - Quy chuẩn biên soạn tài liệu mã nguồn mở chuyên nghiệp trên GitHub.
*   *Mã nguồn mở chuẩn:* Các repo github của các thư viện lớn như `vllm` hoặc `transformers` để học cách tổ chức thư mục mã nguồn.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Tiến hành rà soát mã nguồn (Code Review) trong các tệp tin `scripts/run_baseline.py` và `scripts/plot_results.py`:
    *   Loại bỏ các đoạn code thừa, code comment rác hoặc các biến không sử dụng.
    *   Thêm chú thích giải thích chức năng (Docstrings/Comments) cho các hàm đo đạc bộ nhớ, thời gian và Perplexity.
*   **Bước 2:** Cấu hình tệp tin quản lý thư viện phụ thuộc `requirements.txt` chứa thông tin chính xác về phiên bản của các thư viện được sử dụng trong suốt quá trình chạy thực nghiệm:
    `vllm>=0.7.0`, `torch>=2.2.0`, `pynvml>=11.5.0`, `pandas>=2.1.0`, `matplotlib>=3.8.0`, `plotly>=5.18.0`.
*   **Bước 3:** Tạo tệp tin `.gitignore` để ngăn việc đẩy các file dung lượng lớn không cần thiết lên GitHub (ví dụ: các file log CSV thô dung lượng lớn, các checkpoint mô hình, các file cache Python `.pyc`).
*   **Bước 4:** Hoàn thiện tệp tin hướng dẫn `README.md` bao quát toàn bộ dự án dựa trên khung đã xây dựng, bổ sung phần hướng dẫn cài đặt chi tiết từng bước (Step-by-Step Installation) và hướng dẫn chạy nhanh lệnh đo đạc baseline.

#### 4. Kết quả đầu ra (Expected Output)
*   Repository GitHub được dọn dẹp sạch sẽ, cấu trúc thư mục rõ ràng đúng chuẩn.
*   Tệp tin `requirements.txt` và `README.md` hoàn chỉnh đẩy lên nhánh `main`.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Mã nguồn trên GitHub được tổ chức gọn gàng, không có file thừa hoặc file log rác được đẩy lên.
*   [ ] Thực hiện cài đặt thử nghiệm môi trường từ đầu trên một máy ảo mới bằng lệnh `pip install -r requirements.txt` hoạt động thành công.
*   [ ] Tài liệu `README.md` hiển thị đầy đủ, không bị lỗi định dạng markdown, mô tả chi tiết cách thức chạy lại các lệnh benchmark để tái lập kết quả.

---

### TASK 4: TỔNG DUYỆT NỘI BỘ & NGHIỆM THU (ALL TEAMS - COORDINATION)

*   **Title:** `[REVIEW] Tổ chức họp tổng duyệt nội bộ (Internal Review), Chạy thử Slide bảo vệ & Nghiệm thu toàn dự án`
*   **Module:** Phase 4: Viết bài + chỉnh sửa
*   **Sprint:** Sprint 4 (Week 7)
*   **Team / Role:** Toàn bộ thành viên nhóm (Do Kien Hung - PM / Host)
*   **Priority:** High

#### 1. Mô tả chi tiết Task (Description)
Tổ chức buổi họp tổng duyệt nội bộ (Internal Review / Rehearsal) cho toàn bộ thành viên nhóm trước ngày báo cáo chính thức với hội đồng. Thực hiện chạy thử nghiệm trình bày Slide bảo vệ để căn chỉnh thời gian tối ưu, phân chia vai trò thuyết trình của từng thành viên và rà soát chéo các tài liệu bàn giao để phát hiện sai sót cuối cùng.

#### 2. Tài liệu đọc tham khảo (References)
*   *Mẫu biên bản họp:* Tệp tin `kickoff/minutes_template.md` đã tạo để ghi chép nội dung buổi họp nghiệm thu.

#### 3. Từng bước thực hiện chi tiết (Step-by-Step)
*   **Bước 1:** Bạn (Kiến Hưng) thiết lập lịch họp trực tiếp hoặc trực tuyến dài 60-90 phút cho toàn nhóm.
*   **Bước 2:** Phân chia vai trò thuyết trình slide bảo vệ cuối kỳ:
    *   *Người thuyết trình 1 (Research Lead):* Trình bày phần Đặt vấn đề, Lý do chọn đề tài và Cơ sở lý thuyết.
    *   *Người thuyết trình 2 (Tech Lead):* Trình bày Kiến trúc hệ thống, Giải pháp kỹ thuật và Quy trình đo đạc.
    *   *Người thuyết trình 3 (Data/Analysis):* Trình bày các biểu đồ Kết quả, Phân tích đánh đổi Pareto và Kết luận.
*   **Bước 3:** Tiến hành chạy thử (Rehearsal) thuyết trình slide căn thời gian chính xác trong khoảng 15 phút. Ghi nhận các đóng góp ý kiến của các thành viên để chỉnh sửa ngay các slide bị mập mờ thông tin hoặc quá tải chữ.
*   **Bước 4:** Rà soát chéo các sản phẩm bàn giao cuối cùng:
    *   Team Tech kiểm tra chéo file Báo cáo tiếng Việt xem phần mô tả kỹ thuật đã khớp với mã nguồn thực tế chưa.
    *   Team Research kiểm tra chéo các biểu đồ trong Paper tiếng Anh xem các nhãn chú thích đã đồng nhất chưa.
*   **Bước 5:** Thống nhất phân chia người phụ trách trả lời cho từng nhóm câu hỏi phản biện lý thuyết/thực nghiệm đã chuẩn bị từ trước.

#### 4. Kết quả đầu ra (Expected Output)
*   Biên bản họp tổng duyệt nội bộ `results/Meeting_Minutes_Final_Review.md`.
*   Các tài liệu Slide, Báo cáo Word, Paper PDF được khóa phiên bản (Freeze version) sẵn sàng nộp cho giảng viên hướng dẫn.

#### 5. Tiêu chuẩn hoàn thành (Definition of Done - DoD)
*   [ ] Đã tổ chức thành công buổi họp chạy thử nghiệm slide với sự tham gia đầy đủ của tất cả thành viên trong nhóm.
*   [ ] Phân định rõ ràng vai trò thuyết trình và trách nhiệm trả lời câu hỏi phản biện cho từng thành viên.
*   [ ] Tất cả các sản phẩm bàn giao (Code, Word, Slide, Paper) được lưu trữ tại đúng các thư mục quy định và hoạt động ổn định.
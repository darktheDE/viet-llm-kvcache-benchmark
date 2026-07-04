# KỊCH BẢN THUYẾT TRÌNH ONBOARDING NHÓM (PRESENTATION SCRIPT)

Tài liệu này cung cấp lời thoại chi tiết (speaking script) cho người thuyết trình tương ứng với từng slide trong file **onboarding_presentation.html**. Văn phong được thiết kế dưới dạng ngôn ngữ nói, tự nhiên, gần gũi và dễ hiểu để triển khai dự án cho toàn team.

---

### Slide 1: Welcome & Giới thiệu
* **Lời thoại:**
  > "Chào mọi người! Hôm nay nhóm chúng mình họp để kickoff và onboarding dự án nghiên cứu cuối kỳ môn Học máy quy mô lớn (BDML). Đề tài của chúng ta là: **Benchmarking TurboQuant & KV Cache Compression trên các mô hình ngôn ngữ lớn tiếng Việt**. 
  > Buổi hôm nay mình sẽ đi qua bối cảnh, ### Slide 2: Bối cảnh & Mục tiêu dự án
* **Lời thoại:**
  > "Đầu tiên là về bối cảnh tại sao chúng ta làm đề tài này. Khi chạy các mô hình ngôn ngữ lớn (LLM) với câu hỏi hoặc văn bản dài, có một bộ phận bộ nhớ gọi là KV Cache sẽ phình to ra rất nhanh theo chiều dài của văn bản, gây ra hiện tượng tràn bộ nhớ GPU (Out-of-Memory).
  > Hiện tại có các kỹ thuật nén mới rất hứa hẹn như TurboQuant, PolarQuant hay HQQ giúp tiết kiệm VRAM rất tốt. Tuy nhiên, hầu hết các đánh giá hiện nay chỉ làm trên tiếng Anh. Tiếng Việt của chúng ta có cách ghép vần, token hóa khác biệt nên chưa rõ các phương pháp này nén lại thì chất lượng tiếng Việt sẽ bị giảm thế nào. 
  > Khoảng trống nghiên cứu là chưa có nhiều công trình đánh giá sự suy giảm chất lượng này một cách hệ thống trên tiếng Việt. Mục tiêu của chúng ta là dựng lên một hệ thống đo đạc thực tế để tìm ra điểm Pareto tối ưu — tức là điểm mà nén được nhiều bộ nhớ nhất nhưng chất lượng tiếng Việt ít bị suy giảm nhất để khuyến nghị cho các doanh nghiệp."

---

### Slide 3: Câu hỏi nghiên cứu (RQ1 - RQ4)
* **Lời thoại:**
  > "Để nghiên cứu đi đúng hướng và đạt tiêu chuẩn khoa học cao, nhóm mình sẽ tập trung trả lời 4 câu hỏi nghiên cứu chính:
  > - **RQ1:** Chạy TurboQuant thực tế trên tiếng Việt có giảm được VRAM và tăng tốc độ sinh từ (latency/throughput) không?
  > - **RQ2:** Chất lượng ngôn ngữ (đo bằng Perplexity) bị giảm bao nhiêu điểm ở các mốc nén 3-bit, 4-bit?
  > - **RQ3:** Khi kéo dài độ dài văn bản từ 4k lên 32k tokens thì sự đánh đổi giữa bộ nhớ và chất lượng thay đổi ra sao?
  > - **RQ4:** Cuối cùng, phương pháp nén nào (TurboQuant, PolarQuant, HQQ hay FP8) sẽ là tối ưu và ổn định nhất để đưa vào thực tế?"

---

### Slide 4: Phạm vi thực nghiệm (Scope)
* **Lời thoại:**
  > "Về phạm vi công việc, chúng ta sẽ thực nghiệm trên:
  > - **5 mô hình:** Sailor2-8B (mô hình SEA đa ngôn ngữ của Sea AI Lab, dựa trên Qwen2.5), Qwen2.5-7B, Llama-3.1-8B, URA-LLaMa-3-8B (từ Đại học Bách Khoa HCM), và Vistral-7B làm đối chứng.
  > - **Các mốc nén:** Full KV (BF16) làm chuẩn so sánh, FP8, HQQ, PolarQuant và TurboQuant.
  > - **Dữ liệu:** Bộ benchmark VMLU tiếng Việt, tập dữ liệu VTSNLP và một bộ test-set nhỏ tự curate dài từ 4k đến 16k tokens.
  > - **Chỉ số đo đạc:** Về phần cứng ta đo Peak VRAM lúc prefill và decode, tốc độ đo TTFT và ITL. Về chất lượng ta đo Perplexity và các điểm F1/ROUGE-L."

---

### Slide 5: Kiến trúc Hệ thống Benchmark
* **Lời thoại:**
  > "Đây là kiến trúc hệ thống 4 tầng mà team Tech và team Data sẽ cùng xây dựng:
  > - **Tầng 1 (Data Prep):** Chúng ta dùng NeMo Curator làm sạch dữ liệu tiếng Việt rồi phân nhóm độ dài câu (4k đến 32k).
  > - **Tầng 2 (Serving):** Dùng vLLM nạp mô hình gốc (giữ nguyên trọng số 16-bit), sau đó tích hợp các thuật toán nén KV Cache (FP8, PolarQuant, TurboQuant) thông qua thư viện và nhân CUDA/Triton sẵn có để đo đạc và đánh giá hiệu năng thực tế mà không tự lập trình kernel mới từ đầu.
  > - **Tầng 3 (Monitoring):** Viết script Python hook vào thư viện `pynvml` đo VRAM đỉnh và các mốc thời gian phản hồi, tính toán Perplexity.
  > - **Tầng 4 (Analysis):** Gom tất cả file CSV kết quả, dùng Pandas phân tích và vẽ đồ thị Pareto."� decode, tốc độ đo TTFT và ITL. Về chất lượng ta đo Perplexity và các điểm F1/ROUGE-L."

---

### Slide 5: Kiến trúc Hệ thống Benchmark
* **Lời thoại:**
  > "Đây là kiến trúc hệ thống 4 tầng mà team Tech và team Data sẽ cùng xây dựng:
  > - **Tầng 1 (Data Prep):** Chúng ta dùng NeMo Curator làm sạch dữ liệu tiếng Việt rồi phân nhóm độ dài câu (4k đến 32k).
  > - **Tầng 2 (Serving):** Dùng vLLM nạp mô hình gốc, sau đó gọi các nhân Triton/CUDA của TurboQuant để nén KV Cache.
  > - **Tầng 3 (Monitoring):** Viết script Python hook vào thư viện `pynvml` đo VRAM đỉnh và các mốc thời gian phản hồi, tính toán Perplexity.
  > - **Tầng 4 (Analysis):** Gom tất cả file CSV kết quả, dùng Pandas phân tích và vẽ đồ thị Pareto."

---

### Slide 6: Lộ trình triển khai (Sprints)
* **Lời thoại:**
  > "Chúng ta có lộ trình 7 tuần chia làm 4 Sprints rất rõ ràng:
  > - **Sprint 1 (Tuần 1-2):** Setup môi trường Cloud GPU, tiền xử lý dữ liệu tiếng Việt sạch và chạy thử mốc baseline BF16.
  > - **Sprint 2 (Tuần 3-4):** Hoàn thiện script đo tự động, chạy thực nghiệm diện rộng mốc baseline và dựng khung LaTeX trên Overleaf.
  > - **Sprint 3 (Tuần 5-6):** Team Tech chạy nén toàn bộ các mốc còn lại, Duy xử lý các ca lỗi OOM, Thạch vẽ biểu đồ Pareto và team Research viết phần kết quả.
  > - **Sprint 4 (Tuần 7):** Hoàn thiện báo cáo tiếng Việt 30 trang, slide thuyết trình và bài báo tiếng Anh 6 trang chuẩn IEEE."

---

### Slide 7: Phân bổ nhân sự (RACI)
* **Lời thoại:**
  > "Về phân chia công việc trong nhóm:
  > - Mình (Kiến Hưng) và Phan Trọng Quí sẽ chịu trách nhiệm điều phối tiến độ và tổng hợp báo cáo.
  > - Team Tech gồm Quân (Tech Lead), Việt Anh, Khánh và Duy (chuyên về lượng tử hóa và xử lý lỗi phần cứng).
  > - Team Data gồm Phát (Data Lead), Huy (curate prompt) và Thạch (chuyên vẽ đồ thị kết quả).
  > - Team Research gồm Quốc Anh (Owner ý tưởng) và Phú sẽ lo liệu phần viết paper tiếng Anh trên Overleaf.
  > Để trơn tru, mọi người nhớ cập nhật trạng thái task trên **Plane.so** hàng tuần nhé."

---

### Slide 8: Workflow quản lý tài liệu
* **Lời thoại:**
  > "Để tránh việc tài liệu bị duplicate và lẫn lộn phiên bản, nhóm mình thống nhất quy tắc sau:
  > - **Notion:** Là nơi đọc chính (Knowledge Base), chứa đề cương, lý thuyết và biên bản họp.
  > - **Plane.so:** Chỉ để giao việc, tuyệt đối không viết tài liệu dài trên đây. Chỉ ghi checklist DoD và dán link Notion/Git.
  > - **GitHub:** Chỉ chứa source code, requirements.txt và file `.tex` viết paper. Không đẩy file Word, Excel, PPTX lên Git.
  > - **Google Drive:** Chỉ chứa các file sản phẩm cuối cùng như bản Word báo cáo, Slide PPTX và dataset thô."

---

### Slide 9: Kế hoạch Quản trị Rủi ro
* **Lời thoại:**
  > "Chúng ta cũng lường trước một số rủi ro:
  > - **Về phần cứng:** GPU local của chúng ta không đủ mạnh để chạy vLLM context dài. Giải pháp là chúng ta sẽ góp quỹ thuê GPU đám mây như RunPod/Vast.ai. Thuê 1 card RTX 4090 chỉ mất khoảng 5-10k VNĐ/giờ. Tổng dự án chỉ tiêu hết tầm 130k-250k VNĐ, rất rẻ và khả thi.
  > - **Về lỗi tràn bộ nhớ (OOM):** Khi kéo context lên 32k chắc chắn sẽ bị OOM. Duy sẽ làm helpdesk cấu hình FlashAttention-2, giảm block size hoặc giảm số lượng batch sequence để tối ưu hóa bộ nhớ."

---

### Slide 10: DoD & Kêu gọi hành động
* **Lời thoại:**
  > "Cuối cùng là Tiêu chuẩn hoàn thành (DoD) để nghiệm thu dự án: GitHub repo sạch sẽ có hướng dẫn chạy; Báo cáo Word tiếng Việt đạt trên 30 trang; Slide thuyết trình sẵn sàng; và Paper tiếng Anh trên 6 trang biên dịch không lỗi trên Overleaf.
  > Ngay sau buổi hôm nay, chúng ta bắt đầu **Sprint 1** luôn:
  > - Việt Anh thuê GPU Cloud chạy thử baseline.
  > - Phát làm sạch dữ liệu.
  > - Quốc Anh lên Overleaf viết Proposal.
  > Mọi người có câu hỏi hay góp ý gì cho kế hoạch này không?"

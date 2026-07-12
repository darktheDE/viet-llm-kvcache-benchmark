### 1. Giải quyết vấn đề Phần cứng & Chi phí

*   **Ý kiến của thầy & nhóm:** *"Vấn đề về phần cứng để triển khai training model và áp dụng turboquant... Không biết server của Việt Anh chạy nổi không... Chạy CPU hay GPU?"*
*   **Định hướng kỹ thuật thực tế:**
    *   **Lưu ý quan trọng:** **Nhóm không cần phải tiến hành huấn luyện (Training) hay tinh chỉnh (Fine-tuning) lại mô hình.** TurboQuant, HQQ, hay PolarQuant đều là các kỹ thuật **Post-Training Quantization (PTQ)**. Quy trình này chỉ thực hiện nén trực tiếp trên KV cache hoặc trọng số của mô hình đã có sẵn khi chạy suy luận (Inference). Do đó, **chi phí tính toán gần như bằng 0 so với việc training.**
    *   **Tích hợp kernel & Trọng số:** Trọng số của mô hình được giữ nguyên ở định dạng gốc 16-bit (BF16/FP16), trọng tâm nghiên cứu duy nhất là lượng tử hóa và nén KV cache, không nén trọng số. Các thuật toán nén KV cache (FP8, PolarQuant, TurboQuant) được tích hợp trực tiếp vào inference engine thông qua thư viện và nhân (kernels) CUDA/Triton có sẵn, nhóm không cần tự lập trình hay viết lại các kernel này từ đầu.
    *   **Yêu cầu về máy chủ (Server):** Vì vLLM và TurboQuant sử dụng các nhân tính toán (kernels) viết bằng Triton/CUDA tối ưu cho GPU, **hệ thống bắt buộc phải chạy trên GPU NVIDIA** (kiến trúc Ampere hoặc Ada Lovelace trở lên như RTX 30/40 series, A10, L4, A100). Nếu server của Việt Anh chỉ chạy chủ yếu bằng CPU, hệ thống vLLM sẽ không thể khởi chạy hoặc không đạt hiệu năng thực tế.
    *   **Giải pháp tối ưu chi phí:** Nhóm không cần mua phần cứng vật lý hay đầu tư server đắt đỏ. Bạn có thể hướng dẫn team Technical thuê GPU đám mây (Cloud GPU) trên các nền tảng như **Vast.ai, RunPod, hoặc Lambda Labs**.
        *   Thuê 1 GPU RTX 3090 hoặc RTX 4090 (24GB VRAM - hoàn toàn đủ để chạy các mô hình 7B-8B với context 16k-32k) chỉ mất khoảng **$0.20 - $0.40 / giờ** (khoảng 5.000 - 10.000 VNĐ/giờ).
        *   Tổng thời gian chạy thực nghiệm đo đạc của nhóm dự kiến chỉ mất từ 15 - 25 giờ máy chạy liên tục. Tổng chi phí phần cứng cho cả dự án sẽ chỉ rơi vào khoảng **$5 - $10 (tương đương 130.000 - 250.000 VNĐ)**. Đây là mức chi phí cực kỳ thấp và hoàn toàn khả thi cho một nhóm sinh viên.

---

### 2. Liệt kê các Metrics cụ thể cần đo đạc

Để bài báo khoa học của nhóm đạt tiêu chuẩn cao (hướng tới phân khúc Q1 như thầy mong đợi), các chỉ số đo đạc cần được phân rã chi tiết và chuẩn hóa theo đúng quy chuẩn nghiên cứu hệ thống (System Research) thay vì chỉ liệt kê chung chung:

1.  **Về hiệu năng bộ nhớ (Memory Footprint):**
    *   **Peak VRAM (GB) during Prefill Phase:** Bộ nhớ đỉnh khi xử lý prompt đầu vào (ngữ cảnh dài).
    *   **Peak VRAM (GB) during Decode Phase:** Bộ nhớ đỉnh trong quá trình sinh text (nơi KV cache phình to).
    *   **KV Cache Size (MB):** Dung lượng bộ nhớ thực tế bị chiếm dụng bởi KV cache tương ứng với các độ dài chuỗi (4k, 8k, 16k, 32k).
    *   **KV Cache Compression Ratio (Tỷ lệ nén - %):** Tỷ số dung lượng KV Cache sau nén so với dung lượng gốc (BF16).
    *   **Base VRAM vs. Dynamic VRAM:** Phân tách bộ nhớ nạp mô hình tĩnh (Base Model Memory) và bộ nhớ chạy động (KV Cache, Activation) để làm nổi bật tác dụng của nén KV Cache.
2.  **Về tốc độ suy luận (Inference Speed):**
    *   **Time to First Token (TTFT - mili-giây):** Thời gian phản hồi token đầu tiên (đo hiệu năng của Prefill).
    *   **Inter-Token Latency (ITL - mili-giây/token):** Thời gian trung bình để sinh ra các token tiếp theo (đo hiệu năng của Decode).
    *   **Generation Throughput (tokens/giây):** Đo ở các mức độ tải khác nhau (Batch Size = 1 cho ứng dụng đơn lẻ và Batch Size = 4/8/16 cho server chịu tải).
    *   **GPU Memory Efficiency Index (Tokens/s/MB):** Tỷ lệ throughput giải mã sinh ra trên mỗi MB bộ nhớ động VRAM được cấp phát, đo độ hiệu dụng của phần cứng.
3.  **Về chất lượng ngôn ngữ (Output Quality):**
    *   **Perplexity (PPL):** Đo trên tập dữ liệu tiếng Việt chuẩn (như WikiText-vi hoặc ViWiki) để đánh giá mức độ suy giảm khả năng hiểu ngôn ngữ sau khi nén.
    *   **Task-specific Metrics (Đo theo tác vụ):** Điểm số **F1** và **Exact Match (EM)** trên bộ dữ liệu Vi-SQuAD (Hỏi đáp), hoặc điểm **ROUGE-L** cho tác vụ tóm tắt văn bản dài tiếng Việt.

---

### 3. Danh sách các Vietnamese LLMs & Tập Dữ liệu đã được kiểm chứng (Để đối chiếu)

#### 3.1. Các mô hình tiếng Việt xuất sắc
1.  **`Qwen/Qwen2.5-7B-Instruct-1M` (hoặc bản tinh chỉnh tiếng Việt `Qwen2.5-7B-Instruct-vietnamese`):**
    *   *Lý do:* Qwen2.5 là dòng mô hình mã nguồn mở hàng đầu hiện nay, hỗ trợ tiếng Việt cực kỳ tốt và có cửa sổ ngữ cảnh hỗ trợ lên tới 128k tokens. Đây là mô hình lý tưởng để đo đạc sự thay đổi khi kéo dài ngữ cảnh từ 4k lên 32k.
2.  **`qwen3:8b`:**
    *   *Lý do:* Đã được kiểm chứng đạt điểm số rất cao trên benchmark VMLU tiếng Việt. Hỗ trợ ngữ cảnh dài mặc định lên tới 128k tokens nhờ kiến trúc RoPE điều chỉnh.
3.  **`arcee-ai/Arcee-VyLinh`:**
    *   *Lý do:* Bản thích ứng tiếng Việt dựa trên Mistral-7B, văn phong tiếng Việt rất mượt mà. Mặc dù giới hạn ngữ cảnh gốc là 8k, mô hình này rất thích hợp để làm đối chứng xem khi nén vượt ngưỡng thì chất lượng suy giảm ra sao.
4.  **`llama3.2:3b` (3 Billion parameters):**
    *   *Lý do:* Mô hình nhẹ 3B tham số từ Meta, dựa trên kiến trúc Llama 3.2. Được dùng làm baseline compact để kiểm tra hiệu quả nén KV Cache trên mô hình nhỏ, phù hợp cho môi trường tài nguyên hạn chế.

#### 3.2. Tập dữ liệu cào bổ sung & Nguồn đóng gói sẵn (News & Social Media)
Nhóm sẽ thu thập thêm dữ liệu sách, báo và mạng xã hội tiếng Việt có tính thời sự nóng bằng hai con đường phối hợp:
1.  **Viết Scraper mẫu cào từ báo điện tử:** Cào trực tiếp các bài báo thời sự nóng từ VnExpress / Tuổi Trẻ (sử dụng requests kết hợp BeautifulSoup) để tạo kho ngữ cảnh thời sự.
2.  **Tích hợp các nguồn đóng gói sẵn có tính thời sự cao:**
    *   `vietnews` (Hugging Face Datasets): Dataset tóm tắt tin tức tiếng Việt chất lượng cao chứa hàng ngàn bài viết từ các trang báo lớn.
    *   `binhvq/news-corpus` (GitHub/HF): Corpus báo chí tiếng Việt lớn, đa dạng chủ đề từ 2020-2026.
    *   `UIT-VSFC` & `nhanvtp/vietnamese-social-media-sentiment` (HF): Các bài viết mạng xã hội tiếng Việt thực tế, bình luận nóng về đời sống xã hội.
    *   `wikimedia/wikipedia` (vi subset): Phiên bản tiếng Việt mới nhất của Wikipedia chứa các sự kiện thời sự cập nhật.

---

### 4. Xây dựng mạch xuyên suốt cho bài báo (Coherent Paper Thread)

Để bài báo có tính thuyết phục cao đối với các phản biện (reviewers) của tạp chí lớn, nhóm nên đi theo cấu trúc mạch lập luận (Storyline) sau:

*   **Đặt vấn đề (Introduction):** Việc xử lý ngữ cảnh dài bằng LLM tiếng Việt đang bị nghẽn nghiêm trọng ở bộ nhớ KV cache, đặc biệt trên các phần cứng biên/phổ thông của doanh nghiệp nhỏ.
*   **Khoảng trống nghiên cứu (Research Gap):** Các phương pháp nén tiên tiến (như TurboQuant) được quảng cáo là giữ nguyên chất lượng trên tiếng Anh, nhưng tiếng Việt là ngôn ngữ đơn âm tiết có dấu, cấu trúc token hóa (byte-level BPE) rất khác biệt. Chưa có nhiều công trình đánh giá sự suy giảm chất lượng này một cách hệ thống trên tiếng Việt.
*   **Đóng góp (Contributions):**
    1.  Xây dựng bộ benchmark thực nghiệm đầu tiên đánh giá TurboQuant và các baseline nén KV cache trên các LLM tiếng Việt phổ biến.
    2.  Chỉ ra ranh giới Pareto tối ưu (đường cong đánh đổi giữa dung lượng bộ nhớ tiết kiệm được và độ suy giảm chất lượng ngôn ngữ thực tế).
    3.  Đưa ra khuyến nghị cấu hình nén tối ưu (ví dụ: dùng mốc bit nào, phương pháp nào) cho doanh nghiệp khi triển khai LLM tiếng Việt trong thực tế.

# ANSWER LECTURE

### 1. Giải quyết vấn đề Phần cứng & Chi phí (Lo ngại lớn nhất của nhóm)

*   **Ý kiến của thầy & nhóm:** *"Vấn đề về phần cứng để triển khai training model và áp dụng turboquant... Không biết server của Việt Anh chạy nổi không... Chạy CPU hay GPU?"*
*   **Định hướng kỹ thuật thực tế:**
    *   **Lưu ý quan trọng:** **Nhóm không cần phải tiến hành huấn luyện (Training) hay tinh chỉnh (Fine-tuning) lại mô hình.** TurboQuant, HQQ, hay PolarQuant đều là các kỹ thuật **Post-Training Quantization (PTQ)** [2]. Quy trình này chỉ thực hiện nén trực tiếp trên trọng số hoặc KV cache của mô hình đã có sẵn khi chạy suy luận (Inference) [1, 2]. Do đó, **chi phí tính toán gần như bằng 0 so với việc training.**
    *   **Yêu cầu về máy chủ (Server):** Vì vLLM và TurboQuant sử dụng các nhân tính toán (kernels) viết bằng Triton/CUDA tối ưu cho GPU, **hệ thống bắt buộc phải chạy trên GPU NVIDIA** (kiến trúc Ampere hoặc Ada Lovelace trở lên như RTX 30/40 series, A10, L4, A100). Nếu server của Việt Anh chỉ chạy chủ yếu bằng CPU, hệ thống vLLM sẽ không thể khởi chạy hoặc không đạt hiệu năng thực tế.
    *   **Giải pháp tối ưu chi phí:** Nhóm không cần mua phần cứng vật lý hay đầu tư server đắt đỏ. Bạn có thể hướng dẫn team Technical thuê GPU đám mây (Cloud GPU) trên các nền tảng như **Vast.ai, RunPod, hoặc Lambda Labs**.
        *   Thuê 1 GPU RTX 3090 hoặc RTX 4090 (24GB VRAM - hoàn toàn đủ để chạy các mô hình 7B-8B với context 16k-32k) chỉ mất khoảng **$0.20 - $0.40 / giờ** (khoảng 5.000 - 10.000 VNĐ/giờ).
        *   Tổng thời gian chạy thực nghiệm đo đạc của nhóm dự kiến chỉ mất từ 15 - 25 giờ máy chạy liên tục. Tổng chi phí phần cứng cho cả dự án sẽ chỉ rơi vào khoảng **$5 - $10 (tương đương 130.000 - 250.000 VNĐ)**. Đây là mức chi phí cực kỳ thấp và hoàn toàn khả thi cho một nhóm sinh viên.

---

### 2. Liệt kê các Metrics cụ thể cần đo đạc (Đáp ứng yêu cầu Q1)

Để bài báo khoa học của nhóm đạt tiêu chuẩn cao (hướng tới phân khúc Q1 như thầy mong đợi), các chỉ số đo đạc cần được phân rã chi tiết và chuẩn hóa theo đúng quy chuẩn nghiên cứu hệ thống (System Research) thay vì chỉ liệt kê chung chung:

1.  **Về hiệu năng bộ nhớ (Memory Footprint):**
    *   **Peak VRAM (GB) during Prefill Phase:** Bộ nhớ đỉnh khi xử lý prompt đầu vào (ngữ cảnh dài).
    *   **Peak VRAM (GB) during Decode Phase:** Bộ nhớ đỉnh trong quá trình sinh text (nơi KV cache phình to).
    *   **KV Cache Size (MB):** Dung lượng bộ nhớ thực tế bị chiếm dụng bởi KV cache tương ứng với các độ dài chuỗi (4k, 8k, 16k, 32k).
2.  **Về tốc độ suy luận (Inference Speed):**
    *   **Time to First Token (TTFT - mili-giây):** Thời gian phản hồi token đầu tiên (đo hiệu năng của Prefill).
    *   **Inter-Token Latency (ITL - mili-giây/token):** Thời gian trung bình để sinh ra các token tiếp theo (đo hiệu năng của Decode).
    *   **Generation Throughput (tokens/giây):** Đo ở các mức độ tải khác nhau (Batch Size = 1 cho ứng dụng đơn lẻ và Batch Size = 4/8/16 cho server chịu tải).
3.  **Về chất lượng ngôn ngữ (Output Quality):**
    *   **Perplexity (PPL):** Đo trên tập dữ liệu tiếng Việt chuẩn (như WikiText-vi hoặc ViWiki) để đánh giá mức độ suy giảm khả năng hiểu ngôn ngữ sau khi nén.
    *   **Task-specific Metrics (Đo theo tác vụ):** Điểm số **F1** và **Exact Match (EM)** trên bộ dữ liệu Vi-SQuAD (Hỏi đáp), hoặc điểm **ROUGE-L** cho tác vụ tóm tắt văn bản dài tiếng Việt.

---

### 3. Danh sách các Vietnamese LLMs đã được kiểm chứng (Để đối chiếu)

Thầy yêu cầu tìm kiếm các mô hình đã được kiểm chứng hoạt động tốt trong môi trường tiếng Việt để đối chiếu. Dưới đây là 4 ứng viên xuất sắc nhất, có hỗ trợ ngữ cảnh dài và đã được cộng đồng nghiên cứu kiểm chứng trên bảng xếp hạng **VMLU** (Zalo AI công bố):

1.  **`vinai/PhoGPT-7B5-Instruct` (7.5 Billion parameters):**
    *   *Lý do:* Mô hình thuần Việt do VinAI phát triển, được huấn luyện từ đầu trên 102 tỷ token tiếng Việt. Sử dụng cơ chế ALiBi cho phép ngoại suy độ dài ngữ cảnh tốt. Đây là baseline thuần Việt không thể thiếu.
2.  **`Qwen/Qwen2.5-7B-Instruct` (hoặc bản tinh chỉnh tiếng Việt `Qwen2.5-7B-Instruct-vietnamese`):**
    *   *Lý do:* Qwen2.5 là dòng mô hình mã nguồn mở hàng đầu hiện nay, hỗ trợ tiếng Việt cực kỳ tốt và có cửa sổ ngữ cảnh hỗ trợ lên tới 128k tokens. Đây là mô hình lý tưởng để đo đạc sự thay đổi khi kéo dài ngữ cảnh từ 4k lên 32k.
3.  **`meta-llama/Meta-Llama-3.1-8B-Instruct`:**
    *   *Lý do:* Đã được kiểm chứng đạt điểm số rất cao trên benchmark VMLU tiếng Việt. Hỗ trợ ngữ cảnh dài mặc định lên tới 128k tokens nhờ kiến trúc RoPE điều chỉnh.
4.  **`Viet-Mistral/Vistral-7B-Chat`:**
    *   *Lý do:* Bản thích ứng tiếng Việt dựa trên Mistral-7B, văn phong tiếng Việt rất mượt mà. Mặc dù giới hạn ngữ cảnh gốc là 8k, mô hình này rất thích hợp để làm đối chứng xem khi nén vượt ngưỡng thì chất lượng suy giảm ra sao.

---

### 4. Xây dựng mạch xuyên suốt cho bài báo (Coherent Paper Thread)

Để bài báo có tính thuyết phục cao đối với các phản biện (reviewers) của tạp chí lớn, nhóm nên đi theo cấu trúc mạch lập luận (Storyline) sau:

*   **Đặt vấn đề (Introduction):** Việc xử lý ngữ cảnh dài bằng LLM tiếng Việt đang bị nghẽn nghiêm trọng ở bộ nhớ KV cache, đặc biệt trên các phần cứng biên/phổ thông của doanh nghiệp nhỏ.
*   **Khoảng trống nghiên cứu (Research Gap):** Các phương pháp nén tiên tiến (như TurboQuant) được quảng cáo là giữ nguyên chất lượng trên tiếng Anh, nhưng tiếng Việt là ngôn ngữ đơn âm tiết có dấu, cấu trúc token hóa (byte-level BPE) rất khác biệt. Chưa có công trình nào đánh giá sự suy giảm chất lượng này một cách hệ thống trên tiếng Việt.
*   **Đóng góp (Contributions):**
    1.  Xây dựng bộ benchmark thực nghiệm đầu tiên đánh giá TurboQuant và các baseline nén KV cache trên các LLM tiếng Việt phổ biến.
    2.  Chỉ ra ranh giới Pareto tối ưu (đường cong đánh đổi giữa dung lượng bộ nhớ tiết kiệm được và độ suy giảm chất lượng ngôn ngữ thực tế) [1].
    3.  Đưa ra khuyến nghị cấu hình nén tối ưu (ví dụ: dùng mốc bit nào, phương pháp nào) cho doanh nghiệp khi triển khai LLM tiếng Việt trong thực tế [1].

---

### 5. Hành động tiếp theo cho Hưng (Phân rã task trên `plane.so`)

Với vai trò PM, bạn có thể tạo ngay các task sau lên hệ thống để giải tỏa áp lực hạ tầng cho cả nhóm:

1.  **Task cho Việt Anh (Hạ tầng):** Tìm hiểu cách đăng ký tài khoản và khởi tạo một GPU RTX 4090/3090 trên RunPod hoặc Vast.ai, nạp trước thử $5 - $10 (khoảng 125k - 250k VNĐ) để làm quỹ chạy thử nghiệm.
2.  **Task cho Minh Quân (Tech):** Cài đặt môi trường vLLM trên GPU đám mây vừa tạo và chạy thử nghiệm đo TTFT/ITL với mô hình `PhoGPT-7B5-Instruct` sử dụng tham số `--kv-cache-dtype FP8` (chạy thử baseline trước).
3.  **Task cho Quốc Anh (Research):** Soạn thảo phần đề cương chi tiết của bài báo (Draft Paper Outline) theo mạch lập luận đã thống nhất phía trên [2].
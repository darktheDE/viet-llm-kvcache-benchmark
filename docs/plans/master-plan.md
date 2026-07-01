# BẢN KẾ HOẠCH TỔNG THỂ (MASTER PLAN)
## Dự án: Benchmarking TurboQuant and KV Cache Compression Methods on Vietnamese LLMs
**Môn học:** Ứng dụng dữ liệu lớn: học máy ở quy mô lớn (DBML434077) - HCM-UTE  
**Quản lý tiến độ:** Đỗ Kiến Hưng (Writing & Coordination Lead / Project Manager)

---

## I. MỤC TIÊU CHI CHIẾT & ĐỊNH NGHĨA HOÀN THÀNH (DEFINITION OF DONE - DoD)

Dự án cam kết bàn giao đầy đủ **4 sản phẩm cốt lõi** vào cuối Tuần 7:
1.  **Mã nguồn thực nghiệm (GitHub):** Mã nguồn Python đóng gói sạch, chạy thử nghiệm benchmark tự động thông qua script `run_baseline.py`, có tài liệu hướng dẫn tái lập (`README.md` hoàn chỉnh).
2.  **Báo cáo chi tiết tiếng Việt (Word Document):** Độ dài $\ge 30$ trang, mô tả chi tiết phương pháp luận, kiến trúc, kết quả thử nghiệm và phân tích thực nghiệm.
3.  **Slide thuyết trình bảo vệ (PPTX):** Độ dài 15–25 trang, tóm tắt trực quan các phát hiện chính.
4.  **Bài báo khoa học tiếng Anh (Research Paper):** Độ dài $\ge 6$ trang theo chuẩn cấu trúc IEEE/Springer (hướng tới chuẩn Scopus/WoS).

---

## II. LỘ TRÌNH PHÁT TRIỂN & CÁC KHÚC NGHỊ (TIMELINE & SPRINTS)

Tổng tiến độ dự án kéo dài **7 tuần**, chia làm **4 Phase** chạy theo dạng Sprints (mỗi Sprint 1–2 tuần):

```text
  [Tuần 1-2: Phase 1] -------------> [Tuần 3: Phase 2A] -------------> [Tuần 4-5: Phase 2B]
  Khởi động, Setup & Prep            Baseline Pipeline                 Main Benchmarking
         |                                                                    |
         v                                              ### PHASE 1: KHỞI ĐỘNG, SETUP MÔI TRƯỜNG & CHUẨN BỊ DỮ LIỆU (Tuần 1 - Tuần 2)
*   **Mục tiêu:** Thống nhất đề cương, cài đặt môi trường Cloud GPU, chuẩn bị, cào bổ sung và làm sạch dữ liệu, nghiên cứu tích hợp sớm thuật toán nén.
*   **Nhiệm vụ chi tiết:**
    *   *Quản lý & Điều phối (Kiến Hưng, Việt Anh):* Thiết lập không gian làm việc chung (Drive, Notion, Plane.so), tạo template biên bản họp `kickoff/minutes_template.md`. Lên kế hoạch báo cáo tiến độ hàng tuần vào **Thứ 4** cho giảng viên hướng dẫn.
    *   *Nghiên cứu (Quốc Anh, Trọng Phú):* Hoàn thiện đề cương chi tiết bằng tiếng Anh, viết định nghĩa bài toán (`problem_definition.md`).
    *   *Kỹ thuật (Việt Anh, Minh Quân, Minh Khánh, Quang Duy):* Tạo tài khoản và cài đặt môi trường trên Cloud GPU đã thuê; thiết lập môi trường Conda với CUDA 12.x, PyTorch và vLLM; nghiên cứu tích hợp và chạy thử sớm các nhân thuật toán nén (TurboQuant, PolarQuant, HQQ, FP8) để phát hiện sớm các lỗi xung đột compiler.
    *   *Dữ liệu (Hồ Phát, Hữu Huy, Ngọc Thạch):* Tải các bộ dữ liệu `VMLU`, `VTSNLP`; thực hiện **cào thêm dữ liệu thực tế (sách, báo, mạng xã hội thời sự nóng)**; áp dụng công cụ `NVIDIA NeMo Curator` để chuẩn bị dữ liệu tiếng Việt sạch; thiết kế và đóng gói bộ test-set nhỏ (~10-20 mẫu ngữ cảnh dài) tại file `datasets/test_set_small.json`.

### PHASE 2A: THIẾT LẬP PIPELINE CHUẨN & CHẠY THỬ baseline (Tuần 3)
*   **Mục tiêu:** Xây dựng script đo đạc tự động tích hợp các metrics bổ sung nâng cao và chạy kiểm thử thành công trên 1 mô hình nền tảng BF16 (Full KV Cache).
*   **Nhiệm vụ chi tiết:**
    *   *Kỹ thuật (Minh Quân, Minh Khánh, Quang Duy):* Hoàn thiện script đo đạc `scripts/run_baseline.py` ghi nhận tự động các chỉ số: Peak VRAM (Prefill/Decode), TTFT, ITL, Throughput, Perplexity, và các chỉ số bổ sung (**KV Cache Compression Ratio, GPU Memory Efficiency Index, Base vs Dynamic VRAM**).
    *   *Điều phối & Quản lý (Kiến Hưng):* Tạo trước biểu mẫu CSV ghi nhận kết quả `results/template_log.csv` để thống nhất cấu trúc lưu trữ số liệu.
    *   *Dữ liệu (Hồ Phát, Hữu Huy):* Viết tài liệu hướng dẫn sử dụng dữ liệu `datasets/dataset_brief.md` để bàn giao cho team kỹ thuật.

### PHASE 2B: CHẠY THỰC NGHIỆM ĐỒNG LOẠT (Tuần 4 - Tuần 5)
*   **Mục tiêu:** Thu thập toàn bộ số liệu thực nghiệm trên 4 dòng mô hình mục tiêu đối với tất cả các mốc thuật toán nén trên Cloud GPU.
*   **Nhiệm vụ chi tiết:**
    *   *Kỹ thuật & Thử nghiệm (Nhóm phân vai 1-2 mô hình/người):*
        *   **Cặp 1 (Việt Anh, Minh Quân):** Chạy thực nghiệm trên mô hình `VinaLLaMA-7B-Chat`.
        *   **Cặp 2 (Minh Khánh, Quang Duy):** Chạy thực nghiệm trên mô hình `Qwen2.5-7B-Instruct`.
        *   **Hỗ trợ chung (Duy - Infrastructure):** Đảm nhận vai trò xử lý lỗi (bug-fixing), tối ưu hóa bộ nhớ và giám sát việc chạy thử nghiệm các mốc nén KV cache: FP8, HQQ, PolarQuant, TurboQuant (mốc Full) và TurboQuant (mốc tắt QJL) trên Cloud GPU.
    *   *Đo đạc & Ghi kết quả (Toàn bộ team kỹ thuật):* Ghi nhận số liệu chính xác (gồm cả các metrics nâng cao) vào các file kết quả CSV riêng biệt của từng mô hình theo đúng template chuẩn.

### PHASE 3: TỔNG HỢP SỐ LIỆU & VẼ BIỂU ĐỒ TRỰC QUAN (Tuần 6)
*   **Mục tiêu:** Merge dữ liệu, thực hiện tính toán thống kê và vẽ các đường cong đánh đổi Pareto tối ưu bao gồm cả các metrics bổ sung.
*   **Nhiệm vụ chi tiết:**
    *   *Phân tích (Ngọc Thạch - Phân tích chính, Hồ Phát, Hữu Huy):* Viết script Python `scripts/plot_results.py` để đọc dữ liệu từ các file CSV, tính toán trung bình/độ lệch chuẩn.
    *   *Trực quan hóa (Ngọc Thạch):* Xuất ra các biểu đồ trực quan lưu vào thư mục `results/plots/`:
        *   Biểu đồ 1: Memory (VRAM) vs. Context Length.
        *   Biểu đồ 2: Latency/Throughput vs. Context Length.
        *   Biểu đồ 3: Perplexity vs. Compression Methods (Trade-off Pareto curves).
        *   Biểu đồ 4: **KV Cache Compression Ratio & GPU Memory Efficiency vs. Context Length**.
    *   *Nghiên cứu (Quốc Anh, Trọng Phú):* Phân tích sự suy giảm chất lượng và hiệu suất bộ nhớ dựa trên các chỉ số bổ sung, rút ra nhận xét về sự nhạy cảm của tiếng Việt khi nén sâu.

### PHASE 4: HOÀN THIỆN VĂN BẢN, VIẾT BÁO CÁO & PAPER (Tuần 7)
*   **Mục tiêu:** Hoàn thiện 100% tài liệu bàn giao (có chèn số liệu về dữ liệu thời sự cào mới và chỉ số mới), chạy thử slide thuyết trình và chuẩn bị bảo vệ.
*   **Nhiệm vụ chi tiết:**
    *   *Điều phối & Viết báo cáo (Kiến Hưng, Việt Anh):* Tổng hợp nội dung từ các nhóm để hoàn thiện tài liệu Báo cáo tiếng Việt $\ge 30$ trang.
    *   *Viết bài báo khoa học (Quốc Anh, Trọng Phú, Kiến Hưng):* Hoàn thiện bản thảo bài báo tiếng Anh $\ge 6$ trang trên Overleaf theo cấu trúc IEEE.
    *   *Thiết kế Slide (Cả nhóm):* Thiết kế bộ slide báo cáo súc tích từ 15-25 trang trình bày rõ ràng kiến trúc hệ thống, phương pháp và biểu đồ trade-off thu được.
    *   *Tập dượt phản biện (Kiến Hưng chủ trì):* Chạy thử slide báo cáo trước nhóm, chuẩn bị kịch bản trả lời các câu hỏi phản biện tiềm năng.

---

## III. BẢNG PHÂN BỔ NHÂN SỰ CHI TIẾT (RACI MATRIX)

| Thành viên | Vai trò trong dự án | Phase 1 (Prep) | Phase 2 (Exp) | Phase 3 (Analysis) | Phase 4 (Writing) |
|---|---|:---:|:---:|:---:|:---:|
| **Đỗ Kiến Hưng** | PM / Agile Coordinator | **R** / **A** | **A** | **A** | **R** / **A** |
| **Phan Trọng Quí** | Joint Coordinator / Writing | **R** | **C** | **C** | **R** (Báo cáo) |
| **Hồ Việt Anh** | Technical & Experiment | **R** (Cloud Setup) | **R** (VinaLLaMA) | **C** | **C** |
| **Phạm Minh Quân** | Tech Lead / Optimizer | **R** (CUDA setup) | **R** (VinaLLaMA) | **C** | **C** |
| **Trần Minh Khánh** | Tech runner | **C** | **R** (Qwen) | **C** | **C** |
| **Nguyễn V. Q. Duy** | Quantization / Patching | **C** (vLLM config) | **R** (Qwen/Debug) | **C** | **C** |
| **Nguyễn Hồ Phát** | Data curator | **R** (NeMo Curator) | **A** | **R** | **C** |
| **Huỳnh Hữu Huy** | Data / Prompts | **R** (Test suite) | **A** | **R** | **C** |
| **Huỳnh Ngọc Thạch** | Metric plotting lead | **C** | **A** | **R** (Plotting) | **C** |
| **N. Đăng Quốc Anh** | Research Owner / Idea lead | **R** (Đề cương) | **C** | **R** (Nhận xét) | **R** (Paper EN) |
| **Phan Trọng Phú** | Paper Writer / Review | **R** (References) | **C** | **C** | **R** (Paper EN) |

*Chú thích:* **R** (Responsible - Người thực hiện), **A** (Accountable - Người chịu trách nhiệm/PM), **C** (Consulted - Người hỗ trợ/Tham vấn), **I** (Informed - Người nhận thông tin).

## IV. BẢN KẾ HOẠCH QUẢN TRỊ RỦI RO (RISK MANAGEMENT PLAN)

| Loại rủi ro | Chi tiết rủi ro | Mức độ | Giải pháp phòng ngừa & Khắc phục |
|---|---|:---:|---|
| **Rủi ro Phần cứng** | Server local không có GPU hoặc GPU yếu, không chạy được nhân Triton/CUDA của vLLM. | **Cao** | Sử dụng ngân sách nhóm thuê Cloud GPU (Vast.ai, RunPod) như đã thống nhất. |
| **Rủi ro Lỗi mã nguồn** | vLLM báo lỗi không tương thích phiên bản CUDA hoặc lỗi Out-of-Memory (OOM) khi kéo dài context. | **Trung bình** | - Chỉ định Duy làm "Help desk" xử lý lỗi kỹ thuật.<br>- Thêm tham số `--max-num-batched-tokens 4096` khi chạy và bật PagedAttention. |
| **Rủi ro Khoa học** | Dữ liệu nén thử nghiệm trên tiếng Việt bị lỗi hiển thị phông chữ Unicode sau khi tokenize. | **Trung bình** | Team Data (Phát, Huy) bắt buộc phải kiểm tra kỹ bộ test-set bằng cách giải mã ngược (detokenize) thử 1-2 mẫu trước khi chạy benchmark đại trà. |
| **Rủi ro Tiến độ** | Viết báo cáo tiếng Việt và bài báo tiếng Anh song song bị quá tải vào Tuần 7. | **Cao** | - Kiến Hưng giám sát chặt chẽ tiến độ trên `plane.so`.<br>- Khởi tạo khung tài liệu Overleaf ngay từ Tuần 1 và cập nhật dần kết quả thay vì dồn vào tuần cuối. |
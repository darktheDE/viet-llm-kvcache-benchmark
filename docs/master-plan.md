# BẢN KẾ HOẠCH TỔNG THỂ (MASTER PLAN)
## Dự án: Benchmarking TurboQuant and KV Cache Compression Methods on Vietnamese LLMs
**Môn học:** Ứng dụng dữ liệu lớn: học máy ở quy mô lớn (DBML434077) - HCM-UTE [3]  
**Quản lý tiến độ:** Đỗ Kiến Hưng (Writing & Coordination Lead / Project Manager)

---

## I. MỤC TIÊU CHI CHIẾT & ĐỊNH NGHĨA HOÀN THÀNH (DEFINITION OF DONE - DoD)

Dự án cam kết bàn giao đầy đủ **4 sản phẩm cốt lõi** vào cuối Tuần 7 [2, 3]:
1.  **Mã nguồn thực nghiệm (GitHub):** Mã nguồn Python đóng gói sạch, chạy thử nghiệm benchmark tự động thông qua script `run_baseline.py` [1, 2], có tài liệu hướng dẫn tái lập (`README.md` hoàn chỉnh).
2.  **Báo cáo chi tiết tiếng Việt (Word Document):** Độ dài $\ge 30$ trang, mô tả chi tiết phương pháp luận, kiến trúc, kết quả thử nghiệm và phân tích thực nghiệm [3].
3.  **Slide thuyết trình bảo vệ (PPTX):** Độ dài 15–25 trang, tóm tắt trực quan các phát hiện chính [3].
4.  **Bài báo khoa học tiếng Anh (Research Paper):** Độ dài $\ge 6$ trang theo chuẩn cấu trúc IEEE/Springer (hướng tới chuẩn Scopus/WoS) [3].

---

## II. LỘ TRÌNH PHÁT TRIỂN & CÁC KHÚC NGHỊ (TIMELINE & SPRINTS)

Tổng tiến độ dự án kéo dài **7 tuần**, chia làm **4 Phase** chạy theo dạng Sprints (mỗi Sprint 1–2 tuần) [2]:

```text
  [Tuần 1-2: Phase 1] -------------> [Tuần 3: Phase 2A] -------------> [Tuần 4-5: Phase 2B]
  Khởi động, Setup & Prep            Baseline Pipeline                 Main Benchmarking
         |                                                                    |
         v                                                                    v
  [Tuần 6: Phase 3] <---------------------------------------------------------+
  Data Synthesis & Plotting
         |
         v
  [Tuần 7: Phase 4] --------------> BÀN GIAO & BẢO VỆ CUỐI KỲ
  Paper Writing & Polish
```

### PHASE 1: KHỞI ĐỘNG, SETUP MÔI TRƯỜNG & CHUẨN BỊ DỮ LIỆU (Tuần 1 - Tuần 2) [2]
*   **Mục tiêu:** Thống nhất đề cương [1, 2], cài đặt môi trường Cloud GPU [2], chuẩn bị và làm sạch dữ liệu [1, 2].
*   **Nhiệm vụ chi tiết:**
    *   *Quản lý & Điều phối (Kiến Hưng, Việt Anh):* Thiết lập không gian làm việc chung (Drive, Notion, Plane.so), tạo template biên bản họp `kickoff/minutes_template.md` [1, 2].
    *   *Nghiên cứu (Quốc Anh, Trọng Phú):* Hoàn thiện đề cương chi tiết bằng tiếng Anh [1, 2], viết định nghĩa bài toán (`problem_definition.md`) [1].
    *   *Kỹ thuật (Việt Anh, Minh Quân, Minh Khánh, Quang Duy):* Tạo tài khoản trên nền tảng Cloud GPU (Vast.ai hoặc RunPod) [2]; thiết lập môi trường Conda với CUDA 12.x, PyTorch và vLLM [1, 2]; cấu hình cấu trúc thư mục chuẩn của Git [1, 2].
    *   *Dữ liệu (Hồ Phát, Hữu Huy, Ngọc Thạch):* Tải các bộ dữ liệu `VMLU`, `VTSNLP` [1, 2]; áp dụng công cụ `NVIDIA NeMo Curator` để chuẩn bị dữ liệu tiếng Việt sạch [1, 2]; thiết kế và đóng gói bộ test-set nhỏ (~10-20 mẫu ngữ cảnh dài) tại file `datasets/test_set_small.json` [1, 2].

### PHASE 2A: THIẾT LẬP PIPELINE CHUẨN & CHẠY THỬ baseline (Tuần 3) [2]
*   **Mục tiêu:** Xây dựng script đo đạc tự động [1, 2] và chạy kiểm thử thành công trên 1 mô hình nền tảng BF16 (Full KV Cache) [1, 2].
*   **Nhiệm vụ chi tiết:**
    *   *Kỹ thuật (Minh Quân, Minh Khánh, Quang Duy):* Hoàn thiện script đo đạc `scripts/run_baseline.py` ghi nhận tự động các chỉ số [1, 2]: Peak VRAM (Prefill/Decode), TTFT, ITL, Throughput và Perplexity [1, 2].
    *   *Điều phối & Quản lý (Kiến Hưng):* Tạo trước biểu mẫu CSV ghi nhận kết quả `results/template_log.csv` để thống nhất cấu trúc lưu trữ số liệu [1, 2].
    *   *Dữ liệu (Hồ Phát, Hữu Huy):* Viết tài liệu hướng dẫn sử dụng dữ liệu `datasets/dataset_brief.md` để bàn giao cho team kỹ thuật [1, 2].

### PHASE 2B: CHẠY THỰC NGHIỆM ĐỒNG LOẠT (Tuần 4 - Tuần 5) [2]
*   **Mục tiêu:** Thu thập toàn bộ số liệu thực nghiệm trên 4 dòng mô hình mục tiêu [1, 2].
*   **Nhiệm vụ chi tiết:**
    *   *Kỹ thuật & Thử nghiệm (Nhóm phân vai 1-2 mô hình/người):*
        *   **Cặp 1 (Việt Anh, Minh Quân):** Chạy thực nghiệm trên mô hình `PhoGPT-7B5-Instruct` [2].
        *   **Cặp 2 (Minh Khánh, Quang Duy):** Chạy thực nghiệm trên mô hình `Qwen2.5-7B-Instruct` [2].
        *   **Hỗ trợ chung (Duy - Infrastructure):** Đảm nhận vai trò xử lý lỗi (bug-fixing), tối ưu hóa bộ nhớ và giám sát việc chạy thử nghiệm các mốc nén KV cache [2]: FP8, HQQ, PolarQuant, TurboQuant (mốc Full) và TurboQuant (mốc tắt QJL) [1, 2].
    *   *Đo đạc & Ghi kết quả (Toàn bộ team kỹ thuật):* Ghi nhận số liệu chính xác vào các file kết quả CSV riêng biệt của từng mô hình theo đúng template chuẩn [2].

### PHASE 3: TỔNG HỢP SỐ LIỆU & VẼ BIỂU ĐỒ TRỰC QUAN (Tuần 6) [2]
*   **Mục tiêu:** Merge dữ liệu [2], thực hiện tính toán thống kê và vẽ các đường cong đánh đổi Pareto tối ưu [2].
*   **Nhiệm vụ chi tiết:**
    *   *Phân tích (Ngọc Thạch - Phân tích chính, Hồ Phát, Hữu Huy):* Viết script Python `scripts/plot_results.py` sử dụng pandas và matplotlib/plotly để đọc dữ liệu từ các file CSV [2]; tổng hợp trung bình/độ lệch chuẩn [2].
    *   *Trực quan hóa (Ngọc Thạch):* Xuất ra các biểu đồ trực quan lưu vào thư mục `results/plots/` [1, 2]:
        *   Biểu đồ 1: Memory (VRAM) vs. Context Length (màu sắc biểu thị phương pháp nén) [2].
        *   Biểu đồ 2: Latency/Throughput vs. Context Length [2].
        *   Biểu đồ 3: Perplexity vs. Compression Methods (Trade-off Pareto curves) [2].
    *   *Nghiên cứu (Quốc Anh, Trọng Phú):* Phân tích các mô hình suy giảm chất lượng, rút ra nhận xét cốt lõi về sự nhạy cảm của tiếng Việt khi nén sâu [2].

### PHASE 4: HOÀN THIỆN VĂN BẢN, VIẾT BÁO CÁO & PAPER (Tuần 7) [2]
*   **Mục tiêu:** Hoàn thiện 100% tài liệu bàn giao, chạy thử slide thuyết trình và chuẩn bị bảo vệ [2].
*   **Nhiệm vụ chi tiết:**
    *   *Điều phối & Viết báo cáo (Kiến Hưng, Việt Anh):* Tổng hợp nội dung từ các nhóm để hoàn thiện tài liệu Báo cáo tiếng Việt $\ge 30$ trang [3].
    *   *Viết bài báo khoa học (Quốc Anh, Trọng Phú, Kiến Hưng):* Hoàn thiện bản thảo bài báo tiếng Anh $\ge 6$ trang trên Overleaf theo cấu trúc IEEE [2, 3].
    *   *Thiết kế Slide (Cả nhóm):* Thiết kế bộ slide báo cáo súc tích từ 15-25 trang trình bày rõ ràng kiến trúc hệ thống, phương pháp và biểu đồ trade-off thu được [3].
    *   *Tập dượt phản biện (Kiến Hưng chủ trì):* Chạy thử slide báo cáo trước nhóm, chuẩn bị kịch bản trả lời các câu hỏi phản biện tiềm năng [1].

---

## III. BẢNG PHÂN BỔ NHÂN SỰ CHI TIẾT (RACI MATRIX)

| Thành viên | Vai trò trong dự án | Phase 1 (Prep) | Phase 2 (Exp) | Phase 3 (Analysis) | Phase 4 (Writing) |
|---|---|:---:|:---:|:---:|:---:|
| **Đỗ Kiến Hưng** | PM / Agile Coordinator | **R** / **A** | **A** | **A** | **R** / **A** |
| **Hồ Việt Anh** | Joint Coordinator / Tech | **R** | **R** (PhoGPT) | **C** | **R** (Báo cáo) |
| **Phạm Minh Quân** | Tech Lead / Optimizer | **R** (CUDA setup) | **R** (PhoGPT) | **C** | **C** |
| **Trần Minh Khánh** | Tech runner | **C** | **R** (Qwen) | **C** | **C** |
| **Nguyễn V. Q. Duy** | Quantization / Patching | **C** (vLLM config) | **R** (Qwen/Debug) | **C** | **C** |
| **Nguyễn Hồ Phát** | Data curator | **R** (NeMo Curator) | **A** | **R** | **C** |
| **Huỳnh Hữu Huy** | Data / Prompts | **R** (Test suite) | **A** | **R** | **C** |
| **Huỳnh Ngọc Thạch** | Metric plotting lead | **C** | **A** | **R** (Plotting) | **C** |
| **N. Đăng Quốc Anh** | Research Owner / Idea lead | **R** (Đề cương) | **C** | **R** (Nhận xét) | **R** (Paper EN) |
| **Phan Trọng Phú** | Paper Writer / Review | **R** (References) | **C** | **C** | **R** (Paper EN) |

*Chú thích:* **R** (Responsible - Người thực hiện), **A** (Accountable - Người chịu trách nhiệm/PM), **C** (Consulted - Người hỗ trợ/Tham vấn), **I** (Informed - Người nhận thông tin).

---

## IV. BẢN KẾ HOẠCH QUẢN TRỊ RỦI RO (RISK MANAGEMENT PLAN)

| Loại rủi ro | Chi tiết rủi ro | Mức độ | Giải pháp phòng ngừa & Khắc phục |
|---|---|:---:|---|
| **Rủi ro Phần cứng** | Server local không có GPU hoặc GPU yếu, không chạy được nhân Triton/CUDA của vLLM [2]. | **Cao** | Sử dụng ngân sách nhóm thuê Cloud GPU (Vast.ai, RunPod) [2]. Chi phí dự kiến cực rẻ (~250.000 VNĐ cho cả dự án). |
| **Rủi ro Lỗi mã nguồn** | vLLM báo lỗi không tương thích phiên bản CUDA hoặc lỗi Out-of-Memory (OOM) khi kéo dài context [2]. | **Trung bình** | - Chỉ định Duy làm "Help desk" xử lý lỗi kỹ thuật [2].<br>- Thêm tham số `--max-num-batched-tokens 4096` khi chạy và bật PagedAttention [2]. |
| **Rủi ro Khoa học** | Dữ liệu nén thử nghiệm trên tiếng Việt bị lỗi hiển thị phông chữ Unicode sau khi tokenize [2]. | **Trung bình** | Team Data (Phát, Huy) bắt buộc phải kiểm tra kỹ bộ test-set bằng cách giải mã ngược (detokenize) thử 1-2 mẫu trước khi chạy benchmark đại trà [1, 2]. |
| **Rủi ro Tiến độ** | Viết báo cáo tiếng Việt và bài báo tiếng Anh song song bị quá tải vào Tuần 7 [2]. | **Cao** | - Kiến Hưng giám sát chặt chẽ tiến độ trên `plane.so`.<br>- Khởi tạo khung tài liệu Overleaf ngay từ Tuần 1 và cập nhật dần kết quả thay vì dồn vào tuần cuối [2]. |

---

Bạn hoàn toàn có thể lưu bản Master Plan này vào **Notion Workspace** của nhóm hoặc copy các đầu việc chi tiết trong từng Phase để tạo nhanh các thẻ công việc (Issues/Tasks) trên **Plane.so** cho các thành viên thực hiện. Bạn có muốn tôi hỗ trợ viết chi tiết mô tả công việc (Task Description) cho bất kỳ Sprint cụ thể nào không?
# Thư mục Tài liệu Dự án (Project Documentation)

Thư mục này chứa toàn bộ các tài liệu hướng dẫn, kế hoạch, kiến trúc hệ thống và lộ trình triển khai của dự án **Benchmarking TurboQuant and KV Cache Compression Methods on Vietnamese LLMs**.

---

## Danh mục tài liệu (Documents Directory)

Dưới đây là mô tả chi tiết nội dung và mục đích của từng file trong thư mục `/docs`:

| Tên File | Loại tài liệu | Mô tả chi tiết |
| :--- | :--- | :--- |
| **[sys-arch.md](sys-arch.md)** | **Kiến trúc hệ thống** | Bản mô tả chi tiết kiến trúc benchmark 4 tầng: (1) Tiền xử lý dữ liệu, (2) LLM Serving & Lõi nén, (3) Giám sát & đo đạc chỉ số, (4) Phân tích & Trực quan hóa. |
| **[master-plan.md](master-plan.md)** | **Kế hoạch tổng thể** | Lộ trình tổng thể 7 tuần phát triển dự án, phân bổ nhân sự (ma trận RACI) và kế hoạch quản trị rủi ro hệ thống. |
| **[doc-review.md](doc-review.md)** | **Đánh giá & Review** | Báo cáo đánh giá tính hợp lý, khả thi của tài liệu dự án; kết quả grounding search (sửa đổi các mã arXiv bị lỗi trích dẫn) và đề xuất quy chuẩn workflow tránh trùng lặp tài liệu. |
| **[answer-lecture.md](answer-lecture.md)** | **Giải trình & Đối chiếu** | Các phương án phản hồi ý kiến phản biện của giảng viên hướng dẫn về phần cứng, chi phí, chỉ số đo đạc cụ thể và danh sách mô hình kiểm chứng. |
| **[onboarding_presentation.html](onboarding_presentation.html)** | **Slide Onboarding (HTML)** | Slide thuyết trình onboarding tương tác được thiết kế tối ưu hiển thị (viewport fitting) theo theme Neon Cyber. |
| **[presentation_script.md](presentation_script.md)** | **Kịch bản Onboarding (MD)** | Lời thoại thuyết trình (Speaking Script) chi tiết cho từng slide trong onboarding_presentation.html, văn phong nói tự nhiên, dễ hiểu. |
| **[sprint01.md](sprint01.md)** | **Sprint 01 (Week 1-2)** | Kế hoạch chi tiết giai đoạn khởi động: Thiết lập không gian cộng tác, proposal v2, tiền xử lý dữ liệu bằng NVIDIA NeMo Curator và cấu hình baseline vLLM. |
| **[sprint02.md](sprint02.md)** | **Sprint 02 (Week 3-4)** | Kế hoạch hoàn thiện script đo đạc tự động `run_baseline.py`, chạy thực nghiệm mốc đầu trên PhoGPT-7B5/Qwen2.5 và khởi tạo khung LaTeX bài báo trên Overleaf. |
| **[sprint03.md](sprint03.md)** | **Sprint 03 (Week 5-6)** | Kế hoạch tối ưu hóa VRAM chống lỗi CUDA OOM, tổng hợp số liệu CSV và vẽ biểu đồ Pareto Frontier bằng python; viết chương Kết quả & Thảo luận. |
| **[sprint04.md](sprint04.md)** | **Sprint 04 (Week 7)** | Giai đoạn đóng gói: Hoàn thiện báo cáo Word tiếng Việt (>=30 trang), slide thuyết trình bảo vệ, hoàn chỉnh Paper tiếng Anh (>=6 trang) và tài liệu Q&A. |

---

## Quy tắc cộng tác và Cập nhật tài liệu

Nhóm tuân thủ quy chuẩn quản lý tài liệu tại **[doc-review.md](doc-review.md#3-de-xuat-quy-chuan-workflow-quan-ly-tai-lieu-tranh-duplicate--version-conflict)**:
1. **Thay đổi cấu trúc code/chạy thử:** Cập nhật file hướng dẫn cục bộ (ví dụ: `README.md` của các thư mục chức năng) và cập nhật sơ đồ kiến trúc hệ thống nếu có thay đổi.
2. **Thay đổi task/deadline:** Cập nhật trực tiếp trên **Plane.so** thay vì chỉnh sửa file markdown tại đây để tránh xung đột lịch trình.
3. **Bài viết học thuật:** Viết và cập nhật trực tiếp tại Overleaf, sử dụng file [references.bib](../paper/references.bib) để quản lý trích dẫn.


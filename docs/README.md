# Thư mục Tài liệu Dự án (Project Documentation)

Thư mục này chứa toàn bộ các tài liệu hướng dẫn, kế hoạch, kiến trúc hệ thống và lộ trình triển khai của dự án **Benchmarking TurboQuant and KV Cache Compression Methods on Vietnamese LLMs**.

---

## Danh mục tài liệu (Documents Directory)

Dưới đây là mô tả chi tiết nội dung, người tạo và mục đích của từng file trong thư mục `/docs` và các thư mục con:

| Tên File | Loại tài liệu | Người tạo | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| **[sys-arch.md](sys-arch.md)** | **Kiến trúc hệ thống** | Kien Hung \<kienhung.do1105@gmail.com\> | Bản mô tả chi tiết kiến trúc benchmark 4 tầng: Tiền xử lý, Serving, Giám sát và Trực quan hóa. |
| **[doc-review.md](doc-review.md)** | **Đánh giá & Review** | Kien Hung \<kienhung.do1105@gmail.com\> | Báo cáo đánh giá tính hợp lý của tài liệu, kết quả grounding search và quy chuẩn workflow. |
| **[answer-lecture.md](process/answer-lecture.md)** | **Giải trình & Đối chiếu** | Kien Hung \<kienhung.do1105@gmail.com\> | Các phương án phản hồi ý kiến phản biện của giảng viên về phần cứng, chi phí, chỉ số đo đạc. |
| **[related_works.md](related_works.md)** | **Literature Review** | Kien Hung \<kienhung.do1105@gmail.com\> | Tài liệu tổng quan lý thuyết và nghiên cứu liên quan về nén KV Cache (HQQ, PolarQuant, TurboQuant). |
| **[plans/README.md](plans/README.md)** | **Thư mục Kế hoạch** | *Mới* | Tài liệu kế hoạch tổng thể, lộ trình sprint và các technical todolist. |
| **[presentation/README.md](presentation/README.md)** | **Thư mục Slide Deck** | *Mới* | Tài liệu slide deck onboarding, slide proposal và speak script tương ứng. |
| **[process/README.md](process/README.md)** | **Thư mục Tiến trình** | *Mới* | Tài liệu nhật ký tiến trình đo đạc và phân tích kết quả. |
| **[report/README.md](report/README.md)** | **Thư mục Báo cáo** | *Mới* | Báo cáo kỹ thuật hệ thống benchmark thực tế trên Cloud GPU và báo cáo tiến độ Data Team. |

---

## Quy tắc cộng tác và Cập nhật tài liệu

Nhóm tuân thủ quy chuẩn quản lý tài liệu tại **[doc-review.md](doc-review.md#3-de-xuat-quy-chuan-workflow-quan-ly-tai-lieu-tranh-duplicate--version-conflict)**:
1. **Thay đổi cấu trúc code/chạy thử:** Cập nhật file hướng dẫn cục bộ (ví dụ: `README.md` của các thư mục chức năng) và cập nhật sơ đồ kiến trúc hệ thống nếu có thay đổi.
2. **Thay đổi task/deadline:** Cập nhật trực tiếp trên **Plane.so** thay vì chỉnh sửa file markdown tại đây để tránh xung đột lịch trình.
3. **Bài viết học thuật:** Viết và cập nhật trực tiếp tại Overleaf, sử dụng file [references.bib](../paper/references.bib) để quản lý trích dẫn.


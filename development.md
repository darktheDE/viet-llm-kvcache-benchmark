# DEVELOPMENT GUIDELINE

---

## I. QUY TRÌNH PHÁT TRIỂN TÍNH NĂNG (FEATURE WORKFLOW)

Quy trình triển khai và hoàn thành một Task được thực hiện tuần tự theo mô hình Agile/Git-Flow dưới đây:

### 1. Đọc và nhận Task
* Nhận nhiệm vụ được phân bổ trên bảng Kanban [Plane.so](https://app.plane.so).
* Đọc kỹ các thông tin: **Mô tả**, **Từng bước thực hiện**, và **Tiêu chuẩn hoàn thành** như đã được định nghĩa cụ thể trong các tài liệu Sprint.

### 2. Tạo nhánh làm việc (Branching Strategy)
Mọi đoạn code đóng góp vào kho lưu trữ phải xuất phát từ nhánh `develop` và tuân thủ quy tắc đặt tên:
* Tạo nhánh mới từ nhánh `develop` trên GitHub/Local:
  ```bash
  git checkout develop
  git pull origin develop
  git checkout -b feature/ten-tính-năng
  ```
  *(Ví dụ: `feature/run-baseline-setup`, `feature/pareto-plotting`)*

### 3. Thực thi coding & Cải tiến
* Thực hiện từng bước (Step-by-Step) theo yêu cầu của Task.
* **Quyền cải tiến:** Hướng dẫn trong Task chỉ mang tính chất tham khảo kỹ thuật. Lập trình viên được khuyến khích chỉnh sửa, tối ưu hóa thuật toán hoặc cải thiện cấu trúc file nếu phương án đó đem lại hiệu năng hoặc độ sạch của code tốt hơn.

### 4. Tạo Pull Request (PR) & Review chéo
Khi hoàn thành code và tự kiểm thử (Self-test) thành công ở local:
1. Push nhánh feature lên GitHub:
   ```bash
   git add .
   git commit -m "feat: mô tả ngắn gọn thay đổi"
   git push origin feature/ten-tính-năng
   ```
2. Tạo Pull Request (PR) từ nhánh `feature/...` vào nhánh `develop`.
3. Chỉ định Reviewer vào PR.

### 5. Merge code & Hoàn thành Task
* Khi PR nhận đủ sự phê duyệt (Approved) của reviewer và không có xung đột (Conflict):
  * Thực hiện merge PR vào nhánh `develop`.
  * Chuyển trạng thái thẻ công việc trên **Plane.so** sang `Done`.
  * Xóa nhánh feature tương ứng trên GitHub.
  * Tiếp tục nhận task mới trên Plane.so.

---

## II. CÁC QUY CHUẨN CODING PHỔ BIẾN (CODE STANDARDS)

### 1. Quy chuẩn Python (PEP 8)
* **Đặt tên (Naming Convention):**
  * Tên hàm, biến, thuộc tính: Dạng `snake_case`.
  * Tên Class: Dạng `PascalCase`.
  * Tên hằng số: Chữ in hoa `UPPER_CASE`.
* **Format code:** Sử dụng thụt lề (indentation) bằng **4 khoảng trắng (spaces)**.
* **Docstring & Comments:** 
  * Mọi hàm hoặc class chính phải có docstring giải thích ngắn gọn mục tiêu, các tham số đầu vào (`Args`) và giá trị trả về (`Returns`).
  * Giữ nguyên các comment học thuật giải thích công thức toán học nén KV cache.

### 2. Quản lý Git Commit (Conventional Commits)
Thông điệp commit (Commit message) cần viết rõ ràng theo định dạng:
* `feat:` Khi thêm tính năng mới
* `fix:` Khi sửa lỗi
* `docs:` Khi chỉnh sửa tài liệu
* `refactor:` Khi tối ưu hoặc tái cấu trúc code mà không đổi tính năng.

### 3. Nguyên tắc phát triển phần mềm (Clean Code)
* **KISS (Keep It Simple, Stupid):** Viết code đơn giản, tường minh. Tránh tạo các lớp trừu tượng quá mức không cần thiết cho mục đích benchmark.
* **Don't Repeat Yourself (DRY):** Tránh lặp lại code. Gom các đoạn mã tính toán chỉ số (như đo VRAM, TTFT, PPL) thành các hàm tiện ích dùng chung trong thư mục `scripts/`.
* **Quản lý file phụ thuộc:** Mọi thư viện cài thêm phục vụ chạy code bắt buộc phải được ghi nhận chính xác phiên bản vào file `scripts/requirements.txt`.

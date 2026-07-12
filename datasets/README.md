# Datasets (Tập dữ liệu)

Thư mục này chứa các bộ dữ liệu chuẩn hóa và các bộ kiểm thử nhỏ (định dạng JSON/JSONL) dùng để đánh giá hiệu năng và chất lượng của các mô hình ngôn ngữ lớn tiếng Việt khi nén KV Cache.

## File Metadata

| Tên File / Thư mục | Người tạo | Vai trò / Mục đích |
| :--- | :--- | :--- |
| **[dataset_brief.md](dataset_brief.md)** | QUOC ANH \<quocanh0815@gmail.com\> | Tài liệu mô tả tóm tắt cấu trúc và thông số của Vietnamese Long-Context Test Suite. |
| **[data_quality_checklist.md](data_quality_checklist.md)** | TriH28 \<5.11t1.huynhhuuhuy@gmail.com\> | Bảng kiểm soát chất lượng dữ liệu (format, chiều dài ngữ cảnh, tiêu chí lọc). |
| **[PHAT_test_set_small.json](PHAT_test_set_small.json)** | Kien Hung \<kienhung.do1105@gmail.com\> | File dữ liệu test-set mẫu thu nhỏ được sinh ra bởi Phat. |
| **[static.md](static.md)** | QUOC ANH \<quocanh0815@gmail.com\> | Tài liệu ghi chú và phân tích các thuộc tính tĩnh của dữ liệu. |
| **[test_set_small.json](test_set_small.json)** | QUOC ANH \<quocanh0815@gmail.com\> | Bộ dữ liệu đánh giá ngữ cảnh dài chính thức (các mốc 4k, 8k, 16k context). |
| **[test_set_small.jsonl](test_set_small.jsonl)** | QUOC ANH \<quocanh0815@gmail.com\> | File test_set_small được chuyển đổi sang định dạng dòng (JSON Lines). |
| **[test_set_smoke.json](test_set_smoke.json)** | TriH28 \<5.11t1.huynhhuuhuy@gmail.com\> | Tập dữ liệu smoke test dung lượng cực nhỏ để test nhanh mã chạy. |
| **[test_set_tasks_small.json](test_set_tasks_small.json)** | QUOC ANH \<quocanh0815@gmail.com\> | Tập dữ liệu đánh giá chất lượng trên các tác vụ downstream cụ thể. |
| **[test_set_tasks_small.jsonl](test_set_tasks_small.jsonl)** | QUOC ANH \<quocanh0815@gmail.com\> | Định dạng JSON Lines cho tập test_set_tasks_small. |
| **[test_set_tasks_smoke.json](test_set_tasks_smoke.json)** | TriH28 \<5.11t1.huynhhuuhuy@gmail.com\> | Tập benchmark tác vụ smoke test nhỏ gọn để chạy nhanh pipeline. |
| **[expansion/](expansion/)** | TriH28 \<5.11t1.huynhhuuhuy@gmail.com\> | Thư mục chứa dữ liệu thô phục vụ mở rộng tập test-set. |

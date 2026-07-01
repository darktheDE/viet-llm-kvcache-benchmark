# Scripts (Kịch bản Tự động hóa)

Thư mục này chứa các kịch bản tự động hóa, cấu hình chạy thử, đo đạc chỉ số tài nguyên và mã nguồn vẽ biểu đồ phân tích kết quả benchmark.

## File Metadata

| Tên File / Thư mục | Người tạo | Vai trò / Mục đích |
| :--- | :--- | :--- |
| **[build_long_context_testset.py](build_long_context_testset.py)** | Phat.Nguyen \<hophat0011@gmail.com\> | Tạo tệp JSON dài ngữ cảnh từ kết quả lọc sạch của NeMo. |
| **[clean_with_nemo.py](clean_with_nemo.py)** | Phat.Nguyen \<hophat0011@gmail.com\> | Tiền xử lý, lọc nhiễu và làm sạch dữ liệu văn bản tiếng Việt sử dụng NeMo Curator. |
| **[collect_expansion_data.py](collect_expansion_data.py)** | TriH28 \<5.11t1.huynhhuuhuy@gmail.com\> | Thu thập, tổng hợp và định dạng lại các nguồn dữ liệu bổ sung để mở rộng test-set. |
| **[create_test_set.py](create_test_set.py)** | QUOC ANH \<quocanh0815@gmail.com\> | Xây dựng bộ test-set cơ bản từ các nguồn dữ liệu văn bản ban đầu. |
| **[download_datasets.py](download_datasets.py)** | Phat.Nguyen \<hophat0011@gmail.com\> | Tải dữ liệu tự động từ Hugging Face (VMLU, VTSNLP, v.v.). |
| **[nemo_backend.py](nemo_backend.py)** | Phat.Nguyen \<hophat0011@gmail.com\> | Module wrapper hỗ trợ định tuyến chạy NeMo Curator trên CPU hoặc GPU Docker. |
| **[plot_results.py](plot_results.py)** | HuynhThach1606 \<23133072@student.hcmute.edu.vn\> | Kịch bản vẽ các đồ thị so sánh (VRAM, Latency, Throughput, Pareto frontier). |
| **[run_baseline.py](run_baseline.py)** | Quan-min211 \<minhquan021105@gmail.com\> | Script cốt lõi đo đạc các chỉ số (VRAM, TTFT, ITL, PPL) sử dụng công cụ vLLM. |
| **[run_mock_grid.py](run_mock_grid.py)** | Quan-min211 \<minhquan021105@gmail.com\> | Script mô phỏng Grid Search trên CPU để kiểm tra logic ghi kết quả và định dạng log CSV. |
| **[scrape_news_sample.py](scrape_news_sample.py)** | QUOC ANH \<quocanh0815@gmail.com\> | Cào dữ liệu báo chí mẫu từ nguồn VnExpress để mở rộng ngữ cảnh dài tiếng Việt. |
| **[utils_text.py](utils_text.py)** | Phat.Nguyen \<hophat0011@gmail.com\> | Các hàm tiện ích xử lý text (tính từ, đếm số token, làm sạch khoảng trắng). |
| **[validate_testset.py](validate_testset.py)** | Phat.Nguyen \<hophat0011@gmail.com\> | Kiểm tra tính hợp lệ về cấu trúc JSON và schema của các tập dữ liệu test-set. |
| **[__init__.py](__init__.py)** | Phat.Nguyen \<hophat0011@gmail.com\> | Khởi tạo package Python cho module scripts. |
| **[test/](test/)** | Quan-min211 \<minhquan021105@gmail.com\> | Thư mục chứa các kịch bản chạy benchmark thực tế trên Cloud GPU. |

# GPU Benchmarking Scripts (Kịch bản Đo đạc Thực tế trên GPU)

Thư mục này chứa các kịch bản chính thức chạy trên máy chủ Cloud GPU để đo đạc hiệu năng và chất lượng thực tế của các mô hình ngôn ngữ lớn tiếng Việt dưới các kỹ thuật nén KV Cache khác nhau.

## File Metadata

| Tên File | Người tạo | Vai trò / Mục đích |
| :--- | :--- | :--- |
| **[generate_real_analysis.py](generate_real_analysis.py)** | Quan-min211 \<minhquan021105@gmail.com\> | Tự động sinh ra Jupyter Notebook (`real_benchmark_analysis.ipynb`) phân tích số liệu thực tế thu được. |
| **[run_real_benchmark.py](run_real_benchmark.py)** | Quan-min211 \<minhquan021105@gmail.com\> | Thực hiện chạy benchmark thực tế trên GPU cho một cấu hình mô hình, độ dài ngữ cảnh và chế độ nén nhất định. |
| **[run_real_grid.py](run_real_grid.py)** | Quan-min211 \<minhquan021105@gmail.com\> | Điều phối chạy tự động toàn bộ Grid Search thực tế (lặp qua tất cả các mô hình, độ dài ngữ cảnh và chế độ nén KV Cache). |

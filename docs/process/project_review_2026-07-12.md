# Review mã nguồn và tiến độ dự án ngày 12/07/2026

## 1. Phạm vi review

Review đối chiếu bốn file kế hoạch Sprint với mã nguồn, dữ liệu, tài liệu quy trình,
lịch sử Git và các artefact hiện có trong repository. Chỉ tiêu nào có bằng chứng trực
tiếp mới được chuyển từ `[ ]` sang `[x]`.

Môi trường review không có Python runtime và GPU, vì vậy không thể chạy lại unit test,
dataset validator hay benchmark thực. Các kết luận runtime dựa trên artefact và lịch sử
đã được commit; những tiêu chí cần xác nhận Cloud GPU, Plane.so, Drive, Overleaf hoặc
biên bản họp vẫn giữ nguyên trạng thái chưa hoàn thành.

## 2. Kết quả kiểm kê Sprint

| Sprint | Tiêu chí DoD | Đã xác nhận | Tỷ lệ |
| :--- | ---: | ---: | ---: |
| Sprint 01 | 17 | 4 | 23,5% |
| Sprint 02 | 8 | 1 | 12,5% |
| Sprint 03 | 8 | 0 | 0% |
| Sprint 04 | 12 | 1 | 8,3% |
| **Tổng** | **45** | **6** | **13,3%** |

Các tiêu chí vừa được tick:

1. Sprint 01, Task 3: ba tiêu chí đóng gói JSON, kiểm tra detokenize và tương thích
   đầu vào benchmark. Bằng chứng nằm trong các test-set canonical/smoke,
   `datasets/data_quality_checklist.md`, validator và loader của benchmark.
2. Sprint 01, Task 5: đã thu thập 200 expansion candidates, trong đó 189 bản ghi có
   `source_type=news`, vượt yêu cầu tối thiểu 100 văn bản.
3. Sprint 02, Task 1: CLI benchmark ngắn đã được triển khai và ghi nhận chạy mock
   thành công trong `docs/process/tech_report_task1&2.md` cùng artefact CSV.
4. Sprint 04, Task 3: README gốc đã có hướng dẫn cài đặt, pipeline dữ liệu, lệnh chạy
   benchmark và lệnh tạo biểu đồ.

## 3. Đánh giá mức hoàn thiện thực tế

- **Theo DoD Sprint nghiêm ngặt:** 13,3%. Đây là con số kiểm toán được và không suy
  diễn các hoạt động ngoài repository.
- **Theo mức độ có mã/artefact chức năng:** khoảng 40%. Pipeline dữ liệu, mock/real
  runner, PPL offline, kiểm tra chất lượng generation và plotting đã có; tuy nhiên phần
  benchmark thật và tích hợp nén vẫn chưa được chứng minh end-to-end.
- **Theo sản phẩm bàn giao cuối kỳ:** khoảng 20-25%. Chưa có báo cáo Word/PDF 30 trang,
  slide bảo vệ 15-25 trang, paper IEEE 6 trang và biên bản nghiệm thu cuối.
- **Ước lượng tổng thể có trọng số:** khoảng **30%**.

## 4. Phát hiện code review quan trọng

### Mức nghiêm trọng cao

1. `scripts/run_baseline.py` nhận lựa chọn HQQ và PolarQuant nhưng ánh xạ runtime không
   có hai phương pháp này; cả hai bị rơi về `kv_cache_dtype="auto"`. Kết quả gắn nhãn
   HQQ/PolarQuant vì thế có nguy cơ thực chất là baseline không nén.
2. `requirements.txt` không khai báo các dependency runtime cốt lõi `torch`, `vllm` và
   `pynvml`, đồng thời phần lớn package không được khóa phiên bản. Việc tái lập môi
   trường từ đầu chưa đạt DoD.
3. Thư mục `results/` chưa có dữ liệu benchmark thật: các file compiled/summary chỉ có
   header. Các biểu đồ hiện tại là artefact mock, chưa đủ cơ sở cho kết luận khoa học.
4. Repository đang track nhiều file dữ liệu lớn (`.json`/`.jsonl`) từ khoảng 1 MB đến
   hơn 46 MB, trái với ràng buộc không commit dữ liệu lớn trong `AGENTS.md`.

### Mức nghiêm trọng trung bình

1. Đo VRAM trong `run_baseline.py` chỉ đọc `nvmlDeviceGetMemoryInfo()` sau generation,
   không có sampler nền nên không bảo đảm bắt được peak VRAM trong prefill/decode.
2. Latency hiện là tổng thời gian chia tổng token, chưa đo riêng TTFT và ITL như kế hoạch.
3. CSV benchmark không có `compression_ratio`, `gpu_efficiency_index`, `base_vram_mb`
   và `dynamic_vram_mb`; Task 1 Sprint 02 chưa hoàn thành.
4. `plot_results.py` chưa tính/vẽ Compression Ratio và GPU Memory Efficiency; các file
   tổng hợp hiện rỗng nên Task 2 Sprint 03 chưa đạt DoD đầy đủ.
5. Tài liệu tiến trình có nội dung lỗi thời: nhắc tới model/file notebook cũ hoặc không
   còn tồn tại, và có tuyên bố benchmark hoàn thành trong khi artefact chỉ là mock.

## 5. Thứ tự ưu tiên để hoàn thành dự án

1. Sửa ánh xạ HQQ/PolarQuant/TurboQuant và thêm kiểm tra fail-fast để không bao giờ ghi
   sai nhãn phương pháp nén.
2. Chuẩn hóa `requirements.txt`, tạo môi trường sạch và chạy toàn bộ unit test, validator,
   smoke grid.
3. Hoàn thiện metric collector: peak VRAM nền, TTFT, ITL, base/dynamic VRAM,
   compression ratio và GPU efficiency.
4. Chạy benchmark thật có log provenance trên tối thiểu hai model và các mốc 4k/8k/16k;
   sau đó backfill PPL/quality metrics.
5. Sinh lại bảng và biểu đồ từ dữ liệu thật, rồi mới viết Results/Discussion, báo cáo,
   paper và slide.
6. Chuyển dữ liệu/log lớn ra kho lưu trữ phù hợp, cập nhật `.gitignore`, metadata và
   tài liệu quy trình.

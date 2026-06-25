# Data Quality Checklist - Vietnamese Long-Context Test Suite

Checklist này dùng để kiểm định chất lượng bộ dữ liệu `datasets/test_set_small.json`, `datasets/test_set_small.jsonl` và `datasets/test_set_smoke.json` trước khi bàn giao cho Team Technical chạy benchmark.

## 1. Kiểm tra cú pháp file

* [x] `datasets/test_set_small.json` đọc được bằng `json.load()`.
* [x] `datasets/test_set_small.json` không lỗi cú pháp JSON.
* [x] `datasets/test_set_smoke.json` không lỗi cú pháp JSON.
* [x] `datasets/test_set_small.jsonl` đã được đồng bộ lại từ file JSON mới nhất.
* [x] Số mẫu trong JSON và JSONL trùng nhau.

Lệnh kiểm tra:

```powershell
python -m json.tool datasets/test_set_small.json > check.json
python -m json.tool datasets/test_set_smoke.json > check.json
```

## 2. Kiểm tra Unicode và tiếng Việt

* [x] Đã kiểm tra ký tự lỗi Unicode replacement character `�`.
* [x] Đã loại bỏ 3 mẫu chứa ký tự lỗi `�`.
* [x] File `datasets/test_set_small.json` hiện không còn ký tự `�`.
* [x] Văn bản tiếng Việt có dấu hiển thị bình thường.
* [ ] Chưa phát hiện thêm lỗi mojibake như `Ã`, `áº`, `Æ`.

Lệnh kiểm tra:

```powershell
Select-String -Path datasets/test_set_small.json -Pattern "�"
```

Kết quả mong muốn: không in ra dòng nào.

## 3. Kiểm tra schema dữ liệu

Mỗi sample trong dataset phải có đủ các trường sau:

* [x] `prompt_type`
* [x] `context_length_target`
* [x] `text`
* [x] `expected_output`
* [x] `actual_tokens`

Ràng buộc giá trị:

* [x] `prompt_type` chỉ nhận một trong ba giá trị: `qa`, `retrieval`, `general`.
* [x] `context_length_target` chỉ nhận một trong ba giá trị: `4000`, `8000`, `16000`.
* [x] `text` là chuỗi không rỗng.
* [x] `actual_tokens` là số nguyên dương.
* [x] `expected_output` tồn tại trong mọi sample.

## 4. Kiểm tra số lượng và phân phối dữ liệu

Dataset đầy đủ:

* [x] `datasets/test_set_small.json` có 507 mẫu sau khi làm sạch.
* [x] `datasets/test_set_small.jsonl` có 507 dòng tương ứng.
* [x] Dataset có đủ 3 bucket: 4k, 8k, 16k.
* [x] Dataset có đủ 3 loại tác vụ: `qa`, `retrieval`, `general`.

Smoke test:

* [x] `datasets/test_set_smoke.json` có 15 mẫu.
* [x] Mỗi bucket 4k, 8k, 16k có 5 mẫu.
* [x] Mỗi bucket gồm:

  * 2 mẫu `qa`
  * 2 mẫu `retrieval`
  * 1 mẫu `general`

Lệnh kiểm tra phân phối smoke test:

```powershell
python -c "import json, collections; data=json.load(open('datasets/test_set_smoke.json',encoding='utf-8')); print('total',len(data)); print(collections.Counter((x['context_length_target'],x['prompt_type']) for x in data))"
```

## 5. Kiểm tra logic task QA

Áp dụng cho các sample có:

```text
prompt_type == "qa"
```

Yêu cầu:

* [x] `expected_output` là một chữ cái đáp án.
* [x] Đáp án thuộc tập `A`, `B`, `C`, `D`, `E`.
* [x] Prompt yêu cầu mô hình chỉ trả lời một chữ cái đáp án.
* [ ] Sau khi Team Tech chạy inference, cần kiểm tra output có parse được thành A/B/C/D/E hay không.
* [ ] Tính Exact Match giữa output đã chuẩn hóa và `expected_output`.

Quy tắc chấm đề xuất:

```text
QA Exact Match = số mẫu trả lời đúng / tổng số mẫu QA
```

## 6. Kiểm tra logic task Retrieval

Áp dụng cho các sample có:

```text
prompt_type == "retrieval"
```

Yêu cầu:

* [x] `expected_output` là chuỗi ngắn cần truy xuất.
* [x] Prompt yêu cầu mô hình trả lời ngắn gọn, chỉ trả lời thông tin cần tìm.
* [ ] Sau khi Team Tech chạy inference, kiểm tra `expected_output` có xuất hiện trong output hay không.
* [ ] Ghi nhận lỗi nếu mô hình trả lời sai, không trả lời, hoặc hallucinate thông tin khác.

Quy tắc chấm đề xuất:

```text
Retrieval Hit Rate = số mẫu output chứa expected_output / tổng số mẫu retrieval
```

## 7. Kiểm tra logic task General

Áp dụng cho các sample có:

```text
prompt_type == "general"
```

Yêu cầu:

* [x] Text là văn bản tiếng Việt dài tự nhiên.
* [x] Dùng để tính Perplexity hoặc offline language modeling loss.
* [ ] Không dùng Exact Match cho task `general`.
* [ ] Sau khi Team Tech tích hợp PPL, kiểm tra mẫu `general` có chạy được forward pass không crash.

Quy tắc đánh giá đề xuất:

```text
PPL = exp(cross_entropy_loss)
```

## 8. Kiểm tra tương thích với Team Technical

Team Technical cần xác nhận script benchmark đọc được các trường:

* [ ] `prompt_type`
* [ ] `context_length_target`
* [ ] `text`
* [ ] `expected_output`
* [ ] `actual_tokens`

Yêu cầu trước khi benchmark đầy đủ:

* [ ] Team Tech chạy thử thành công `datasets/test_set_smoke.json`.
* [ ] Script benchmark không lỗi khi đọc 15 mẫu smoke test.
* [ ] Script benchmark lọc được mẫu theo bucket 4k, 8k, 16k.
* [ ] Script benchmark lọc được mẫu theo `prompt_type`.
* [ ] Output sinh ra được lưu vào CSV kết quả.

## 9. Kiểm tra file kết quả CSV sau khi benchmark

Sau khi Team Technical chạy benchmark, cần kiểm tra file kết quả có các trường tối thiểu:

* [ ] `model`
* [ ] `kv_cache_type`
* [ ] `context_length`
* [ ] `prompt_type`
* [ ] `peak_memory_mb`
* [ ] `latency_ms_per_token`
* [ ] `throughput_tokens_per_s`
* [ ] `perplexity`
* [ ] `status`
* [ ] `error_message`

Yêu cầu chất lượng CSV:

* [ ] Không lệch cột.
* [ ] Không thiếu header.
* [ ] Các giá trị số parse được bằng `pandas`.
* [ ] Các case lỗi OOM phải được ghi rõ là `OOM`, không để trống.
* [ ] Không có dòng kết quả bị duplicate ngoài ý muốn.
* [ ] Không có giá trị âm ở các cột memory, latency, throughput hoặc perplexity.

## 10. Trạng thái bàn giao

Trạng thái hiện tại:

* [x] Dataset full đã làm sạch Unicode.
* [x] Dataset full còn 507 mẫu hợp lệ.
* [x] JSONL đã đồng bộ lại từ JSON.
* [x] Smoke test 15 mẫu đã được tạo.
* [x] Dataset brief đã cập nhật trạng thái kiểm định.
* [ ] Chờ Team Technical xác nhận chạy được smoke test.
* [ ] Chờ reviewer approve Pull Request.
* [ ] Chờ merge Pull Request vào nhánh chính.

Kết luận tạm thời:

```text
Dataset status: READY FOR TECH SMOKE TEST
Final handoff status: PENDING TECH CONFIRMATION
```

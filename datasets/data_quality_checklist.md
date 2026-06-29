# Data Quality Checklist - Vietnamese Long-Context Test Suite

Checklist này dùng để kiểm định chất lượng bộ dữ liệu canonical long-context (`test_set_small.json`, `test_set_smoke.json`) và bộ dữ liệu task benchmark (`test_set_tasks_small.json`, `test_set_tasks_smoke.json`) trước khi chạy benchmark.

---

## 1. Kiểm tra cú pháp và tích hợp file

* [x] Cả file canonical và file task đều đọc được bằng `json.load()` không lỗi cú pháp JSON.
* [x] Đã tạo các bản ghi test-set smoke tương ứng để kiểm tra nhanh.
* [x] Số lượng mẫu trong các file JSON/JSONL đồng bộ và chính xác.

Lệnh kiểm tra cú pháp JSON:
```powershell
python -m json.tool datasets/test_set_small.json > check_canonical.json
python -m json.tool datasets/test_set_tasks_small.json > check_tasks.json
```

---

## 2. Kiểm tra Unicode và lỗi ký tự tiếng Việt

* [x] Không chứa ký tự replacement character "" (Unicode `\ufffd`).
* [x] Đã loại bỏ hoàn toàn các mẫu bị lỗi mojibake hoặc lỗi giải mã Unicode.
* [x] Hiển thị đúng tiếng Việt có dấu khi detokenize.

Lệnh kiểm tra ký tự "":
```powershell
# PowerShell
Select-String -Path datasets/test_set_small.json -Pattern ""
Select-String -Path datasets/test_set_tasks_small.json -Pattern ""

# Python (nếu PowerShell không nhận dạng được ký tự đặc biệt)
python -c "import json; data=json.load(open('datasets/test_set_small.json', encoding='utf-8')); has_err = any('\ufffd' in str(s) for s in (data if isinstance(data, list) else data.get('samples', []))); print('Chứa ký tự lỗi:', has_err)"
```
*(Kết quả mong muốn: không tìm thấy dòng kết quả nào hoặc in ra `Chứa ký tự lỗi: False`)*

---

## 3. Kiểm tra Schema dữ liệu

### A. Canonical Long-Context Schema (`datasets/test_set_small.json`)
Tệp tin phải ở dạng top-level JSON object chứa các khóa:
* `dataset_name`
* `version`
* `language`
* `created_by`
* `description`
* `mode`
* `tokenizer`
* `samples` (Danh sách các mẫu)

Mỗi sample trong danh sách `samples` phải có:
* `id`
* `source`
* `context_group` (nhận giá trị `"4k"`, `"8k"`, hoặc `"16k"`)
* `target_tokens`
* `actual_tokens`
* `text`
* `metadata`

### B. Task Benchmark Schema (`datasets/test_set_tasks_small.json`)
Tệp tin ở dạng danh sách các mẫu (hoặc đối tượng có trường `samples`), mỗi mẫu chứa:
* `prompt_type` (nhận giá trị `"qa"`, `"retrieval"`, hoặc `"general"`)
* `context_length_target` (nhận giá trị `4000`, `8000`, hoặc `16000`)
* `text` (văn bản chứa prompt đầy đủ)
* `expected_output`
* `actual_tokens`
* `metadata` (bao gồm `source`, `domain`, `subject`)

Lệnh kiểm duyệt tự động qua validator:
```bash
# Validate Canonical Long-Context Schema
python scripts/validate_testset.py --input datasets/test_set_small.json --schema long_context

# Validate Task Benchmark Schema
python scripts/validate_testset.py --input datasets/test_set_tasks_small.json --schema task
```

---

## 4. Kiểm tra phân phối và số lượng mẫu

### A. Bộ dữ liệu Canonical Long-Context
* [x] Full mode: Từ 10-20 mẫu tổng thể.
* [x] Full mode: Ít nhất 3 mẫu cho mỗi nhóm (`4k`, `8k`, `16k`).

### B. Bộ dữ liệu Task Benchmark
* [x] Full mode (`test_set_tasks_small.json`): Chứa 507 mẫu.
* [x] Smoke mode (`test_set_tasks_smoke.json`): Chứa 15 mẫu.
* [x] Phân phối smoke mode: Mỗi mốc ngữ cảnh (4000, 8000, 16000) có đúng 5 mẫu (2 QA, 2 Retrieval, 1 General).

Lệnh kiểm tra phân phối smoke task:
```powershell
python -c "import json, collections; data=json.load(open('datasets/test_set_tasks_smoke.json',encoding='utf-8')); print('total',len(data)); print(collections.Counter((x['context_length_target'],x['prompt_type']) for x in data))"
```

---

## 5. Ràng buộc Logic Tác vụ (Task constraints)

### Task QA (`prompt_type == "qa"`)
* [x] `expected_output` là chữ cái đáp án trắc nghiệm duy nhất (`A`, `B`, `C`, `D` hoặc `E`).
* [x] Metadata chứa thông tin phân loại chủ đề/lĩnh vực (ví dụ: `domain: STEM`, `subject: math`).

### Task Retrieval (`prompt_type == "retrieval"`)
* [x] `expected_output` là chuỗi ngắn cần truy xuất (không rỗng).

### Task General (`prompt_type == "general"`)
* [x] Dành cho Perplexity (PPL). `expected_output` được phép để rỗng.

---

## 6. Trạng thái bàn giao

```text
Dataset Canonical: READY (JSON Object with samples)
Dataset Tasks: READY (JSON List of QA/Retrieval/General tasks)
Validation Status: AUTOMATED SCHEMA CHECKS IMPLEMENTED
```

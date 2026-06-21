Bạn là senior data engineering reviewer. Hãy kiểm tra repo hiện tại và xác định chính xác pipeline `hybrid_nemo_python` đang chia trách nhiệm như thế nào: bước nào thực sự dùng NVIDIA NeMo Curator, bước nào dùng Python custom logic.

## Bối cảnh

Project đã có pipeline Data Curation tiếng Việt với các file chính:

```text
scripts/clean_with_nemo.py
scripts/nemo_backend.py
scripts/utils_text.py
scripts/build_long_context_testset.py
scripts/validate_testset.py
datasets/dataset_brief.md
README.md
```

Agent trước đó báo cáo rằng pipeline đang dùng backend:

```text
hybrid_nemo_python
```

với NeMo Curator cho các bước generic curation và Python cho Vietnamese-specific filters/dedup/metadata. Tôi cần bạn kiểm chứng lại bằng cách đọc code thật, không chỉ dựa vào mô tả trước đó.

## Nhiệm vụ

Hãy đọc trực tiếp các file:

```text
scripts/clean_with_nemo.py
scripts/nemo_backend.py
scripts/utils_text.py
scripts/build_long_context_testset.py
scripts/validate_testset.py
datasets/dataset_brief.md
README.md
```

Sau đó trả lời rõ:

1. Những bước nào thật sự gọi API NVIDIA NeMo Curator.
2. Những bước nào dùng Python custom.
3. Những bước nào thuộc cleaning/filtering.
4. Những bước nào thuộc packaging/benchmark/validation, không nên tính là NeMo cleaning.
5. Metadata output hiện ghi backend/steps có chính xác với code không.
6. Có chỗ nào mô tả quá tay, ví dụ nói “NeMo làm toàn bộ pipeline” nhưng thực tế là hybrid không.

## Lệnh kiểm chứng nên chạy

Chạy hoặc dùng lệnh tương đương:

```bash
grep -R "nemo_curator\|DocumentBatch\|UnicodeReformatter\|NewlineNormalizer\|WordCountFilter\|UrlsFilter\|NonAlphaNumericFilter\|WhiteSpaceFilter\|cleaning_backend\|nemo_curator_steps\|python_fallback_steps" -n scripts datasets README.md
```

Chạy thử cleaning để lấy metadata thật:

```bash
docker compose run --rm data-pipeline python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl --backend auto
```

Sau đó thống kê metadata:

```bash
docker compose run --rm data-pipeline python - <<'PY'
import json
from collections import Counter

backend = Counter()
nemo_steps = Counter()
python_steps = Counter()

with open("data/processed/cleaned.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        md = r.get("metadata", {})
        backend[md.get("cleaning_backend", "missing")] += 1
        for s in md.get("nemo_curator_steps", []):
            nemo_steps[s] += 1
        for s in md.get("python_fallback_steps", []):
            python_steps[s] += 1

print("backend counts:")
for k, v in backend.items():
    print(k, v)

print("\nNeMo Curator steps:")
for k, v in nemo_steps.items():
    print(k, v)

print("\nPython custom/fallback steps:")
for k, v in python_steps.items():
    print(k, v)
PY
```

Nếu có lỗi vì thiếu raw data, hãy dùng dataset hiện có hoặc chạy lại:

```bash
docker compose run --rm data-pipeline python scripts/download_datasets.py --max-records-per-source 200
docker compose run --rm data-pipeline python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl --backend auto
```

## Output yêu cầu

Trả về báo cáo dạng bảng:

| Nhóm bước | Bước cụ thể | Dùng NeMo hay Python | File/code liên quan | Bằng chứng | Ghi chú |
| --------- | ----------- | -------------------- | ------------------- | ---------- | ------- |

Phân nhóm tối thiểu:

### A. NeMo Curator steps

Liệt kê chính xác các API/class/function NeMo được gọi, ví dụ nếu có:

```text
DocumentBatch[pandas_dataframe]
Modify[UnicodeReformatter(normalization=NFC)]
Modify[NewlineNormalizer]
DocumentFilter[word_count:WordCountFilter]
DocumentFilter[urls_ratio:UrlsFilter]
DocumentFilter[alpha_numeric:NonAlphaNumericFilter]
DocumentFilter[white_space:WhiteSpaceFilter]
```

### B. Python custom cleaning/filtering steps

Liệt kê chính xác các bước còn chạy bằng Python, ví dụ:

```text
Vietnamese signal detection
replacement char filter
custom character length filter
exact dedup
near dedup
metadata enrichment
```

### C. Benchmark packaging/validation steps

Liệt kê những bước không thuộc NeMo cleaning, ví dụ:

```text
build 4k/8k/16k JSON test-set
tokenize with Qwen tokenizer
detokenize validation
schema validation
dataset_brief generation/update
```

## Đánh giá bắt buộc

Sau bảng, hãy kết luận:

1. Pipeline hiện tại có đúng là `hybrid_nemo_python` không?
2. Có thể gọi là “NeMo Curator pipeline” không, hay chỉ nên gọi là “hybrid NeMo Curator + Python custom pipeline”?
3. Phần nào không nên tuyên bố là do NeMo làm?
4. Nếu muốn NeMo hóa thêm, nên chuyển bước nào thành custom NeMo-compatible filters?
5. Có cần chỉnh `dataset_brief.md` hoặc README để mô tả chính xác hơn không?

## Nếu tài liệu mô tả chưa chính xác

Nếu phát hiện `dataset_brief.md` hoặc README mô tả sai, hãy đề xuất patch nội dung, nhưng chưa tự sửa nếu chưa được xác nhận.

## Quy tắc

* Không được phán đoán theo tên file.
* Phải đọc code thật.
* Không được nói “NeMo dùng hết” nếu dedup/build/validate vẫn là Python.
* Không được nói “chỉ import NeMo” nếu code thật có gọi API NeMo.
* Không xóa file hoặc sửa code trong bước review này.

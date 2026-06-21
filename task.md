Bạn là senior data engineering reviewer kiêm technical writer. Hãy tạo file báo cáo tiến độ cho Data Team tại:

```text
datateam_report.md
```

## Bối cảnh

Project đang thực hiện Task Data Pipeline tiếng Việt cho benchmark LLM/KV Cache. Data Team đã triển khai pipeline Docker + NVIDIA NeMo Curator để tạo bộ test-set dài ngữ cảnh tiếng Việt.

Các thành phần đã được triển khai gồm:

* Docker setup chạy được trên Linux container từ Windows.
* Pipeline tải dữ liệu từ Hugging Face.
* Pipeline làm sạch dữ liệu bằng backend hybrid NeMo Curator + Python.
* Custom NeMo-compatible filters cho tiếng Việt.
* Bộ test-set `datasets/test_set_small.json`.
* Tài liệu `datasets/dataset_brief.md`.
* Script validate dataset.

## Nhiệm vụ của bạn

Hãy đọc repo hiện tại, kiểm tra các file thật, sau đó viết báo cáo tiến độ vào `datateam_report.md`.

Không được chỉ dựa vào mô tả dưới đây. Phải kiểm chứng bằng code/file hiện có.

## Các file cần đọc

Đọc tối thiểu các file sau nếu tồn tại:

```text
Dockerfile
docker-compose.yml
requirements.txt
README.md
datasets/dataset_brief.md
datasets/test_set_small.json
scripts/download_datasets.py
scripts/clean_with_nemo.py
scripts/nemo_backend.py
scripts/utils_text.py
scripts/build_long_context_testset.py
scripts/validate_testset.py
```

Nếu file nào không tồn tại, ghi rõ trong báo cáo.

## Nội dung báo cáo cần có

Tạo `datateam_report.md` bằng tiếng Việt, cấu trúc như sau:

```markdown
# Báo cáo tiến độ Data Team

## 1. Tóm tắt trạng thái

## 2. Các hạng mục đã hoàn thành

## 3. Kiến trúc pipeline hiện tại

## 4. Docker & môi trường chạy

## 5. Nguồn dữ liệu

## 6. Quy trình làm sạch dữ liệu

## 7. Phân chia trách nhiệm NeMo Curator và Python

## 8. Test-set đầu ra

## 9. Kết quả kiểm thử và validation

## 10. Hạn chế hiện tại

## 11. Rủi ro kỹ thuật còn lại

## 12. Đề xuất bước tiếp theo

## 13. Kết luận
```

## Thông tin cần kiểm chứng và đưa vào báo cáo

### 1. Docker & môi trường

Kiểm tra và ghi rõ:

* Dockerfile dùng base image gì.
* Python version nếu có thể chạy kiểm tra.
* `requirements.txt` có `nemo-curator[text-cpu]==1.2.0` hay không.
* Docker service tên gì trong `docker-compose.yml`.
* Pipeline có bắt buộc GPU không.
* Lệnh build và chạy container.

Nên chạy:

```bash
docker compose config
docker compose run --rm data-pipeline python --version
docker compose run --rm data-pipeline python -c "import nemo_curator; print('nemo curator OK')"
```

Nếu không chạy được vì môi trường local, ghi rõ là chưa chạy lại được và dựa vào file cấu hình.

### 2. Pipeline scripts

Kiểm tra các script chính và mô tả vai trò:

```text
scripts/download_datasets.py
scripts/clean_with_nemo.py
scripts/nemo_backend.py
scripts/utils_text.py
scripts/build_long_context_testset.py
scripts/validate_testset.py
```

Mỗi script cần mô tả ngắn:

* Input.
* Output.
* Vai trò.
* Cách chạy.

### 3. Hybrid NeMo Curator + Python

Cần phân biệt rõ:

#### Phần dùng NeMo Curator

Kiểm tra trong code và liệt kê chính xác các bước, nếu có:

```text
DocumentBatch[pandas_dataframe]
Modify[UnicodeReformatter(normalization=NFC)]
Modify[NewlineNormalizer]
Modify[ProjectFtfyFixText]
DocumentFilter[WordCountFilter]
DocumentFilter[UrlsFilter]
DocumentFilter[NonAlphaNumericFilter]
DocumentFilter[WhiteSpaceFilter]
Custom NeMo-compatible filters:
- MinCharacterCountFilter
- ReplacementCharacterFilter
- LetterRatioFilter
- StrangeSymbolRatioFilter
- VietnameseSignalFilter
```

#### Phần dùng Python

Kiểm tra trong code và liệt kê phần còn dùng Python, ví dụ:

```text
exact_dedup
near_dedup
I/O orchestration
metadata writing
build 4k/8k/16k test-set
tokenize/detokenize validation
dataset brief/report generation
```

Không được viết rằng “100% pipeline làm bằng NeMo” nếu Python vẫn dùng cho dedup, packaging hoặc validation.

Cách mô tả đúng:

```text
Pipeline hiện là hybrid NeMo-compatible curation: NeMo backend xử lý phần lớn cleaning/filtering, bao gồm custom Vietnamese quality filters; Python giữ vai trò orchestration, corpus-level deduplication, packaging và validation.
```

### 4. Dataset output

Đọc `datasets/test_set_small.json` và thống kê:

* `dataset_name`
* `version`
* `language`
* tokenizer sử dụng
* tổng số samples
* số samples theo nhóm `4k`, `8k`, `16k`
* min/max/avg token mỗi nhóm
* source dataset chính

Nếu có thể, chạy:

```bash
python scripts/validate_testset.py --input datasets/test_set_small.json
```

Hoặc qua Docker:

```bash
docker compose run --rm data-pipeline python scripts/validate_testset.py --input datasets/test_set_small.json
```

Ghi kết quả validation vào báo cáo.

### 5. Nguồn dữ liệu

Kiểm tra `dataset_brief.md` và metadata trong JSON để ghi rõ:

* Dataset nào đã dùng thành công.
* Dataset nào chưa dùng được.
* Lý do nếu có.

Theo trạng thái gần nhất:

```text
VTSNLP/vietnamese_curated_dataset đã dùng thành công.
5760/vmlu có thể gặp lỗi 401/permission trong lần chạy trước.
V-Bench chưa tích hợp vì chưa xác định được dataset ID ổn định.
```

Nhưng hãy kiểm chứng lại trong file hiện có.

### 6. Hạn chế và rủi ro

Báo cáo cần ghi rõ, không che giấu:

* Nếu artifact hiện tại chỉ lấy từ một nguồn dữ liệu.
* Nếu VMLU/V-Bench chưa tích hợp được.
* Nếu dedup vẫn là Python.
* Nếu custom filters là NeMo-compatible nhưng không phải built-in NVIDIA filters.
* Nếu chưa chạy lại được Docker validation ở máy hiện tại.

### 7. Lệnh tái lập

Đưa vào báo cáo các lệnh tái lập:

```bash
docker compose build
docker compose run --rm data-pipeline bash
```

Bên trong container:

```bash
python scripts/download_datasets.py --max-records-per-source 5000
python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl --backend auto
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json
python scripts/validate_testset.py --input datasets/test_set_small.json
```

### 8. Chất lượng báo cáo

Báo cáo phải:

* Viết bằng tiếng Việt.
* Rõ ràng, có cấu trúc.
* Có bảng tóm tắt nếu phù hợp.
* Nêu trạng thái theo kiểu `Hoàn thành`, `Một phần`, `Chưa hoàn thành`.
* Phân biệt rõ “đã kiểm chứng bằng lệnh chạy” và “ghi nhận từ file/tài liệu”.
* Không phóng đại mức độ hoàn thành.

## Kiểm tra sau khi tạo report

Sau khi viết `datateam_report.md`, hãy chạy:

```bash
python - <<'PY'
from pathlib import Path
p = Path("datateam_report.md")
assert p.exists(), "datateam_report.md does not exist"
text = p.read_text(encoding="utf-8")
required = [
    "# Báo cáo tiến độ Data Team",
    "## 1. Tóm tắt trạng thái",
    "## 7. Phân chia trách nhiệm NeMo Curator và Python",
    "## 10. Hạn chế hiện tại",
    "## 13. Kết luận",
]
for r in required:
    assert r in text, f"Missing section: {r}"
print("datateam_report.md OK")
PY
```

## Output cần trả về sau khi hoàn thành

Sau khi tạo xong file, hãy báo cáo lại:

1. File đã tạo/sửa.
2. Các file đã đọc để tổng hợp.
3. Có chạy lại được validation không.
4. Tóm tắt 5-7 gạch đầu dòng về tiến độ Data Team.
5. Các hạn chế còn lại.
6. Commit message đề xuất.

Commit message đề xuất:

```text
docs(data): add Data Team progress report
```

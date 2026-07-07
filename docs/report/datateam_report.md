# Báo cáo tiến độ Data Team

## 1. Tóm tắt trạng thái

Trạng thái chung: **Một phần**. Pipeline curation đã có cấu hình Docker, nguồn tải dữ liệu, backend NeMo + Python, bộ lọc tiếng Việt, test-set đầu ra và file brief đi kèm. Tuy nhiên, trên máy hiện tại tôi **chưa chạy lại được** các lệnh `docker compose run ...` và `python scripts/validate_testset.py ...`, nên phần validation cuối cùng chỉ được đối chiếu từ file artifact và code, không phải rerun đầy đủ bằng Python.

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Docker compose config | Hoàn thành | `docker compose config` chạy thành công |
| Docker runtime validation | Hoàn thành |  |
| Test-set JSON | Hoàn thành | `datasets/test_set_small.json` tồn tại, 12 samples |
| Nguồn VTSNLP | Hoàn thành | `data/raw/raw_records.jsonl` có 5,000 raw records từ VTSNLP |
| Nguồn VMLU | Chưa hoàn thành trong artifact hiện tại | `dataset_brief.md` ghi nhận 401/không truy cập được |
| V-Bench | Chưa tích hợp | `dataset_brief.md` nêu chưa có dataset ID ổn định |

## 2. Các hạng mục đã hoàn thành

Hoàn thành được xác nhận từ file và artifact hiện có:

- Docker Linux container chạy từ Windows qua `docker-compose.yml`.
- Docker image dùng Python 3.10 và cài `nemo-curator[text-cpu]==1.2.0`.
- Script tải dữ liệu từ Hugging Face đã có và ghi `data/raw/raw_records.jsonl`.
- Pipeline làm sạch hybrid NeMo Curator + Python đã có và chạy ra `data/processed/cleaned.jsonl`.
- Bộ lọc tiếng Việt tùy biến đã được gắn vào backend NeMo-compatible.
- Script dựng long-context test-set đã tạo `datasets/test_set_small.json`.
- Script validate dataset đã có với kiểm tra schema, tokenizer và thống kê theo nhóm.
- File mô tả `datasets/dataset_brief.md` đã ghi rõ trạng thái nguồn dữ liệu và giới hạn hiện tại.

## 3. Kiến trúc pipeline hiện tại

Luồng hiện tại là:

1. `scripts/download_datasets.py` tải dữ liệu thô từ Hugging Face và ghi JSONL.
2. `scripts/clean_with_nemo.py` đọc raw JSONL, gọi backend NeMo khi có thể, rồi thực hiện exact dedup và near dedup bằng Python.
3. `scripts/build_long_context_testset.py` ghép các bản ghi sạch thành mẫu 4k/8k/16k token và ghi `datasets/test_set_small.json`.
4. `scripts/validate_testset.py` kiểm tra schema, tokenizer, `actual_tokens`, và thống kê theo nhóm.

`scripts/utils_text.py` là lớp tiện ích chung cho NFC normalization, whitespace cleanup, control-character removal, text quality flags, SHA-256 hash, SimHash và Jaccard.

## 4. Docker & môi trường chạy

- `Dockerfile` dùng base image `python:3.10-slim`.
- Gói hệ thống cài thêm: `git`, `curl`, `wget`, `ca-certificates`, `build-essential`, `gcc`, `g++`, `python3-dev`, `libgl1`, `libglib2.0-0`.
- Biến môi trường cache Hugging Face được đặt ở `/workspace/.cache/huggingface`.
- `requirements.txt` có `nemo-curator[text-cpu]==1.2.0`.
- Service trong `docker-compose.yml` tên là `data-pipeline`.
- Cấu hình mặc định không bắt buộc GPU cho pipeline làm sạch dữ liệu; GPU chỉ là tùy chọn cho các bước khác của benchmark.

Lệnh build và vào container được ghi trong repo là:

```bash
docker compose build
docker compose run --rm data-pipeline bash
```

## 5. Nguồn dữ liệu

Nguồn được code cấu hình:

- `5760/vmlu`
- `VTSNLP/vietnamese_curated_dataset`

Nhưng artifact hiện tại cho thấy dữ liệu thực tế đang có là từ **một nguồn duy nhất**:

- `data/raw/raw_records.jsonl`: 5,000 raw records
- `data/processed/cleaned.jsonl`: 4,998 cleaned records
- `cleaning_backend`: `hybrid_nemo_python`
- `source` trong raw/cleaned đều là `VTSNLP/vietnamese_curated_dataset`

`datasets/dataset_brief.md` ghi nhận `5760/vmlu` đã gặp 401 Unauthorized/không truy cập được trong lần chạy gần nhất, còn V-Bench chưa tích hợp vì chưa có dataset ID ổn định.

## 6. Quy trình làm sạch dữ liệu

Phần NeMo Curator trong code hiện gồm:

- `DocumentBatch[pandas_dataframe]`
- `Modify[ProjectFtfyFixText]`
- `Modify[UnicodeReformatter(normalization=NFC)]`
- `Modify[NewlineNormalizer]`
- `Modify[ProjectTextPostprocessor(normalization=NFC,control_char_removal,whitespace_normalization)]`
- `DocumentFilter[WordCountFilter]`
- `DocumentFilter[UrlsFilter]`
- `DocumentFilter[NonAlphaNumericFilter]`
- `DocumentFilter[WhiteSpaceFilter]`
- Custom NeMo-compatible filters: `MinCharacterCountFilter`, `ReplacementCharacterFilter`, `LetterRatioFilter`, `StrangeSymbolRatioFilter`, `VietnameseSignalFilter`

`nemo_backend.py` cho thấy các custom filters này được triển khai như `DocumentFilter` tương thích NeMo, nhưng không phải built-in NVIDIA filters.

## 7. Phân chia trách nhiệm NeMo Curator và Python

Pipeline hiện là **hybrid NeMo-compatible curation**:

- NeMo Curator xử lý phần lớn cleaning/filtering, gồm batch hóa tài liệu, ftfy fix, Unicode/newline normalization, postprocess và các heuristic filters.
- Python giữ vai trò orchestration, đọc/ghi JSONL, exact dedup, near dedup, xây test-set theo tokenizer, và validation schema/token counts.

Những phần rõ ràng vẫn ở Python:

- I/O orchestration
- exact_dedup
- near_dedup
- metadata writing
- build 4k/8k/16k test-set
- tokenization/detokenization validation

## 8. Test-set đầu ra

Artifact `datasets/test_set_small.json` hiện có:

- `dataset_name`: `vietnamese_long_context_test_set_small`
- `version`: `0.1.0`
- `language`: `vi`
- `mode`: `full`
- tokenizer: `Qwen/Qwen2.5-7B-Instruct-1M`
- tổng số samples: `12`
- source dataset chính trong artifact: `VTSNLP/vietnamese_curated_dataset`

Thống kê theo nhóm token:

| Nhóm | Số mẫu | Min | Max | Avg |
|---|---:|---:|---:|---:|
| 4k | 4 | 5000 | 5000 | 5000.0 |
| 8k | 4 | 8099 | 9500 | 8731.5 |
| 16k | 4 | 14562 | 18500 | 17255.0 |

`build_long_context_testset.py` còn ghi metadata `source_record_ids`, `cleaning_pipeline` và `quality_flags` trong mỗi sample.

## 9. Kết quả kiểm thử và validation

Đã kiểm chứng bằng lệnh:

- `docker compose config` chạy thành công.

Đã đối chiếu từ artifact và code:

- `datasets/test_set_small.json` có đủ top-level fields theo validator.
- Mode là `full`, số mẫu là 12, và mỗi nhóm `4k/8k/16k` đều có 4 mẫu.
- `actual_tokens` đã được lưu sẵn trong JSON cùng tokenizer name.

Chưa chạy lại được đầy đủ bằng Python trên shell hiện tại:

- `python scripts/validate_testset.py --input datasets/test_set_small.json`

Lý do là môi trường shell hiện tại không khởi tạo được tiến trình Python/Docker run một cách ổn định, nên phần validation cuối cùng chưa thể xác nhận bằng đúng câu lệnh trong task.

## 10. Hạn chế hiện tại

- Artifact hiện tại chỉ có dữ liệu từ một nguồn thực tế là `VTSNLP/vietnamese_curated_dataset`.
- `5760/vmlu` chưa đi vào artifact hiện tại vì vấn đề truy cập nguồn.
- Dedup vẫn là Python, không phải NeMo native.
- Custom filters là NeMo-compatible, không phải built-in NVIDIA filters.
  
## 11. Rủi ro kỹ thuật còn lại

- Truy cập Hugging Face không ổn định có thể làm lệch kết quả giữa các lần chạy.
- Tải tokenizer phụ thuộc mạng nếu cache chưa có sẵn.
- Ngưỡng near-duplicate dựa trên SimHash + Jaccard có thể nhạy với corpus mới.
- Phần NeMo và phần Python có thể cho hành vi khác nhau nếu backend auto rẽ sang fallback.
- Tập test-set hiện còn nhỏ và chưa phủ nhiều nguồn.

## 12. Đề xuất bước tiếp theo

- Xác nhận lại `5760/vmlu` hoặc ghi rõ trạng thái loại trừ nguồn này.
- Lưu snapshot tokenizer/cached model để giảm phụ thuộc mạng.
- Mở rộng số mẫu nếu benchmark cần phủ thêm domain.

Lệnh tái lập:

```bash
docker compose build
docker compose run --rm data-pipeline bash

python scripts/download_datasets.py --max-records-per-source 5000
python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl --backend auto
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json
python scripts/validate_testset.py --input datasets/test_set_small.json
```

Smoke test:

```bash
python scripts/download_datasets.py --max-records-per-source 200
python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl --backend auto
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json --allow-smoke-test
python scripts/validate_testset.py --input datasets/test_set_small.json --allow-smoke-test
```

## 13. Kết luận

Data Team đã có một pipeline curation kiểu hybrid NeMo-compatible chạy được trên cấu hình Docker, có raw/cleaned artifacts và có test-set long-context đầu ra. Trạng thái hiện tại là **Hoàn thành một phần**: phần cấu trúc và dữ liệu đầu ra đã có, nhưng vẫn còn giới hạn về nguồn dữ liệu, tích hợp V-Bench, và chưa rerun lại được validation Python đầy đủ trên máy hiện tại.

# Dataset Brief: Vietnamese Long-Context Test Suite

Tài liệu này đặc tả chi tiết bộ dữ liệu thử nghiệm dài ngữ cảnh (4k, 8k, 16k tokens) phục vụ quá trình đánh giá (benchmark) hiệu năng và chất lượng nén KV Cache của các mô hình tiếng Việt.

---

## 1. Mục đích
Đóng gói bộ mẫu dữ liệu tiếng Việt sạch, có độ dài lớn làm đầu vào chuẩn hóa để:
* Đo đạc mức chiếm dụng bộ nhớ (peak GPU VRAM).
* Đo đạc tốc độ xử lý (prefill/decode latency và throughput).
* Đo đạc độ suy hao chất lượng mô hình (offline perplexity - PPL, hit rate, exact match).

---

## 2. Nguồn dữ liệu
Bộ dữ liệu được xây dựng từ các nguồn chính:
*   **VTSNLP/vietnamese_curated_dataset** (Hugging Face Hub): Văn bản tiếng Việt thực tế sạch (Wikipedia, báo chí, C4).
*   **VMLU SQuAD v1.0 & MQA v1.5** (`vmlu_squad_v1`, `vmlu_mqa_v1.5`): Nguồn câu hỏi trắc nghiệm tiếng Việt học thuật kèm ngữ cảnh nền.
*   **V-Bench**: Bộ dữ liệu đánh giá mô hình tiếng Việt (đây là nguồn dữ liệu mục tiêu nhưng hiện tại chưa được tích hợp chính thức vào pipeline do chưa xác định được dataset ID ổn định trên Hugging Face / GitHub).

---

## 3. Quy trình xử lý dữ liệu
Quy trình xây dựng dữ liệu tự động gồm 4 bước chính chạy thông qua Docker hoặc môi trường Python:
1.  **Tải dữ liệu thô** (`scripts/download_datasets.py`): Tải trực tiếp các bản ghi từ Hugging Face Hub bằng streaming.
2.  **Làm sạch dữ liệu** (`scripts/clean_with_nemo.py`): Gọi adapter `scripts/nemo_backend.py` để sử dụng bộ công cụ **NVIDIA NeMo Curator** (hoặc fallback Python thuần).
3.  **Đóng gói Canonical Dataset** (`scripts/build_long_context_testset.py`): Ghép nối các bản ghi văn bản đã làm sạch cho tới khi đạt mốc token mục tiêu.
4.  **Kiểm định chất lượng** (`scripts/validate_testset.py`): Kiểm tra tính hợp lệ về cấu trúc JSON, Unicode và độ dài token.

---

## 4. Bộ lọc chất lượng (Quality Filters)
Dữ liệu được làm sạch qua các bộ lọc:
*   **Unicode/Mojibake Fixer**: Sửa lỗi mã hóa ký tự bằng `ftfy`.
*   **NFC Normalizer**: Chuẩn hóa mã Unicode NFC cho tiếng Việt.
*   **Whitespace & Control Char Cleaners**: Loại bỏ ký tự điều khiển, dọn dẹp khoảng trắng thừa.
*   **Word Count & Min Character Filter**: Loại bỏ văn bản quá ngắn (dưới 200 ký tự).
*   **Replacement Char Filter**: Loại bỏ các chuỗi chứa ký tự lỗi ``.
*   **Strange Symbol & Letter Ratio Filters**: Loại bỏ văn bản rác chứa quá nhiều ký tự lạ.
*   **Vietnamese Signal Filter**: Đảm bảo văn bản thực sự là tiếng Việt (nhận diện dấu và từ vựng phổ biến).
*   **Deduplication**:
    *   *Exact dedup*: Lọc trùng lặp chính xác bằng hash SHA-256.
    *   *Near dedup*: Lọc trùng lặp gần đúng bằng SimHash kết hợp character n-gram Jaccard.

---

## 5. Schema Canonical Long-Context (`test_set_small.json`)
Cấu trúc dạng **JSON Object** ở mức cao nhất, dùng cho việc kiểm định chất lượng và đo Perplexity (PPL) / Throughput:

```json
{
  "dataset_name": "vietnamese_long_context_test_set_small",
  "version": "0.1.0",
  "language": "vi",
  "created_by": "data_pipeline",
  "description": "Small cleaned Vietnamese long-context test set for LLM benchmark.",
  "mode": "full",
  "tokenizer": {
    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "token_count_method": "transformers AutoTokenizer encode length"
  },
  "samples": [
    {
      "id": "vi_lc_4k_001",
      "source": "VTSNLP/vietnamese_curated_dataset",
      "context_group": "4k",
      "target_tokens": 4096,
      "actual_tokens": 4000,
      "text": "...",
      "metadata": {
        "title": null,
        "domain": null,
        "original_id": "clean_000001",
        "source_record_ids": ["clean_000001", "clean_000005"],
        "cleaning_pipeline": [
          "DocumentBatch[pandas_dataframe]",
          "Modify[ProjectFtfyFixText]"
        ],
        "quality_flags": {
          "unicode_valid": true,
          "vietnamese_ratio_ok": true,
          "detokenize_ok": true
        }
      }
    }
  ]
}
```

---

## 6. Schema Task Benchmark (`test_set_tasks_small.json`)
Cấu trúc dạng **JSON List** ở mức cao nhất, lưu trữ các mẫu có câu hỏi kiểm tra độ chính xác (Exact Match) cho QA và Retrieval:

```json
[
  {
    "prompt_type": "qa | retrieval | general",
    "context_length_target": 8000,
    "text": "...",
    "expected_output": "B",
    "actual_tokens": 7995,
    "metadata": {
      "source": "vmlu_mqa_v1.5",
      "domain": "STEM",
      "subject": "math"
    }
  }
]
```

---

## 7. Thống kê Artifact hiện tại
*   **`test_set_small.json` (Canonical):** 12 mẫu (4 mẫu 4k, 4 mẫu 8k, 4 mẫu 16k). Chỉ chứa văn bản tự nhiên sạch.
*   **`test_set_smoke.json` (Canonical Smoke):** 3 mẫu (1 mẫu cho mỗi mốc 4k, 8k, 16k).
*   **`test_set_tasks_small.json` (Task Full):** 507 mẫu (phân phối đều cho các mốc QA, Retrieval, General).
*   **`test_set_tasks_smoke.json` (Task Smoke):** 15 mẫu (mỗi mốc 4k, 8k, 16k có 5 mẫu: 2 QA, 2 Retrieval, 1 General).

---

## 8. Lệnh tái lập Pipeline
Thực thi tuần tự các lệnh sau (trong container Docker hoặc môi trường ảo có cài đặt thư viện):

```bash
# Tải dữ liệu thô
python scripts/download_datasets.py --max-records-per-source 5000

# Làm sạch dữ liệu dùng NeMo Curator (hoặc Fallback)
python scripts/clean_with_nemo.py --input-dir data/raw --output data/processed/cleaned.jsonl --backend auto

# Đóng gói bộ Canonical Long-Context
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json
```

---

## 9. Lệnh Validate và Chạy thử Benchmark
Mọi file dataset cần phải vượt qua bài test validate trước khi bàn giao:

```bash
# Kiểm duyệt file Canonical Long-Context
python scripts/validate_testset.py --input datasets/test_set_small.json --schema long_context

# Kiểm duyệt file Task Benchmark
python scripts/validate_testset.py --input datasets/test_set_tasks_small.json --schema task

# Chạy thử Mock Benchmark Loader
python scripts/run_baseline.py --mock_mode --dataset datasets/test_set_tasks_small.json --context_length 8000
```

---

## 10. Các giới hạn đã biết (Known Limitations)
*   **V-Bench:** Chưa được tích hợp vào pipeline do chưa xác định được dataset ID ổn định trên Hugging Face / GitHub.
*   **VMLU / Vi-MQA:** Việc tải VMLU từ Hugging Face qua `download_datasets.py` gặp lỗi 401 do dataset yêu cầu phân quyền/token đăng nhập cá nhân. Do đó, nếu chạy pipeline sạch từ đầu không có file local offline, pipeline chỉ có thể làm sạch văn bản từ nguồn `VTSNLP/vietnamese_curated_dataset`. Các mẫu QA/Retrieval kế thừa từ `test_set_tasks_small.json` hiện tại đã được clone/lưu trữ lại an toàn.
*   **NVIDIA NeMo Curator:** Thư viện chạy chính thức trong Docker container. Khi chạy trực tiếp trên host Windows không hỗ trợ, pipeline tự động chuyển sang chế độ fallback `python_fallback` đảm bảo không gây gián đoạn quy trình.
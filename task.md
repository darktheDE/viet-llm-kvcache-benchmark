Bạn đã hoàn thành Docker setup và Data Pipeline, nhưng hiện tại `scripts/clean_with_nemo.py` mới chỉ import được `nemo_curator` rồi dùng Python fallback filters. Điều này chưa đủ vì task yêu cầu sử dụng NVIDIA NeMo Curator để làm sạch/lọc dữ liệu.

Hãy cập nhật pipeline để tích hợp NeMo Curator thật sự ở mức khả dụng với package hiện tại:

```text
nemo-curator[text-cpu]==1.2.0
```

Không làm lại Dockerfile, docker-compose.yml hoặc requirements.txt trừ khi thật sự bắt buộc.

## 1. Mục tiêu sửa

Cập nhật:

```text
scripts/clean_with_nemo.py
datasets/dataset_brief.md
README.md nếu cần
```

Có thể thêm helper:

```text
scripts/nemo_backend.py
```

Mục tiêu là `clean_with_nemo.py` phải thật sự thử dùng API của NeMo Curator để xử lý dataset. Python fallback chỉ được dùng khi API NeMo không khả dụng hoặc một bước cụ thể không hỗ trợ.

## 2. Yêu cầu quan trọng

Không được chỉ làm:

```python
import nemo_curator
```

rồi gọi đó là dùng NeMo.

Phải inspect package hiện tại trong container để tìm API thật:

```bash
python - <<'PY'
import nemo_curator
import pkgutil
print("nemo_curator:", nemo_curator)
print("submodules:")
for m in pkgutil.walk_packages(nemo_curator.__path__, nemo_curator.__name__ + "."):
    name = m.name
    if any(k in name.lower() for k in ["filter", "dedup", "document", "dataset", "modifier", "classifier"]):
        print(name)
PY
```

Sau đó kiểm tra class/function dùng được:

```bash
python - <<'PY'
import inspect
# import các module NeMo Curator tìm được rồi print signature class/function chính
PY
```

Dựa trên API thực tế của version đã cài, hãy tích hợp NeMo Curator vào pipeline.

## 3. Backend design bắt buộc

Trong `scripts/clean_with_nemo.py`, thiết kế backend rõ ràng:

```text
backend = "nemo_curator" nếu dùng được NeMo Curator cho bước xử lý chính
backend = "hybrid_nemo_python" nếu NeMo dùng một phần, Python dùng một phần
backend = "python_fallback" nếu không dùng được API NeMo
```

Mỗi record output phải có metadata:

```json
{
  "cleaning_backend": "nemo_curator hoặc hybrid_nemo_python hoặc python_fallback",
  "nemo_curator_available": true,
  "nemo_curator_steps": [
    "..."
  ],
  "python_fallback_steps": [
    "..."
  ]
}
```

## 4. Tối thiểu NeMo Curator phải tham gia vào một trong các nhóm sau

Hãy cố gắng dùng NeMo Curator trực tiếp cho ít nhất một bước thật:

1. Document dataset representation/loading.
2. Document filtering.
3. Text cleaning/modification.
4. Exact deduplication.
5. Fuzzy/near deduplication.
6. Quality filtering.

Nếu NeMo Curator có sẵn class filter/modifier/dedup phù hợp, hãy dùng nó.

Nếu API NeMo Curator không có filter tiếng Việt chuyên biệt, vẫn có thể dùng NeMo cho phần generic document pipeline/dataset/filtering, rồi dùng Python custom filters cho tiếng Việt.

## 5. Không được phá pipeline hiện tại

Các lệnh cũ vẫn phải chạy:

```bash
docker compose run --rm data-pipeline python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl
docker compose run --rm data-pipeline python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json
docker compose run --rm data-pipeline python scripts/validate_testset.py --input datasets/test_set_small.json
```

Nếu tích hợp NeMo làm full pipeline bị lỗi, thêm flag:

```bash
--backend auto
--backend nemo
--backend python
```

Ý nghĩa:

```text
--backend auto    ưu tiên NeMo, lỗi thì fallback Python
--backend nemo    bắt buộc dùng NeMo, lỗi thì fail
--backend python  chỉ dùng Python fallback
```

Default phải là:

```text
--backend auto
```

## 6. Validation mới

Sau khi sửa, chạy:

```bash
docker compose run --rm data-pipeline python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl --backend auto
```

Sau đó kiểm tra metadata:

```bash
python - <<'PY'
import json
from collections import Counter

counter = Counter()
nemo_steps = Counter()

with open("data/processed/cleaned.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        md = r.get("metadata", {})
        counter[md.get("cleaning_backend", "missing")] += 1
        for s in md.get("nemo_curator_steps", []):
            nemo_steps[s] += 1

print("backend counts:", counter)
print("nemo steps:", nemo_steps)
PY
```

Yêu cầu:

* Nếu `backend counts` chỉ toàn `python_fallback`, thì task chưa đạt yêu cầu NeMo.
* Cần có ít nhất một phần records dùng `nemo_curator` hoặc `hybrid_nemo_python`.
* Nếu thật sự không thể dùng API NeMo sau khi inspect, phải lưu log lỗi chi tiết và ghi rõ trong `dataset_brief.md`.

## 7. Cập nhật dataset_brief.md

Cập nhật phần quy trình xử lý.

Không được viết mơ hồ:

```text
Sử dụng NeMo Curator
```

Phải viết cụ thể:

```text
Pipeline chạy với backend hybrid_nemo_python. NeMo Curator được dùng cho các bước: ...
Python custom filters được dùng cho các bước tiếng Việt chuyên biệt: ...
```

Nếu vẫn fallback:

```text
NeMo Curator chỉ import được nhưng API xử lý chưa được tích hợp thành công trong phiên bản này.
```

Nhưng đây là trạng thái chưa đạt yêu cầu chính.

## 8. Kết quả cần báo cáo

Sau khi hoàn tất, trả về:

1. File đã sửa.
2. API/module NeMo Curator đã inspect được.
3. NeMo Curator được dùng ở bước nào.
4. Backend counts sau cleaning.
5. Số record dùng `nemo_curator`, `hybrid_nemo_python`, `python_fallback`.
6. Kết quả build test-set và validate lại.
7. Nếu không thể dùng NeMo API, đưa lỗi cụ thể và lý do kỹ thuật, không nói chung chung.

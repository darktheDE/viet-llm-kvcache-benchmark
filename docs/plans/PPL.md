# Kế hoạch Tái cấu trúc PPL & Quality Logging

Tài liệu này ghi lại trạng thái hiện tại của task **PPL & Quality Logging Refactor**, các phần đã triển khai, cách dùng script mới, và các vấn đề còn tồn đọng.

Ngày cập nhật: 2026-07-04

---

## 1. Mục tiêu kỹ thuật

Benchmark KV cache compression cần tách rõ 2 việc:

1. Model đang benchmark sinh `generated_text`.
2. PPL được tính bằng reference model BF16 gốc, không dùng logprobs của model đang bị nén KV cache.

Ngoài PPL, hệ thống cần lưu thêm quality diagnostics để phát hiện output lỗi khi nén sâu:

- `repetition_flag`
- `gibberish_flag`
- `repeated_ngram_ratio`
- `special_char_ratio`
- `output_length`
- `quality_warning`

Các cột số không được ghi sentinel string như `"OOM"`, `"ERROR"`, `"N/A"`. Khi lỗi, cột số để trống và nguyên nhân lỗi ghi vào status/error field riêng.

---

## 2. Trạng thái triển khai hiện tại

### 2.1. Đã làm trong code

#### Phase 2: Offline PPL script

Đã thêm script:

```text
scripts/compute_ppl_offline.py
```

Chức năng chính:

- Đọc raw CSV benchmark.
- Đọc JSONL generated outputs.
- Map CSV row với JSONL record theo `sample_id`, hoặc theo prefix `sample_id__s{i}` cho các dòng CSV aggregate hiện tại.
- Load tokenizer + reference model bằng `scripts/utils_ppl.py`.
- Tính PPL bằng reference model, không dùng logprobs của model bị nén.
- Ghi ra CSV mới, không overwrite mặc định.
- Hỗ trợ `--overwrite`.
- Hỗ trợ `--resume` để skip row đã có `ppl_status=OK`.
- Ghi lỗi từng sample vào `ppl_status` và `ppl_error`, không crash toàn bộ batch.
- Numeric lỗi để trống.
- Có progress log theo `--progress_every`.

CLI:

```bash
python scripts/compute_ppl_offline.py \
  --input_csv results/<raw_result>.csv \
  --input_jsonl results/<raw_result>_generated.jsonl \
  --output_csv results/<raw_result>_with_ppl.csv \
  --reference_model <model_or_path> \
  --device cuda \
  --dtype bf16 \
  --stride 512 \
  --ppl_mode conditional
```

#### Phase 3: Chuẩn hóa policy tính PPL

Đã thêm helper:

```text
scripts/utils_ppl.py
```

Policy mặc định:

- `--ppl_mode conditional`: score generated tokens conditioned on `prompt_text`.
- Ghép `prompt_text + generated_text`.
- Mask loss trên prompt tokens, chỉ score generated tokens.

Fallback:

- `--ppl_mode generated_only`: chỉ score `generated_text`.
- Dùng khi thiếu prompt hoặc khi conditional mode quá nặng trong môi trường hạn chế VRAM.

Trạng thái lỗi được chuẩn hóa:

- `OK`
- `EMPTY`
- `OOM`
- `ERROR`

#### Phase 4: Quality flags

Đã thêm helper:

```text
scripts/utils_generation_quality.py
```

Helper trả về các field:

```text
repetition_flag
gibberish_flag
repeated_ngram_ratio
special_char_ratio
output_length
quality_warning
```

Đã tích hợp vào:

```text
scripts/run_baseline.py
```

Khi chạy real benchmark, JSONL generated output sẽ lưu thêm quality fields cho từng sample. CSV summary row cũng có quality fields dạng aggregate.

Mock mode đã được sửa để không ghi PPL giả nữa. Thay vào đó:

```text
ppl_status=SKIPPED_MOCK
perplexity=""
```

#### Phase 6: Tests và validation

Đã thêm tests:

```text
tests/test_generation_quality.py
tests/test_compute_ppl_offline.py
```

Coverage hiện có:

- Text bình thường.
- Text rỗng.
- Text lặp.
- Text nhiều ký tự đặc biệt.
- Mapping CSV aggregate row sang JSONL sample records.
- Aggregate PPL theo token-weighted loss.
- Synthetic backfill với fake PPL function.
- CSV output tự thêm cột PPL/quality.

Đã chạy:

```bash
python -m py_compile scripts/run_baseline.py scripts/utils_generation_quality.py scripts/utils_ppl.py scripts/compute_ppl_offline.py
python -m unittest discover -s tests -p "test_*.py"
python scripts/compute_ppl_offline.py --help
python scripts/run_baseline.py --mock_mode --dataset datasets/test_set_smoke.json --context_length 4000 --max_new_tokens 4 --output results/_tmp_phase246_smoke.csv
```

Kết quả:

```text
8 tests OK
CLI help OK
mock smoke benchmark OK
```

File smoke tạm `results/_tmp_phase246_smoke.csv` đã được xóa sau khi kiểm tra.

---

## 3. Schema hiện tại

### 3.1. CSV mới từ `scripts/run_baseline.py`

CSV header hiện tại:

```text
model,kv_cache_type,context_length,peak_memory_mb,latency_ms_per_token,throughput_tokens_per_s,perplexity,ppl_loss,ppl_tokens,ppl_status,ppl_error,status,sample_id,output_path,repetition_flag,gibberish_flag,repeated_ngram_ratio,special_char_ratio,output_length,quality_warning
```

Ghi chú:

- Schema này rộng hơn schema 10 cột ban đầu.
- `perplexity`, `ppl_loss`, `ppl_tokens` để trống trước khi backfill.
- `ppl_status` cho biết trạng thái PPL.
- Quality fields được ghi trực tiếp vào CSV để downstream có thể filter nhanh.

### 3.2. JSONL generated outputs

Mỗi record JSONL chứa tối thiểu:

```json
{
  "sample_id": "sail/Sailor2-8B-Chat__FP16__8000__s0",
  "prompt_text": "Văn bản đầu vào gốc...",
  "generated_text": "Văn bản model sinh ra...",
  "generated_tokens": 128,
  "model": "sail/Sailor2-8B-Chat",
  "dataset": "datasets/test_set_small.json",
  "context_length": 8000,
  "kv_cache_type": "FP16",
  "kv_cache_dtype": "auto",
  "max_new_tokens": 128,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "seed": null,
  "status": "OK",
  "error_message": null,
  "repetition_flag": false,
  "gibberish_flag": false,
  "repeated_ngram_ratio": 0.0,
  "special_char_ratio": 0.0,
  "output_length": 128,
  "quality_warning": ""
}
```

---

## 4. Cách chạy backfill PPL

Ví dụ:

```bash
python scripts/compute_ppl_offline.py \
  --input_csv results/template_log_real_run.csv \
  --input_jsonl results/template_log_real_run_generated.jsonl \
  --output_csv results/template_log_real_run_with_ppl.csv \
  --reference_model sail/Sailor2-8B-Chat \
  --device cuda \
  --dtype bf16 \
  --stride 512 \
  --ppl_mode conditional
```

Resume:

```bash
python scripts/compute_ppl_offline.py \
  --input_csv results/template_log_real_run.csv \
  --input_jsonl results/template_log_real_run_generated.jsonl \
  --output_csv results/template_log_real_run_with_ppl.csv \
  --reference_model sail/Sailor2-8B-Chat \
  --device cuda \
  --dtype bf16 \
  --stride 512 \
  --ppl_mode conditional \
  --resume
```

Nếu conditional PPL bị OOM:

```bash
python scripts/compute_ppl_offline.py \
  --input_csv results/template_log_real_run.csv \
  --input_jsonl results/template_log_real_run_generated.jsonl \
  --output_csv results/template_log_real_run_with_ppl.csv \
  --reference_model sail/Sailor2-8B-Chat \
  --device cuda \
  --dtype bf16 \
  --stride 512 \
  --ppl_mode generated_only
```

---

## 5. Những vấn đề còn tồn đọng

### 5.1. Phase 1 chưa chuẩn hóa triệt để artifact linkage

Hiện `run_baseline.py` vẫn ghi CSV theo dạng summary row cho cả run:

```text
sample_id=<model>__<kv_cache_type>__<context_length>
```

Trong khi JSONL lưu từng sample:

```text
sample_id=<model>__<kv_cache_type>__<context_length>__s0
sample_id=<model>__<kv_cache_type>__<context_length>__s1
```

`compute_ppl_offline.py` đã hỗ trợ map bằng prefix để backfill được với schema hiện tại. Tuy nhiên thiết kế bền vững hơn là:

- ghi per-sample CSV row; hoặc
- thêm `output_record_id`; hoặc
- định nghĩa rõ CSV summary row sẽ nhận PPL aggregate từ nhiều JSONL records.

### 5.2. Phase 5 chưa làm: `plot_results.py`

Theo yêu cầu gần nhất, chỉ làm Phase 2, 3, 4, 6 nên chưa sửa:

```text
scripts/plot_results.py
```

Tồn đọng:

- Vẫn có rủi ro glob mù `*.csv`.
- Cần exclude `all_results_compiled.csv`, `all_results_summary.csv`, `template_log.csv`, `*_compiled.csv`, `*_summary.csv`.
- Cần không crash nếu thiếu `perplexity` hoặc quality columns.
- Cần cảnh báo rõ khi numeric parse bị coerce.

### 5.3. `scripts/test/run_real_benchmark.py` chưa được đồng bộ

File này không được sửa trong lần triển khai Phase 2/3/4/6 để tránh đụng file thực nghiệm/team khác.

Tồn đọng đã biết:

- Chưa tích hợp `scripts/utils_generation_quality.py`.
- Chưa dùng schema CSV mới có `ppl_status`, `ppl_error`, quality fields.
- Một nhánh load model error vẫn ghi `"ERROR"` vào numeric fields.
- Cần team sở hữu file này xác nhận trước khi sửa tiếp.

### 5.4. Kết quả cũ trong `results/` chưa được backfill

Các file hiện có như:

```text
results/template_log_demo_run.csv
results/all_results_compiled.csv
results/all_results_summary.csv
```

không phải raw generated-output artifacts đầy đủ cho PPL chuẩn.

Đặc biệt `template_log_demo_run.csv` là mock/demo và có các giá trị PPL giả từ phiên bản cũ. Không nên dùng các PPL này làm kết quả chất lượng thật.

### 5.5. Chưa test với reference model thật

Đã test bằng unit/synthetic và smoke CLI, nhưng chưa chạy:

- load BF16 reference model thật;
- tính PPL trên GPU;
- kiểm tra OOM thực tế với prompt 8k/16k/32k.

Cần chạy validation trên máy có GPU và model cache phù hợp.

### 5.6. CSV schema đã rộng hơn kế hoạch ban đầu

Kế hoạch ban đầu nói CSV 10 cột. Code hiện tại dùng CSV 20 cột để lưu cả PPL status/error và quality flags.

Đây là thay đổi có chủ đích để downstream filter nhanh hơn, nhưng cần cập nhật các consumer cũ nếu chúng vẫn giả định 10 cột.

---

## 6. Checklist hiện tại

- [x] Thêm `scripts/compute_ppl_offline.py`.
- [x] Thêm `scripts/utils_ppl.py`.
- [x] Hỗ trợ PPL `conditional`.
- [x] Hỗ trợ PPL `generated_only`.
- [x] Không dùng logprobs của model bị nén để tính PPL.
- [x] Không overwrite output CSV mặc định.
- [x] Có `--overwrite`.
- [x] Có `--resume`.
- [x] Ghi `ppl_status` và `ppl_error`.
- [x] Thêm `scripts/utils_generation_quality.py`.
- [x] Tích hợp quality flags vào `scripts/run_baseline.py`.
- [x] Mock mode không ghi PPL giả nữa.
- [x] Thêm unit/synthetic tests cho Phase 6.
- [ ] Chuẩn hóa per-sample CSV linkage hoặc `output_record_id`.
- [ ] Đồng bộ `scripts/test/run_real_benchmark.py`.
- [ ] Sửa `scripts/plot_results.py` để chỉ đọc raw benchmark CSV.
- [ ] Backfill lại kết quả thật sau khi có JSONL generated outputs.
- [ ] Chạy validation với BF16 reference model thật trên GPU.

---

## 7. Khuyến nghị bước tiếp theo

Ưu tiên tiếp theo:

1. Chạy một real benchmark nhỏ để tạo CSV + JSONL theo schema mới.
2. Chạy `compute_ppl_offline.py` với reference model BF16 trên output đó.
3. Kiểm tra CSV backfilled có PPL đúng dòng, quality flags đầy đủ.
4. Sau khi xác nhận, làm tiếp Phase 1 hoặc Phase 5 tùy nhu cầu team:
   - Phase 1 nếu cần backfill per-sample chính xác và lâu dài.
   - Phase 5 nếu cần plot/report ngay mà tránh đọc nhầm compiled/summary CSV.

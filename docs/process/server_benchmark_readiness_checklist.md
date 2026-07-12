# Checklist sẵn sàng thuê GPU và chạy benchmark thật

Ngày audit: 12/07/2026  
Phạm vi: **toàn bộ repository**, gồm source code, test, dataset, Docker/dependency,
model config, experiment config, results, plotting, PPL và tài liệu vận hành.  
Trạng thái: **NO-GO cho full benchmark trả phí**.

Tài liệu này là nguồn quyết định GO/NO-GO cấp dự án. Một commit mới không tự động làm
thay đổi trạng thái GO; toàn bộ gate liên quan phải được chạy lại và lưu bằng chứng.

## 1. Kết luận điều hành

Repository đã có nền tảng dữ liệu, mock pipeline, real runner, grid runner, VRAM sampler,
PPL offline và plotting. Tuy nhiên dự án chưa đủ điều kiện chạy full grid có giá trị khoa
học vì vẫn còn lỗi làm sai nhãn phương pháp nén, sai thống kê trạng thái, sai giới hạn
context, thiếu dependency GPU tái lập, thiếu metrics cốt lõi và chưa có real result.

Có thể thuê server cho **một phiên preflight/smoke có giới hạn thời gian** sau khi tất cả
mục P0 được hoàn thành. Không chạy full grid cho tới khi P0, P1 và P2 đều PASS.

## 2. Bằng chứng audit toàn dự án

### 2.1 Kết quả đã chạy local

| Kiểm tra | Kết quả | Kết luận |
| :--- | :--- | :--- |
| `python -m unittest discover -s tests -v` | 8/8 test PASS | PPL backfill và generation quality utility đạt test hiện có |
| `python -m compileall -q scripts tests` | PASS | Không có lỗi cú pháp Python |
| Canonical JSON parse bằng PowerShell | 12 full + 3 smoke sample, không có text rỗng | Cấu trúc JSON cơ bản đọc được |
| Task JSON parse bằng PowerShell | 507 full + 15 smoke sample, không có text rỗng | Cấu trúc task cơ bản đọc được |
| `validate_testset.py` | BLOCKED: thiếu `transformers` local | Chưa xác nhận validator end-to-end |
| Mock FP16 4k trên smoke set | FAIL: `Unexpected UTF-8 BOM` | Entrypoint `scripts/run_baseline.py` chưa đọc được dataset canonical hiện tại |
| `plot_results.py` trên dòng lỗi mock | Sinh 4 PNG và cảnh báo legend rỗng | Plotter không fail khi không có dòng `OK`; có nguy cơ tạo biểu đồ rỗng gây hiểu nhầm |

### 2.2 Dataset inventory

| Dataset | Số mẫu | Phân bố |
| :--- | ---: | :--- |
| `test_set_small.json` | 12 | 4k: 4, 8k: 4, 16k: 4 |
| `test_set_smoke.json` | 3 | 4k: 1, 8k: 1, 16k: 1 |
| `test_set_tasks_small.json` | 507 | 4k: 169, 8k: 168, 16k: 170 |
| `test_set_tasks_smoke.json` | 15 | Mỗi context: 2 QA, 2 Retrieval, 1 General |

Khoảng `actual_tokens` của canonical set:

| Bucket | Min | Max | Nhận xét |
| :--- | ---: | ---: | :--- |
| 4k | 5.000 | 5.000 | Vượt nhãn 4k |
| 8k | 8.099 | 9.500 | Có mẫu vượt buffer 1.024 token |
| 16k | 14.562 | 18.500 | Có mẫu vượt `max_model_len=17.152` hiện tại |

Ba file `PHAT_test_set_small.json`, `test_set_small.json` và `test_set_smoke.json` có
UTF-8 BOM. Real runner mới dùng `utf-8-sig`, nhưng entrypoint chính `run_baseline.py`
vẫn dùng `utf-8`, tạo hành vi khác nhau giữa hai pipeline.

### 2.3 Artefact và khả năng tái lập

- `results/all_results_compiled.csv`, `all_results_summary.csv` và
  `template_log_demo_run.csv` chỉ có header, chưa có real data.
- `configs/` và `experiments/` mới chỉ có README, chưa có manifest cấu hình hay
  provenance thực nghiệm.
- Dockerfile hiện là **CPU data-pipeline image**, không phải GPU benchmark image.
- `requirements.txt` thiếu dependency runtime GPU cốt lõi và phần lớn package không
  khóa phiên bản.
- Repository đang track nhiều file dữ liệu lớn khoảng 19-44 MB, trái boundary trong
  `AGENTS.md` và làm clone/upload server chậm.
- Có hai implementation benchmark khác nhau:
  `scripts/run_baseline.py` và `scripts/test/run_real_benchmark.py`; diff khoảng 673 dòng.
  Chưa có một canonical entrypoint duy nhất.

## 3. Issue register

### P0 - Có thể làm sai kết quả hoặc lãng phí toàn bộ phiên GPU

| ID | Vấn đề | Bằng chứng/phạm vi | Điều kiện đóng |
| :--- | :--- | :--- | :--- |
| P0-01 | HQQ bị thay bằng `int4_per_token_head` nhưng vẫn ghi nhãn HQQ | `scripts/test/run_real_benchmark.py` | Có implementation HQQ thật hoặc loại HQQ khỏi grid |
| P0-02 | PolarQuant bị thay bằng FP8 nhưng vẫn ghi nhãn PolarQuant | Real runner | Có PolarQuant thật hoặc loại khỏi grid |
| P0-03 | Model mặc định của real runner chưa đồng bộ với bộ benchmark 4-model chính thức | Real runner CLI | Default là model hợp lệ và có unit test CLI |
| P0-04 | Child runner ghi lỗi nhưng thường trả exit code 0 | Real runner + grid | Grid đọc status chuẩn hoặc child trả non-zero |
| P0-05 | Grid có thể đếm LOAD_ERROR/OOM/ERROR thành success | `returncode == 0` | Test tự động chứng minh phân loại đúng mọi status |
| P0-06 | Context thực tế vượt `max_model_len` | Dataset 8k/16k | Tokenize/truncate theo từng model, ghi prompt tokens |
| P0-07 | `run_baseline.py` không đọc UTF-8 BOM | Canonical/smoke JSON | Tất cả loader dùng chính sách encoding thống nhất |
| P0-08 | Hai benchmark runner lệch logic/schema/model | `scripts/` và `scripts/test/` | Chọn một canonical runner, runner còn lại bỏ/deprecate |
| P0-09 | Không có environment GPU khóa phiên bản | requirements/Docker | Có lock file hoặc GPU image digest đã smoke-test |
| P0-10 | Chưa chốt method matrix khoa học | README, plans, runner | Manifest nêu đúng model/method/preset/context |

### P1 - Metrics hoặc vận hành chưa đạt mục tiêu nghiên cứu

| ID | Vấn đề | Điều kiện đóng |
| :--- | :--- | :--- |
| P1-01 | Latency aggregate đang được gọi là ITL | Đo TTFT và ITL theo timestamp/engine metric |
| P1-02 | Chưa ghi base/dynamic VRAM | Thêm hai trường và test công thức |
| P1-03 | Chưa có compression ratio/GPU efficiency | Thêm schema, công thức, unit test |
| P1-04 | CSV của hai runner không cùng schema | Một schema versioned duy nhất |
| P1-05 | Không có per-sample metric/repeat đầy đủ | Ghi sample rows và tối thiểu 3 repeat ở pilot |
| P1-06 | Seed đang là `None` | Set và persist seed/sampling params |
| P1-07 | Grid không resume/skip/anti-duplicate | Có manifest và idempotent resume |
| P1-08 | Timeout 600 giây gồm cả download/load/inference | Tách timeout hoặc cấu hình theo phase |
| P1-09 | Plotter tạo biểu đồ khi không có valid row | Fail non-zero nếu không có dữ liệu `OK` |
| P1-10 | Plotter dùng `contains("OK")` | Chỉ chấp nhận status chính xác theo enum/schema |
| P1-11 | PPL có code/test nhưng chưa chạy trên pilot GPU | Backfill pilot thành công và join đúng CSV |
| P1-12 | EM/F1/ROUGE chưa tích hợp end-to-end | Có evaluator và test trên task smoke set |

### P2 - Tính nhất quán, tài liệu và quản trị artefact

| ID | Vấn đề | Điều kiện đóng |
| :--- | :--- | :--- |
| P2-01 | Scope tài liệu và runner phải thống nhất bộ 4 model chính thức | Cập nhật scope đồng bộ trên toàn repo |
| P2-02 | Real runner docstring/ví dụ lệnh không được còn nhắc Gemma đã bị loại | Tài liệu và CLI khớp code |
| P2-03 | Grid in “75 cấu hình” nhưng thực tế 60 | Tính/hiển thị total động ở mọi chỗ |
| P2-04 | Config/experiments chưa có file thực tế | Có manifest, environment và run record |
| P2-05 | Result hiện chỉ có header/mock plot | Real pilot artefact được lưu ngoài Git đúng quy định |
| P2-06 | File dữ liệu lớn đang được Git track | Chuyển storage phù hợp và cập nhật `.gitignore` |
| P2-07 | Test không bao phủ runner/grid/plot/validator | Có test lỗi, OOM, BOM, context và empty-data |
| P2-08 | Python local là 3.12 nhưng project khai báo 3.10 | Chốt và test một version chính thức |

## 4. Những thành phần đã đạt mức code-level

- [x] Có canonical và task dataset full/smoke với số mẫu/phân bố cơ bản hợp lệ.
- [x] Có xử lý `torch.cuda.OutOfMemoryError` trong runner.
- [x] Có background NVML sampler trong real runner.
- [x] Có lưu generated text JSONL để PPL offline.
- [x] Có PPL resume/backfill và 4 unit test PASS.
- [x] Có generation-quality utility và 4 unit test PASS.
- [x] Toàn bộ file Python compile thành công trên Python 3.12.5.
- [x] Có cơ chế lấy Hugging Face token từ environment.
- [x] Có basic CSV aggregation và bốn basic plot.

Các dấu `[x]` ở đây chỉ xác nhận thành phần code tồn tại/đạt test hiện có; chúng không
đồng nghĩa dự án đã GO cho benchmark GPU.

## 5. P0 checklist - Hoàn thành trước khi thuê server

### 5.1 Chốt kiến trúc benchmark

- [ ] Chọn một canonical entrypoint, đề xuất `scripts/run_benchmark.py` hoặc một tên rõ
  ràng ngoài thư mục `scripts/test/`.
- [ ] Chọn một canonical grid runner dùng cùng module/config với entrypoint.
- [ ] Loại/deprecate runner cũ để README và người vận hành không gọi nhầm.
- [ ] Tạo `configs/benchmark_matrix.yaml` chứa model, HF repo, method, dtype preset,
  context, samples, repeats và output schema version.
- [ ] Grid đọc manifest thay vì hard-code danh sách riêng.

### 5.2 Đảm bảo fidelity của phương pháp

- [ ] FP16/BF16 baseline ghi rõ cache dtype thực tế.
- [ ] FP8 được xác nhận bằng log engine.
- [ ] TurboQuant preset được chốt và ghi đúng tên thực tế vào CSV.
- [ ] Không chạy HQQ nếu chưa có HQQ KV-cache implementation thật.
- [ ] Không chạy PolarQuant nếu chưa có PolarQuant implementation thật.
- [ ] Fail-fast nếu method không được hỗ trợ; tuyệt đối không fallback nhưng giữ nhãn cũ.
- [ ] Lưu `requested_method`, `effective_kv_cache_dtype` và backend vào mỗi record.

### 5.3 Dataset và token budget

- [ ] Tất cả JSON loader dùng encoding thống nhất (`utf-8-sig` hoặc loại BOM khỏi file).
- [ ] Validator full và smoke chạy PASS trong environment chính thức.
- [ ] Tokenize bằng tokenizer của từng model trước inference.
- [ ] Ghi `actual_prompt_tokens`, `generated_tokens` và `max_model_len`.
- [ ] Chọn và tài liệu hóa policy truncate/pad/bucket.
- [ ] Fail nếu bucket rỗng; không âm thầm dùng sample bucket khác.
- [ ] Bảo đảm `prompt_tokens + max_new_tokens <= max_model_len` cho từng request.

### 5.4 Status, schema và điều phối

- [ ] Định nghĩa status enum: `OK`, `OOM`, `LOAD_ERROR`, `DATASET_ERROR`, `TIMEOUT`,
  `INFERENCE_ERROR`, `UNSUPPORTED`.
- [ ] Mỗi lỗi có exit code hoặc structured result mà grid đọc được.
- [ ] Có unit test chứng minh grid không đếm lỗi thành success.
- [ ] CSV/JSONL có schema version và header thống nhất.
- [ ] Có `run_id` và config fingerprint chống duplicate.
- [ ] Có resume/skip cấu hình đã hoàn thành.
- [ ] Không ghi chuỗi `ERROR` vào cột numeric.

### 5.5 Environment GPU tái lập

- [ ] Tạo GPU requirements/lock riêng, khóa `vllm`, `torch`, `transformers`,
  `huggingface_hub`, `nvidia-ml-py/pynvml` và analysis dependencies.
- [ ] Tạo GPU Dockerfile/image riêng; không dùng CPU data-pipeline image.
- [ ] Ghi OS, Python, CUDA, driver, GPU compute capability và image digest.
- [ ] Có `scripts/preflight_gpu.py` kiểm tra import, GPU, disk, model access và dtype.
- [ ] Chạy unit test, validator, mock smoke trong chính GPU image trước khi tính giờ thuê.

## 6. P1 checklist - Test local/container trước phiên GPU

- [x] 8 unit test hiện tại PASS.
- [x] Python compile PASS.
- [ ] Cài đầy đủ dependency từ lock file trong environment sạch.
- [ ] Hai validator smoke PASS.
- [ ] Hai validator full PASS.
- [ ] Mock baseline FP16 4k tạo status `MOCK_OK`, không phải dataset error.
- [ ] Mock grid hoàn tất đúng tổng số cấu hình và không duplicate.
- [ ] Plotter fail khi input không có dòng `OK`.
- [ ] Plotter sinh đủ plot khi dùng fixture có dữ liệu hợp lệ.
- [ ] Test BOM cho mọi loader.
- [ ] Test prompt dài hơn max model length.
- [ ] Test status/exit code cho load error, dataset error, OOM và timeout.
- [ ] Test PPL backfill trên fixture CSV/JSONL end-to-end.

## 7. P2 checklist - Phiên GPU smoke có giới hạn

Chỉ bắt đầu phần này khi P0 và P1 đã PASS.

### 7.1 Preflight server

- [ ] `nvidia-smi` nhận đúng GPU/VRAM và driver.
- [ ] Có tối thiểu 100 GB disk trống hoặc dung lượng phù hợp manifest.
- [ ] Có đủ host RAM và shared memory; Docker dùng `--ipc=host` nếu cần.
- [ ] `HF_TOKEN` nằm trong environment, không xuất hiện trong log/command commit.
- [ ] Model gated đã được cấp quyền trước khi bắt đầu tính tiền.
- [ ] Lưu Git SHA, dirty status, image digest, Python, CUDA, PyTorch và vLLM version.

### 7.2 Thứ tự smoke bắt buộc

- [ ] 1 model nhỏ/ổn định, FP16, 1 sample 4k, 16 new tokens.
- [ ] Cùng model/config với FP8.
- [ ] Cùng model/config với từng TurboQuant preset đã chốt.
- [ ] Xác nhận log backend/dtype đúng với label CSV.
- [ ] Test một model 8k rồi 16k, kiểm tra actual token budget.
- [ ] Test OOM có kiểm soát; grid phải ghi OOM, không tăng success.
- [ ] Test model ID sai; grid phải ghi LOAD_ERROR.
- [ ] Restart grid và xác nhận resume không chạy trùng.

### 7.3 Metric acceptance

- [ ] Peak VRAM có base, inference peak và dynamic delta.
- [ ] TTFT được đo riêng.
- [ ] ITL được đo từ decode token timing, không phải average end-to-end latency.
- [ ] Throughput ghi rõ aggregate/request-level.
- [ ] Compression ratio và GPU efficiency có công thức/version.
- [ ] Per-sample rows đủ prompt/generated token count.
- [ ] Sampling seed/params được lưu và chạy lại cho kết quả nhất quán.
- [ ] PPL offline backfill pilot thành công.
- [ ] EM/F1 trên task smoke set thành công nếu nằm trong scope chính thức.

## 8. P3 checklist - Gate trước full grid

- [ ] Pilot đủ model đại diện, method và 4k/8k/16k.
- [ ] Mỗi config pilot có tối thiểu 3 repeat, tính được mean/std.
- [ ] Ước lượng thời gian/chi phí từ pilot và đặt hard budget.
- [ ] Manifest chỉ chứa tổ hợp đã smoke PASS.
- [ ] Backup CSV/JSONL/log định kỳ sang persistent storage.
- [ ] Có monitor tiến độ theo config count, không theo process exit code đơn thuần.
- [ ] Có stop condition khi error rate/OOM/cost vượt ngưỡng.
- [ ] PPL và plotting đọc được pilot real data.
- [ ] Người phụ trách ghi quyết định GO kèm Git SHA, image digest và manifest version.

## 9. P4 checklist - Sau full grid

- [ ] Tổng `OK + OOM + ERROR + TIMEOUT + UNSUPPORTED` bằng tổng manifest.
- [ ] Không có method label khác effective backend/dtype.
- [ ] Không có duplicate config fingerprint.
- [ ] PPL/quality metric hoàn tất hoặc có lý do rõ cho dòng lỗi.
- [ ] Mean/std được tính từ repeat hợp lệ, warm-up được xử lý nhất quán.
- [ ] Plot/tables được sinh lại chỉ từ real data.
- [ ] Lưu provenance, chi phí và log thực nghiệm.
- [ ] Cập nhật Sprint/Plane chỉ sau khi artefact được peer-review.

## 10. Lệnh kiểm tra đề xuất

### Local/container

```bash
python --version
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
python scripts/validate_testset.py --input datasets/test_set_smoke.json --schema long_context --allow-smoke-test
python scripts/validate_testset.py --input datasets/test_set_tasks_smoke.json --schema task --allow-smoke-test
```

### GPU server preflight

```bash
nvidia-smi
python --version
python -c "import torch, vllm, pynvml; print(torch.__version__, vllm.__version__); print(torch.cuda.get_device_name(0))"
df -h
git rev-parse HEAD
git status --short
```

Lệnh smoke cuối cùng phải gọi canonical entrypoint sau khi P0-08 được đóng; không dùng
hai runner hiện tại thay thế cho nhau.

## 11. Quy tắc quyết định GO/NO-GO

- **NO-GO:** còn bất kỳ P0 nào chưa đóng.
- **SMOKE-ONLY:** P0 và P1 PASS, nhưng P2 chưa hoàn thành.
- **PILOT-READY:** P0-P2 PASS, được phép chạy pilot có giới hạn chi phí.
- **FULL-GRID GO:** P0-P3 PASS và có phê duyệt ghi Git SHA/image/manifest.
- **RESULTS ACCEPTED:** P4 PASS và peer review xác nhận method fidelity + provenance.

Trạng thái tại lần audit này: **NO-GO**.

Bạn là senior DevOps/Data Engineering agent. Hãy tạo bộ Docker setup có thể tái lập môi trường NeMo Curator cho project này, để bất kỳ người nào clone source code về cũng có thể build và chạy pipeline giống môi trường hiện tại của tôi.

## Mục tiêu

Không dùng lại container local của tôi. Không dùng `docker commit` làm giải pháp chính.

Yêu cầu là viết Dockerfile và các file cấu hình sao cho:

```bash
git clone <repo>
cd <repo>
docker compose build
docker compose run --rm data-pipeline bash
```

sẽ tạo được môi trường Linux có đầy đủ dependency để chạy pipeline xử lý dữ liệu tiếng Việt bằng NVIDIA NeMo Curator.

## Bối cảnh

* Host chính của tôi là Windows.
* NeMo Curator cần chạy trong Linux container.
* Pipeline xử lý text phải chạy được ở CPU mode.
* GPU là tùy chọn, không bắt buộc.
* Trước đó tôi đã từng cài thành công trong Docker và kiểm tra được:

```bash
python -c "import nemo_curator; print('nemo curator OK')"
```

* Tuy nhiên người khác clone repo sẽ không có container cũ của tôi, nên Dockerfile phải tự cài lại từ đầu.

## File cần tạo hoặc cập nhật

Hãy tạo/cập nhật các file sau:

```text
Dockerfile
docker-compose.yml
requirements.txt
.dockerignore
README.md
```

Nếu file đã có sẵn, hãy đọc trước rồi cập nhật cẩn thận. Không xóa nội dung quan trọng.

---

# 1. Dockerfile

Tạo `Dockerfile` có khả năng build lại môi trường từ đầu.

Ưu tiên dùng base image nhẹ và ổn định:

```dockerfile
FROM python:3.10-slim
```

Nếu `nemo-curator` cần nhiều dependency hệ thống hơn, có thể dùng:

```dockerfile
FROM ubuntu:22.04
```

nhưng phải tự cài Python 3.10.

Dockerfile phải có:

```dockerfile
WORKDIR /workspace
```

Cài system dependencies tối thiểu:

```bash
git
curl
wget
ca-certificates
build-essential
gcc
g++
python3-dev
libgl1
libglib2.0-0
```

Set environment variables:

```dockerfile
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
```

Cài dependencies từ `requirements.txt`:

```dockerfile
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
```

Tạo sẵn thư mục:

```dockerfile
RUN mkdir -p \
    data/raw \
    data/interim \
    data/processed \
    datasets \
    scripts \
    .cache/huggingface
```

Cuối file:

```dockerfile
CMD ["/bin/bash"]
```

---

# 2. requirements.txt

Tạo `requirements.txt` đủ để chạy pipeline.

Bắt buộc có các package:

```text
datasets
transformers
tokenizers
sentencepiece
accelerate
pandas
numpy
pyarrow
tqdm
regex
ftfy
scikit-learn
jsonschema
matplotlib
plotly
jupyter
ipykernel
nemo-curator
```

Nếu `nemo-curator` trên pip không cài được, hãy thử kiểm tra tên package thực tế. Có thể thử:

```text
nemo-curator
```

hoặc package NVIDIA tương ứng nếu tên package đã thay đổi.

Yêu cầu quan trọng:

* Không pin version quá chặt nếu không cần.
* Nhưng nếu build bị lỗi do version conflict, hãy pin version để build ổn định.
* Ghi rõ trong README version Python và package chính.
* Không để Dockerfile phụ thuộc vào môi trường local của tôi.

---

# 3. docker-compose.yml

Tạo service:

```yaml
services:
  data-pipeline:
    build:
      context: .
      dockerfile: Dockerfile
    image: vietnamese-data-pipeline:latest
    container_name: vietnamese-data-pipeline
    working_dir: /workspace
    volumes:
      - .:/workspace
      - hf-cache:/workspace/.cache/huggingface
    environment:
      - PYTHONUNBUFFERED=1
      - HF_HOME=/workspace/.cache/huggingface
      - TRANSFORMERS_CACHE=/workspace/.cache/huggingface
      - HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
    stdin_open: true
    tty: true
    command: bash

volumes:
  hf-cache:
```

Không bật GPU mặc định. Nếu muốn hỗ trợ GPU, thêm hướng dẫn riêng trong README hoặc tạo service phụ `data-pipeline-gpu`, nhưng service mặc định phải chạy được trên máy không có GPU.

---

# 4. .dockerignore

Tạo `.dockerignore` để build nhanh, tránh copy dữ liệu nặng:

```text
.git
__pycache__
*.pyc
.ipynb_checkpoints
data/raw
data/interim
data/processed
datasets/*.json
.cache
.venv
venv
env
*.log
results
```

Không ignore:

```text
scripts/
Dockerfile
docker-compose.yml
requirements.txt
README.md
```

---

# 5. README.md

Cập nhật README với section:

```markdown
## Docker setup for Vietnamese Data Pipeline
```

Nội dung phải có:

### Yêu cầu

* Docker Desktop trên Windows/macOS/Linux.
* Internet để tải Python packages và Hugging Face datasets.
* GPU không bắt buộc.

### Build image

PowerShell/CMD/Linux đều dùng được:

```bash
docker compose build
```

### Vào container

```bash
docker compose run --rm data-pipeline bash
```

### Kiểm tra môi trường

```bash
python --version
python -c "import pandas, datasets, transformers; print('basic deps OK')"
python -c "import nemo_curator; print('nemo curator OK')"
```

### Chạy pipeline full

```bash
python scripts/download_datasets.py --max-records-per-source 5000
python scripts/clean_with_nemo.py --input-dir data/raw --output data/processed/cleaned.jsonl
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json
python scripts/validate_testset.py --input datasets/test_set_small.json
```

### Chạy smoke test

```bash
python scripts/download_datasets.py --max-records-per-source 200
python scripts/clean_with_nemo.py --input-dir data/raw --output data/processed/cleaned.jsonl
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json --allow-smoke-test
python scripts/validate_testset.py --input datasets/test_set_small.json --allow-smoke-test
```

### GPU tùy chọn

Nếu máy có NVIDIA GPU và đã cài NVIDIA Container Toolkit, có thể chạy:

PowerShell:

```powershell
docker run --rm -it --gpus all -v ${PWD}:/workspace vietnamese-data-pipeline:latest bash
```

CMD:

```cmd
docker run --rm -it --gpus all -v "%cd%:/workspace" vietnamese-data-pipeline:latest bash
```

Nhưng phải ghi rõ: pipeline text cleaning không bắt buộc GPU.

---

# 6. Kiểm thử bắt buộc

Sau khi viết file, hãy chạy:

```bash
docker compose config
docker compose build
docker compose run --rm data-pipeline python --version
docker compose run --rm data-pipeline python -c "import pandas, datasets, transformers; print('basic deps OK')"
docker compose run --rm data-pipeline python -c "import nemo_curator; print('nemo curator OK')"
```

Nếu build lỗi ở `nemo-curator`, hãy sửa Dockerfile/requirements cho đến khi build được, hoặc ghi rõ phương án thay thế trong README.

Không được trả về kết quả “xong” nếu chưa có hướng dẫn xử lý lỗi.

---

# 7. Kết quả cần báo cáo

Sau khi hoàn thành, hãy trả về:

1. Danh sách file đã tạo/sửa.
2. Nội dung Dockerfile cuối cùng.
3. Nội dung docker-compose.yml cuối cùng.
4. Nội dung requirements.txt cuối cùng.
5. Lệnh build.
6. Lệnh vào container.
7. Lệnh kiểm tra `nemo_curator`.
8. Kết quả test thực tế hoặc lỗi còn tồn tại.
9. Ghi chú nếu môi trường này khác môi trường local cũ của tôi.

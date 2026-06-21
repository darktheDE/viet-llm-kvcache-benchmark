FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

RUN mkdir -p \
    data/raw \
    data/interim \
    data/processed \
    datasets \
    scripts \
    .cache/huggingface

CMD ["/bin/bash"]

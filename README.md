# Benchmarking TurboQuant and KV Cache Compression Methods on Vietnamese Large Language Models

[![Course](https://img.shields.io/badge/Course-DBML434077-blue)](https://github.com/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20vLLM%20%7C%20llama.cpp-orange)](https://github.com/)
[![Status](https://img.shields.io/badge/Status-In--Progress-yellow)](https://github.com/)

This repository contains the official implementation, datasets, experimental configurations, and benchmarking results for our research project on evaluating **TurboQuant** and various **Key-Value (KV) Cache Compression** techniques on Vietnamese Large Language Models (LLMs).

This project is conducted as the final group research project for the course **Big Data Applications: Machine Learning at Scale (DBML434077) at Ho Chi Minh City University of Technology and Engineering (HCM-UTE)**.

---

> [!IMPORTANT]
> **Developer Quick Start:** Please read the **[Development Guideline](development.md)** before writing code, creating branches, or submitting Pull Requests.

## Table of Contents
- [Project Overview](#-project-overview)
- [Research Questions](#-research-questions)
- [System Architecture & Methodology](#-system-architecture--methodology)
- [Benchmark Scope](#-benchmark-scope)
  - [Models](#models)
  - [Compression Methods](#compression-methods)
  - [Datasets](#datasets)
  - [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#-project-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Docker setup for Vietnamese Data Pipeline](#docker-setup-for-vietnamese-data-pipeline)
- [Quick Start](#-quick-start)
- [Workflow & Timeline](#-workflow--timeline)
- [Contributors](#-contributors)
- [Course Information](#-course-information)
- [References](#-references)

---

## Project Overview

Deploying Large Language Models (LLMs) for real-world Vietnamese applications (e.g., long-document processing, context-aware Q&A, and conversational assistants) is growing rapidly. However, during autoregressive generation, the **Key-Value (KV) Cache** scales linearly with context length, becoming a critical memory bottleneck, especially in VRAM-constrained hardware environments.

While recent KV Cache compression and weight quantization techniques such as **TurboQuant**, **RaBit-Q**, **PolarQuant**, and **HQQ** show significant promise in reducing memory footprint and speeding up inference, most existing benchmarks focus on English or generic datasets. It remains systematically unexamined how these methods perform on Vietnamese LLMs, where tokenization and linguistic structures differ significantly.

This project aims to fill this gap by establishing a **systematic, reproducible empirical benchmark** to evaluate the trade-offs between memory footprint, inference latency, throughput, and output quality (perplexity) across various context lengths for Vietnamese LLMs.

---

## Research Questions

Our empirical study is guided by four primary research questions:
*   **RQ1:** Does TurboQuant significantly reduce the memory footprint and inference latency of Vietnamese LLMs compared to the Full KV Cache baseline?
*   **RQ2:** To what extent does the output quality (measured by perplexity and downstream metrics) degrade under different compression ratios across various text domains?
*   **RQ3:** How do the trade-offs between hardware efficiency and output quality behave as context length scales up (from 4k to 32k tokens)?
*   **RQ4:** Which compression strategy (TurboQuant vs. baseline methods like RaBit-Q, PolarQuant, HQQ) offers the most stable pareto-optimal boundary for production-level Vietnamese LLM deployment?

---

## System Architecture & Methodology

The benchmark pipeline consists of three core stages:
1.  **Data Curation & Preprocessing:** Compiling standardized long-context Vietnamese inputs and utilizing **NVIDIA NeMo Curator** to clean and format the datasets.
2.  **Inference Pipeline Execution:** Deploying target models with selected inference frameworks (e.g., `vLLM`, `llama.cpp`, or `TGI`) configured with either Full KV Cache, weight quantization (TurboQuant), or baseline KV Cache compression methods.
3.  **Measurement & Logging:** Utilizing automated scripts to capture peak GPU VRAM consumption, latency per token, decoding throughput, and model perplexity.

---

## Benchmark Scope

### Models
We evaluate the methods across **4 to 6 Vietnamese LLMs** (including foundational and instruction-tuned variants):
*   `PhoGPT-7B` (or equivalent foundational Vietnamese models)
*   `Qwen-VN` (Vietnamese adapted variants)
*   `Llama-VN` (Vietnamese adapted variants)
*   `URA-LLaMa-3-8B` (Vietnamese LLM developed by ura-hcmut)

### Compression Methods
*   **Baseline:** Full KV Cache (uncompressed, BF16/FP16)
*   **Primary Quantization Method:** TurboQuant (with native vLLM presets: `turboquant_4bit_nc` and `turboquant_3bit_nc`)
*   **KV Cache Compression Baselines (widely supported in serving engines):**
    *   FP8 KV Cache (Standard default baseline)
    *   HQQ (Half-Quadratic Quantization - via Marlin kernels)
    *   PolarQuant (via PolarEngine/vLLM plugin)

### Datasets
Experiments are conducted on standard, high-quality Vietnamese datasets designed for long-context evaluations:
*   `VMLU` (Comprehensive academic & general benchmark)
*   `V-Bench`
*   `VTSNLP/vietnamese_curated_dataset` & `VTSNLP/instruct_general_dataset` (Hugging Face curated sets) 
*   Custom long-context test suite (~10–20 standard samples spanning 4k, 8k, and 16k context lengths)

### Evaluation Metrics
*   **Memory Footprint:** Peak VRAM usage during generation (MB/GB)
*   **Inference Performance:** Latency per token (ms/token) and generation throughput (tokens/second)
*   **Output Quality:** Perplexity (PPL) as the primary linguistic metric, supplemented by task-specific metrics (Exact Match, F1, ROUGE) where applicable

---

## Project Structure

*   [configs/](configs/) - Model-specific & engine execution configurations.
*   [datasets/](datasets/) - Standardized datasets and small test suite (JSON/JSONL).
    *   [test_set_small.json](datasets/test_set_small.json) - 10-20 sample test suite spanning 4k, 8k, 16k contexts.
    *   [dataset_brief.md](datasets/dataset_brief.md) - Data definitions and guidelines.
*   [experiments/](experiments/) - Hardware-specific execution configurations and environment logs.
*   [paper/](paper/) - LaTeX source code for the English research paper draft.
*   [results/](results/) - Logged experiment outputs and visualization charts.
    *   [template_log.csv](results/template_log.csv) - Uniform logging template for execution records.
    *   [plots/](results/plots/) - Trade-off charts (Memory vs. PPL, Latency vs. Context).
*   [scripts/](scripts/) - Automated run scripts, instrumentation, and plotting tools.
    *   [run_baseline.py](scripts/run_baseline.py) - Script to execute inference and measure performance metrics.
    *   [plot_results.py](scripts/plot_results.py) - Script to generate trade-off visualizations.

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/darktheDE/viet-llm-kvcache-benchmark.git
cd viet-llm-kvcache-benchmark
```

### 2. Set Up Conda Environment
Ensure you have a CUDA-enabled GPU (typically requiring 16-24 GB VRAM for local execution).
```bash
conda create -n dbml_project python=3.10 -y
conda activate dbml_project
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
# Install framework-specific packages (e.g., vllm, llama-cpp-python, hqq) as guided by local scripts
```
> **Technical Note on Engine Support**: 
> - As of mid-2026, **TurboQuant** and **PolarQuant** are natively supported/plugged into **vLLM** and **SGLang** via custom Triton/CUDA kernels.
> - **FP8 KV** is included as our primary production-grade baseline, substituting algorithms without native inference kernel support (e.g., pure vector search quantization methods) to ensure realistic latency and throughput metrics.
---

## Docker setup for Vietnamese Data Pipeline

This Docker setup rebuilds a Linux environment for the Vietnamese data pipeline from source files in this repository. It uses Python 3.10 and installs NeMo Curator with the CPU text-processing extra so the default service can run on machines without an NVIDIA GPU.

### Requirements

* Docker Desktop on Windows/macOS/Linux.
* Internet access to download Python packages and Hugging Face datasets.
* GPU is optional.

### Build image

The same command works in PowerShell, CMD, and Linux shells:

```bash
docker compose build
```

### Enter container

```bash
docker compose run --rm data-pipeline bash
```

### Verify environment

```bash
python --version
python -c "import pandas, datasets, transformers; print('basic deps OK')"
python -c "import nemo_curator; print('nemo curator OK')"
```

Main package versions:

* Python: 3.10 from `python:3.10-slim`.
* NeMo Curator: `nemo-curator[text-cpu]==1.2.0`.
* `datasets`, `transformers`, `tokenizers`, `sentencepiece`, `accelerate`, `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `matplotlib`, `plotly`, `jupyter`, and `ipykernel` are installed from `requirements.txt`.

### Run full pipeline

```bash
# 1. Download raw records
python scripts/download_datasets.py --max-records-per-source 5000

# 2. Clean records using NeMo Curator
python scripts/clean_with_nemo.py --input-dir data/raw --output data/processed/cleaned.jsonl --backend auto

# 3. Build canonical long-context test suite
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json

# 4. Validate canonical dataset
python scripts/validate_testset.py --input datasets/test_set_small.json --schema long_context

# 5. Validate task benchmark dataset
python scripts/validate_testset.py --input datasets/test_set_tasks_small.json --schema task
```

### Run smoke test

```bash
# 1. Download smoke records
python scripts/download_datasets.py --max-records-per-source 200

# 2. Clean smoke records
python scripts/clean_with_nemo.py --input-dir data/raw --output data/processed/cleaned.jsonl --backend auto

# 3. Build canonical smoke test suite
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_smoke.json --allow-smoke-test

# 4. Validate canonical smoke dataset
python scripts/validate_testset.py --input datasets/test_set_smoke.json --schema long_context --allow-smoke-test

# 5. Validate task smoke dataset
python scripts/validate_testset.py --input datasets/test_set_tasks_smoke.json --schema task --allow-smoke-test
```

### Pipeline status

The data pipeline scripts are available in `scripts/`:

* `download_datasets.py` downloads `5760/vmlu` and `VTSNLP/vietnamese_curated_dataset` from Hugging Face into JSONL.
* `clean_with_nemo.py` defaults to `--backend auto`, builds a NeMo Curator `DocumentBatch`, runs NeMo `Modify` stages for ftfy cleanup, Unicode/newline normalization, control-character removal, and whitespace normalization, then runs built-in plus project custom NeMo-compatible `DocumentFilter` checks. Python remains responsible for stateful exact and near deduplication.
* `build_long_context_testset.py` builds `datasets/test_set_small.json` with a Transformers tokenizer.
* `validate_testset.py` validates schema, token counts, and group-level token statistics.

Cleaning backend flags:

* `--backend auto`: prefer NeMo Curator; fall back to Python if the installed API is unavailable.
* `--backend nemo`: require NeMo Curator API initialization and processing.
* `--backend python`: skip NeMo Curator and run Python-only cleaning.

`datasets/dataset_brief.md` documents the sources, processing rules, Docker commands, and current limitations.

### Optional GPU

The text-cleaning pipeline does not require a GPU. If a machine has an NVIDIA GPU and NVIDIA Container Toolkit installed, build the image first and then run one of the commands below.

PowerShell:

```powershell
docker run --rm -it --gpus all -v ${PWD}:/workspace vietnamese-data-pipeline:latest bash
```

CMD:

```cmd
docker run --rm -it --gpus all -v "%cd%:/workspace" vietnamese-data-pipeline:latest bash
```

If GPU-accelerated NeMo Curator text modules are needed, update `requirements.txt` from `nemo-curator[text-cpu]` to the CUDA text extra supported by the target machine and rebuild the image.

## Quick Start

To run a baseline measurement with uncompressed Full KV Cache on a selected model, execute:

```bash
python scripts/run_baseline.py \
    --model "VinAI/PhoGPT-7B-Instruct" \
    --dataset "datasets/test_set_small.json" \
    --context_length 8000 \
    --max_new_tokens 128 \
    --output "results/phogpt_baseline.csv"
```

The script will automatically measure and append the following fields to your local results file:
*   `peak_memory_mb`
*   `latency_ms_per_token`
*   `throughput_tokens_per_s`
*   `perplexity`

---

## Workflow & Timeline

This project is developed iteratively over a **7-week Agile workflow**:

```
+---------------------------------------------------------------------------------+
| Weeks 1-2: Setup & Scope Validation                                             |
| - Finalize scope, install packages, establish baseline pipeline (Full KV cache) |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| Weeks 3-5: Run Experiments                                                      |
| - Execute TurboQuant and baselines on 4-6 models across context sizes           |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| Week 6: Data Synthesis & Plotting                                               |
| - Aggregate result CSVs, generate trade-off charts, analyze behavior curves     |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| Week 7: Final Paper Writing & Polish                                            |
| - Complete English research draft (LaTeX) and internal technical peer review    |
+---------------------------------------------------------------------------------+
```

---

## Contributors

This project is a collaborative effort by **Group 1**:

| Name | Primary Team | Primary Role / Responsibility |
|---|---|---|
| **Do Kien Hung** | Writing & Coordination | Project Manager, Agile/Scrum Coordinator, Document Manager |
| **Phan Trong Qui** | Writing & Coordination | Joint Coordinator, Documentation & Peer Review |
| **Ho Viet Anh** | Technical & Experiment | Joint Coordination, Technical Setup, DataOps/DevOps Engineer |
| **Pham Minh Quan** | Technical & Experiment | Infrastructure Setup, Inference pipeline optimization |
| **Tran Minh Khanh** | Technical & Experiment | Local GPU Environment, Baseline Runner |
| **Nguyen Van Quang Duy** | Technical & Experiment | Quantization Setup (TurboQuant config & patching) |
| **Nguyen Ho Phat** | Data & Analysis | Dataset curator, NeMo Curator Preprocessing lead |
| **Huynh Huu Huy** | Data & Analysis | Small test set curation, prompt formatting |
| **Huynh Ngoc Thach** | Data & Analysis | Metric plotting script engineer |
| **Nguyen Dang Quoc Anh** | Research & Scope | Literature review, theoretical framework definition, Project Owner |
| **Phan Trong Phu** | Research & Scope | Reference indexing, paper draft structuring |

---

## Course Information

*   **Course Name:** Big Data Applications: Machine Learning at Scale
*   **Course Code:** DBML434077
*   **Institution:** Ho Chi Minh City University of Technology and Engineering
*   **Instructor:** Dr. Le Ngoc Hieu

---

## References

1. Leskovec, J., et al. (2020). *Mining of Massive Datasets*, 3rd Edition, Cambridge University Press.
2. Damji, J. S., et al. (2020). *Learning Spark*, 2nd Edition, O’Reilly.
3. Chambers, B., & Zaharia, M. (2018). *Spark: The Definitive Guide*, O'Reilly.
4. James, G., et al. (2021). *An Introduction to Statistical Learning*, 2nd Edition, Springer.
5. Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. arXiv preprint arXiv:2504.19874.
6. Wu, S., et al. (2025). *PolarQuant: Leveraging Polar Transformation for Efficient Key Cache Quantization and Decoding Acceleration*. NeurIPS 2025. arXiv preprint arXiv:2502.00527.
7. Kwon, W., Li, Z., Stoica, I., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. arXiv preprint arXiv:2309.06180.
8. Micikevicius, P., et al. (2022). *FP8 Quantization for Deep Learning*. arXiv preprint arXiv:2209.05433.
9. Badri, H., & Shaji, A. (2023-2024). *Half-Quadratic Quantization (HQQ)*. Mobius Labs. [github.com/mobiusml/hqq](https://github.com/mobiusml/hqq).
10. *KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs* (2026). arXiv preprint arXiv:2602.05929.

***

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

```text
.
├── configs/                # Configuration files for models and compression frameworks
├── datasets/               # Standardized datasets and small test suite (JSON/JSONL)
│   ├── test_set_small.json # 10-20 sample test suite spanning 4k, 8k, 16k contexts
│   └── dataset_brief.md    # Data definitions and guidelines for the technical team
├── experiments/            # Hardware-specific execution configurations and environment logs
├── paper/                  # LaTeX source code for the English research paper draft
├── results/                # Logged experiment outputs and visualization charts
│   ├── template_log.csv    # Uniform logging template for execution records
│   └── plots/              # Trade-off charts (Memory vs. PPL, Latency vs. Context)
└── scripts/                # Automated run scripts, instrumentation, and plotting tools
    ├── run_baseline.py     # Script to execute inference and measure performance metrics
    └── plot_results.py     # Script to generate trade-off visualizations
```

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/darktheDE/vietnamese-llm-benchmark.git
cd vietnamese-llm-benchmark
```

### 2. Set Up Conda Environment
Ensure you have a CUDA-enabled GPU (typically requiring 16-24 GB VRAM for local execution).
```bash
conda create -n dbml_project python=3.10 -y
conda activate dbml_project
```

### 3. Install Requirements
```bash
pip install -r scripts/requirements.txt
# Install framework-specific packages (e.g., vllm, llama-cpp-python, hqq) as guided by local scripts
```
> **Technical Note on Engine Support**: 
> - As of mid-2026, **TurboQuant** and **PolarQuant** are natively supported/plugged into **vLLM** and **SGLang** via custom Triton/CUDA kernels.
> - **FP8 KV** is included as our primary production-grade baseline, substituting algorithms without native inference kernel support (e.g., pure vector search quantization methods) to ensure realistic latency and throughput metrics.
---

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
| **Ho Viet Anh** | Writing & Coordination / Tech | Joint Coordination, Technical Setup, DataOps/DevOps Engineer |
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

1.  Leskovec, J., et al. (2020). *Mining of Massive Datasets*, 3rd Edition, Cambridge University Press.
2.  Damji, J. S., et al. (2020). *Learning Spark*, 2nd Edition, O’Reilly.
3.  Chambers, B., & Zaharia, M. (2018). *Spark: The Definitive Guide*, O'Reilly.
4.  James, G., et al. (2021). *An Introduction to Statistical Learning*, 2nd Edition, Springer.

***
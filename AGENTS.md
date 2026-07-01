# AGENTS.md

> [!NOTE]
> This file is a machine-readable specification designed for AI coding agents. It serves as a dedicated instruction guide to ensure consistency, safety, and efficiency during development.

---

## 1. Project Overview & Context
This repository contains the official implementation, datasets, experimental configurations, and benchmarking results for evaluating **TurboQuant** and various **Key-Value (KV) Cache Compression** techniques on Vietnamese Large Language Models (LLMs).

*   **Primary Goal:** Establish a reproducible empirical benchmark for memory footprint (Peak VRAM), inference latency (TTFT and ITL), decoding throughput (tokens/s), and output quality (Perplexity - PPL, downstream task metrics) across context lengths (4k, 8k, 16k, 32k) for Vietnamese LLMs.
*   **Evaluation Models:**
    1.  `VinAI/PhoGPT-7B5-Instruct` (Pure Vietnamese adaptation with ALiBi attention)
    2.  `Qwen/Qwen2.5-7B-Instruct` (Adapted multilingual baseline with long context support)
    3.  `meta-llama/Meta-Llama-3.1-8B-Instruct` (Global standard baseline)
    4.  `ura-hcmut/URA-LLaMa-3-8B` (Vietnamese continual pre-trained model)
    5.  `Viet-Mistral/Vistral-7B-Chat` (High-performing cross-lingual model)
*   **Compression Methods:**
    *   **Baseline:** Full KV Cache (uncompressed, BF16/FP16)
    *   **FP8:** 8-bit floating point standard baseline
    *   **HQQ:** Half-Quadratic Quantization (via Marlin kernels)
    *   **PolarQuant:** Polar coordinate projection (via PolarEngine/Triton)
    *   **TurboQuant:** PolarQuant with QJL (Quantized Johnson-Lindenstrauss) error compensation (presets: `turboquant_4bit_nc`, `turboquant_3bit_nc`)

---

## 2. Tech Stack & Environment
*   **Language & Version:** Python 3.10
*   **Core Libraries:** PyTorch, vLLM, llama.cpp, HQQ, NVIDIA NeMo Curator, `pynvml` (for GPU profiling), `pandas` (for analysis), `matplotlib`/`plotly` (for visualization).
*   **Infrastructure:** RunPod / Vast.ai (RTX 3090/4090 or L4 GPUs with minimum 24GB VRAM).
*   **Local Setup:**
    ```bash
    conda create -n dbml_project python=3.10 -y
    conda activate dbml_project
    pip install -r requirements.txt
    ```

---

## 3. Project Directory Structure
*   [configs/](configs) - Model-specific & engine execution configurations.
*   [datasets/](datasets) - Small test suites, schema definitions, and benchmarks.
    *   [test_set_small.json](datasets/test_set_small.json) - Small canonical long-context test suite (4k, 8k, 16k context).
    *   [test_set_smoke.json](datasets/test_set_smoke.json) - Smoke test suite for quick local code runs.
    *   [test_set_tasks_small.json](datasets/test_set_tasks_small.json) - Task benchmark dataset.
    *   [dataset_brief.md](datasets/dataset_brief.md) - Documentation on datasets.
*   [experiments/](experiments) - Environment configuration files and execution logs.
*   [paper/](paper) - LaTeX paper drafts and reference lists (`references.bib`).
*   [results/](results) - CSV results and graphical reports.
    *   [template_log.csv](results/template_log.csv) - Uniform logging schema for execution records.
    *   [plots/](results/plots) - Trade-off charts (latency vs context, Pareto frontiers, VRAM usage).
*   [scripts/](scripts) - Core automated scripts and helper modules.
    *   [run_baseline.py](scripts/run_baseline.py) - Script to execute inference and measure performance.
    *   [plot_results.py](scripts/plot_results.py) - Script to generate Pareto-optimal charts.
    *   [validate_testset.py](scripts/validate_testset.py) - Check json schema and length validations.
    *   [nemo_backend.py](scripts/nemo_backend.py) - NeMo Curator wrapper backend.
    *   [download_datasets.py](scripts/download_datasets.py) - HF downloader for VMLU & VTSNLP.
    *   [clean_with_nemo.py](scripts/clean_with_nemo.py) - Text cleaning script.
    *   [build_long_context_testset.py](scripts/build_long_context_testset.py) - Creates canonical JSONs.
    *   [run_mock_grid.py](scripts/run_mock_grid.py) - Mock baseline runner for quick validation.
    *   [test/](scripts/test) - Actual execution files (`run_real_benchmark.py`, `run_real_grid.py`, `generate_real_analysis.py`).

---

## 4. Setup & Execution Commands

### Data Pipeline (Docker / CPU only)
*   **Build image:**
    ```bash
    docker compose build
    ```
*   **Enter container:**
    ```bash
    docker compose run --rm data-pipeline bash
    ```
*   **Run full data pipeline:**
    ```bash
    python scripts/download_datasets.py --max-records-per-source 5000
    python scripts/clean_with_nemo.py --input-dir data/raw --output data/processed/cleaned.jsonl --backend auto
    python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json
    python scripts/validate_testset.py --input datasets/test_set_small.json --schema long_context
    python scripts/validate_testset.py --input datasets/test_set_tasks_small.json --schema task
    ```
*   **Run smoke test data pipeline:**
    ```bash
    python scripts/download_datasets.py --max-records-per-source 200
    python scripts/clean_with_nemo.py --input-dir data/raw --output data/processed/cleaned.jsonl --backend auto
    python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_smoke.json --allow-smoke-test
    python scripts/validate_testset.py --input datasets/test_set_smoke.json --schema long_context --allow-smoke-test
    python scripts/validate_testset.py --input datasets/test_set_tasks_smoke.json --schema task --allow-smoke-test
    ```

### Benchmark Running & Logging
*   **Run baseline (Full KV Cache):**
    ```bash
    python scripts/run_baseline.py --model "VinAI/PhoGPT-7B-Instruct" --dataset "datasets/test_set_small.json" --context_length 8000 --max_new_tokens 128 --output "results/phogpt_baseline.csv"
    ```
*   **Run Mock Baseline (For validation without GPU):**
    ```bash
    python scripts/run_mock_grid.py
    ```

### Visualization & Plots
*   **Generate Pareto & latency graphs:**
    ```bash
    python scripts/plot_results.py
    ```

---

## 5. Coding Standards & Conventions

### Python Style Rules (PEP 8)
*   **Naming Conventions:**
    *   Variables, attributes, and function names: `snake_case` (e.g., `peak_memory_mb`, `get_gpu_memory`).
    *   Class names: `PascalCase` (e.g., `MetricTracer`, `DatasetValidator`).
    *   Constants: `UPPER_CASE` (e.g., `DEFAULT_CONTEXT_LENGTH`, `MAX_TOKENS`).
*   **Formatting:** Use exactly **4 spaces** for indentation. Never use tabs.
*   **Documentation:** All main classes and functions must include clear Docstrings. Outline parameters (`Args`) and return values (`Returns`).
*   **KISS & DRY:** Keep implementation simple. Abstract reusable calculations (e.g., VRAM hooks, ITL time measurements) into unified helper files in [scripts/](scripts).

---

## 6. Git Workflow & Conventional Commits

*   **Branching Strategy:** Create new features branching off `develop` and prefix the name with `feature/` (e.g., `feature/add-quant-kernels`).
*   **Commit Messages (Conventional Commits style):**
    *   `feat: <description>` for new features.
    *   `fix: <description>` for bug fixes.
    *   `docs: <description>` for documentation changes.
    *   `refactor: <description>` for structure edits without behavioral changes.
*   **Dependency Management:** Add packages to [requirements.txt](requirements.txt) with precise versioning.

---

## 7. Boundaries & Constraints

### ALWAYS
*   **Catch CUDA Out of Memory errors gracefully:** Include `try...except torch.cuda.OutOfMemoryError` blocks around model load and generate steps. Log `"OOM"` as the cell value in the output CSV, ensuring the script moves to the next grid search config without crashing.
*   **Preserve academic comments:** Keep all existing comments, theoretical explanations, and mathematical formulas explaining KV Cache compression, Polar coordinates, or QJL error compensation.
*   **Maintain docstrings:** Do not change or remove existing, unrelated function docstrings.
*   **Verify schemas:** Ensure generated datasets/logs conform to existing keys and CSV layouts.

### NEVER
*   **Do NOT commit large files:** Never commit generated logs (`.csv`), large data samples (`.json`, `.jsonl`), or model weights (`.bin`, `.safetensors`) to Git. Keep them local or in `.gitignore`.
*   **Do NOT simplify academic logic:** Do not rewrite or modify underlying Triton or CUDA quantization kernels unless explicitly instructed.
*   **Do NOT use tabs:** Ensure the editor is configured to convert tabs to 4 spaces.

### ASK FIRST
*   *Ask before introducing complex external libraries.* Keep dependencies minimal and easy to build.
*   *Ask before altering the schema* of [template_log.csv](results/template_log.csv) or [test_set_small.json](datasets/test_set_small.json).

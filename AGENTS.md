# AGENTS.md

## Project Overview
This repository contains the official implementation, datasets, experimental configurations, and benchmarking results for evaluating **TurboQuant** and various **Key-Value (KV) Cache Compression** techniques on Vietnamese Large Language Models (LLMs).
- **Core Stack:** Python 3.10, PyTorch, vLLM, llama.cpp, HQQ, NeMo Curator.
- **Goal:** Establish a reproducible empirical benchmark for memory footprint, inference latency, throughput, and perplexity across context lengths (4k, 8k, 16k) for Vietnamese LLMs.

## Dev Environment & Setup
- **Environment:** Conda environment with `python=3.10` is expected.
- **Dependencies:** Install requirements via `pip install -r requirements.txt`.
- **Data Pipeline (Docker):** 
  - Build: `docker compose build`
  - Run container: `docker compose run --rm data-pipeline bash`
  - Run full data pipeline: `python scripts/download_datasets.py`, `python scripts/clean_with_nemo.py`, `python scripts/build_long_context_testset.py`, `python scripts/validate_testset.py`.

## Coding Standards & Preferences (PEP 8)
- **Naming Conventions:**
  - `snake_case` for variables, attributes, and function names.
  - `PascalCase` for class names.
  - `UPPER_CASE` for constants.
- **Formatting:** Use **4 spaces** for indentation. No tabs.
- **Documentation:** All main functions and classes must include clear docstrings specifying `Args` and `Returns`.
- **Comments:** **CRITICAL:** Preserve all academic comments and mathematical formulas explaining KV cache compression algorithms.
- **Code Organization:** Follow KISS and DRY principles. Abstract reusable metrics calculations (e.g., VRAM, TTFT, PPL) into common utility functions under the `scripts/` directory.

## Testing & Execution
- **Run Baseline:** Use `scripts/run_baseline.py` for testing Full KV Cache performance.
  - Example: `python scripts/run_baseline.py --model "vilm/vinallama-7b-chat" --dataset "datasets/test_set_small.json" --context_length 8000 --max_new_tokens 128 --output "results/baseline.csv"`
- Always test locally before creating Pull Requests.

## Version Control & Git Workflow
- **Branching Strategy:** All new code must be branched from `develop` and prefixed with `feature/` (e.g., `feature/ten-tinh-nang`).
- **Commit Messages (Conventional Commits):**
  - `feat:` for new features.
  - `fix:` for bug fixes.
  - `docs:` for documentation updates.
  - `refactor:` for code restructuring without changing functionality.
- **Dependencies:** If your code requires new packages, ensure they are added to `requirements.txt` with specific versions.

## Boundaries & Constraints
- **Agent Limitations:** Do not rewrite or simplify the underlying academic quantization logic or CUDA/Triton kernels unless explicitly instructed.
- **Data Safety:** Never commit large datasets (`.json`, `.jsonl`), generated logs, or model weights to the repository. Only commit sample/test sets (like `test_set_small.json`) and plotting scripts.

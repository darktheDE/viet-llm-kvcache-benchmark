# LITERATURE REVIEW & RELATED WORKS
## Benchmarking KV Cache Compression on Large Language Models

This document compiles the key research publications, theoretical frameworks, and state-of-the-art KV Cache compression papers from the past 5 years (2022–2026). It serves as the academic foundation for Group 1's research paper draft.

---

### 1. Key Papers in KV Cache Quantization & Vector Quantization

#### A. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (2025)
*   **Authors:** Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research / UT)
*   **arXiv Identifier:** [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
*   **Core Theory:** Presents a data-oblivious online vector quantization algorithm designed to minimize both mean-squared error (MSE) and inner product distortion. It employs a two-stage process:
    1.  Applying a random rotation to input vectors to force a concentrated Beta distribution on coordinates, allowing near-optimal scalar quantization.
    2.  Compensating for residual errors in estimating query-key inner products by utilizing a 1-bit **Quantized Johnson-Lindenstrauss (QJL)** projection.
*   **Relevance:** The primary target quantization algorithm for this benchmark, evaluated for its distortion-reduction properties on Vietnamese language sequences.

#### B. PolarQuant: Leveraging Polar Transformation for Efficient Key Cache Quantization and Decoding Acceleration (2025)
*   **Authors:** Songhao Wu, Ang Lv, Xiao Feng, Yufei Zhang, Xun Zhang, Guojun Yin, Wei Lin, Rui Yan (Peking University, ByteDance)
*   **arXiv Identifier:** [arXiv:2502.00527](https://arxiv.org/abs/2502.00527) (Presented at NeurIPS 2025)
*   **Core Theory:** Identifies that Key-Value embeddings under RoPE (Rotary Position Embedding) exhibit severe outliers when represented in Cartesian coordinates. To solve this, PolarQuant transforms vectors into Polar Coordinates (radius and angle). 
    *   Angles have a stable, highly concentrated analytical distribution which can be quantized uniformly without storing expensive scaling parameters.
    *   Allows efficient decoding by converting query-key inner products into table lookups.
*   **Relevance:** Serves as the structural foundation for TurboQuant and acts as our secondary low-bit baseline.

#### C. PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression (2026)
*   **Author:** Caio Vicentino
*   **arXiv Identifier:** [arXiv:2603.29078](https://arxiv.org/abs/2603.29078) (Note: Rebranded to **HLWQ** - Hadamard-Lloyd Weight Quantization to avoid collision)
*   **Core Theory:** A companion work focusing on Gaussian weight quantization (rather than KV cache) utilizing Walsh-Hadamard rotations. 
*   **Relevance:** Important to reference to distinguish between *Weight Quantization* and *KV Cache Quantization* in the literature review section.

#### D. PolarQuant: Quantizing KV Caches with Polar Transformation (2025)
*   **Authors:** Insu Han, Praneeth Kacham, Amin Karbasi, Vahab Mirrokni, Amir Zandieh (KAIST, Google Research, Yale University)
*   **arXiv Identifier:** [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
*   **Core Theory:** Uses random preconditioning followed by polar coordinate conversion. Since angles in this transformed space have a tight, concentrated distribution, they can be quantized uniformly without storing data-dependent scaling parameters.
*   **Relevance:** Directly serves as the mathematical foundation for TurboQuant's coordinate rotation and error compensation framework.

---

### 2. High-Performance Serving & Memory Management

#### A. Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)
*   **Authors:** Woosuk Kwon, Zhuohan Li, Ion Stoica, et al. (UC Berkeley, Stanford, UCSD)
*   **SOSP 2023:** [Paper Link / arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
*   **Core Theory:** Introduces **PagedAttention**, an algorithm that stores KV cache in non-contiguous physical memory blocks, mapping them via logical block tables (similar to paging in OS virtual memory).
*   **Relevance:** The fundamental engine logic powering **vLLM** and **SGLang**, which are utilized to load models and run Triton/CUDA kernels in our technical setup.

#### B. FP8 Quantization for Deep Learning (2022)
*   **Authors:** Paulius Micikevicius, et al. (NVIDIA, Intel, Arm)
*   **arXiv Identifier:** [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
*   **Core Theory:** Establishes the binary format and scaling algorithms for 8-bit floating point representations (E4M3 and E5M2).
*   **Relevance:** Serves as our primary industrial baseline for FP8 KV Cache compression.

#### C. Half-Quadratic Quantization (HQQ) (2023-2024)
*   **Authors:** Hicham Badri, Appu Shaji (Mobius Labs)
*   **Official Implementation:** [github.com/mobiusml/hqq](https://github.com/mobiusml/hqq)
*   **Core Theory:** A data-free quantization framework optimizing scale and zero-points for uniform grids using a half-quadratic algorithm.
*   **Relevance:** Utilized for comparison as a high-speed, data-free post-training weight/cache quantization baseline.

---

### 3. Benchmarks & Evaluation Corpora

#### A. KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs (2026)
*   **arXiv Identifier:** [arXiv:2602.05929](https://arxiv.org/abs/2602.05929)
*   **Core Theory:** Evaluates how KV-cache exhibits different levels of low-rank structure across diverse languages and data domains.
*   **Relevance:** Establishes the methodology for measuring language perplexity degradation vs. context scaling.

#### B. VMLU: A High-Quality Vietnamese Language Understanding Benchmark (2024)
*   **Source:** Zalo AI / Community Leaderboard
*   **Relevance:** The primary Vietnamese evaluation suite used in our study to measure output quality on QA and reasoning tasks.

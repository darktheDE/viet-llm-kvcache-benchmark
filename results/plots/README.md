# Results — Figures & Captions

> **Cách dùng:** các caption dưới đây viết sẵn bằng tiếng Anh để dán thẳng vào bản LaTeX/paper.
> Mỗi mục gồm: tên file hình, caption chính, và các *caveat* bắt buộc phải giữ để tránh bị reviewer bắt lỗi.
> Số liệu trích từ `all_results_compiled.csv` (real runs, GPU A100 80 GB). Trạng thái `OK` only.

## Conventions (state once in the paper)

- **Models:** Qwen3-8B, Qwen2.5-7B, Mistral-7B-v0.3, Phi-4-mini, Gemma-3-4B (weights kept at FP16/BF16; only the KV cache is quantized).
- **Context lengths:** 4k / 8k / 16k tokens. The 32k bucket was dropped: it exceeds several models' maximum context window (~26k), producing configuration errors rather than out-of-memory events.
- **Decoding:** greedy (temperature = 0), `max_new_tokens = 128`.
- **Perplexity** is computed offline by an uncompressed reference model over each method's generated text. Reported as the **median** across samples for scaling lines (robust to a small failure tail, n = 2–4 per cell) and as the **mean** for the size trade-off (so the tail is reflected). Both choices are stated per figure.
- **PolarQuant** runs through the vLLM `fp8` fallback in this setup, so it is numerically equivalent to FP8 and its points coincide with FP8 throughout.

---

## Figure 1 — Peak VRAM vs context (`vram_vs_context.pdf`)

**Caption.** Figure 1: Measured peak GPU memory versus context length for each KV-cache method. All methods sit within a <2% band just below the pre-allocated pool line (~90% of the 80 GB device) and are essentially indistinguishable. This is expected: vLLM reserves the KV-cache pool up front via `gpu_memory_utilization`, so *measured* peak VRAM reflects the reserved pool, not the true KV-cache footprint. Consequently, peak VRAM cannot be used to compare compression methods on this platform; we instead report the analytical KV-cache footprint in Figure 3.

**Caveat (keep):** frame this as a *methodological finding*, never as "compression saves/does not save memory." The y-axis starts at 0 with the GPU-capacity and pool reference lines precisely so the flatness is visible.

---

## Figure 2 — Latency, throughput & perplexity vs context (`scaling_grid.pdf`)

**Caption.** Figure 2: Scaling behaviour of each KV-cache method with context length. Left: per-token latency; middle: decode throughput; right: perplexity (median line; faint markers are individual samples). The 4-bit methods incur a decode-speed penalty (HQQ most severe: latency rises to ~56 ms/token and throughput falls to ~18 tokens/s at 16k), while FP8/PolarQuant track the FP16 baseline. Perplexity stays near baseline for TurboQuant across all lengths but for HQQ collapses at 16k (median 10.2), indicating a length-dependent failure mode.

**Caveat (keep):** note n = 2–4 per cell; the individual-sample markers are shown deliberately instead of error bars.

---

## Figure 3 — Perplexity vs theoretical KV-cache size (`pareto_ppl_vs_kvsize.pdf`)

**Caption.** Figure 3: Quality–compression trade-off at 16k context. The x-axis is the **analytical** KV-cache size relative to FP16 (bit-width / 16; FP8 and PolarQuant = 50%, HQQ and TurboQuant = 25%), which is platform-independent and unaffected by the pool artefact of Figure 1; the y-axis is mean perplexity. TurboQuant is Pareto-optimal — at a 4× smaller cache it matches or beats the FP16 baseline (1.39 vs 1.67) — whereas HQQ, at the same 4-bit budget, is dominated (10.17). FP8 and PolarQuant coincide at 50% size with perplexity ~2.2.

**Caveat (keep):** state that the size axis is analytical (nominal bit-width, ignoring per-group scale/zero metadata) and that PolarQuant overlaps FP8.

---

## Figure 4 — Theoretical compression & effective throughput (`compression_efficiency.pdf`)

**Caption.** Figure 4: Left — theoretical KV-cache compression ratio versus FP16 (2× for the 8-bit methods, 4× for the 4-bit methods). Right — *effective throughput*, defined as measured decode throughput multiplied by the compression ratio, an upper-bound proxy for serving capacity in the memory-bound regime where a smaller cache admits proportionally larger batches. TurboQuant leads on this proxy at every context length.

**Caveat (keep):** the effective-throughput metric is **quality-agnostic** — it rewards HQQ's 4× ratio despite HQQ's degraded output (see Figure 6). Say this explicitly in the caption so HQQ is not misread as a good operating point.

---

## Figure 5 — Perplexity vs decode speed (`pareto_ppl_vs_speed.pdf`)

**Caption.** Figure 5: Measured quality–speed trade-off at 16k context (median perplexity vs median throughput). The frontier comprises FP8/PolarQuant (fastest, ~62 tokens/s, perplexity ~2.6) and TurboQuant (best quality, perplexity 1.39, at ~46 tokens/s); FP16 lies between them. HQQ is dominated on both axes (slowest and worst quality). TurboQuant therefore buys baseline-level quality at roughly a 25% throughput cost relative to the 8-bit methods.

**Caveat (keep):** PolarQuant coincides with FP8.

---

## Figure 6 — Output quality relative to the FP16 baseline (`quality_gibberish_vs_repetition.pdf`)

**Caption.** Figure 6: Vietnamese output quality by KV-cache method, computed directly from generated text and referenced to the uncompressed FP16 baseline (dashed line). Left — junk-token ("gibberish") ratio: only HQQ rises markedly above baseline (23.6% vs 10.1%, with 18% of its generations mostly gibberish), while TurboQuant is at or below baseline (8.0%). Right — repeated-3-gram ratio is essentially flat across all methods including FP16 (~30–34%), confirming that the repetition observed under greedy decoding is an artefact of the task/decoding setup rather than of compression.

**Caveat (keep):** this figure is why the original binary `repetition_flag` was discarded — it saturated (fired on ~79% of runs, including the uncompressed baseline) and mislabelled TurboQuant.

---

## Key numbers (16k context)

| Method | KV size (% FP16) | Compression | PPL (mean) | PPL (median) | Throughput (tok/s) | Gibberish % | Mostly-gibberish % |
| :--- | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| FP16 (baseline) | 100 | 1× | 1.67 | 1.51 | 61.2 | 10.1 | 3 |
| FP8 | 50 | 2× | 2.21 | 2.56 | 62.5 | 11.4 | 6 |
| PolarQuant (= FP8 fallback) | 50 | 2× | 2.23 | 2.63 | 62.5 | 13.6 | 7 |
| HQQ | 25 | 4× | 10.17 | 10.17 | 18.1 | 23.6 | 18 |
| **TurboQuant** | **25** | **4×** | **1.39** | **1.39** | **45.9** | **8.0** | **2** |

*(Gibberish and mostly-gibberish shares are pooled across contexts; n = 8–15 runs per method.)*

## Headline claims (supported by the figures)

1. TurboQuant achieves a 4× theoretical KV-cache reduction while preserving baseline-level quality (perplexity 1.39 vs FP16 1.67) — Figures 3, 6.
2. At the same 4-bit budget, TurboQuant outperforms HQQ by ~7× in perplexity; HQQ collapses at 16k context — Figures 2, 3.
3. The cost of TurboQuant's quality is ~25% lower decode throughput; the 8-bit methods (FP8/PolarQuant) are faster but lower quality — Figure 5.
4. On this platform, measured peak VRAM is pool-dominated and cannot distinguish methods; compression benefit must be argued analytically — Figures 1, 4.

## Methodological caveats to disclose

- Measured peak VRAM reflects vLLM's pre-allocated pool, not the KV-cache footprint (Figure 1).
- Perplexity is measured on the models' own greedy generations, which rewards repetitive text; it is a relative indicator across methods, not an absolute language-quality score.
- PolarQuant executes as an FP8 fallback and is not evaluated as a distinct low-bit kernel here.
- Small sample size (n = 8–15 per method) and incomplete model × method coverage; treat differences within the FP8/PolarQuant/FP16 cluster as inconclusive.
- Verify the TurboQuant bit-width used (4-bit → 4×; 3-bit → ~5.3×) and update `METHOD_BITS` and Figures 3–4 accordingly.

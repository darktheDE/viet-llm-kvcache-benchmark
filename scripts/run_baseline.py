from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Fix encoding cho Windows PowerShell (cp1252 không hỗ trợ tiếng Việt)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

try:
    from vllm import LLM, SamplingParams
    import pynvml
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_GPU_STACK = True
except ImportError:
    HAS_GPU_STACK = False
    print("WARNING: Không tìm thấy thư viện vLLM, pynvml, PyTorch hoặc transformers. Chuyển sang MOCK_MODE.")

from scripts.utils_csv import append_log_row, ensure_log_header
from scripts.utils_generation_quality import analyze_generated_text
from scripts.utils_ppl import compute_perplexity


SUPPORTED_MODELS = [
    "vilm/vinallama-7b-chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "ura-hcmut/URA-LLaMa-3-8B",
    "Viet-Mistral/Vistral-7B-Chat",
]

KV_CACHE_DTYPE_MAP = {
    "FP16": "auto",
    "FP8": "fp8",
    "HQQ": "hqq_4bit",
    "PolarQuant": "polarquant_4bit",
    "TurboQuant": "turboquant_4bit_nc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KV Cache Compression on Vietnamese LLMs")
    parser.add_argument("--model", type=str, default="vilm/vinallama-7b-chat", choices=SUPPORTED_MODELS)
    parser.add_argument("--dataset", type=str, default="datasets/test_set_small.json")
    parser.add_argument("--context_length", type=int, default=8000)
    parser.add_argument(
        "--kv_cache_type",
        type=str,
        default="FP16",
        choices=["FP16", "FP8", "HQQ", "PolarQuant", "TurboQuant"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default="results/raw_benchmark_log.csv")
    parser.add_argument("--mock_mode", action="store_true")
    parser.add_argument("--reference_model", type=str, default=None)
    parser.add_argument("--reference_tokenizer", type=str, default=None)
    parser.add_argument("--ppl_stride", type=int, default=512)
    parser.add_argument("--ppl_max_length", type=int, default=None)
    return parser.parse_args()


def load_dataset_samples(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    raise ValueError(f"Invalid dataset format in {path}: expected list or dict with 'samples' key")


def filter_samples_by_length(samples: list[dict[str, Any]], context_length: int) -> list[dict[str, Any]]:
    if context_length <= 5000:
        target_group = "4k"
        target_len = 4000
    elif context_length <= 10000:
        target_group = "8k"
        target_len = 8000
    else:
        target_group = "16k"
        target_len = 16000

    filtered: list[dict[str, Any]] = []
    for item in samples:
        cg = item.get("context_group")
        clt = item.get("context_length_target")
        tt = item.get("target_tokens")

        match = False
        if cg is not None and str(cg).lower() == target_group.lower():
            match = True
        elif clt is not None:
            try:
                match = int(clt) == target_len
            except (TypeError, ValueError):
                match = False
        elif tt is not None:
            try:
                tt_int = int(tt)
            except (TypeError, ValueError):
                tt_int = -1
            match = abs(tt_int - target_len) < 1000 or (
                target_len == 4000 and tt_int == 4096
            ) or (
                target_len == 8000 and tt_int == 8192
            ) or (
                target_len == 16000 and tt_int == 16384
            )

        if match:
            filtered.append(item)

    return filtered or samples


def load_reference_model(model_name: str | None, tokenizer_name: str | None) -> tuple[Any, Any, Any]:
    if not HAS_GPU_STACK:
        return None, None, None

    candidate_model = model_name
    candidate_tokenizer = tokenizer_name or model_name
    if not candidate_model:
        return None, None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(candidate_tokenizer, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        reference_model = AutoModelForCausalLM.from_pretrained(
            candidate_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        reference_model.eval()
        device = next(reference_model.parameters()).device
        return reference_model, tokenizer, device
    except Exception as exc:
        print(f"[WARN] Khong tai duoc reference model BF16: {exc}")
        return None, None, None


def load_dataset_prompt_text(sample: dict[str, Any]) -> str:
    for key in ("text", "prompt", "content", "input"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_quality_row(text: str) -> dict[str, Any]:
    metrics = analyze_generated_text(text)
    return {
        "repetition_flag": metrics["repetition_flag"],
        "gibberish_flag": metrics["gibberish_flag"],
        "repeated_ngram_ratio": metrics["repeated_ngram_ratio"],
        "special_char_ratio": metrics["special_char_ratio"],
        "output_length": metrics["output_length"],
        "quality_warning": metrics["quality_warning"],
    }


def build_row_base(args: argparse.Namespace, sample: dict[str, Any], sample_id: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "dataset": args.dataset,
        "sample_id": sample_id,
        "kv_cache_type": args.kv_cache_type,
        "kv_cache_dtype": KV_CACHE_DTYPE_MAP.get(args.kv_cache_type, "auto"),
        "context_length": args.context_length,
        "peak_memory_mb": None,
        "latency_ms_per_token": None,
        "throughput_tokens_per_s": None,
        "generated_tokens": None,
        "perplexity": None,
        "ppl_loss": None,
        "ppl_tokens": 0,
        "ppl_status": "SKIPPED",
        "ppl_error": "",
        "repetition_flag": False,
        "gibberish_flag": False,
        "repeated_ngram_ratio": None,
        "special_char_ratio": None,
        "output_length": None,
        "quality_warning": "",
        "status": "PENDING",
        "error_message": "",
        "_sample_text": load_dataset_prompt_text(sample),
    }


def setup_csv(output_path: str) -> None:
    ensure_log_header(output_path)


def run_mock_benchmark(args: argparse.Namespace) -> None:
    print("\n=======================================================")
    print("   DEMO BENCHMARK KV CACHE COMPRESSION (MOCK MODE)")
    print("=======================================================\n")

    try:
        samples = load_dataset_samples(args.dataset)
        filtered_samples = filter_samples_by_length(samples, args.context_length)
    except Exception as exc:
        row = {
            "model": args.model,
            "dataset": args.dataset,
            "sample_id": 0,
            "kv_cache_type": args.kv_cache_type,
            "kv_cache_dtype": KV_CACHE_DTYPE_MAP.get(args.kv_cache_type, "auto"),
            "context_length": args.context_length,
            "peak_memory_mb": None,
            "latency_ms_per_token": None,
            "throughput_tokens_per_s": None,
            "generated_tokens": None,
            "perplexity": None,
            "ppl_loss": None,
            "ppl_tokens": 0,
            "ppl_status": "SKIPPED_MOCK",
            "ppl_error": "",
            "repetition_flag": False,
            "gibberish_flag": False,
            "repeated_ngram_ratio": None,
            "special_char_ratio": None,
            "output_length": None,
            "quality_warning": "",
            "status": "DATASET_LOAD_ERROR",
            "error_message": str(exc),
        }
        append_log_row(args.output, row)
        print(f"  -> Lỗi nạp dataset: {exc}")
        return

    num_samples = min(len(filtered_samples), 3)
    print(f"  -> Đã nạp {num_samples} mẫu văn bản.")

    for idx, sample in enumerate(filtered_samples[:num_samples]):
        prompt_text = load_dataset_prompt_text(sample)
        if not prompt_text:
            quality = build_quality_row("")
        else:
            mock_text = f"{prompt_text[:256]} ... mock generation"
            quality = build_quality_row(mock_text)

        base_vram = 14000
        if args.kv_cache_type == "TurboQuant":
            peak_vram = base_vram + args.context_length * 0.1
        elif args.kv_cache_type == "FP16":
            peak_vram = base_vram + args.context_length * 0.4
        else:
            peak_vram = base_vram + args.context_length * 0.2

        latency = 30.5 + (args.context_length / 1000)
        if args.kv_cache_type != "FP16":
            latency += 2.5
        throughput = 1000 / latency if latency > 0 else None
        generated_tokens = args.max_new_tokens

        row = {
            "model": args.model,
            "dataset": args.dataset,
            "sample_id": idx,
            "kv_cache_type": args.kv_cache_type,
            "kv_cache_dtype": KV_CACHE_DTYPE_MAP.get(args.kv_cache_type, "auto"),
            "context_length": args.context_length,
            "peak_memory_mb": round(peak_vram, 2),
            "latency_ms_per_token": round(latency, 2),
            "throughput_tokens_per_s": round(throughput, 2) if throughput is not None else None,
            "generated_tokens": generated_tokens,
            "perplexity": None,
            "ppl_loss": None,
            "ppl_tokens": 0,
            "ppl_status": "SKIPPED_MOCK",
            "ppl_error": "",
            "repetition_flag": quality["repetition_flag"],
            "gibberish_flag": quality["gibberish_flag"],
            "repeated_ngram_ratio": quality["repeated_ngram_ratio"],
            "special_char_ratio": quality["special_char_ratio"],
            "output_length": quality["output_length"],
            "quality_warning": quality["quality_warning"],
            "status": "MOCK_OK",
            "error_message": "",
        }
        append_log_row(args.output, row)

    print("  -> MOCK benchmark hoàn tất.")


def _extract_generated_text(output: Any) -> str:
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    return ""


def run_real_benchmark(args: argparse.Namespace) -> None:
    if not HAS_GPU_STACK:
        run_mock_benchmark(args)
        return

    print("--- THỰC THI BENCHMARK TRÊN GPU ---")
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    dtype_mapping = KV_CACHE_DTYPE_MAP
    kv_cache_dtype = dtype_mapping.get(args.kv_cache_type, "auto")

    print(f"Đang tải mô hình {args.model} với kv_cache_dtype={kv_cache_dtype}...")
    try:
        llm = LLM(
            model=args.model,
            kv_cache_dtype=kv_cache_dtype,
            max_model_len=args.context_length,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=4096,
            max_num_seqs=2,
            trust_remote_code=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        append_log_row(
            args.output,
            {
                "model": args.model,
                "dataset": args.dataset,
                "sample_id": 0,
                "kv_cache_type": args.kv_cache_type,
                "kv_cache_dtype": kv_cache_dtype,
                "context_length": args.context_length,
                "peak_memory_mb": None,
                "latency_ms_per_token": None,
                "throughput_tokens_per_s": None,
                "generated_tokens": None,
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": 0,
                "ppl_status": "OOM",
                "ppl_error": str(exc),
                "repetition_flag": False,
                "gibberish_flag": False,
                "repeated_ngram_ratio": None,
                "special_char_ratio": None,
                "output_length": None,
                "quality_warning": "",
                "status": "OOM",
                "error_message": str(exc),
            },
        )
        return
    except Exception as exc:
        append_log_row(
            args.output,
            {
                "model": args.model,
                "dataset": args.dataset,
                "sample_id": 0,
                "kv_cache_type": args.kv_cache_type,
                "kv_cache_dtype": kv_cache_dtype,
                "context_length": args.context_length,
                "peak_memory_mb": None,
                "latency_ms_per_token": None,
                "throughput_tokens_per_s": None,
                "generated_tokens": None,
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": 0,
                "ppl_status": "ERROR",
                "ppl_error": str(exc),
                "repetition_flag": False,
                "gibberish_flag": False,
                "repeated_ngram_ratio": None,
                "special_char_ratio": None,
                "output_length": None,
                "quality_warning": "",
                "status": "LOAD_ERROR",
                "error_message": str(exc),
            },
        )
        return

    reference_model = None
    reference_tokenizer = None
    reference_device = None
    if args.reference_model:
        reference_model, reference_tokenizer, reference_device = load_reference_model(
            args.reference_model,
            args.reference_tokenizer or args.reference_model,
        )
    else:
        print("[WARN] Không có reference_model nên PPL sẽ được skip.")

    try:
        samples = load_dataset_samples(args.dataset)
        filtered_samples = filter_samples_by_length(samples, args.context_length)
    except Exception as exc:
        append_log_row(
            args.output,
            {
                "model": args.model,
                "dataset": args.dataset,
                "sample_id": 0,
                "kv_cache_type": args.kv_cache_type,
                "kv_cache_dtype": kv_cache_dtype,
                "context_length": args.context_length,
                "peak_memory_mb": None,
                "latency_ms_per_token": None,
                "throughput_tokens_per_s": None,
                "generated_tokens": None,
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": 0,
                "ppl_status": "ERROR",
                "ppl_error": str(exc),
                "repetition_flag": False,
                "gibberish_flag": False,
                "repeated_ngram_ratio": None,
                "special_char_ratio": None,
                "output_length": None,
                "quality_warning": "",
                "status": "DATASET_LOAD_ERROR",
                "error_message": str(exc),
            },
        )
        return

    prompts = [load_dataset_prompt_text(item) for item in filtered_samples[:5]]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        append_log_row(
            args.output,
            {
                "model": args.model,
                "dataset": args.dataset,
                "sample_id": 0,
                "kv_cache_type": args.kv_cache_type,
                "kv_cache_dtype": kv_cache_dtype,
                "context_length": args.context_length,
                "peak_memory_mb": None,
                "latency_ms_per_token": None,
                "throughput_tokens_per_s": None,
                "generated_tokens": None,
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": 0,
                "ppl_status": "EMPTY",
                "ppl_error": "no prompts found in dataset",
                "repetition_flag": False,
                "gibberish_flag": False,
                "repeated_ngram_ratio": None,
                "special_char_ratio": None,
                "output_length": None,
                "quality_warning": "",
                "status": "EMPTY_DATASET",
                "error_message": "no prompts found in dataset",
            },
        )
        return

    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for sample_id, prompt in enumerate(prompts):
            start_time = time.time()
            output = llm.generate([prompt], sampling_params)[0]
            end_time = time.time()

            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            peak_vram_mb = info.used / (1024 * 1024)
            generated_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
            total_time = end_time - start_time
            throughput = generated_tokens / total_time if total_time > 0 else None
            latency = (total_time / generated_tokens) * 1000 if generated_tokens > 0 else None
            generated_text = output.outputs[0].text if output.outputs and getattr(output.outputs[0], "text", None) else ""
            quality = build_quality_row(generated_text)

            if reference_model is not None and reference_tokenizer is not None and reference_device is not None:
                ppl_result = compute_perplexity(
                    reference_model=reference_model,
                    tokenizer=reference_tokenizer,
                    text=generated_text,
                    device=reference_device,
                    max_length=args.ppl_max_length,
                    stride=args.ppl_stride,
                )
            else:
                ppl_result = {
                    "perplexity": None,
                    "ppl_loss": None,
                    "ppl_tokens": 0,
                    "ppl_status": "SKIPPED_NO_REFERENCE",
                    "ppl_error": "reference model not available",
                }

            row = {
                "model": args.model,
                "dataset": args.dataset,
                "sample_id": sample_id,
                "kv_cache_type": args.kv_cache_type,
                "kv_cache_dtype": kv_cache_dtype,
                "context_length": args.context_length,
                "peak_memory_mb": round(peak_vram_mb, 2),
                "latency_ms_per_token": round(latency, 2) if latency is not None else None,
                "throughput_tokens_per_s": round(throughput, 2) if throughput is not None else None,
                "generated_tokens": generated_tokens,
                "perplexity": ppl_result["perplexity"],
                "ppl_loss": ppl_result["ppl_loss"],
                "ppl_tokens": ppl_result["ppl_tokens"],
                "ppl_status": ppl_result["ppl_status"],
                "ppl_error": ppl_result["ppl_error"],
                "repetition_flag": quality["repetition_flag"],
                "gibberish_flag": quality["gibberish_flag"],
                "repeated_ngram_ratio": quality["repeated_ngram_ratio"],
                "special_char_ratio": quality["special_char_ratio"],
                "output_length": quality["output_length"],
                "quality_warning": quality["quality_warning"],
                "status": "OK",
                "error_message": "",
            }
            append_log_row(args.output, row)

    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        append_log_row(
            args.output,
            {
                "model": args.model,
                "dataset": args.dataset,
                "sample_id": 0,
                "kv_cache_type": args.kv_cache_type,
                "kv_cache_dtype": kv_cache_dtype,
                "context_length": args.context_length,
                "peak_memory_mb": None,
                "latency_ms_per_token": None,
                "throughput_tokens_per_s": None,
                "generated_tokens": None,
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": 0,
                "ppl_status": "OOM",
                "ppl_error": str(exc),
                "repetition_flag": False,
                "gibberish_flag": False,
                "repeated_ngram_ratio": None,
                "special_char_ratio": None,
                "output_length": None,
                "quality_warning": "",
                "status": "OOM",
                "error_message": str(exc),
            },
        )
    except Exception as exc:
        append_log_row(
            args.output,
            {
                "model": args.model,
                "dataset": args.dataset,
                "sample_id": 0,
                "kv_cache_type": args.kv_cache_type,
                "kv_cache_dtype": kv_cache_dtype,
                "context_length": args.context_length,
                "peak_memory_mb": None,
                "latency_ms_per_token": None,
                "throughput_tokens_per_s": None,
                "generated_tokens": None,
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": 0,
                "ppl_status": "ERROR",
                "ppl_error": str(exc),
                "repetition_flag": False,
                "gibberish_flag": False,
                "repeated_ngram_ratio": None,
                "special_char_ratio": None,
                "output_length": None,
                "quality_warning": "",
                "status": "ERROR",
                "error_message": str(exc),
            },
        )


def main() -> None:
    args = parse_args()
    setup_csv(args.output)

    if args.mock_mode or not HAS_GPU_STACK:
        run_mock_benchmark(args)
    else:
        run_real_benchmark(args)


if __name__ == "__main__":
    main()

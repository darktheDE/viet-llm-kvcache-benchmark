"""
Optimized worker script to benchmark Mistral 7B on a single KV cache compression method.
Loads the model ONCE with maximum context length room (16000 * 1.4 ratio + 128 + 512 = 23040),
then sequentially runs context lengths [4000, 8000, 16000] to save load time.
"""

import argparse
import json
import csv
import time
import os
import sys
import gc
import math
import subprocess
import threading
from pathlib import Path

# ============================================================
# 1. Import GPU libraries
# ============================================================
try:
    from vllm import LLM, SamplingParams
    import pynvml
    import torch
    import psutil
except ImportError as e:
    print(f"LOI NGHIEM TRONG: Thieu thu vien bat buoc - {e}")
    print("Hay cai dat: pip install vllm pynvml torch psutil")
    sys.exit(1)

try:
    from huggingface_hub import login as hf_login
except ImportError:
    hf_login = None

# Model Configuration
MODEL_NAME = "mistral:7b-instruct-v0.3-fp16"
HF_MODEL_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_RATIO = 1.4  # Mistral tokenizer expansion factor for Vietnamese text

KV_CACHE_DTYPE_MAP = {
    "FP16": "auto",
    "FP8": "fp8",
    "HQQ": "int4_per_token_head",
    "PolarQuant": "fp8",
    "TurboQuant": "turboquant_4bit_nc",
    "TurboQuant_3bit": "turboquant_3bit_nc",
}

CSV_HEADER = [
    "model", "kv_cache_type", "context_length",
    "peak_memory_mb", "latency_ms_per_token",
    "throughput_tokens_per_s", "perplexity", "status",
    "sample_id", "output_path"
]

class VRAMMonitor:
    """Background thread to poll VRAM usage via pynvml."""
    def __init__(self, gpu_index=0, interval_ms=50):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.interval = interval_ms / 1000.0
        self.peak_mb = 0.0
        self._running = False
        self._thread = None

    def _monitor_loop(self):
        while self._running:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                current_mb = info.used / (1024 * 1024)
                if current_mb > self.peak_mb:
                    self.peak_mb = current_mb
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_mb = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return round(self.peak_mb, 2)


def configure_model_access(hf_token=None, pull_ollama=False):
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        if hf_login is not None:
            hf_login(token=token, add_to_git_credential=False)
        print("  -> Hugging Face token da duoc nap.")

    if pull_ollama:
        try:
            subprocess.run(["ollama", "pull", MODEL_NAME], check=True, timeout=1800)
            print(f"  -> Ollama model da san sang: {MODEL_NAME}")
        except FileNotFoundError:
            print("  -> Khong tim thay lenh ollama; bo qua.")
        except Exception as exc:
            print(f"  -> Canh bao: ollama pull that bai: {exc}")

    return HF_MODEL_REPO


def setup_csv(output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(output_path)
    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)


def log_result(output_path, result_dict):
    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            result_dict.get("model"),
            result_dict.get("kv_cache_type"),
            result_dict.get("context_length"),
            result_dict.get("peak_memory_mb", ""),
            result_dict.get("latency_ms_per_token", ""),
            result_dict.get("throughput_tokens_per_s", ""),
            result_dict.get("perplexity", ""),
            result_dict.get("status", "OK"),
            result_dict.get("sample_id", ""),
            result_dict.get("output_path", ""),
        ])


def persist_generated_texts(jsonl_path, records):
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, mode='a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Mistral 7B Benchmark - Single Method")
    parser.add_argument("--kv_cache_type", type=str, default="FP16", choices=["FP16", "FP8", "HQQ", "PolarQuant", "TurboQuant"], help="KV Cache method to evaluate")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--output", type=str, default="../../results/template_log_real_run.csv", help="Output CSV path")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to run per context length")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--pull_ollama", action="store_true", default=False, help="Pull Ollama model before running")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  OPTIMIZED MISTRAL WORKER")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Method:  {args.kv_cache_type}")
    print(f"{'='*60}\n")

    # --- Bước 1: Nạp Dataset ---
    if args.dataset is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset = os.path.join(script_dir, "../../datasets/test_set_small.json")
    if not os.path.isabs(args.dataset):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset = os.path.join(script_dir, args.dataset)
    print(f"  -> Path dataset: {args.dataset}")

    try:
        with open(args.dataset, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict) and "samples" in data:
            all_samples = data["samples"]
        elif isinstance(data, list):
            all_samples = data
        else:
            raise ValueError("Format dataset khong hop le.")
    except Exception as e:
        print(f"  LOI load dataset: {e}")
        sys.exit(1)

    setup_csv(args.output)
    jsonl_path = args.output.replace(".csv", "_generated.jsonl")

    # --- Bước 2: Khởi tạo VRAM Monitor ---
    vram_monitor = VRAMMonitor(gpu_index=0, interval_ms=50)

    # --- Bước 3: Tải Model qua vLLM ---
    kv_dtype = KV_CACHE_DTYPE_MAP.get(args.kv_cache_type, "auto")
    vllm_model = configure_model_access(hf_token=args.hf_token, pull_ollama=args.pull_ollama)

    # Đặt max_model_len đủ room cho context length lớn nhất (16000) nhân với expansion ratio
    max_ctx_length = 16000
    max_len = int(max_ctx_length * MISTRAL_RATIO) + args.max_new_tokens + 512
    print(f"  -> Loading model with max_model_len={max_len} (ctx=16000 x {MISTRAL_RATIO} + new={args.max_new_tokens} + buf=512)...")

    llm = None
    try:
        torch.cuda.reset_peak_memory_stats()
        llm = LLM(
            model=vllm_model,
            kv_cache_dtype=kv_dtype,
            max_model_len=max_len,
            gpu_memory_util_organization=0.85, # Note: using gpu_memory_utilization, wait let me use correct param
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=max(max_len, 4096),
            max_num_seqs=2,
            trust_remote_code=True,
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        )
        print("  -> Model da loaded vao GPU.")
    except Exception as e:
        print(f"  LOI khi tai model: {e}")
        # Log error cho ca 3 context lengths
        for ctx in [4000, 8000, 16000]:
            result = {
                "model": MODEL_NAME,
                "kv_cache_type": args.kv_cache_type,
                "context_length": ctx,
                "peak_memory_mb": "ERROR",
                "latency_ms_per_token": "ERROR",
                "throughput_tokens_per_s": "ERROR",
                "perplexity": "ERROR",
                "status": f"LOAD_ERROR: {e}",
                "sample_id": f"mistral:7b-instruct-v0.3-fp16__{args.kv_cache_type}__{ctx}",
                "output_path": jsonl_path
            }
            log_result(args.output, result)
        return

    # --- Bước 4: Chạy benchmark tuần tự cho các Context Lengths ---
    context_lengths = [4000, 8000, 16000]
    context_group_map = {4000: "4k", 8000: "8k", 16000: "16k"}

    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    for ctx in context_lengths:
        target_group = context_group_map.get(ctx)
        filtered = [s for s in all_samples if s.get("context_group") == target_group]
        if not filtered:
            print(f"  -> Canh bao: Khong co sample cho context_group={target_group}, dung tat ca")
            filtered = all_samples

        samples = filtered[:args.num_samples]
        prompts = [item["text"] for item in samples]
        print(f"\n--------------------------------------------------")
        print(f"  Run: Context={ctx} | Samples={len(prompts)}")
        print(f"--------------------------------------------------")

        try:
            torch.cuda.reset_peak_memory_stats()
            vram_monitor.start()

            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            end_time = time.time()

            peak_vram_mb = vram_monitor.stop()
            torch_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            final_peak = max(peak_vram_mb, torch_peak_mb)

            total_generated_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
            total_time = end_time - start_time

            if total_generated_tokens > 0 and total_time > 0:
                latency = (total_time / total_generated_tokens) * 1000
                throughput = total_generated_tokens / total_time
            else:
                latency = 0
                throughput = 0

            print(f"  -> Inference hoan tat!")
            print(f"     * Peak VRAM:  {round(final_peak, 2)} MB")
            print(f"     * Latency:    {round(latency, 2)} ms/token")
            print(f"     * Throughput: {round(throughput, 2)} tokens/s")

            # Ghi ket qua vao CSV
            sample_id = f"mistral:7b-instruct-v0.3-fp16__{args.kv_cache_type}__{ctx}"
            result = {
                "model": MODEL_NAME,
                "kv_cache_type": args.kv_cache_type,
                "context_length": ctx,
                "peak_memory_mb": round(final_peak, 2),
                "latency_ms_per_token": round(latency, 2),
                "throughput_tokens_per_s": round(throughput, 2),
                "perplexity": "",
                "status": "OK",
                "sample_id": sample_id,
                "output_path": jsonl_path
            }
            log_result(args.output, result)

            # Persist generated text
            records = []
            for i, out in enumerate(outputs):
                records.append({
                    "sample_id": f"{sample_id}__s{i}",
                    "prompt_text": prompts[i],
                    "generated_text": out.outputs[0].text,
                    "generated_tokens": len(out.outputs[0].token_ids),
                    "model": MODEL_NAME,
                    "dataset": args.dataset,
                    "context_length": ctx,
                    "kv_cache_type": args.kv_cache_type,
                    "kv_cache_dtype": kv_dtype,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "seed": None,
                    "status": "OK",
                    "error_message": None
                })
            persist_generated_texts(jsonl_path, records)

        except torch.cuda.OutOfMemoryError as oom:
            print(f"  -> [OOM] Tran bo nho GPU khi chay context={ctx}: {oom}")
            vram_monitor.stop()
            result = {
                "model": MODEL_NAME,
                "kv_cache_type": args.kv_cache_type,
                "context_length": ctx,
                "peak_memory_mb": "OOM",
                "latency_ms_per_token": "OOM",
                "throughput_tokens_per_s": "OOM",
                "perplexity": "ERROR",
                "status": "OOM",
                "sample_id": f"mistral:7b-instruct-v0.3-fp16__{args.kv_cache_type}__{ctx}",
                "output_path": jsonl_path
            }
            log_result(args.output, result)
        except Exception as e:
            print(f"  -> [ERROR] Loi khi chay context={ctx}: {e}")
            vram_monitor.stop()
            result = {
                "model": MODEL_NAME,
                "kv_cache_type": args.kv_cache_type,
                "context_length": ctx,
                "peak_memory_mb": "ERROR",
                "latency_ms_per_token": "ERROR",
                "throughput_tokens_per_s": "ERROR",
                "perplexity": "ERROR",
                "status": f"RUN_ERROR: {e}",
                "sample_id": f"mistral:7b-instruct-v0.3-fp16__{args.kv_cache_type}__{ctx}",
                "output_path": jsonl_path
            }
            log_result(args.output, result)

    # Clean up
    print("\nGiai phong VRAM...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("Done worker.")

if __name__ == "__main__":
    main()

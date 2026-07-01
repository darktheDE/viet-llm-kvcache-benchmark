"""
Script đo đạc Benchmark THỰC TẾ trên GPU (Real Mode).
Yêu cầu: Máy chủ có GPU >= 24GB VRAM, đã cài vLLM, pynvml, PyTorch.

File này là "Công nhân thực thi" — chạy đo đạc cho ĐÚNG 1 cấu hình
(1 Model + 1 KV Cache Type + 1 Context Length). Được gọi bởi
run_real_grid.py (Quản đốc) hoặc chạy lẻ từ dòng lệnh.

Kết quả ghi vào: ../../results/real_benchmark_log.csv

Cách chạy lẻ:
    python scripts/test/run_real_benchmark.py \
        --model "vilm/vinallama-7b-chat" \
        --kv_cache_type FP16 \
        --context_length 8000

Cách chạy Grid (tự động 75 cấu hình):
    python scripts/test/run_real_grid.py
"""

import argparse
import json
import csv
import time
import os
import sys
import math
import threading
from pathlib import Path

# ============================================================
# 1. Import thư viện GPU (BẮT BUỘC phải có trên máy chạy thật)
# ============================================================
try:
    from vllm import LLM, SamplingParams
    import pynvml
    import torch
except ImportError as e:
    print(f"LOI NGHIEM TRONG: Thieu thu vien bat buoc - {e}")
    print("Hay cai dat: pip install vllm pynvml torch")
    sys.exit(1)

# ============================================================
# 2. Danh sách Model và Cấu hình
# ============================================================
SUPPORTED_MODELS = [
    "vilm/vinallama-7b-chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "ura-hcmut/URA-LLaMa-3-8B",
    "Viet-Mistral/Vistral-7B-Chat"
]

# Ánh xạ tên phương pháp nén -> tham số kv_cache_dtype của vLLM
KV_CACHE_DTYPE_MAP = {
    "FP16": "auto",
    "FP8": "fp8",
    "HQQ": "hqq_4bit",
    "PolarQuant": "polarquant_4bit",
    "TurboQuant": "turboquant_4bit_nc",
}


def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Real GPU Benchmark - KV Cache Compression trên Vietnamese LLMs"
    )
    parser.add_argument(
        "--model", type=str, default="vilm/vinallama-7b-chat",
        choices=SUPPORTED_MODELS, help="Ten mo hinh can benchmark"
    )
    parser.add_argument(
        "--dataset", type=str, default="../../datasets/test_set_small.json",
        help="Duong dan den tap du lieu"
    )
    parser.add_argument(
        "--context_length", type=int, default=8000,
        help="Do dai ngu canh toi da (Max Model Len)"
    )
    parser.add_argument(
        "--kv_cache_type", type=str, default="FP16",
        choices=["FP16", "FP8", "HQQ", "PolarQuant", "TurboQuant"],
        help="Phuong phap nen KV Cache"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="So token toi da can sinh (Decode phase)"
    )
    parser.add_argument(
        "--output", type=str, default="../../results/real_benchmark_log.csv",
        help="Duong dan luu ket qua CSV"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="So luong mau van ban lay tu dataset de test"
    )
    return parser.parse_args()


# ============================================================
# 3. Hàm tiện ích ghi CSV
# ============================================================
def setup_csv(output_path):
    """Tạo file CSV với header nếu chưa tồn tại."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(output_path)
    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model", "kv_cache_type", "context_length",
                "peak_memory_mb", "latency_ms_per_token",
                "throughput_tokens_per_s", "perplexity", "status"
            ])


def log_result(output_path, result_dict):
    """Ghi 1 dòng kết quả vào file CSV."""
    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            result_dict.get("model"),
            result_dict.get("kv_cache_type"),
            result_dict.get("context_length"),
            result_dict.get("peak_memory_mb"),
            result_dict.get("latency_ms_per_token"),
            result_dict.get("throughput_tokens_per_s"),
            result_dict.get("perplexity", "N/A"),
            result_dict.get("status", "OK")
        ])


# ============================================================
# 4. Luồng đo Peak VRAM liên tục (Background Thread)
#    Theo đúng yêu cầu [TECH]_todolist.md Bước 3:
#    "Thiết lập luồng đo song song để lấy Peak VRAM cao nhất"
# ============================================================
class VRAMMonitor:
    """Luồng chạy nền liên tục ghi nhận Peak VRAM cao nhất."""

    def __init__(self, gpu_index=0, interval_ms=50):
        """
        Args:
            gpu_index: Chỉ số GPU cần đo (mặc định GPU 0).
            interval_ms: Tần suất lấy mẫu VRAM (mili-giây).
        """
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.interval = interval_ms / 1000.0
        self.peak_mb = 0.0
        self._running = False
        self._thread = None

    def _monitor_loop(self):
        """Vòng lặp đo VRAM liên tục cho đến khi stop() được gọi."""
        while self._running:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            current_mb = info.used / (1024 * 1024)
            if current_mb > self.peak_mb:
                self.peak_mb = current_mb
            time.sleep(self.interval)

    def start(self):
        """Bắt đầu đo VRAM nền."""
        self.peak_mb = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Dừng đo và trả về Peak VRAM (MB)."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return round(self.peak_mb, 2)


# ============================================================
# 5. Tính Perplexity (PPL) thật từ Loss của mô hình
#    PPL = exp(average_negative_log_likelihood)
# ============================================================
def compute_perplexity(llm, prompts, sampling_params):
    """
    Tính Perplexity gần đúng dựa trên log-probabilities từ vLLM.

    Returns:
        Giá trị PPL trung bình trên tất cả các prompts.
        Trả về "N/A" nếu không tính được.
    """
    try:
        # Yêu cầu vLLM trả về log_probs cho mỗi token sinh ra
        ppl_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=1  # Lấy log-prob của các token trong prompt
        )
        outputs = llm.generate(prompts[:2], ppl_params)  # Chạy 2 mẫu để tính PPL

        total_log_prob = 0.0
        total_tokens = 0
        for output in outputs:
            if output.prompt_logprobs:
                for token_logprob in output.prompt_logprobs:
                    if token_logprob is not None:
                        # Lấy log-prob cao nhất (token thực tế)
                        for token_id, logprob_obj in token_logprob.items():
                            total_log_prob += logprob_obj.logprob
                            total_tokens += 1
                            break  # Chỉ lấy token đầu tiên (token thực tế)

        if total_tokens > 0:
            avg_neg_log_likelihood = -total_log_prob / total_tokens
            ppl = math.exp(avg_neg_log_likelihood)
            return round(ppl, 2)
        else:
            return "N/A"
    except Exception as e:
        print(f"  [WARN] Khong tinh duoc PPL: {e}")
        return "N/A"


# ============================================================
# 6. Hàm chính: Chạy Benchmark THỰC TẾ trên GPU
# ============================================================
def run_real_benchmark(args):
    """
    Thực thi đo đạc trên GPU thật với vLLM.
    Đo: Peak VRAM, Latency (ITL), Throughput, Perplexity.
    """
    print(f"\n{'='*60}")
    print(f"  REAL GPU BENCHMARK")
    print(f"  Model:   {args.model}")
    print(f"  Method:  {args.kv_cache_type}")
    print(f"  Context: {args.context_length} tokens")
    print(f"{'='*60}\n")

    # --- Bước 1: Nạp Dataset ---
    print("[1/5] Nap du lieu tu dataset...")
    try:
        with open(args.dataset, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item["text"] for item in data[:args.num_samples]]
        print(f"  -> Da nap {len(prompts)} mau van ban.")
    except FileNotFoundError:
        print(f"  LOI: Khong tim thay file {args.dataset}")
        sys.exit(1)

    # --- Bước 2: Khởi tạo VRAM Monitor ---
    print("[2/5] Khoi tao he thong do VRAM (pynvml background thread)...")
    vram_monitor = VRAMMonitor(gpu_index=0, interval_ms=50)

    # --- Bước 3: Tải Model qua vLLM ---
    kv_dtype = KV_CACHE_DTYPE_MAP.get(args.kv_cache_type, "auto")
    print(f"[3/5] Tai mo hinh vao GPU voi kv_cache_dtype={kv_dtype}...")
    try:
        # Reset peak memory tracking của PyTorch
        torch.cuda.reset_peak_memory_stats()

        llm = LLM(
            model=args.model,
            kv_cache_dtype=kv_dtype,
            max_model_len=args.context_length,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=4096,
            max_num_seqs=2,
            trust_remote_code=True
        )
        print(f"  -> Mo hinh da san sang tren GPU.")
    except Exception as e:
        print(f"  LOI tai mo hinh: {e}")
        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": "ERROR",
            "latency_ms_per_token": "ERROR",
            "throughput_tokens_per_s": "ERROR",
            "perplexity": "ERROR",
            "status": f"LOAD_ERROR: {e}"
        }
        log_result(args.output, result)
        return

    # --- Bước 4: Chạy Inference & Đo đạc ---
    print(f"[4/5] Bat dau Inference ({args.max_new_tokens} tokens/mau)...")
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    try:
        # Bật VRAM Monitor trước khi chạy Inference
        vram_monitor.start()

        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        # Dừng VRAM Monitor và lấy Peak
        peak_vram_mb = vram_monitor.stop()

        # Tính Latency & Throughput
        total_generated_tokens = sum(
            len(out.outputs[0].token_ids) for out in outputs
        )
        total_time = end_time - start_time

        if total_generated_tokens > 0 and total_time > 0:
            latency = (total_time / total_generated_tokens) * 1000  # ms/token
            throughput = total_generated_tokens / total_time  # tokens/s
        else:
            latency = 0
            throughput = 0

        # Cũng lấy Peak từ PyTorch để cross-check
        torch_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        final_peak = max(peak_vram_mb, torch_peak_mb)

        print(f"  -> Inference hoan tat!")
        print(f"     * Tokens sinh ra: {total_generated_tokens}")
        print(f"     * Thoi gian:      {round(total_time, 2)}s")
        print(f"     * Peak VRAM:      {round(final_peak, 2)} MB")
        print(f"     * Latency:        {round(latency, 2)} ms/token")
        print(f"     * Throughput:     {round(throughput, 2)} tokens/s")

        # --- Tính PPL ---
        print("[5/5] Tinh Perplexity (PPL)...")
        ppl = compute_perplexity(llm, prompts, sampling_params)
        print(f"     * PPL:            {ppl}")

        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": round(final_peak, 2),
            "latency_ms_per_token": round(latency, 2),
            "throughput_tokens_per_s": round(throughput, 2),
            "perplexity": ppl,
            "status": "OK"
        }

    except torch.cuda.OutOfMemoryError:
        vram_monitor.stop()
        print("  LOI: CUDA Out of Memory (OOM)!")
        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": "OOM",
            "latency_ms_per_token": "OOM",
            "throughput_tokens_per_s": "OOM",
            "perplexity": "N/A",
            "status": "OOM"
        }

    except Exception as e:
        vram_monitor.stop()
        print(f"  LOI khong xac dinh: {e}")
        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": "ERROR",
            "latency_ms_per_token": "ERROR",
            "throughput_tokens_per_s": "ERROR",
            "perplexity": "ERROR",
            "status": f"ERROR: {e}"
        }

    # --- Ghi kết quả ---
    log_result(args.output, result)
    print(f"\nKet qua da duoc ghi vao: {args.output}")

    # Giải phóng bộ nhớ GPU
    del llm
    torch.cuda.empty_cache()
    print("Da giai phong VRAM. San sang cho cau hinh tiep theo.\n")


def main():
    args = parse_args()
    setup_csv(args.output)
    run_real_benchmark(args)


if __name__ == "__main__":
    main()

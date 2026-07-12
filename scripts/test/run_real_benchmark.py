"""
Script đo đạc Benchmark THỰC TẾ trên GPU (Real Mode).
Yêu cầu: Máy chủ có GPU >= 24GB VRAM, đã cài vLLM, pynvml, PyTorch.

File này là "Công nhân thực thi" — chạy đo đạc cho ĐÚNG 1 cấu hình
(1 Model + 1 KV Cache Type + 1 Context Length). Được gọi bởi
run_real_grid.py (Quản đốc) hoặc chạy lẻ từ dòng lệnh.

Kết quả ghi vào: ../../results/template_log_real_run.csv

Cách chạy lẻ:
    python scripts/test/run_real_benchmark.py \
        --model "gemma4:e4b-it-bf16" \
        --kv_cache_type FP16 \
        --context_length 16000

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
import subprocess
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

try:
    from huggingface_hub import login as hf_login
except ImportError:
    hf_login = None

# ============================================================
# 2. Danh sách Model và Cấu hình
# ============================================================
SUPPORTED_MODELS = [
    "qwen3:8b-fp16",
    "llama3.1:8b-instruct-fp16",
    "mistral:7b-instruct-v0.3-fp16",
    "qwen2.5:7b-instruct-fp16",
]

OLLAMA_TO_HF_MODEL = {
    "qwen3:8b-fp16": "Qwen/Qwen3-8B",
    "llama3.1:8b-instruct-fp16": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral:7b-instruct-v0.3-fp16": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5:7b-instruct-fp16": "Qwen/Qwen2.5-7B-Instruct",
}


def resolve_model_for_vllm(model_name):
    """Map Ollama aliases to Hugging Face repos for vLLM runs.

    Args:
        model_name: Ollama model alias or Hugging Face repo id.

    Returns:
        Hugging Face repo id used by vLLM.
    """
    return OLLAMA_TO_HF_MODEL.get(model_name, model_name)


def configure_model_access(model_name, hf_token=None, pull_ollama=False):
    """Prepare Hugging Face token and optional Ollama pull on rented machines.

    Args:
        model_name: User-facing model alias or Hugging Face repo id.
        hf_token: Optional Hugging Face token. If omitted, HF_TOKEN or
            HUGGING_FACE_HUB_TOKEN from the environment is used.
        pull_ollama: Whether to run `ollama pull` for Ollama aliases.

    Returns:
        Hugging Face repo id resolved for vLLM.
    """
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        if hf_login is not None:
            hf_login(token=token, add_to_git_credential=False)
        print("  -> Hugging Face token da duoc nap tu --hf_token/HF_TOKEN.")

    if pull_ollama and model_name in OLLAMA_TO_HF_MODEL:
        try:
            subprocess.run(["ollama", "pull", model_name], check=True, timeout=1800)
            print(f"  -> Ollama model da san sang: {model_name}")
        except FileNotFoundError:
            print("  -> Khong tim thay lenh ollama; bo qua buoc ollama pull.")
        except Exception as exc:
            print(f"  -> Canh bao: ollama pull that bai: {exc}")

    return resolve_model_for_vllm(model_name)

# Ánh xạ tên phương pháp nén -> tham số kv_cache_dtype của vLLM
KV_CACHE_DTYPE_MAP = {
    "FP16": "auto",
    "FP8": "fp8",
    "HQQ": "int4_per_token_head",       # hqq_4bit not in vLLM 0.25; closest alternative
    "PolarQuant": "fp8",                 # polarquant_4bit not in vLLM 0.25; fallback to fp8
    "TurboQuant": "turboquant_4bit_nc",  # Built-in TurboQuant with 4-bit, no compensation
    "TurboQuant_3bit": "turboquant_3bit_nc",  # Extra: 3-bit TurboQuant variant
}


def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Real GPU Benchmark - KV Cache Compression trên Vietnamese LLMs"
    )
    parser.add_argument(
        "--model", type=str, default="gemma4:e4b",
        choices=SUPPORTED_MODELS, help="Ten mo hinh can benchmark"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Duong dan den tap du lieu (mac dinh: datasets/test_set_small.json tu goc project)"
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
        "--output", type=str, default="../../results/template_log_real_run.csv",
        help="Duong dan luu ket qua CSV"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="So luong mau van ban lay tu dataset de test"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Hugging Face access token; mac dinh doc tu HF_TOKEN/HUGGING_FACE_HUB_TOKEN"
    )
    parser.add_argument(
        "--pull_ollama", action="store_true", default=False,
        help="Pull Ollama model before running (default: False, use HuggingFace)"
    )
    return parser.parse_args()


# ============================================================
# 3. Hàm tiện ích ghi CSV
# ============================================================
# Header CSV mở rộng: thêm sample_id, output_path để ghép JSONL cho PPL backfill
CSV_HEADER = [
    "model", "kv_cache_type", "context_length",
    "peak_memory_mb", "latency_ms_per_token",
    "throughput_tokens_per_s", "perplexity", "status",
    "sample_id", "output_path"
]


def setup_csv(output_path):
    """Tạo file CSV với header nếu chưa tồn tại."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(output_path)
    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)


def log_result(output_path, result_dict):
    """Ghi 1 dòng kết quả vào file CSV."""
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
    """
    Lưu generated_text vào file JSONL để phục vụ backfill PPL offline.

    Mỗi dòng JSONL chứa:
      - sample_id: khóa ổn định để ghép lại với CSV
      - prompt_text: đoạn prompt gốc (để sau đổi cách tính PPL prompt+completion)
      - generated_text: văn bản model sinh ra
      - generated_tokens: số token đã sinh
      - model, dataset, context_length, kv_cache_type, kv_cache_dtype
      - max_new_tokens, temperature, top_p, top_k
      - status, error_message

    Args:
        jsonl_path: Đường dẫn file JSONL đầu ra.
        records: Danh sách dict chứa thông tin cần ghi.
    """
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, mode='a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


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
# 5. Tính Perplexity (PPL)
#    LƯU Ý: PPL cần được tính OFFLINE bằng model tham chiếu BF16.
#    Hàm compute_perplexity() cũ đã bị XÓA vì tính PPL trên chính
#    model bị nén là sai nguyên lý (self-evaluation bias).
#    Thay vào đó, script sẽ lưu generated_text vào JSONL để
#    backfill PPL sau bằng script compute_ppl_offline.py.
# ============================================================


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
    # Resolve dataset path: absolute or relative to project root
    if args.dataset is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset = os.path.join(script_dir, "../../datasets/test_set_small.json")
    if not os.path.isabs(args.dataset):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset = os.path.join(script_dir, args.dataset)
    print(f"  -> Dataset path: {args.dataset}")
    try:
        with open(args.dataset, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            
        if isinstance(data, dict) and "samples" in data:
            all_samples = data["samples"]
        elif isinstance(data, list):
            all_samples = data
        else:
            raise ValueError("Invalid dataset format. Expected list or dict with 'samples' key.")

        # Filter samples by context_group matching requested context_length
        context_group_map = {4000: "4k", 8000: "8k", 16000: "16k", 32000: "32k"}
        target_group = context_group_map.get(args.context_length)
        if target_group:
            filtered = [s for s in all_samples if s.get("context_group") == target_group]
            if filtered:
                samples = filtered
                print(f"  -> Filtered to {len(samples)} samples with context_group={target_group}")
            else:
                print(f"  -> Canh bao: Khong co sample cho context_group={target_group}, dung tat ca")
                samples = all_samples
        else:
            samples = all_samples

        samples = samples[:args.num_samples]
        prompts = [item["text"] for item in samples]
        print(f"  -> Da nap {len(prompts)} mau van ban.")
    except FileNotFoundError:
        print(f"  LOI: Khong tim thay file {args.dataset}")
        sys.exit(1)

    # --- Bước 2: Khởi tạo VRAM Monitor ---
    print("[2/5] Khoi tao he thong do VRAM (pynvml background thread)...")
    vram_monitor = VRAMMonitor(gpu_index=0, interval_ms=50)

    # --- Bước 3: Tải Model qua vLLM ---
    kv_dtype = KV_CACHE_DTYPE_MAP.get(args.kv_cache_type, "auto")

    # Resolve HF token: CLI arg -> env HF_TOKEN -> env HUGGING_FACE_HUB_TOKEN
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print(f"  -> Su dung HuggingFace token (gated model).")
    else:
        print(f"  -> Khong co HF token. Neu model bi gated, se bao loi.")

    print(f"[3/5] Tai mo hinh vao GPU voi kv_cache_dtype={kv_dtype}...")
    try:
        # Reset peak memory tracking của PyTorch
        torch.cuda.reset_peak_memory_stats()
        vllm_model = configure_model_access(
            args.model,
            hf_token=args.hf_token,
            pull_ollama=args.pull_ollama,
        )
        print(f"  -> vLLM repo: {vllm_model}")

        # max_model_len needs room for prompt tokens + generated tokens + buffer
        # Sample actual_tokens may exceed target context_length due to tokenizer differences
        max_len = args.context_length + args.max_new_tokens + 1024
        print(f"  -> max_model_len={max_len} (ctx={args.context_length} + new={args.max_new_tokens} + buf=4096)")
        llm = LLM(
            model=vllm_model,
            kv_cache_dtype=kv_dtype,
            max_model_len=max_len,
            gpu_memory_utilization=0.98,
            # max_num_batched_tokens phai >= max_model_len
            max_num_batched_tokens=max(max_len, 4096),
            max_num_seqs=2,
            trust_remote_code=True,
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
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

    # Đường dẫn JSONL lưu generated_text cho backfill PPL
    jsonl_path = args.output.replace(".csv", "_generated.jsonl")

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

        # --- Bước 5: Persist generated_text vào JSONL cho backfill PPL ---
        print("[5/5] Luu generated_text vao JSONL de backfill PPL sau...")
        kv_dtype = KV_CACHE_DTYPE_MAP.get(args.kv_cache_type, "auto")
        jsonl_records = []
        for i, out in enumerate(outputs):
            sample_id = f"{args.model}__{args.kv_cache_type}__{args.context_length}__s{i}"
            generated_text = out.outputs[0].text
            gen_tokens = len(out.outputs[0].token_ids)
            jsonl_records.append({
                "sample_id": sample_id,
                "prompt_text": prompts[i],
                "generated_text": generated_text,
                "generated_tokens": gen_tokens,
                "model": args.model,
                "dataset": args.dataset,
                "context_length": args.context_length,
                "kv_cache_type": args.kv_cache_type,
                "kv_cache_dtype": kv_dtype,
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "seed": None,
                "status": "OK",
                "error_message": None,
            })
        persist_generated_texts(jsonl_path, jsonl_records)
        print(f"     * Da luu {len(jsonl_records)} mau vao: {jsonl_path}")

        # Ghi 1 dòng tổng hợp vào CSV (PPL để trống, sẽ backfill sau)
        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": round(final_peak, 2),
            "latency_ms_per_token": round(latency, 2),
            "throughput_tokens_per_s": round(throughput, 2),
            "perplexity": "",  # Sẽ backfill PPL offline sau
            "status": "OK",
            "sample_id": f"{args.model}__{args.kv_cache_type}__{args.context_length}",
            "output_path": jsonl_path,
        }

    except torch.cuda.OutOfMemoryError:
        vram_monitor.stop()
        print("  LOI: CUDA Out of Memory (OOM)!")
        # Cột số để trống, ghi OOM vào cột status
        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": "",
            "latency_ms_per_token": "",
            "throughput_tokens_per_s": "",
            "perplexity": "",
            "status": "OOM",
            "sample_id": "",
            "output_path": "",
        }

    except Exception as e:
        vram_monitor.stop()
        print(f"  LOI khong xac dinh: {e}")
        # Cột số để trống, ghi chi tiết lỗi vào cột status
        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": "",
            "latency_ms_per_token": "",
            "throughput_tokens_per_s": "",
            "perplexity": "",
            "status": f"ERROR: {e}",
            "sample_id": "",
            "output_path": "",
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

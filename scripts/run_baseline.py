import argparse
import json
import csv
import time
import os
import sys

# Fix encoding cho Windows PowerShell (cp1252 không hỗ trợ tiếng Việt)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
import random
from pathlib import Path

# Thử import vLLM và pynvml. Nếu không có (chạy local/không GPU) thì chuyển sang Mock Mode.
try:
    from vllm import LLM, SamplingParams
    import pynvml
    import torch
    MOCK_MODE = False
except ImportError:
    MOCK_MODE = True
    print("WARNING: Không tìm thấy thư viện vLLM, pynvml hoặc PyTorch. Chuyển sang MOCK_MODE (Chế độ giả lập).")

SUPPORTED_MODELS = [
    "vilm/vinallama-7b-chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "ura-hcmut/URA-LLaMa-3-8B",
    "Viet-Mistral/Vistral-7B-Chat"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KV Cache Compression on Vietnamese LLMs")
    parser.add_argument("--model", type=str, default="vilm/vinallama-7b-chat", 
                        help="Tên mô hình cần benchmark", choices=SUPPORTED_MODELS)
    parser.add_argument("--dataset", type=str, default="datasets/test_set_small.json", help="Đường dẫn đến tập dữ liệu")
    parser.add_argument("--context_length", type=int, default=8000, help="Độ dài ngữ cảnh tối đa (Max Model Len)")
    parser.add_argument("--kv_cache_type", type=str, default="FP16", 
                        choices=["FP16", "FP8", "HQQ", "PolarQuant", "TurboQuant"], help="Phương pháp nén KV Cache")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Số token tối đa cần sinh (Decode)")
    parser.add_argument("--output", type=str, default="results/template_log.csv", help="Đường dẫn lưu kết quả CSV")
    parser.add_argument("--mock_mode", action="store_true", help="Ép buộc chạy ở chế độ giả lập (Mock Mode) không cần GPU")
    return parser.parse_args()

def setup_csv(output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(output_path)
    
    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "kv_cache_type", "context_length", "peak_memory_mb", "latency_ms_per_token", "throughput_tokens_per_s", "perplexity", "status"])

def log_result(output_path, result_dict):
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

def load_dataset_samples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "samples" in data:
        return data["samples"]
    else:
        raise ValueError(f"Invalid dataset format in {path}: expected list or dict with 'samples' key")

def filter_samples_by_length(samples: list[dict], context_length: int) -> list[dict]:
    # Determine the target bucket
    if context_length <= 5000:
        target_group = "4k"
        target_len = 4000
    elif context_length <= 10000:
        target_group = "8k"
        target_len = 8000
    else:
        target_group = "16k"
        target_len = 16000
        
    filtered = []
    for item in samples:
        cg = item.get("context_group")
        clt = item.get("context_length_target")
        tt = item.get("target_tokens")
        
        match = False
        if cg is not None and str(cg).lower() == target_group.lower():
            match = True
        elif clt is not None and int(clt) == target_len:
            match = True
        elif tt is not None and (abs(int(tt) - target_len) < 1000 or (target_len == 4000 and int(tt) == 4096) or (target_len == 8000 and int(tt) == 8192) or (target_len == 16000 and int(tt) == 16384)):
            match = True
            
        if match:
            filtered.append(item)
            
    if not filtered:
        filtered = samples
    return filtered

def run_mock_benchmark(args):
    print(f"\n=======================================================")
    print(f"   🚀 DEMO BENCHMARK KV CACHE COMPRESSION (MOCK MODE)  ")
    print(f"=======================================================\n")
    
    # Bước 1: Nạp Dữ Liệu
    print(f"[1. Nạp Dữ Liệu] Đang đọc file dataset: {args.dataset}")
    try:
        samples = load_dataset_samples(args.dataset)
        filtered_samples = filter_samples_by_length(samples, args.context_length)
        num_samples = min(len(filtered_samples), 3) # Chạy thử 3 mẫu
        print(f"  -> Thành công! Đã nạp {num_samples} mẫu văn bản tiếng Việt dài sau khi lọc.")
    except Exception as e:
        print(f"  -> Lỗi nạp dataset: {e}")
        # Ghi log lỗi vào CSV
        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": 0.0,
            "latency_ms_per_token": 0.0,
            "throughput_tokens_per_s": 0.0,
            "perplexity": "N/A",
            "status": "DATASET_LOAD_ERROR"
        }
        log_result(args.output, result)
        print(f"  -> Đã ghi nhận lỗi load dataset vào: {args.output}")
        sys.exit(1)
    time.sleep(1)
        
    # Bước 2: Tải Mô hình (16-bit)
    print(f"\n[2. Tải Mô hình (16-bit)] Đang tải trọng số mô hình...")
    print(f"  -> Model: {args.model}")
    print(f"  -> Context Length Target: {args.context_length} tokens")
    time.sleep(1.5)
    print(f"  -> Hoàn tất tải mô hình vào VRAM (giả lập 15GB).")
    
    # Bước 3: Kích hoạt Lõi Nén KV Cache
    print(f"\n[3. Kích hoạt Lõi Nén KV Cache]")
    print(f"  -> Khởi tạo Engine vLLM với thuật toán nén: {args.kv_cache_type}")
    if args.kv_cache_type == "TurboQuant":
        print(f"  -> [TurboQuant Active]: Đang bóp nghẹt bộ nhớ tạm từ 16-bit xuống 4-bit (real-time).")
    elif args.kv_cache_type == "FP16":
        print(f"  -> [FP16 Baseline]: Chạy nguyên bản không nén.")
    time.sleep(1)
    
    # Bước 4: Khai thác & Đo lường
    print(f"\n[4. Khai thác & Đo lường] Đang tiến hành Inference và sinh {args.max_new_tokens} tokens...")
    time.sleep(2) # Giả lập prefill & decode
    
    # Sinh thông số giả lập minh hoạ
    base_vram = 14000 # 14GB base
    if args.kv_cache_type == "TurboQuant":
        peak_vram = base_vram + args.context_length * 0.1
    elif args.kv_cache_type == "FP16":
        peak_vram = base_vram + args.context_length * 0.4
    else:
        peak_vram = base_vram + args.context_length * 0.2
        
    latency = 30.5 + (args.context_length / 1000)
    if args.kv_cache_type != "FP16":
        latency += 2.5 # Nén có thể tốn thêm tí time giả định
        
    throughput = 1000 / latency
    perplexity = round(random.uniform(5.0, 8.0), 2)
    
    print(f"  -> Inference hoàn tất! pynvml đã ghi nhận các chỉ số hệ thống:")
    print(f"     * Peak VRAM: {round(peak_vram, 2)} MB")
    print(f"     * Latency:   {round(latency, 2)} ms/token")
    print(f"     * PPL:       {perplexity}")
    
    # Bước 5: Ghi Log CSV
    print(f"\n[5. Ghi Log CSV] Đang định dạng và lưu kết quả...")
    result = {
        "model": args.model,
        "kv_cache_type": args.kv_cache_type,
        "context_length": args.context_length,
        "peak_memory_mb": round(peak_vram, 2),
        "latency_ms_per_token": round(latency, 2),
        "throughput_tokens_per_s": round(throughput, 2),
        "perplexity": perplexity,
        "status": "MOCK_OK"
    }
    
    log_result(args.output, result)
    print(f"  -> Đã append thành công 1 dòng log vào tệp: {args.output}")
    print(f"\n✅ DEMO PIPELINE HOÀN TẤT TRỌN VẸN!")

def run_real_benchmark(args):
    print(f"--- THỰC THI BENCHMARK TRÊN GPU ---")
    # Khởi tạo pynvml để đo VRAM
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Cấu hình kv_cache_dtype cho vLLM
    dtype_mapping = {
        "FP16": "auto",
        "FP8": "fp8",
        "TurboQuant": "turboquant_4bit_nc",
        # HQQ và PolarQuant có thể yêu cầu plugin hoặc cấu hình biến môi trường riêng
    }
    kv_cache_dtype = dtype_mapping.get(args.kv_cache_type, "auto")
    
    print(f"Đang tải mô hình {args.model} với kv_cache_dtype={kv_cache_dtype}...")
    try:
        # Khởi tạo vLLM. 
        # Tối ưu hóa VRAM: gpu_memory_utilization cao, max_num_seqs nhỏ
        llm = LLM(
            model=args.model,
            kv_cache_dtype=kv_cache_dtype,
            max_model_len=args.context_length,
            gpu_memory_utilization=0.98,
            max_num_batched_tokens=4096, # Tránh lỗi phân mảnh Qwen
            max_num_seqs=2,
            trust_remote_code=True
        )
        
        # Thiết lập SamplingParams
        sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
        
        # Load dataset
        try:
            samples = load_dataset_samples(args.dataset)
            filtered_samples = filter_samples_by_length(samples, args.context_length)
        except Exception as e:
            print(f"Lỗi nạp dataset: {e}")
            result = {
                "model": args.model,
                "kv_cache_type": args.kv_cache_type,
                "context_length": args.context_length,
                "peak_memory_mb": 0.0,
                "latency_ms_per_token": 0.0,
                "throughput_tokens_per_s": 0.0,
                "perplexity": "N/A",
                "status": "DATASET_LOAD_ERROR"
            }
            log_result(args.output, result)
            sys.exit(1)
            
        # Lọc các mẫu theo đúng context_length_target (tuỳ chọn)
        prompts = [item["text"] for item in filtered_samples[:5]] # Chạy 5 mẫu ví dụ
        
        # Thực hiện Generation
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        # Đo Peak VRAM
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        peak_vram_mb = info.used / (1024 * 1024)
        
        # Tính toán throughput
        total_generated_tokens = sum([len(out.outputs[0].token_ids) for out in outputs])
        total_time = end_time - start_time
        throughput = total_generated_tokens / total_time if total_time > 0 else 0
        latency = (total_time / total_generated_tokens) * 1000 if total_generated_tokens > 0 else 0
        
        result = {
            "model": args.model,
            "kv_cache_type": args.kv_cache_type,
            "context_length": args.context_length,
            "peak_memory_mb": round(peak_vram_mb, 2),
            "latency_ms_per_token": round(latency, 2),
            "throughput_tokens_per_s": round(throughput, 2),
            "perplexity": "N/A", # Sẽ cần chạy 1 pipeline khác để đo PPL riêng
            "status": "OK"
        }
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"LỖI OOM KHI CHẠY MỐC {args.context_length} tokens!")
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
        print(f"Lỗi không xác định: {e}")
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
        
    log_result(args.output, result)
    print(f"Hoàn tất! Kết quả đã được lưu vào {args.output}")


def main():
    args = parse_args()
    setup_csv(args.output)
    
    if MOCK_MODE or args.mock_mode:
        run_mock_benchmark(args)
    else:
        run_real_benchmark(args)

if __name__ == "__main__":
    main()

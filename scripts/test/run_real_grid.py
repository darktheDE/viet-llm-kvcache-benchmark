"""
Script Quản đốc: Chạy tự động toàn bộ Grid Search THỰC TẾ trên GPU.
Gọi run_real_benchmark.py cho từng cấu hình (Model x Method x Context).

Kết quả xuất ra: ../../results/template_log_real_run.csv

Yêu cầu: Máy chủ GPU Cloud (RunPod/Vast.ai) đã cài vLLM, pynvml.

Cách chạy:
    python scripts/test/run_real_grid.py
"""

import argparse
import subprocess
import time
import os
import sys

MODELS = [
    "qwen3:8b-fp16",
    "llama3.1:8b-instruct-fp16",
    "mistral:7b-instruct-v0.3-fp16",
    "qwen2.5:7b-instruct-fp16",
]

KV_CACHE_TYPES = ["FP16", "FP8", "HQQ", "PolarQuant", "TurboQuant"]
CONTEXT_LENGTHS = [4000, 8000, 16000]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "../../results/template_log_real_run.csv")
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "run_real_benchmark.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid Search orchestrator - chay tu dong 75 cau hinh benchmark"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace access token cho model gated (hoac dat env HF_TOKEN)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve HF token: CLI arg -> env HF_TOKEN -> env HUGGING_FACE_HUB_TOKEN
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    total = len(MODELS) * len(KV_CACHE_TYPES) * len(CONTEXT_LENGTHS)
    print("=" * 60)
    print("  REAL GPU BENCHMARK - GRID SEARCH (75 cau hinh)")
    print("=" * 60)
    print(f"Tong so cau hinh: {total}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Models: {len(MODELS)} | Methods: {len(KV_CACHE_TYPES)} | Contexts: {len(CONTEXT_LENGTHS)}")
    if hf_token:
        print(f"HF Token: *** (co su dung)")
    print("=" * 60 + "\n")

    count = 1
    success_count = 0
    oom_count = 0
    error_count = 0

    start_all = time.time()

    for model in MODELS:
        for kv_type in KV_CACHE_TYPES:
            for ctx in CONTEXT_LENGTHS:
                print(f"\n[{count}/{total}] Model={model} | Method={kv_type} | Context={ctx}")
                print("-" * 50)

                cmd = [
                    sys.executable, SCRIPT_PATH,
                    "--model", model,
                    "--kv_cache_type", kv_type,
                    "--context_length", str(ctx),
                    "--output", OUTPUT_CSV,
                    "--num_samples", "5",
                    "--max_new_tokens", "128",
                    "--dataset", os.path.join(SCRIPT_DIR, "../../datasets/test_set_small.json"),
                ]
                if hf_token:
                    cmd.extend(["--hf_token", hf_token])

                try:
                    env = os.environ.copy()
                    env["PYTHONUTF8"] = "1"
                    if hf_token:
                        env["HF_TOKEN"] = hf_token

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        env=env,
                        timeout=600  # Timeout 10 phut cho moi cau hinh
                    )

                    # In output cua subprocess
                    if result.stdout:
                        print(result.stdout[-500:])  # In 500 ky tu cuoi

                    if result.returncode == 0:
                        print(f"  [OK] Hoan tat thanh cong.")
                        success_count += 1
                    else:
                        # Kiểm tra xem có phải OOM không
                        if "OOM" in result.stdout or "OutOfMemory" in result.stderr:
                            print(f"  [OOM] Tran bo nho GPU.")
                            oom_count += 1
                        else:
                            print(f"  [ERROR] Return code: {result.returncode}")
                            if result.stderr:
                                print(result.stderr[-300:])
                            error_count += 1

                except subprocess.TimeoutExpired:
                    print(f"  [TIMEOUT] Vuot qua 10 phut, bo qua.")
                    error_count += 1
                except Exception as e:
                    print(f"  [ERROR] Loi he thong: {e}")
                    error_count += 1

                count += 1

    end_all = time.time()
    elapsed = round(end_all - start_all, 1)

    # Báo cáo tổng kết
    print("\n" + "=" * 60)
    print("  TONG KET BENCHMARK")
    print("=" * 60)
    print(f"  Thanh cong:  {success_count}/{total}")
    print(f"  OOM:         {oom_count}/{total}")
    print(f"  Loi khac:    {error_count}/{total}")
    print(f"  Tong thoi gian: {elapsed}s ({round(elapsed/60, 1)} phut)")
    print(f"  Ket qua CSV: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()

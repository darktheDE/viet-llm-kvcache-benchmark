"""
Script Quản đốc riêng cho 2 model còn lại: Llama 3.1 & Mistral 7B
Tăng buffer lên 8192 để fix lỗi tokenizer Mistral.
Yêu cầu HF token có quyền truy cập meta-llama/Llama-3.1-8B-Instruct.

Cách chạy:
    python scripts/test/run_real_grid_extra.py
"""

import argparse
import subprocess
import time
import os
import sys

MODELS = [
    "llama3.1:8b-instruct-fp16",
    "mistral:7b-instruct-v0.3-fp16",
]

KV_CACHE_TYPES = ["FP16", "FP8", "HQQ", "PolarQuant", "TurboQuant"]
CONTEXT_LENGTHS = [4000, 8000, 16000]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "../../results/template_log_real_run_extra.csv")
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "run_real_benchmark_extra.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid Search cho Llama 3.1 & Mistral 7B"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace access token cho model gated"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    total = len(MODELS) * len(KV_CACHE_TYPES) * len(CONTEXT_LENGTHS)
    print("=" * 60)
    print("  REAL GPU BENCHMARK - EXTRA MODELS (30 cau hinh)")
    print("=" * 60)
    print(f"Tong so cau hinh: {total}")
    print(f"Models: Llama 3.1 8B, Mistral 7B")
    print(f"Output CSV: {OUTPUT_CSV}")
    if hf_token:
        print(f"HF Token: *** (co su dung)")
    print("=" * 60 + "\n")

    count = 1
    success_count = 0

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
                    "--buffer", "8192",  # Buffer lon hon cho Mistral tokenizer
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
                        timeout=900
                    )

                    if result.stdout:
                        print(result.stdout[-500:])

                    if result.returncode == 0:
                        print(f"  [OK] Hoan tat thanh cong.")
                        success_count += 1
                    else:
                        if "OOM" in result.stdout or "OutOfMemory" in result.stderr:
                            print(f"  [OOM] Tran bo nho GPU.")
                        else:
                            print(f"  [ERROR] Return code: {result.returncode}")
                            if result.stderr:
                                print(result.stderr[-300:])

                except subprocess.TimeoutExpired:
                    print(f"  [TIMEOUT] Vuot qua 15 phut, bo qua.")
                except Exception as e:
                    print(f"  [ERROR] Loi he thong: {e}")

                count += 1

    end_all = time.time()
    elapsed = round(end_all - start_all, 1)

    print("\n" + "=" * 60)
    print("  TONG KET BENCHMARK EXTRA")
    print("=" * 60)
    print(f"  Thanh cong:  {success_count}/{total}")
    print(f"  Tong thoi gian: {elapsed}s ({round(elapsed/60, 1)} phut)")
    print(f"  Ket qua CSV: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()

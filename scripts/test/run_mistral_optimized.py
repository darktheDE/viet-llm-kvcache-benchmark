"""
Optimized Orchestrator script to run all Mistral benchmarks.
Launches run_mistral_single_method.py as a subprocess for each KV Cache type,
allowing the model to load only ONCE per method (benchmarking 4k, 8k, 16k context lengths in a single session).
"""

import argparse
import subprocess
import time
import os
import sys

KV_CACHE_TYPES = ["FP16", "FP8", "HQQ", "PolarQuant", "TurboQuant"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "../../results/template_log_real_run.csv")
WORKER_PATH = os.path.join(SCRIPT_DIR, "run_mistral_single_method.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Mistral Grid Search Orchestrator")
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace access token cho model gated (hoac dat env HF_TOKEN)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="So luong mau van ban de benchmark moi context group"
    )
    parser.add_argument(
        "--pull_ollama", action="store_true", default=False,
        help="Chay ollama pull truoc khi benchmark"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    total = len(KV_CACHE_TYPES)
    print("=" * 60)
    print("  OPTIMIZED MISTRAL GRID SEARCH - LOAD ONCE PER METHOD")
    print("=" * 60)
    print(f"Tong so methods: {total} (Moi method se chay 3 contexts: 4k, 8k, 16k)")
    print(f"Output CSV: {OUTPUT_CSV}")
    if hf_token:
        print(f"HF Token: *** (co su dung)")
    print("=" * 60 + "\n")

    start_all = time.time()

    success_count = 0
    error_count = 0

    for i, kv_type in enumerate(KV_CACHE_TYPES, 1):
        print(f"\n[{i}/{total}] Khoi dong sub-process cho Method={kv_type}...")
        print("-" * 50)

        cmd = [
            sys.executable, WORKER_PATH,
            "--kv_cache_type", kv_type,
            "--output", OUTPUT_CSV,
            "--num_samples", str(args.num_samples),
            "--dataset", os.path.join(SCRIPT_DIR, "../../datasets/test_set_small.json"),
        ]
        if hf_token:
            cmd.extend(["--hf_token", hf_token])
        if args.pull_ollama:
            cmd.append("--pull_ollama")

        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            if hf_token:
                env["HF_TOKEN"] = hf_token

            result = subprocess.run(
                cmd,
                capture_output=False,  # Cho phep in output truc tiep ra console trong luc chay
                env=env,
                timeout=1800  # Timeout 30 phut cho moi method (gom 3 contexts)
            )

            if result.returncode == 0:
                print(f"  [OK] Method={kv_type} hoan tat ca 3 context lengths.")
                success_count += 1
            else:
                print(f"  [ERROR] Sub-process error code: {result.returncode}")
                error_count += 1

        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] Sub-process vuot qua 30 phut.")
            error_count += 1
        except Exception as e:
            print(f"  [ERROR] Loi he thong khi goi sub-process: {e}")
            error_count += 1

    end_all = time.time()
    elapsed = round(end_all - start_all, 1)

    print("\n" + "=" * 60)
    print("  TONG KET BENCHMARK MISTRAL")
    print("=" * 60)
    print(f"  Hoan tat: {success_count}/{total} methods")
    print(f"  That bai: {error_count}/{total} methods")
    print(f"  Tong thoi gian: {elapsed}s ({round(elapsed/60, 1)} phut)")
    print(f"  Ket qua CSV: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()

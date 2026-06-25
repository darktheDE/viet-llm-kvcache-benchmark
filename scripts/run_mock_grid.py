import subprocess
import time
import os

MODELS = [
    "VinAI/PhoGPT-7B5-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "ura-hcmut/URA-LLaMa-3-8B",
    "Viet-Mistral/Vistral-7B-Chat"
]

KV_CACHE_TYPES = ["FP16", "FP8", "HQQ", "PolarQuant", "TurboQuant"]
CONTEXT_LENGTHS = [4000, 8000, 16000]

def main():
    print("=======================================================")
    print("🚀 KHỞI ĐỘNG HỆ THỐNG MOCK BENCHMARK HÀNG LOẠT (GRID SEARCH)")
    print("=======================================================\n")
    print(f"Tổng số cấu hình sẽ chạy: {len(MODELS) * len(KV_CACHE_TYPES) * len(CONTEXT_LENGTHS)}")
    print("Dữ liệu sẽ được append liên tục vào results/template_log.csv\n")
    
    time.sleep(2)

    count = 1
    total = len(MODELS) * len(KV_CACHE_TYPES) * len(CONTEXT_LENGTHS)
    
    for model in MODELS:
        for kv_type in KV_CACHE_TYPES:
            for ctx in CONTEXT_LENGTHS:
                print(f"\n[{count}/{total}] Đang chạy: Model={model} | Method={kv_type} | Context={ctx}")
                
                cmd = [
                    "python", "scripts/run_baseline.py",
                    "--model", model,
                    "--kv_cache_type", kv_type,
                    "--context_length", str(ctx),
                    "--mock_mode"
                ]
                
                # Chạy process con và chờ kết quả
                try:
                    env = os.environ.copy()
                    env["PYTHONUTF8"] = "1"
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
                    if result.returncode == 0:
                        print("  ✅ Đã ghi nhận log thành công.")
                    else:
                        print(f"  ❌ Cảnh báo: Có lỗi xảy ra trong quá trình chạy (Return code: {result.returncode})")
                        print(result.stderr)
                except Exception as e:
                    print(f"  ❌ Lỗi hệ thống khi gọi script: {e}")
                
                count += 1
                
    print("\n=======================================================")
    print("🎉 HOÀN TẤT QUÁ TRÌNH MOCK BENCHMARK TOÀN DIỆN!")
    print("Dữ liệu CSV đã sẵn sàng tại: results/template_log.csv")
    print("Bạn có thể dùng file này để vẽ biểu đồ so sánh hiệu năng.")
    print("=======================================================")

if __name__ == "__main__":
    main()

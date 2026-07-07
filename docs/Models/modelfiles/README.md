# Hướng dẫn thiết lập Context Window cho Benchmark

## Tại sao cần thiết lập num_ctx?

Khi nạp văn bản > 4K token vào Ollama, hệ thống sẽ **tự động cắt (truncate)**
về giới hạn mặc định của phần mềm (thường chỉ 2048–4096 token). Nếu không
khai báo `num_ctx`, mọi kết quả đo đạc ở mốc 8K/16K đều **vô nghĩa** vì
model thực ra chỉ nhìn thấy < 4K token đầu tiên.

---

## Cách 1: Ollama — Dùng Modelfile (khuyến nghị cho benchmark local)

Các Modelfile đã được chuẩn bị sẵn trong thư mục này. Chạy lần lượt:

```bash
# 1. Build custom model version với num_ctx=32K
ollama create gemma4-bench -f docs/Models/modelfiles/Modelfile.gemma4
ollama create qwen3-bench  -f docs/Models/modelfiles/Modelfile.qwen3
ollama create llama32-bench -f docs/Models/modelfiles/Modelfile.llama32

# 2. Xác nhận đã tạo thành công
ollama list

# 3. Kiểm tra context window
ollama show gemma4-bench --modelfile
```

Sau đó chạy benchmark bằng tên alias mới (`gemma4-bench`, v.v.).

---

## Cách 2: vLLM — Truyền max_model_len trực tiếp (ĐANG DÙNG cho real benchmark)

Script `scripts/test/run_real_benchmark.py` đã truyền `max_model_len=args.context_length`
trực tiếp vào vLLM engine. Không cần Modelfile khi dùng vLLM.

```python
llm = LLM(
    model=vllm_model,
    max_model_len=args.context_length,   # <-- đây chính là num_ctx của vLLM
    kv_cache_dtype=kv_dtype,
    max_num_batched_tokens=args.context_length,  # phải >= max_model_len
    ...
)
```

> **Lưu ý:** `max_num_batched_tokens` phải >= `max_model_len`, nếu không vLLM
> sẽ báo lỗi. Script đã được cập nhật để set `max_num_batched_tokens=args.context_length`.

---

## Cách 3: Hugging Face Transformers (nếu chạy trực tiếp không qua vLLM)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Arcee-VyLinh")
model = AutoModelForCausalLM.from_pretrained(
    "arcee-ai/Arcee-VyLinh",
    torch_dtype="bfloat16",
    device_map="auto",
)

# Khai báo max_new_tokens và truncation tường minh
inputs = tokenizer(
    long_text,
    return_tensors="pt",
    truncation=False,       # KHÔNG truncate — để model thấy đủ context
    max_length=None,        # Không giới hạn input length
).to("cuda")
```

---

## Bảng giới hạn context của các model trong benchmark

| Model | Context gốc | `num_ctx` khuyến nghị |
|-------|-------------|----------------------|
| `gemma4:e4b` | 128K | 32768 |
| `qwen3:8b` | 128K | 32768 |
| `llama3.2:3b` | 128K | 32768 |
| `arcee-ai/Arcee-VyLinh` | 128K | 32768 (HF trực tiếp, không cần Modelfile) |
| `Qwen/Qwen2.5-7B-Instruct-1M` | 1M | 32768 (để benchmark 16K, set vừa đủ) |

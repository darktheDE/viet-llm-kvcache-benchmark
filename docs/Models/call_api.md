# Ollama API Reference — Gọi Model qua `127.0.0.1:11434`

> Tài liệu này tổng hợp từ [docs.ollama.com](https://docs.ollama.com) (đọc ngày 08/07/2026).
> Mục đích: Hướng dẫn cách gọi API Ollama để chạy benchmark LLM (TTFT, ITL, throughput, VRAM)
> trên project **viet-llm-kvcache-benchmark**.

---

## 1. Tổng quan

Sau khi cài đặt Ollama, API được phục vụ mặc định tại:

```
http://127.0.0.1:11434/api
```

> **Lưu ý:** API không cần xác thực khi chạy local. Không cần `Authorization` header.

---

## 2. Các Endpoint chính

| Endpoint | Method | Mô tả |
|---|---|---|
| `/api/generate` | POST | Generate text từ prompt (completion) |
| `/api/chat` | POST | Chat multi-turn với lịch sử hội thoại |
| `/api/embed` | POST | Tạo vector embeddings |
| `/api/tags` | GET | Liệt kê tất cả model đã pull về |
| `/api/ps` | GET | Liệt kê model đang chạy trong bộ nhớ |
| `/api/version` | GET | Lấy version Ollama đang chạy |
| `/v1/chat/completions` | POST | OpenAI-compatible endpoint |
| `/v1/models` | GET | Liệt kê model (OpenAI-compatible) |

---

## 3. `POST /api/generate` — Generate Response

Dùng để **benchmark completion** (đo TTFT, ITL, throughput).

### Request Body

| Field | Type | Bắt buộc | Mô tả |
|---|---|---|---|
| `model` | string | ✅ | Tên model (ví dụ: `qwen3:8b`) |
| `prompt` | string | | Prompt đầu vào |
| `stream` | boolean | | `true` (mặc định) = streaming; `false` = trả về 1 lần |
| `system` | string | | System prompt |
| `options` | object | | Tuỳ chọn generation (xem bên dưới) |
| `keep_alive` | string/number | | Thời gian giữ model trong RAM (ví dụ: `"5m"`, `0`) |
| `think` | bool/string | | Bật thinking mode (`true`, `"high"`, `"medium"`, `"low"`, `"max"`) |
| `raw` | boolean | | `true` = trả về raw output, không áp dụng template |
| `format` | string/object | | Structured output: `"json"` hoặc JSON Schema object |
| `images` | array | | Danh sách ảnh base64 (cho vision models) |
| `logprobs` | boolean | | Trả về log-probabilities của output tokens |
| `top_logprobs` | integer | | Số tokens top-N cần trả khi logprobs bật |

### `options` — ModelOptions

| Field | Type | Mô tả |
|---|---|---|
| `temperature` | float | Độ ngẫu nhiên (0.0–2.0); thấp = deterministic |
| `top_k` | integer | Giới hạn top-K tokens khi sampling |
| `top_p` | float | Ngưỡng nucleus sampling |
| `min_p` | float | Ngưỡng xác suất tối thiểu |
| `seed` | integer | Seed cố định → reproducible output |
| `num_ctx` | integer | Context window size (số tokens) |
| `num_predict` | integer | Số token tối đa sinh ra |
| `stop` | string/array | Chuỗi dừng generation |

### Response Body (non-streaming)

| Field | Type | Mô tả |
|---|---|---|
| `model` | string | Tên model |
| `created_at` | string | Timestamp ISO 8601 |
| `response` | string | Text trả về |
| `done` | boolean | `true` khi hoàn tất |
| `done_reason` | string | Lý do kết thúc (ví dụ: `"stop"`) |
| `total_duration` | integer | Tổng thời gian (nanoseconds) |
| `load_duration` | integer | Thời gian load model (ns) |
| `prompt_eval_count` | integer | Số tokens đầu vào (prompt) |
| `prompt_eval_duration` | integer | Thời gian xử lý prompt (ns) — **dùng tính TTFT** |
| `eval_count` | integer | Số tokens đầu ra — **dùng tính throughput** |
| `eval_duration` | integer | Thời gian sinh token (ns) — **dùng tính ITL** |

### Ví dụ cURL

```bash
# Non-streaming (benchmark đo tổng thời gian)
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "qwen3:8b-fp16",
  "prompt": "Giai thich KV Cache compression la gi?",
  "stream": false,
  "options": {
    "num_ctx": 8192,
    "num_predict": 128,
    "temperature": 0.0,
    "seed": 42
  }
}'
```

```bash
# Streaming (đo TTFT = thời gian nhận được chunk đầu tiên)
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "qwen3:8b-fp16",
  "prompt": "Tom tat doan van sau: ...",
  "stream": true,
  "options": {
    "num_ctx": 16384,
    "num_predict": 256
  }
}'
```

```bash
# Load model vào VRAM (không cần prompt)
curl http://127.0.0.1:11434/api/generate -d '{"model": "llama3.1:8b-instruct-fp16"}'

# Unload model khỏi VRAM ngay lập tức
curl http://127.0.0.1:11434/api/generate -d '{"model": "llama3.1:8b-instruct-fp16", "keep_alive": 0}'
```

### Ví dụ Python cho Benchmark

```python
import requests
import time


BASE_URL = "http://127.0.0.1:11434"


def call_generate(model: str, prompt: str, context_length: int, max_new_tokens: int) -> dict:
    """
    Goi /api/generate de do TTFT, ITL, throughput.

    Args:
        model: Ten Ollama model (vi du: "qwen3:8b").
        prompt: Prompt dau vao.
        context_length: Context window size (num_ctx).
        max_new_tokens: So tokens toi da duoc sinh ra.

    Returns:
        dict chua metrics: ttft_ms, itl_ms, throughput_tps,
        total_duration_ms, wall_clock_ms, prompt_tokens, output_tokens.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": context_length,
            "num_predict": max_new_tokens,
            "temperature": 0.0,
            "seed": 42,
        }
    }

    start = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=600)
    wall_clock_ms = (time.perf_counter() - start) * 1000

    resp.raise_for_status()
    data = resp.json()

    # Ollama tra ve timing theo nanoseconds -> chuyen sang ms
    prompt_eval_duration_ms = data.get("prompt_eval_duration", 0) / 1_000_000
    eval_duration_ms        = data.get("eval_duration", 0) / 1_000_000
    total_duration_ms       = data.get("total_duration", 0) / 1_000_000

    output_tokens = data.get("eval_count", 0)
    prompt_tokens = data.get("prompt_eval_count", 0)

    # TTFT proxy tu prefill duration (chinh xac hon khi dung streaming)
    ttft_ms = prompt_eval_duration_ms

    # ITL (Inter-Token Latency) trung binh
    itl_ms = (eval_duration_ms / output_tokens) if output_tokens > 0 else 0.0

    # Throughput (tokens/s) trong qua trinh decode
    throughput_tps = (output_tokens / eval_duration_ms * 1000) if eval_duration_ms > 0 else 0.0

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "itl_ms": round(itl_ms, 4),
        "throughput_tps": round(throughput_tps, 2),
        "total_duration_ms": round(total_duration_ms, 2),
        "wall_clock_ms": round(wall_clock_ms, 2),
        "response_text": data.get("response", ""),
    }
```

---

## 4. `POST /api/chat` — Chat Completion

Dùng cho multi-turn conversation và đo benchmark với system prompt.

### Request Body

| Field | Type | Bắt buộc | Mô tả |
|---|---|---|---|
| `model` | string | ✅ | Tên model |
| `messages` | array | ✅ | Lịch sử hội thoại |
| `stream` | boolean | | Streaming (default: `true`) |
| `options` | object | | Tương tự `/api/generate` |
| `format` | string/object | | `"json"` hoặc JSON Schema |
| `think` | bool/string | | Thinking mode |
| `tools` | array | | Function/tool calling |
| `keep_alive` | string/number | | Thời gian giữ model |
| `logprobs` | boolean | | Log-probabilities |
| `top_logprobs` | integer | | Số top tokens khi logprobs bật |

### Schema `messages`

```json
[
  {
    "role": "system",
    "content": "Ban la mot tro ly AI tieng Viet."
  },
  {
    "role": "user",
    "content": "Giai thich KV Cache la gi?"
  }
]
```

Các role hợp lệ: `system`, `user`, `assistant`, `tool`.

### Ví dụ cURL

```bash
# Chat non-streaming
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [
    {"role": "system", "content": "Tra loi bang tieng Viet."},
    {"role": "user", "content": "KV Cache compression co y nghia gi trong LLM inference?"}
  ],
  "stream": false,
  "options": {
    "num_ctx": 8192,
    "num_predict": 256,
    "temperature": 0.0
  }
}'
```

### Ví dụ Python

```python
import requests


def call_chat(
    model: str,
    messages: list,
    context_length: int = 8192,
    max_new_tokens: int = 256
) -> dict:
    """
    Goi /api/chat voi danh sach messages.

    Args:
        model: Ten Ollama model.
        messages: List[dict] voi keys 'role' va 'content'.
        context_length: Context window size.
        max_new_tokens: So token toi da sinh ra.

    Returns:
        dict chua response text va metrics.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": context_length,
            "num_predict": max_new_tokens,
            "temperature": 0.0,
            "seed": 42,
        }
    }

    resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    output_tokens      = data.get("eval_count", 0)
    eval_duration_ms   = data.get("eval_duration", 0) / 1_000_000
    prompt_eval_dur_ms = data.get("prompt_eval_duration", 0) / 1_000_000

    return {
        "model": data.get("model"),
        "content": data["message"]["content"],
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": output_tokens,
        "ttft_ms": round(prompt_eval_dur_ms, 2),
        "itl_ms": round((eval_duration_ms / output_tokens) if output_tokens > 0 else 0, 4),
        "throughput_tps": round(
            (output_tokens / eval_duration_ms * 1000) if eval_duration_ms > 0 else 0, 2
        ),
        "total_duration_ms": round(data.get("total_duration", 0) / 1_000_000, 2),
    }
```

---

## 5. `GET /api/tags` — Liệt kê Models

```bash
curl http://127.0.0.1:11434/api/tags
```

Trả về danh sách tất cả models đã pull về local, kèm `size`, `digest`, `modified_at`.

---

## 6. `GET /api/ps` — Models đang Running

```bash
curl http://127.0.0.1:11434/api/ps
```

Trả về model nào đang được load trong VRAM, thời gian expire (`expires_at`), và `size_vram`.

---

## 7. `POST /api/embed` — Generate Embeddings

```bash
# Embed một câu
curl http://127.0.0.1:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": "Day la cau tieng Viet can embed."
}'

# Embed nhiều câu cùng lúc
curl http://127.0.0.1:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": ["Cau mot", "Cau hai"]
}'
```

Response trả về mảng `embeddings` (float arrays) và timing metrics.

---

## 8. OpenAI-Compatible API

Ollama hỗ trợ OpenAI-compatible endpoint tại `/v1/`. Dùng khi muốn tích hợp với tooling
sẵn có (LangChain, OpenAI SDK, v.v.).

### Base URL

```
http://127.0.0.1:11434/v1/
```

### `POST /v1/chat/completions`

```bash
curl -X POST http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:8b",
    "messages": [{"role": "user", "content": "Xin chao!"}],
    "stream": false,
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

### Dùng OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1/",
    api_key="ollama",  # bat buoc nhung bi ignore
)

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[
        {"role": "system", "content": "Tra loi bang tieng Viet."},
        {"role": "user", "content": "Giai thich KV Cache compression la gi?"}
    ],
    stream=False,
    temperature=0.0,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

### Các Endpoint OpenAI-Compatible được hỗ trợ

| Endpoint | Mô tả |
|---|---|
| `POST /v1/chat/completions` | Chat completions đầy đủ (streaming, tools, vision) |
| `POST /v1/completions` | Text completion (prompt → completion) |
| `GET /v1/models` | Liệt kê models |
| `GET /v1/models/{model}` | Lấy chi tiết một model |
| `POST /v1/embeddings` | Vector embeddings |
| `POST /v1/responses` | OpenAI Responses API (từ Ollama v0.13.3) |

---

## 9. Đo Metrics Benchmark từ API Response

Tất cả endpoints trả về các trường timing (**đơn vị: nanoseconds**):

| Trường | Ý nghĩa | Dùng để đo |
|---|---|---|
| `total_duration` | Tổng thời gian từ request đến response | Wall-clock time |
| `load_duration` | Thời gian load model vào VRAM | Cold start overhead |
| `prompt_eval_count` | Số input tokens | Context length thực tế |
| `prompt_eval_duration` | Thời gian encode prompt | Prefill latency ≈ TTFT |
| `eval_count` | Số output tokens | Throughput baseline |
| `eval_duration` | Thời gian decode tokens | ITL = eval_duration / eval_count |

### Công thức tính metrics

```python
NS_TO_MS = 1_000_000  # nanoseconds -> milliseconds

prompt_eval_duration_ms = data["prompt_eval_duration"] / NS_TO_MS
eval_duration_ms        = data["eval_duration"] / NS_TO_MS
total_duration_ms       = data["total_duration"] / NS_TO_MS

output_tokens = data["eval_count"]
prompt_tokens = data["prompt_eval_count"]

# Time To First Token (TTFT) — proxy tu prefill duration
ttft_ms = prompt_eval_duration_ms

# Inter-Token Latency (ITL) trung binh (ms/token)
itl_ms = eval_duration_ms / output_tokens if output_tokens > 0 else 0.0

# Decoding Throughput (tokens/s)
throughput_tps = output_tokens / (eval_duration_ms / 1000) if eval_duration_ms > 0 else 0.0
```

> **Lưu ý:** Với streaming mode, metrics chỉ có ở chunk cuối cùng (khi `done == true`).

---

## 10. Ví dụ Response đầy đủ (Non-Streaming)

```json
{
  "model": "qwen3:8b",
  "created_at": "2025-10-17T23:14:07.414671Z",
  "response": "KV Cache la co che luu tru...",
  "done": true,
  "done_reason": "stop",
  "total_duration": 174560334,
  "load_duration": 101397084,
  "prompt_eval_count": 32,
  "prompt_eval_duration": 13074791,
  "eval_count": 128,
  "eval_duration": 52479709
}
```

---

## 11. Quản lý Model trong VRAM

### Load model vào VRAM trước khi benchmark

```python
import requests


def preload_model(model: str, keep_alive: str = "10m") -> None:
    """Load model vao VRAM, giu trong keep_alive."""
    requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "keep_alive": keep_alive},
        timeout=120,
    )


def unload_model(model: str) -> None:
    """Giai phong VRAM bang cach unload model ngay lap tuc."""
    requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "keep_alive": 0},
        timeout=30,
    )
```

---

## 12. Mapping Model Name cho Project

| Model | Ollama Tag | Backend | Precision | HuggingFace Repo |
|---|---|---|---|---|
| Qwen3 8B | `qwen3:8b-fp16` | Ollama | FP16 | Qwen/Qwen3-8B |
| Llama 3.1 8B Instruct | `llama3.1:8b-instruct-fp16` | Ollama | FP16 | meta-llama/Llama-3.1-8B-Instruct |
| Mistral 7B Instruct v0.3 | `mistral:7b-instruct-v0.3-fp16` | Ollama | FP16 | mistralai/Mistral-7B-Instruct-v0.3 |
| Qwen2.5 7B Instruct | `qwen2.5:7b-instruct-fp16` | Ollama | FP16 | Qwen/Qwen2.5-7B-Instruct |

> **Lưu ý:** Tất cả 4 model trong benchmark chính thức đều chạy qua **Ollama** (không cần vLLM). Gemma 4 không còn nằm trong scope benchmark hiện tại.
> Để benchmark với vLLM, thay `BASE_URL` thành `http://127.0.0.1:8000` và dùng endpoint `/v1/chat/completions`.

---

## 13. Error Handling

```python
import requests


def safe_call_generate(model: str, prompt: str, **kwargs) -> dict | None:
    """
    Goi /api/generate voi xu ly loi day du.

    Returns:
        dict chua metrics hoac None neu loi.
    """
    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, **kwargs},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.ConnectionError:
        print("[ERROR] Ollama khong chay. Khoi dong bang: ollama serve")
        return None

    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout khi goi model {model}")
        return None

    except requests.exceptions.HTTPError as e:
        data = e.response.json() if e.response else {}
        code = e.response.status_code if e.response else "?"
        print(f"[ERROR] HTTP {code}: {data.get('error', str(e))}")
        return None
```

### Các lỗi thường gặp

| HTTP Status | Nguyên nhân | Giải pháp |
|---|---|---|
| `404` | Model chưa được pull | Chạy `ollama pull <model>` |
| `500` | Lỗi server / VRAM OOM | Kiểm tra VRAM; giảm `num_ctx` |
| `Connection refused` | Ollama chưa khởi động | Chạy `ollama serve` |

---

## 14. Kiểm tra Ollama đang chạy

```bash
# Ping server (trả về "Ollama is running")
curl http://127.0.0.1:11434/

# Xem version
curl http://127.0.0.1:11434/api/version
```

```python
import requests


def check_ollama_running(base_url: str = "http://127.0.0.1:11434") -> bool:
    """Kiem tra Ollama server co dang chay khong."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
```

---

## Tham khảo

- [Ollama API Introduction](https://docs.ollama.com/api/introduction)
- [Generate API](https://docs.ollama.com/api/generate)
- [Chat API](https://docs.ollama.com/api/chat)
- [Embeddings API](https://docs.ollama.com/api/embed)
- [Usage Metrics](https://docs.ollama.com/api/usage)
- [OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Streaming](https://docs.ollama.com/api/streaming)
- [OpenAPI Spec](https://docs.ollama.com/openapi.yaml)

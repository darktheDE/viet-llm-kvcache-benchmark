# Dataset Brief - Vietnamese LLM KV Cache Benchmark

Tai lieu nay mo ta bo du lieu test suite duoc tao tu dong de phuc vu danh gia (benchmark) TurboQuant va cac phuong phap nen KV Cache.

## 1. Nguon du lieu su dung
*   **VMLU SQuAD v1.0** (`vmlu_squad_v1`): Cung cap cac doan van canh tu nhien (contexts) bang tieng Viet chat luong cao lam tai lieu nen/tai lieu nhieu (distractor).
*   **VMLU MQA v1.5** (`vmlu_mqa_v1.5`): Cung cap cac cau hoi trac nghiem tieng Viet hoc thuat kem dap an chuan xac.
*   **VTSNLP/vietnamese_curated_dataset** (HF Hub): Cung cap tap du lieu tieng Viet thuc te (Wikipedia, bao chi, C4) duoc loc sach boi NeMo Curator.

## 2. Cac moc cau hinh (Buckets)
Bo du lieu gom 3 moc do dai ngu canh muc tieu:
*   **4,000 tokens**
*   **8,000 tokens**
*   **16,000 tokens**

Sau khi kiem dinh Unicode va loai bo 3 mau loi ky tu `�`, bo du lieu hien tai con **507 mau**.
Ngoai ra, nhom tao them file `datasets/test_set_smoke.json` gom **15 mau** de Team Tech chay thu nhanh pipeline truoc khi benchmark day du.

## 3. Huong dan su dung bo du lieu (Developer & Runner Guidelines)

Bo du lieu duoc dong goi duoi dang JSON: `datasets/test_set_small.json` va JSONL: `datasets/test_set_small.jsonl`.
Duoi day la huong dan su dung trong cac script Python benchmark cua Team Tech:

### A. Doc du lieu tu JSON
```python
import json

with open("datasets/test_set_small.json", "r", encoding="utf-8") as f:
    test_suite = json.load(f)

for sample in test_suite:
    prompt_type = sample["prompt_type"]          # 'qa' | 'retrieval' | 'general'
    target_length = sample["context_length_target"] # 4000 | 8000 | 16000
    prompt_text = sample["text"]                  # Text dau vao day du truyen cho vLLM
    expected_output = sample["expected_output"]  # Chieu khoa / Dap an de so khop
    
    # Truyen prompt_text cho vLLM de sinh tu
    # So khop dau ra sinh ra voi expected_output neu la qa/retrieval
```

### B. Cac luu y quan trong khi Benchmark
1.  **Do dac chat luong (Exact Match - EM):**
    *   Doi voi tac vu **`qa`**: expected_output la chu cai hoa duy nhat (vi du: `B`). Ban can cau hinh sampling params cua vLLM voi `max_tokens=2` de lay ra dap an nhanh chong va kiem tra Exact Match.
    *   Doi voi tac vu **`retrieval`**: expected_output la mot tu/so cu the (vi du: `Tom` hoac `992831`). Ban nen lay ra khoang `max_tokens=10` va kiem tra xem chuoi expected_output co nam trong ket qua sinh ra cua mo hinh hay khong (substring match).
2.  **Do dac Perplexity (PPL):**
    *   Uu tien dung cac mau **`general`** vi day la van ban tieng Viet ghep tu nhien dai (chu yeu tu VTSNLP), rat sach va phu hop de danh gia kha nang tiep thu ngon ngu cua mo hinh qua loss ma khong can sinh tu (offline perplexity evaluation).
3.  **Kiem soat VRAM va tranh OOM:**
    *   Khi load cac mau 16,000 tokens, can cau hinh vLLM bat `gpu_memory_utilization` cao (0.95 - 0.98) va dieu chinh `max_num_seqs=1` de ngan VRAM phinh to trong luc decode.

## 4. Cau truc File JSON
Moi mau trong file `test_set_small.json` co cau truc:
```json
{
  "prompt_type": "qa|retrieval|general",
  "context_length_target": 8000,
  "text": "...",
  "expected_output": "...",
  "actual_tokens": 7985
}
```

*   `text` la prompt dau vao day du truyen truc tiep cho inference engine.
*   `expected_output` la cau tra loi mong muon.
*   `actual_tokens` la so luong token thuc te duoc cat va dem bang tokenizer Qwen.

## 5. Trang thai kiem dinh du lieu

Bo du lieu da duoc kiem tra cu phap JSON bang lenh:

```powershell
python -m json.tool datasets/test_set_small.json > check.json
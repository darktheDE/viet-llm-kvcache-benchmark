# Dataset Brief: Vietnamese Long-Context Test Set Small

## 1. Muc dich

Bo du lieu nay la test-set nho cho benchmark mo hinh ngon ngu tieng Viet trong bai toan long-context. Muc tieu la tao cac mau van ban sach, co do dai gan 4k/8k/16k token de dung cho cac phep do chat luong, bo nho va hieu nang cua LLM.

## 2. Nguon du lieu

Pipeline hien tai tai du lieu tu Hugging Face:

* `5760/vmlu`
* `VTSNLP/vietnamese_curated_dataset`

Trong lan chay kiem thu ngay 2026-06-21, `5760/vmlu` tra ve 401 Unauthorized / khong truy cap duoc tren Hugging Face, nen pipeline da warning va tiep tuc voi `VTSNLP/vietnamese_curated_dataset`. Full output hien tai duoc tao tu 5,000 raw records cua `VTSNLP/vietnamese_curated_dataset`.

V-Bench chua duoc tich hop trong phien ban nay vi chua xac dinh duoc dataset ID Hugging Face/GitHub on dinh.

## 3. Quy trinh xu ly

Quy trinh tai lap gom cac buoc:

1. `scripts/download_datasets.py` tai raw records bang `datasets.load_dataset`, tu detect field text va ghi `data/raw/raw_records.jsonl`.
2. `scripts/clean_with_nemo.py --backend auto` chay voi backend `hybrid_nemo_python`. NeMo Curator duoc dung that su cho document representation/loading bang `DocumentBatch[pandas_dataframe]`, text modification bang `Modify[ProjectFtfyFixText]`, `Modify[UnicodeReformatter(normalization=NFC)]`, `Modify[NewlineNormalizer]` va `Modify[ProjectTextPostprocessor(normalization=NFC,control_char_removal,whitespace_normalization)]`. Built-in NeMo `DocumentFilter` xu ly word count, URL ratio, alpha-numeric ratio va whitespace ratio; cac rule rieng cua project cho min characters, replacement char, letter ratio, strange symbol ratio va Vietnamese signal duoc chuyen thanh custom NeMo-compatible `DocumentFilter`. Python chi con xu ly exact dedup va near dedup trong nhanh hybrid.
3. `scripts/build_long_context_testset.py` dung `transformers.AutoTokenizer` de ghep cac record thanh mau long-context va ghi `datasets/test_set_small.json`.
4. `scripts/validate_testset.py` validate schema, token count va thong ke theo nhom.

## 4. Bo loc chat luong

Cleaning pipeline hien tai gom:

* NeMo Curator `DocumentBatch` voi pandas DataFrame.
* NeMo Curator `Modify[ProjectFtfyFixText]`.
* NeMo Curator `Modify[UnicodeReformatter(normalization=NFC)]`.
* NeMo Curator `Modify[NewlineNormalizer]`.
* NeMo Curator `Modify[ProjectTextPostprocessor(normalization=NFC,control_char_removal,whitespace_normalization)]`.
* Built-in NeMo heuristic `DocumentFilter` cho word count, URL ratio, alpha-numeric ratio va whitespace ratio.
* Custom NeMo-compatible `DocumentFilter[min_characters:MinCharacterCountFilter]` loai text duoi 200 ky tu.
* Custom NeMo-compatible `DocumentFilter[replacement_char_free:ReplacementCharacterFilter]` loai text co replacement character.
* Custom NeMo-compatible `DocumentFilter[letter_ratio:LetterRatioFilter]` loai text co ty le chu cai thap.
* Custom NeMo-compatible `DocumentFilter[strange_symbol_ratio:StrangeSymbolRatioFilter]` loai text co qua nhieu symbol la.
* Custom NeMo-compatible `DocumentFilter[vietnamese_signal:VietnameseSignalFilter]` uu tien text co dau hieu tieng Viet bang ky tu co dau hoac tu tieng Viet pho bien.
* Python exact dedup bang SHA-256 hash.
* Python near dedup bang SimHash ket hop character n-gram Jaccard.

NeMo Curator import duoc trong Docker image va duoc tich hop qua API thuc te cua `nemo-curator[text-cpu]==1.2.0`: `nemo_curator.tasks.document.DocumentBatch`, `nemo_curator.stages.text.modifiers.modifier.Modify`, `nemo_curator.stages.text.modifiers.unicode.unicode_reformatter.UnicodeReformatter`, `nemo_curator.stages.text.modifiers.string.newline_normalizer.NewlineNormalizer`, `nemo_curator.stages.text.filters.doc_filter.DocumentFilter`, va cac heuristic filters trong `nemo_curator.stages.text.filters.heuristic.string`. Cac filter tieng Viet/rule rieng cua project la custom `DocumentFilter` tuong thich NeMo, khong phai built-in filter cua NVIDIA. Python custom logic con lai trong nhanh hybrid la stateful dedup.

## 5. Cau truc JSON

Top-level schema:

```json
{
  "dataset_name": "vietnamese_long_context_test_set_small",
  "version": "0.1.0",
  "language": "vi",
  "created_by": "data_pipeline",
  "description": "Small cleaned Vietnamese long-context test set for LLM benchmark.",
  "mode": "full",
  "tokenizer": {
    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "token_count_method": "transformers AutoTokenizer encode length"
  },
  "samples": []
}
```

Moi sample co cac field chinh:

* `id`
* `source`
* `context_group`
* `target_tokens`
* `actual_tokens`
* `text`
* `metadata`

## 6. Thong ke test-set

Thong ke artifact `datasets/test_set_small.json` sau full/medium run:

* Mode: `full`.
* Tokenizer su dung: `Qwen/Qwen2.5-7B-Instruct`.
* Tong so mau: 12.
* Nhom 4k: 4 mau, 5,000-5,000 tokens, trung binh 5,000.0 tokens.
* Nhom 8k: 4 mau, 8,099-9,500 tokens, trung binh 8,731.5 tokens.
* Nhom 16k: 4 mau, 14,562-18,500 tokens, trung binh 17,255.0 tokens.
* Khoang token thuc te toan bo: 5,000-18,500 tokens.
* Raw records tai duoc: 5,000.
* Cleaned records: 4,998.
* Cleaning backend counts sau lan chay `--backend auto`: `hybrid_nemo_python=4,998`, `nemo_curator=0`, `python_fallback=0`.
* NeMo steps da ghi trong metadata moi record: `DocumentBatch[pandas_dataframe]`, `Modify[ProjectFtfyFixText]`, `Modify[UnicodeReformatter(normalization=NFC)]`, `Modify[NewlineNormalizer]`, `Modify[ProjectTextPostprocessor(normalization=NFC,control_char_removal,whitespace_normalization)]`, `DocumentFilter[word_count:WordCountFilter]`, `DocumentFilter[urls_ratio:UrlsFilter]`, `DocumentFilter[alpha_numeric:NonAlphaNumericFilter]`, `DocumentFilter[white_space:WhiteSpaceFilter]`, `DocumentFilter[min_characters:MinCharacterCountFilter]`, `DocumentFilter[replacement_char_free:ReplacementCharacterFilter]`, `DocumentFilter[letter_ratio:LetterRatioFilter]`, `DocumentFilter[strange_symbol_ratio:StrangeSymbolRatioFilter]`, `DocumentFilter[vietnamese_signal:VietnameseSignalFilter]`.

## 7. Cach chay tai lap bang Docker

```bash
docker compose build
docker compose run --rm data-pipeline bash
python scripts/download_datasets.py --max-records-per-source 5000
python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl --backend auto
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json
python scripts/validate_testset.py --input datasets/test_set_small.json
```

Smoke test:

```bash
python scripts/download_datasets.py --max-records-per-source 200
python scripts/clean_with_nemo.py --input data/raw/raw_records.jsonl --output data/processed/cleaned.jsonl --backend auto
python scripts/build_long_context_testset.py --input data/processed/cleaned.jsonl --output datasets/test_set_small.json --allow-smoke-test
python scripts/validate_testset.py --input datasets/test_set_small.json --allow-smoke-test
```

## 8. Ghi chu tuong thich benchmark

Test-set nay duoc thiet ke cho benchmark long-context trong repo, khong phai benchmark chat/instruction hoan chinh. Truong `context_group` giup chia mau theo muc token 4k, 8k va 16k. Truong `actual_tokens` duoc tinh bang tokenizer trong top-level JSON.

## 9. Han che hien tai

* V-Bench chua duoc tich hop trong phien ban nay vi chua xac dinh duoc dataset ID Hugging Face/GitHub on dinh.
* `5760/vmlu` nam trong danh sach nguon nhung khong truy cap duoc trong lan chay kiem thu ngay 2026-06-21, nen artifact hien tai chua co mau tu nguon nay.
* Cleaning chay backend `hybrid_nemo_python`: NeMo Curator adapter xu ly document batch, ftfy/Unicode/newline/postprocess modifiers, built-in heuristic filters va custom NeMo-compatible filters cho cac rule tieng Viet; Python xu ly exact/near dedup.
* Full mode da dat schema va so mau yeu cau, nhung nguon du lieu hien tai chi den tu mot dataset do gioi han truy cap cua `5760/vmlu`.

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
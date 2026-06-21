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
2. `scripts/clean_with_nemo.py --backend auto` chay voi backend `hybrid_nemo_python`. NeMo Curator duoc dung that su cho document representation/loading bang `DocumentBatch[pandas_dataframe]`, text modification bang `Modify[UnicodeReformatter(normalization=NFC)]` va `Modify[NewlineNormalizer]`, cung generic quality filtering bang `DocumentFilter[word_count:WordCountFilter]`, `DocumentFilter[urls_ratio:UrlsFilter]`, `DocumentFilter[alpha_numeric:NonAlphaNumericFilter]`, `DocumentFilter[white_space:WhiteSpaceFilter]`. Python custom filters tiep tuc xu ly cac buoc tieng Viet chuyen biet va dedup.
3. `scripts/build_long_context_testset.py` dung `transformers.AutoTokenizer` de ghep cac record thanh mau long-context va ghi `datasets/test_set_small.json`.
4. `scripts/validate_testset.py` validate schema, token count va thong ke theo nhom.

## 4. Bo loc chat luong

Cleaning pipeline hien tai gom:

* NeMo Curator `DocumentBatch` voi pandas DataFrame.
* NeMo Curator `Modify[UnicodeReformatter(normalization=NFC)]`.
* NeMo Curator `Modify[NewlineNormalizer]`.
* NeMo Curator heuristic `DocumentFilter` cho word count, URL ratio, alpha-numeric ratio va whitespace ratio.
* Unicode normalization NFC.
* `ftfy.fix_text`.
* Xoa control characters.
* Chuan hoa whitespace.
* Loai text duoi 200 ky tu.
* Loai text co replacement character.
* Loai text co ty le chu cai thap hoac qua nhieu symbol la.
* Uu tien text co dau hieu tieng Viet bang ky tu co dau hoac tu tieng Viet pho bien.
* Exact dedup bang SHA-256 hash.
* Near dedup bang SimHash ket hop character n-gram Jaccard.

NeMo Curator import duoc trong Docker image va duoc tich hop qua API thuc te cua `nemo-curator[text-cpu]==1.2.0`: `nemo_curator.tasks.document.DocumentBatch`, `nemo_curator.stages.text.modifiers.modifier.Modify`, `nemo_curator.stages.text.modifiers.unicode.unicode_reformatter.UnicodeReformatter`, `nemo_curator.stages.text.modifiers.string.newline_normalizer.NewlineNormalizer`, va cac heuristic filters trong `nemo_curator.stages.text.filters.heuristic.string`. Python custom filters duoc dung cho cac buoc tieng Viet chuyen biet: dau hieu tieng Viet, replacement char, length threshold, exact dedup va near dedup.

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
* NeMo steps da ghi trong metadata moi record: `DocumentBatch[pandas_dataframe]`, `Modify[UnicodeReformatter(normalization=NFC)]`, `Modify[NewlineNormalizer]`, `DocumentFilter[word_count:WordCountFilter]`, `DocumentFilter[urls_ratio:UrlsFilter]`, `DocumentFilter[alpha_numeric:NonAlphaNumericFilter]`, `DocumentFilter[white_space:WhiteSpaceFilter]`.

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
* Cleaning chay backend `hybrid_nemo_python`: NeMo Curator xu ly document batch, Unicode/newline modification va generic heuristic filters; Python xu ly cac filter tieng Viet va dedup.
* Full mode da dat schema va so mau yeu cau, nhung nguon du lieu hien tai chi den tu mot dataset do gioi han truy cap cua `5760/vmlu`.

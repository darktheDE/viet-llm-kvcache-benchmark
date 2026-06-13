import os
import json
import random
import re
import unicodedata
from tqdm import tqdm

# Cau hinh duong dan
DATASETS_DIR = "datasets"
SQUAD_FILE = os.path.join(DATASETS_DIR, "vmlu_squad_v1", "vi_squad_benchmark_question_only.json")
DIALOGUE_FILE = os.path.join(DATASETS_DIR, "vmlu_dialog_v1", "vi_dialogue_question_only.json")
MQA_DIR = os.path.join(DATASETS_DIR, "vmlu_mqa_v1.5")
OUTPUT_FILE = os.path.join(DATASETS_DIR, "test_set_small.json")
OUTPUT_FILE_JSONL = os.path.join(DATASETS_DIR, "test_set_small.jsonl")
BRIEF_FILE = os.path.join(DATASETS_DIR, "dataset_brief.md")

# Fallback Tokenizer setup
try:
    from transformers import AutoTokenizer
    print("Loading tokenizer Qwen/Qwen2.5-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    def count_tokens(text):
        return len(tokenizer.encode(text))
except Exception as e:
    print(f"Khong the load tokenizer Qwen: {e}. Su dung ham dem token gan dung (word-based fallback).")
    tokenizer = None
    def count_tokens(text):
        words = text.split()
        return int(len(words) * 1.3) + 1

# Cac moc context length muc tieu
CONTEXT_BUCKETS = [4000, 8000, 16000]

# Quy mo mau thu nghiem
QA_SAMPLES_PER_BUCKET = 70
RETRIEVAL_SAMPLES_PER_BUCKET = 70
GENERAL_SAMPLES_PER_BUCKET = 30

def normalize_vietnamese(text):
    """Chuan hoa bang ma Unicode NFC cho tieng Viet va loai bo khoang trang thua"""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clip_text_to_tokens(text, max_tokens):
    """Cat van ban dat dung so luong token muc tieu ma khong vuot qua"""
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text)
            if len(ids) <= max_tokens:
                return text, len(ids)
            clipped_text = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
            actual_clipped_len = len(tokenizer.encode(clipped_text))
            return clipped_text, actual_clipped_len
        except Exception:
            pass
            
    # Fallback word-based
    words = text.split()
    target_words_len = int(max_tokens / 1.8)  # Dung ty le 1.8 an toan hon doi voi Qwen tieng Viet
    clipped_text = " ".join(words[:target_words_len])
    return clipped_text, int(len(clipped_text.split()) * 1.8)

def clean_and_load_squad():
    print(f"Dang doc du lieu SQuAD tu {SQUAD_FILE}...")
    if not os.path.exists(SQUAD_FILE):
        raise FileNotFoundError(f"Khong tim thay file {SQUAD_FILE}")
    
    with open(SQUAD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    contexts = []
    for item in data.get("data", []):
        ctx = normalize_vietnamese(item.get("context", ""))
        if ctx and len(ctx) > 50:
            contexts.append(ctx)
            
    contexts = list(set(contexts))
    print(f"Tim thay {len(contexts)} doan ngu canh SQuAD doc ban.")
    return contexts

def clean_and_load_mqa():
    print(f"Dang doc du lieu MQA tu {MQA_DIR}...")
    mqa_samples = []
    
    files_to_read = ["test.jsonl", "valid.jsonl", "dev.jsonl"]
    for file_name in files_to_read:
        file_path = os.path.join(MQA_DIR, file_name)
        if os.path.exists(file_path):
            print(f"Dang doc tu file {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            question = normalize_vietnamese(sample.get("question", ""))
                            choices = [normalize_vietnamese(c) for c in sample.get("choices", [])]
                            answer = sample.get("answer", "").strip()
                            if question and choices and answer:
                                mqa_samples.append({
                                    "question": question,
                                    "choices": choices,
                                    "answer": answer
                                })
                        except json.JSONDecodeError:
                            continue
            if len(mqa_samples) >= 300:
                break
                
    print(f"Tim thay {len(mqa_samples)} cau hoi trac nghiem MQA hop le.")
    return mqa_samples

def clean_and_load_vtsnlp():
    print("Dang ket noi va load du lieu VTSNLP/vietnamese_curated_dataset tu Hugging Face...")
    vtsnlp_contexts = []
    try:
        from datasets import load_dataset
        ds = load_dataset("VTSNLP/vietnamese_curated_dataset", split="train", streaming=True)
        
        count = 0
        max_samples = 600
        iterator = iter(ds)
        
        for _ in range(max_samples * 2):
            try:
                sample = next(iterator)
                text = sample.get("text", "")
                text = normalize_vietnamese(text)
                if text and len(text) > 100:
                    vtsnlp_contexts.append(text)
                    count += 1
                if count >= max_samples:
                    break
            except StopIteration:
                break
        print(f"Lay thanh cong {len(vtsnlp_contexts)} doan van ban tu VTSNLP.")
    except Exception as e:
        print(f"Loi khi tai VTSNLP tu HF Hub: {e}. Se su dung SQuAD lam fallback cho VTSNLP.")
    
    return vtsnlp_contexts

# Danh sach cac template kim (needles) va cau hoi tuong ung cho bai test Retrieval
NEEDLE_TEMPLATES = [
    {
        "needle": "Luu y quan trong: Ten chu meo cung cua CEO la {value}.",
        "question": "Ten chu meo cung cua CEO la gi?",
        "answer": "{value}"
    },
    {
        "needle": "Ma so dang nhap bi mat cua nhan vien van hanh he thong la {value}.",
        "question": "Ma so dang nhap bi mat cua nhan vien van hanh he thong la gi?",
        "answer": "{value}"
    },
    {
        "needle": "Dia chi email lien he khan cap cua phong bao mat la {value}.",
        "question": "Dia chi email lien he khan cap cua phong bao mat la gi?",
        "answer": "{value}"
    },
    {
        "needle": "Mat khau WiFi cua phong hop VIP tang 5 la {value}.",
        "question": "Mat khau WiFi cua phong hop VIP tang 5 la gi?",
        "answer": "{value}"
    },
    {
        "needle": "Ma OTP xac thuc giao dich tai chinh hien tai cua cong ty la {value}.",
        "question": "Ma OTP xac thuc giao dich tai chinh hien tai cua cong ty la gi?",
        "answer": "{value}"
    },
    {
        "needle": "So dien thoai duong day nong ho tro khan cap khach hang la {value}.",
        "question": "So dien thoai duong day nong ho tro khan cap khach hang la so nao?",
        "answer": "{value}"
    },
    {
        "needle": "Ten doi tac chien luoc ky ket hop dong thuong mai nam nay la {value}.",
        "question": "Ten doi tac chien luoc ky ket hop dong thuong mai nam nay la gi?",
        "answer": "{value}"
    },
    {
        "needle": "Ma du an nghien cuu bao mat cap cao duoc phe duyet la {value}.",
        "question": "Ma du an nghien cuu bao mat cap cao duoc phe duyet la gi?",
        "answer": "{value}"
    },
    {
        "needle": "Ten thanh pho dien ra hoi nghi khach hang thuong nien cua tap doan la {value}.",
        "question": "Ten thanh pho dien ra hoi nghi khach hang thuong nien cua tap doan la gi?",
        "answer": "{value}"
    },
    {
        "needle": "Ma so bao hiem xa hoi cua nhan su truong phong la {value}.",
        "question": "Ma so bao hiem xa hoi cua nhan su truong phong la gi?",
        "answer": "{value}"
    }
]

RANDOM_VALUES = [
    "Tom", "Luna", "Milo", "Kiki", "889912", "SECURE-99A", "admin_backup@security.vn",
    "vip_wifi_2026", "992831", "18001091", "Vingroup", "Viettel", "Project-TurboQuant",
    "Da Nang", "Nha Trang", "BHXH-9928172"
]

def generate_retrieval_sample(target_len, combined_contexts):
    """Tao mau Needle In A Haystack"""
    template = random.choice(NEEDLE_TEMPLATES)
    val = str(random.randint(100000, 999999)) if random.random() > 0.5 else random.choice(RANDOM_VALUES)
    
    needle_text = template["needle"].format(value=val)
    question = template["question"]
    expected_answer = template["answer"].format(value=val)
    
    prompt_header = "Duoi day la mot tai lieu dai chua nhieu doan thong tin khac nhau. Hay doc ky tai lieu nay va tra loi cau hoi o duoi cung mot cach ngan gon, chinh xac nhat dua tren noi dung tai lieu.\n\nTAI LIEU:\n"
    prompt_footer = f"\n\nDua vao tai lieu tren, hay tra loi cau hoi sau day mot cach ngan gon nhat (chi tra loi thong tin can tim, khong viet them tu khac):\nCâu hỏi: {question}\nTra loi:"
    
    # Tru bot 5 token du phong sai so ghep chuoi
    fixed_tokens = count_tokens(prompt_header) + count_tokens(prompt_footer) + count_tokens(needle_text) + 5
    remaining_tokens = target_len - fixed_tokens
    
    selected_contexts = []
    current_tokens = 0
    
    shuffled_contexts = list(combined_contexts)
    random.shuffle(shuffled_contexts)
    
    for ctx in shuffled_contexts:
        ctx_tokens = count_tokens(ctx)
        if current_tokens + ctx_tokens < remaining_tokens:
            selected_contexts.append(ctx)
            current_tokens += ctx_tokens
        else:
            needed = remaining_tokens - current_tokens
            if needed > 5:
                clipped_ctx, actual_clipped_len = clip_text_to_tokens(ctx, needed)
                selected_contexts.append(clipped_ctx)
                current_tokens += actual_clipped_len
            break
            
    if len(selected_contexts) > 3:
        insert_idx = random.randint(1, len(selected_contexts) - 2)
        selected_contexts.insert(insert_idx, needle_text)
    else:
        selected_contexts.append(needle_text)
        
    full_text = prompt_header + "\n\n".join(selected_contexts) + prompt_footer
    return full_text, expected_answer

def generate_qa_sample(target_len, combined_contexts, mqa_sample):
    """Tao mau QA trac nghiem trong ngu canh dai"""
    question = mqa_sample["question"]
    choices = mqa_sample["choices"]
    expected_answer = mqa_sample["answer"]
    
    choices_str = "\n".join([f"- {c}" for c in choices])
    
    prompt_header = "Duoi day la tai lieu tham khao duoc tong hop tu nhieu nguon khac nhau. Hay doc ky tai lieu nay va tra loi cau hoi trac nghiem o cuoi.\n\nTAI LIEU THAM KHAO:\n"
    prompt_footer = f"\n\nDua vao tai lieu tham khao tren va kien thuc cua ban, hay tra loi cau hoi trac nghiem sau day bang cach chon mot chu cai dap an dung duy nhat (A, B, C, D hoac E). Chi tra loi chu cai cua dap an (vi du: A).\n\nCâu hỏi: {question}\nCac lua chon:\n{choices_str}\nDap an la:"
    
    fixed_tokens = count_tokens(prompt_header) + count_tokens(prompt_footer) + 5
    remaining_tokens = target_len - fixed_tokens
    
    selected_contexts = []
    current_tokens = 0
    
    shuffled_contexts = list(combined_contexts)
    random.shuffle(shuffled_contexts)
    
    for ctx in shuffled_contexts:
        ctx_tokens = count_tokens(ctx)
        if current_tokens + ctx_tokens < remaining_tokens:
            selected_contexts.append(ctx)
            current_tokens += ctx_tokens
        else:
            needed = remaining_tokens - current_tokens
            if needed > 5:
                clipped_ctx, actual_clipped_len = clip_text_to_tokens(ctx, needed)
                selected_contexts.append(clipped_ctx)
                current_tokens += actual_clipped_len
            break
            
    full_text = prompt_header + "\n\n".join(selected_contexts) + prompt_footer
    return full_text, expected_answer

def generate_general_sample(target_len, combined_contexts):
    """Tao mau van ban tu nhien dai de do Perplexity (PPL)"""
    prompt_header = "Duoi day la mot bai doc tong hop tieng Viet dai ngu canh:\n\n"
    prompt_footer = "\n\n(Het bai doc)"
    
    fixed_tokens = count_tokens(prompt_header) + count_tokens(prompt_footer) + 5
    remaining_tokens = target_len - fixed_tokens
    
    selected_contexts = []
    current_tokens = 0
    
    shuffled_contexts = list(combined_contexts)
    random.shuffle(shuffled_contexts)
    
    for ctx in shuffled_contexts:
        ctx_tokens = count_tokens(ctx)
        if current_tokens + ctx_tokens < remaining_tokens:
            selected_contexts.append(ctx)
            current_tokens += ctx_tokens
        else:
            needed = remaining_tokens - current_tokens
            if needed > 5:
                clipped_ctx, actual_clipped_len = clip_text_to_tokens(ctx, needed)
                selected_contexts.append(clipped_ctx)
                current_tokens += actual_clipped_len
            break
            
    full_text = prompt_header + "\n\n".join(selected_contexts) + prompt_footer
    return full_text, ""

def main():
    print("=== PIPELINE XU LY & CHUAN HOA DU LIEU BENCHMARK ===")
    
    try:
        squad_contexts = clean_and_load_squad()
        mqa_samples = clean_and_load_mqa()
        vtsnlp_contexts = clean_and_load_vtsnlp()
    except Exception as e:
        print(f"Loi khi load du lieu: {e}")
        return
        
    # Neu khong co VTSNLP, dung SQuAD lam fallback
    if not vtsnlp_contexts:
        vtsnlp_contexts = squad_contexts
        print("Su dung SQuAD lam fallback cho VTSNLP.")
        
    combined_contexts = squad_contexts + vtsnlp_contexts
    
    test_set_small = []
    
    for bucket in CONTEXT_BUCKETS:
        print(f"\n--- Dang tao du lieu cho moc {bucket} tokens ---")
        
        # 1. Tao mau QA trac nghiem
        print(f"Tao mau QA ({QA_SAMPLES_PER_BUCKET} mau)...")
        qa_count = 0
        while qa_count < QA_SAMPLES_PER_BUCKET:
            random.shuffle(mqa_samples)
            for mqa_sample in mqa_samples:
                if qa_count >= QA_SAMPLES_PER_BUCKET:
                    break
                text, answer = generate_qa_sample(bucket, combined_contexts, mqa_sample)
                tokens = count_tokens(text)
                test_set_small.append({
                    "prompt_type": "qa",
                    "context_length_target": bucket,
                    "text": text,
                    "expected_output": answer,
                    "actual_tokens": tokens
                })
                qa_count += 1
            
        # 2. Tao mau Retrieval
        print(f"Tao mau Retrieval ({RETRIEVAL_SAMPLES_PER_BUCKET} mau)...")
        ret_count = 0
        for _ in range(RETRIEVAL_SAMPLES_PER_BUCKET):
            text, answer = generate_retrieval_sample(bucket, combined_contexts)
            tokens = count_tokens(text)
            test_set_small.append({
                "prompt_type": "retrieval",
                "context_length_target": bucket,
                "text": text,
                "expected_output": answer,
                "actual_tokens": tokens
            })
            ret_count += 1
            
        # 3. Tao mau General
        print(f"Tao mau General ({GENERAL_SAMPLES_PER_BUCKET} mau)...")
        gen_count = 0
        for _ in range(GENERAL_SAMPLES_PER_BUCKET):
            text, answer = generate_general_sample(bucket, vtsnlp_contexts)
            tokens = count_tokens(text)
            test_set_small.append({
                "prompt_type": "general",
                "context_length_target": bucket,
                "text": text,
                "expected_output": answer,
                "actual_tokens": tokens
            })
            gen_count += 1
            
        print(f"Hoan thanh moc {bucket} tokens: {qa_count} QA, {ret_count} Retrieval, {gen_count} General.")

    # Ghi file JSON
    print(f"\nDang ghi ket qua ra file {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set_small, f, ensure_ascii=False, indent=2)
        
    # Ghi file JSONL
    print(f"Dang ghi ket qua ra file {OUTPUT_FILE_JSONL}...")
    with open(OUTPUT_FILE_JSONL, "w", encoding="utf-8") as f:
        for item in test_set_small:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Tao dataset_brief.md
    print(f"Dang ghi tai lieu huong dan ra file {BRIEF_FILE}...")
    brief_content = f"""# Dataset Brief - Vietnamese LLM KV Cache Benchmark

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

Moi moc do dai chua **{QA_SAMPLES_PER_BUCKET + RETRIEVAL_SAMPLES_PER_BUCKET + GENERAL_SAMPLES_PER_BUCKET} mau**, tong cong toan bo test suite la **{3 * (QA_SAMPLES_PER_BUCKET + RETRIEVAL_SAMPLES_PER_BUCKET + GENERAL_SAMPLES_PER_BUCKET)} mau**.

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
{{
  "prompt_type": "qa|retrieval|general",
  "context_length_target": 8000,
  "text": "...",
  "expected_output": "...",
  "actual_tokens": 7985
}}
```

*   `text` la prompt dau vao day du truyen truc tiep cho inference engine.
*   `expected_output` la cau tra loi mong muon.
*   `actual_tokens` la so luong token thuc te duoc cat va dem bang tokenizer Qwen.
"""
    with open(BRIEF_FILE, "w", encoding="utf-8") as f:
        f.write(brief_content)

    print("\n=== PIPELINE HOAN THANH THANH CONG ===")
    print(f"Tong so mau duoc tao: {len(test_set_small)}")
    print(f"File JSON: {OUTPUT_FILE}")
    print(f"File JSONL: {OUTPUT_FILE_JSONL}")
    print(f"File Brief: {BRIEF_FILE}")

if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


WHITESPACE_RE = re.compile(r"\s+")
VIETNAMESE_ACCENT_RE = re.compile(
    r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
VIETNAMESE_COMMON_WORD_RE = re.compile(
    r"\b(và|của|là|trong|một|các|được|cho|không)\b",
    re.IGNORECASE,
)
ALLOWED_SYMBOLS = set(".,;:!?-–—_()[]{}\"'“”‘’/\\%&+*=<>@#|~`^$€£¥…")


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL line") from exc
            if isinstance(value, dict):
                yield value


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    output_path = ensure_parent(path)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def remove_control_chars(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cf"} or ch in "\n\t "
    )


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def has_replacement_char(text: str) -> bool:
    return "\ufffd" in text or "ï¿½" in text


def has_vietnamese_signal(text: str) -> bool:
    return bool(VIETNAMESE_ACCENT_RE.search(text) or VIETNAMESE_COMMON_WORD_RE.search(text))


def text_quality_flags(text: str) -> dict[str, bool | float]:
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return {
            "unicode_valid": False,
            "vietnamese_ratio_ok": False,
            "replacement_char_free": False,
            "letter_ratio": 0.0,
            "symbol_ratio": 1.0,
            "has_vietnamese_signal": False,
        }

    letters = sum(1 for ch in non_space if ch.isalpha())
    strange_symbols = sum(
        1
        for ch in non_space
        if not ch.isalnum() and not ch.isspace() and ch not in ALLOWED_SYMBOLS
    )
    letter_ratio = letters / len(non_space)
    symbol_ratio = strange_symbols / len(non_space)
    replacement_char_free = not has_replacement_char(text)
    vietnamese_signal = has_vietnamese_signal(text)

    return {
        "unicode_valid": replacement_char_free,
        "vietnamese_ratio_ok": letter_ratio >= 0.45 and symbol_ratio <= 0.20 and vietnamese_signal,
        "replacement_char_free": replacement_char_free,
        "letter_ratio": round(letter_ratio, 4),
        "symbol_ratio": round(symbol_ratio, 4),
        "has_vietnamese_signal": vietnamese_signal,
    }


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def char_ngrams(text: str, n: int = 5) -> set[str]:
    compact = normalize_whitespace(text.lower())
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(0, len(compact) - n + 1)}


def simhash_from_features(features: Iterable[str], bits: int = 64) -> int:
    vector = [0] * bits
    for feature in features:
        digest = int(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(bits):
            vector[bit] += 1 if digest & (1 << bit) else -1
    fingerprint = 0
    for bit, value in enumerate(vector):
        if value >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

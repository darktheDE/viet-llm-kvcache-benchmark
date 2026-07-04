from __future__ import annotations

import re
from collections import Counter

from scripts.utils_text import text_quality_flags


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _longest_run(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = 1
    current = 1
    for prev, cur in zip(tokens, tokens[1:]):
        if cur == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _repeated_ngram_ratio(tokens: list[str], n: int = 4) -> float:
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum((count - 1) for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def analyze_generated_text(text: str) -> dict[str, object]:
    """Inspect generated text for repetition and gibberish signals."""

    normalized_text = text or ""
    stripped = normalized_text.strip()
    tokens = _tokenize(stripped)
    quality_flags = text_quality_flags(normalized_text)

    output_length = len(stripped)
    special_chars = sum(
        1
        for ch in stripped
        if not ch.isalnum() and not ch.isspace()
    )
    special_char_ratio = special_chars / max(len(stripped), 1)
    repeated_ngram_ratio = _repeated_ngram_ratio(tokens)
    longest_run = _longest_run(tokens)

    repetition_flag = repeated_ngram_ratio >= 0.3 or longest_run >= 5
    gibberish_flag = (
        special_char_ratio >= 0.25
        or not stripped
        or "�" in stripped
        or bool(quality_flags.get("replacement_char_free") is False)
    )

    warnings: list[str] = []
    if not stripped:
        warnings.append("empty_output")
    if output_length < 5 and stripped:
        warnings.append("too_short")
    if repetition_flag:
        warnings.append("repetition")
    if gibberish_flag:
        warnings.append("gibberish")

    return {
        "repetition_flag": repetition_flag,
        "gibberish_flag": gibberish_flag,
        "repeated_ngram_ratio": round(repeated_ngram_ratio, 4),
        "special_char_ratio": round(special_char_ratio, 4),
        "output_length": output_length,
        "quality_warning": ";".join(dict.fromkeys(warnings)),
    }

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from scripts.utils_text import has_replacement_char, text_quality_flags


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
QUALITY_FIELDS = [
    "repetition_flag",
    "gibberish_flag",
    "repeated_ngram_ratio",
    "special_char_ratio",
    "output_length",
    "quality_warning",
]


def _tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _repeated_ngram_ratio(tokens: list[str], ngram_size: int = 4) -> float:
    if len(tokens) < ngram_size:
        return 0.0

    ngrams = [
        tuple(tokens[index : index + ngram_size])
        for index in range(0, len(tokens) - ngram_size + 1)
    ]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams) if ngrams else 0.0


def _longest_repeated_token_run(tokens: list[str]) -> int:
    if not tokens:
        return 0

    longest = 1
    current = 1
    previous = tokens[0]
    for token in tokens[1:]:
        if token == previous:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
            previous = token
    return max(longest, current)


def _special_char_ratio(text: str) -> float:
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 0.0

    special_chars = sum(1 for char in non_space if not char.isalnum())
    return special_chars / len(non_space)


def analyze_generated_text(text: str) -> dict[str, Any]:
    """Return lightweight quality diagnostics for generated text."""

    normalized = text or ""
    stripped = normalized.strip()
    output_length = len(stripped)
    tokens = _tokenize_words(stripped)

    repeated_ngram_ratio = _repeated_ngram_ratio(tokens)
    repeated_run = _longest_repeated_token_run(tokens)
    special_char_ratio = _special_char_ratio(stripped)
    base_flags = text_quality_flags(stripped)

    warnings: list[str] = []
    repetition_flag = repeated_ngram_ratio >= 0.30 or repeated_run >= 6
    gibberish_flag = (
        output_length == 0
        or has_replacement_char(stripped)
        or special_char_ratio >= 0.30
        or (
            output_length >= 20
            and not bool(base_flags.get("unicode_valid", True))
        )
        or (
            output_length >= 20
            and float(base_flags.get("letter_ratio", 1.0)) < 0.30
        )
    )

    if output_length == 0:
        warnings.append("empty_output")
    elif output_length < 5:
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


def aggregate_quality_metrics(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample quality diagnostics for one CSV summary row."""

    values = list(metrics)
    if not values:
        return {
            "repetition_flag": False,
            "gibberish_flag": False,
            "repeated_ngram_ratio": "",
            "special_char_ratio": "",
            "output_length": "",
            "quality_warning": "",
        }

    warnings: list[str] = []
    for item in values:
        warning = str(item.get("quality_warning", "") or "")
        warnings.extend(part for part in warning.split(";") if part)

    return {
        "repetition_flag": any(bool(item.get("repetition_flag")) for item in values),
        "gibberish_flag": any(bool(item.get("gibberish_flag")) for item in values),
        "repeated_ngram_ratio": round(
            sum(float(item.get("repeated_ngram_ratio") or 0.0) for item in values)
            / len(values),
            4,
        ),
        "special_char_ratio": round(
            sum(float(item.get("special_char_ratio") or 0.0) for item in values)
            / len(values),
            4,
        ),
        "output_length": round(
            sum(int(item.get("output_length") or 0) for item in values) / len(values),
            2,
        ),
        "quality_warning": ";".join(dict.fromkeys(warnings)),
    }


def skipped_quality_metrics(reason: str) -> dict[str, Any]:
    """Return explicit quality fields when no real generated text exists."""

    return {
        "repetition_flag": False,
        "gibberish_flag": False,
        "repeated_ngram_ratio": "",
        "special_char_ratio": "",
        "output_length": "",
        "quality_warning": reason,
    }

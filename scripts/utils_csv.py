from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping


CANONICAL_LOG_FIELDS = [
    "model",
    "dataset",
    "sample_id",
    "kv_cache_type",
    "kv_cache_dtype",
    "context_length",
    "peak_memory_mb",
    "latency_ms_per_token",
    "throughput_tokens_per_s",
    "generated_tokens",
    "perplexity",
    "ppl_loss",
    "ppl_tokens",
    "ppl_status",
    "ppl_error",
    "repetition_flag",
    "gibberish_flag",
    "repeated_ngram_ratio",
    "special_char_ratio",
    "output_length",
    "quality_warning",
    "status",
    "error_message",
]

INT_LOG_FIELDS = {
    "sample_id",
    "context_length",
    "generated_tokens",
    "ppl_tokens",
    "output_length",
}

FLOAT_LOG_FIELDS = {
    "peak_memory_mb",
    "latency_ms_per_token",
    "throughput_tokens_per_s",
    "perplexity",
    "ppl_loss",
    "repeated_ngram_ratio",
    "special_char_ratio",
}

BOOL_LOG_FIELDS = {
    "repetition_flag",
    "gibberish_flag",
}

STRING_LOG_FIELDS = {
    field
    for field in CANONICAL_LOG_FIELDS
    if field not in INT_LOG_FIELDS
    and field not in FLOAT_LOG_FIELDS
    and field not in BOOL_LOG_FIELDS
}

DEFAULT_RAW_LOG_PATH = Path("results/raw_benchmark_log.csv")

_MISSING_TEXT_VALUES = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "nan",
    "inf",
    "-inf",
    "oom",
    "error",
}

_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_FALSE_VALUES = {"0", "false", "f", "no", "n"}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _is_missing_text(value: str) -> bool:
    return value.strip().lower() in _MISSING_TEXT_VALUES


def _format_int(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return str(int(value))

    text = _as_text(value)
    if _is_missing_text(text):
        return ""
    try:
        number = float(text)
    except ValueError:
        return ""
    if not math.isfinite(number):
        return ""
    return str(int(number))


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return str(number)

    text = _as_text(value)
    if _is_missing_text(text):
        return ""
    try:
        number = float(text)
    except ValueError:
        return ""
    if not math.isfinite(number):
        return ""
    return str(number)


def _format_bool(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"

    text = _as_text(value).lower()
    if text in _TRUE_VALUES:
        return "true"
    if text in _FALSE_VALUES:
        return "false"
    return ""


def _format_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return _as_text(value)


def normalize_log_row(row: Mapping[str, Any]) -> dict[str, str]:
    """Normalize a benchmark row into canonical CSV-safe strings."""

    normalized: dict[str, str] = {}
    for field in CANONICAL_LOG_FIELDS:
        value = row.get(field)
        if field in INT_LOG_FIELDS:
            normalized[field] = _format_int(value)
        elif field in FLOAT_LOG_FIELDS:
            normalized[field] = _format_float(value)
        elif field in BOOL_LOG_FIELDS:
            normalized[field] = _format_bool(value)
        else:
            normalized[field] = _format_text(value)
    return normalized


def ensure_log_header(output_path: str | Path) -> Path:
    """Create the canonical CSV header if the file is missing or empty."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CANONICAL_LOG_FIELDS)
        return path

    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().strip("\r\n")

    if not first_line:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CANONICAL_LOG_FIELDS)
        return path

    existing_header = next(csv.reader([first_line.lstrip("\ufeff")]))
    if existing_header != CANONICAL_LOG_FIELDS:
        raise ValueError(
            f"CSV schema mismatch in {path}: expected canonical header "
            f"{CANONICAL_LOG_FIELDS}, got {existing_header}"
        )
    return path


def append_log_row(output_path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one canonical row to a CSV file."""

    path = ensure_log_header(output_path)
    normalized = normalize_log_row(row)

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CANONICAL_LOG_FIELDS,
            extrasaction="ignore",
            restval="",
            lineterminator="\n",
        )
        writer.writerow(normalized)

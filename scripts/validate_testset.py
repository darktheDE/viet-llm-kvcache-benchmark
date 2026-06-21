from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer

try:
    from scripts.utils_text import has_replacement_char
except ModuleNotFoundError:
    from utils_text import has_replacement_char


LOGGER = logging.getLogger("validate_testset")

REQUIRED_TOP_LEVEL = ["dataset_name", "version", "language", "tokenizer", "samples"]
REQUIRED_SAMPLE_FIELDS = [
    "id",
    "source",
    "context_group",
    "target_tokens",
    "actual_tokens",
    "text",
    "metadata",
]
FULL_MIN_GROUP_COUNTS = {"4k": 3, "8k": 3, "16k": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Vietnamese long-context test set JSON.")
    parser.add_argument("--input", default="datasets/test_set_small.json", help="Input JSON file.")
    parser.add_argument("--allow-smoke-test", action="store_true", help="Allow smoke mode validation.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON value must be an object.")
    return data


def load_tokenizer(data: dict[str, Any]):
    tokenizer_info = data.get("tokenizer") or {}
    tokenizer_name = tokenizer_info.get("name_or_path")
    if not tokenizer_name:
        raise ValueError("tokenizer.name_or_path is missing.")
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def validate_top_level(data: dict[str, Any], errors: list[str]) -> None:
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.append(f"Missing top-level field: {field}")
    if not isinstance(data.get("samples"), list):
        errors.append("Top-level samples must be a list.")


def validate_mode(data: dict[str, Any], allow_smoke_test: bool, errors: list[str], warnings: list[str]) -> None:
    samples = data.get("samples") or []
    mode = data.get("mode", "full")
    group_counts = defaultdict(int)
    for sample in samples:
        if isinstance(sample, dict):
            group_counts[sample.get("context_group")] += 1

    if mode == "smoke_test":
        if allow_smoke_test:
            warnings.append("Smoke mode enabled: fewer or shorter samples are accepted.")
        else:
            errors.append("Dataset is smoke_test mode but --allow-smoke-test was not provided.")
        return

    if not (10 <= len(samples) <= 20):
        errors.append(f"Full mode requires 10-20 samples, found {len(samples)}.")
    for group_name, minimum in FULL_MIN_GROUP_COUNTS.items():
        if group_counts[group_name] < minimum:
            errors.append(
                f"Full mode requires at least {minimum} samples for group {group_name}, "
                f"found {group_counts[group_name]}."
            )


def validate_samples(data: dict[str, Any], tokenizer, errors: list[str]) -> dict[str, list[int]]:
    group_tokens: dict[str, list[int]] = defaultdict(list)
    samples = data.get("samples") or []

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            errors.append(f"Sample {index} is not an object.")
            continue
        for field in REQUIRED_SAMPLE_FIELDS:
            if field not in sample:
                errors.append(f"Sample {index} missing field: {field}")

        text = str(sample.get("text") or "")
        if not text.strip():
            errors.append(f"Sample {index} has empty text.")
            continue
        if has_replacement_char(text):
            errors.append(f"Sample {index} contains replacement character.")

        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            if not decoded.strip():
                errors.append(f"Sample {index} decode result is empty.")
            recomputed = len(token_ids)
        except Exception as exc:
            errors.append(f"Sample {index} tokenization/decode failed: {exc}")
            continue

        actual = sample.get("actual_tokens")
        if not isinstance(actual, int):
            errors.append(f"Sample {index} actual_tokens must be an integer.")
            continue

        tolerance = max(8, int(max(actual, recomputed) * 0.02))
        if abs(actual - recomputed) > tolerance:
            errors.append(
                f"Sample {index} actual_tokens mismatch: stored={actual} recomputed={recomputed}."
            )

        group_tokens[str(sample.get("context_group"))].append(recomputed)

    return group_tokens


def print_summary(group_tokens: dict[str, list[int]]) -> None:
    print("group\tcount\tmin_tokens\tmax_tokens\tavg_tokens")
    for group_name in sorted(group_tokens):
        values = group_tokens[group_name]
        if not values:
            continue
        print(
            f"{group_name}\t{len(values)}\t{min(values)}\t{max(values)}\t{mean(values):.1f}"
        )


def main() -> int:
    args = parse_args()
    configure_logging()

    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = load_json(args.input)
    except Exception as exc:
        LOGGER.error("JSON parse failed: %s", exc)
        return 1

    validate_top_level(data, errors)

    try:
        tokenizer = load_tokenizer(data)
    except Exception as exc:
        LOGGER.error("Tokenizer load failed: %s", exc)
        return 1

    validate_mode(data, args.allow_smoke_test, errors, warnings)
    group_tokens = validate_samples(data, tokenizer, errors)

    for warning in warnings:
        LOGGER.warning(warning)
    print_summary(group_tokens)

    if errors:
        for error in errors:
            LOGGER.error(error)
        return 1

    LOGGER.info("Validation passed for %s", args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

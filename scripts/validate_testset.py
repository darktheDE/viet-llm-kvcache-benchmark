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
    parser.add_argument(
        "--schema",
        choices=["long_context", "task"],
        default="long_context",
        help="Schema format to validate: 'long_context' or 'task'.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_json(path: str | Path, schema: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if schema == "long_context" and not isinstance(data, dict):
        raise ValueError("Top-level JSON value must be an object for long_context schema.")
    return data


def load_tokenizer(data: Any):
    tokenizer_name = None
    if isinstance(data, dict):
        tokenizer_info = data.get("tokenizer") or {}
        tokenizer_name = tokenizer_info.get("name_or_path")
    if not tokenizer_name:
        tokenizer_name = "Qwen/Qwen2.5-7B-Instruct-1M"
        LOGGER.info("No tokenizer specified in dataset, falling back to default: %s", tokenizer_name)
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception as exc:
        LOGGER.warning("Could not load tokenizer %s: %s. Trying fallback...", tokenizer_name, exc)
        return AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)


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


def validate_task_schema(data: Any, tokenizer, allow_smoke_test: bool, errors: list[str], warnings: list[str]) -> dict[str, list[int]]:
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        if "samples" in data:
            samples = data["samples"]
            if not isinstance(samples, list):
                errors.append("Top-level 'samples' field must be a list.")
                return {}
        else:
            errors.append("Top-level object is a dict but does not contain a 'samples' key.")
            return {}
    else:
        errors.append("Dataset must be a list or a dict containing a 'samples' list.")
        return {}

    num_samples = len(samples)
    if allow_smoke_test:
        if num_samples < 5:
            errors.append(f"Task smoke dataset should have at least 5 samples, found {num_samples}.")
        
        # Validate exact smoke distribution:
        # each context_length_target in 4000, 8000, 16000 must have 2 qa, 2 retrieval, 1 general.
        distribution = defaultdict(int)
        for sample in samples:
            if isinstance(sample, dict):
                clt = sample.get("context_length_target")
                pt = sample.get("prompt_type")
                distribution[(clt, pt)] += 1
        
        for clt in [4000, 8000, 16000]:
            if distribution[(clt, "qa")] != 2:
                errors.append(f"Smoke test must have exactly 2 'qa' samples for target {clt}, found {distribution[(clt, 'qa')]}")
            if distribution[(clt, "retrieval")] != 2:
                errors.append(f"Smoke test must have exactly 2 'retrieval' samples for target {clt}, found {distribution[(clt, 'retrieval')]}")
            if distribution[(clt, "general")] != 1:
                errors.append(f"Smoke test must have exactly 1 'general' sample for target {clt}, found {distribution[(clt, 'general')]}")
    else:
        if num_samples < 10:
            errors.append(f"Task full dataset should have at least 10 samples, found {num_samples}.")

    group_tokens = defaultdict(list)
    REQUIRED_TASK_FIELDS = ["prompt_type", "context_length_target", "text", "expected_output", "actual_tokens", "metadata"]

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            errors.append(f"Sample {index} is not an object.")
            continue

        for field in REQUIRED_TASK_FIELDS:
            if field not in sample:
                errors.append(f"Sample {index} missing field: {field}")

        # Metadata validation
        metadata = sample.get("metadata")
        if "metadata" in sample:
            if not isinstance(metadata, dict):
                errors.append(f"Sample {index} metadata must be an object/dict.")
            else:
                for subfield in ["source", "domain", "subject"]:
                    if subfield not in metadata:
                        errors.append(f"Sample {index} metadata missing subfield: {subfield}")
                    elif not isinstance(metadata.get(subfield), str) or not metadata.get(subfield).strip():
                        errors.append(f"Sample {index} metadata field '{subfield}' must be a non-empty string.")

        prompt_type = sample.get("prompt_type")
        if prompt_type not in ["qa", "retrieval", "general"]:
            errors.append(f"Sample {index} invalid prompt_type: {prompt_type}")

        context_length_target = sample.get("context_length_target")
        if context_length_target not in [4000, 8000, 16000]:
            errors.append(f"Sample {index} has invalid context_length_target: {context_length_target}")

        text = str(sample.get("text") or "")
        if not text.strip():
            errors.append(f"Sample {index} text is empty.")
            continue

        if has_replacement_char(text):
            errors.append(f"Sample {index} contains replacement character.")

        expected_output = sample.get("expected_output")
        if prompt_type == "qa":
            if expected_output not in ["A", "B", "C", "D", "E"]:
                errors.append(f"Sample {index} expected_output for QA must be A, B, C, D, or E, found: {expected_output}")
        elif prompt_type == "retrieval":
            if not isinstance(expected_output, str) or not expected_output.strip():
                errors.append(f"Sample {index} expected_output for Retrieval must be a non-empty string, found: {expected_output}")

        actual = sample.get("actual_tokens")
        if not isinstance(actual, int):
            errors.append(f"Sample {index} actual_tokens must be an integer.")
            continue

        if tokenizer is not None:
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                if not decoded.strip():
                    errors.append(f"Sample {index} decode result is empty.")
                recomputed = len(token_ids)

                tolerance = max(8, int(max(actual, recomputed) * 0.05))
                if abs(actual - recomputed) > tolerance:
                    errors.append(
                        f"Sample {index} actual_tokens mismatch: stored={actual} recomputed={recomputed}."
                    )
                group_tokens[f"{context_length_target}_{prompt_type}"].append(recomputed)
            except Exception as exc:
                errors.append(f"Sample {index} tokenization/decode failed: {exc}")
        else:
            group_tokens[f"{context_length_target}_{prompt_type}"].append(actual)

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
        data = load_json(args.input, args.schema)
    except Exception as exc:
        LOGGER.error("JSON load failed: %s", exc)
        return 1

    try:
        tokenizer = load_tokenizer(data)
    except Exception as exc:
        LOGGER.error("Tokenizer load failed: %s. Proceeding without tokenizer checks.", exc)
        tokenizer = None

    if args.schema == "long_context":
        validate_top_level(data, errors)
        validate_mode(data, args.allow_smoke_test, errors, warnings)
        group_tokens = validate_samples(data, tokenizer, errors)
    elif args.schema == "task":
        group_tokens = validate_task_schema(data, tokenizer, args.allow_smoke_test, errors, warnings)
    else:
        LOGGER.error("Invalid schema choice: %s", args.schema)
        return 1

    for warning in warnings:
        LOGGER.warning(warning)
    print_summary(group_tokens)

    if errors:
        for error in errors:
            LOGGER.error(error)
        return 1

    LOGGER.info("Validation passed for %s under schema %s", args.input, args.schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

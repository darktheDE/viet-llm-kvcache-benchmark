from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable

from datasets import get_dataset_config_names, load_dataset

try:
    from scripts.utils_text import ensure_parent, to_jsonable, write_jsonl
except ModuleNotFoundError:
    from utils_text import ensure_parent, to_jsonable, write_jsonl


LOGGER = logging.getLogger("download_datasets")

DATASET_SOURCES = [
    "5760/vmlu",
    "VTSNLP/vietnamese_curated_dataset",
]

TEXT_FIELDS = [
    "text",
    "content",
    "context",
    "document",
    "passage",
    "question",
    "answer",
    "explanation",
    "choices",
    "options",
]
QUESTION_FIELDS = ["question", "query", "prompt"]
ANSWER_FIELDS = ["answer", "label", "correct_answer", "target"]
EXPLANATION_FIELDS = ["explanation", "rationale", "solution"]
OPTION_KEYS = ["A", "B", "C", "D", "E", "a", "b", "c", "d", "e"]
SPLIT_PRIORITY = ["train", "validation", "valid", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw Vietnamese datasets from Hugging Face.")
    parser.add_argument("--output-dir", default="data/raw", help="Directory for raw JSONL output.")
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=5000,
        help="Maximum raw records to emit per Hugging Face source.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            rendered = stringify_value(item)
            if rendered:
                parts.append(f"{key}: {rendered}")
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        parts = [stringify_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value).strip()


def first_non_empty(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = stringify_value(row.get(field))
        if value:
            return value
    return ""


def format_options(row: dict[str, Any]) -> list[str]:
    options_value = row.get("choices", row.get("options"))
    rendered: list[str] = []

    if isinstance(options_value, dict):
        for index, key in enumerate(sorted(options_value.keys())):
            label = str(key).upper()
            value = stringify_value(options_value[key])
            if value:
                rendered.append(f"{label}. {value}")
    elif isinstance(options_value, (list, tuple)):
        labels = ["A", "B", "C", "D", "E", "F"]
        for index, item in enumerate(options_value):
            value = stringify_value(item)
            if value:
                label = labels[index] if index < len(labels) else str(index + 1)
                rendered.append(f"{label}. {value}")
    elif isinstance(options_value, str) and options_value.strip():
        rendered.append(options_value.strip())

    if rendered:
        return rendered

    for key in OPTION_KEYS:
        value = stringify_value(row.get(key))
        if value:
            rendered.append(f"{key.upper()}. {value}")
    return rendered


def benchmark_text(row: dict[str, Any]) -> str:
    question = first_non_empty(row, QUESTION_FIELDS)
    options = format_options(row)
    answer = first_non_empty(row, ANSWER_FIELDS)
    explanation = first_non_empty(row, EXPLANATION_FIELDS)

    if not any([question, options, answer, explanation]):
        return ""

    parts: list[str] = []
    if question:
        parts.append(f"Câu hỏi: {question}")
    parts.extend(options)
    if answer:
        parts.append(f"Đáp án: {answer}")
    if explanation:
        parts.append(f"Giải thích: {explanation}")
    return "\n".join(parts)


def extract_text(row: dict[str, Any]) -> str:
    benchmark = benchmark_text(row)
    parts: list[str] = [benchmark] if benchmark else []

    for field in TEXT_FIELDS:
        if field in {"question", "answer", "explanation", "choices", "options"} and benchmark:
            continue
        value = stringify_value(row.get(field))
        if value and value not in parts:
            if field in {"context", "document", "passage"}:
                parts.insert(0, value)
            else:
                parts.append(value)

    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def dataset_configs(source_id: str) -> list[str | None]:
    try:
        return get_dataset_config_names(source_id, trust_remote_code=True) or [None]
    except TypeError:
        try:
            return get_dataset_config_names(source_id) or [None]
        except Exception as exc:
            LOGGER.warning("Could not list configs for %s: %s.", source_id, exc)
            return []
    except Exception as exc:
        LOGGER.warning("Could not list configs for %s: %s.", source_id, exc)
        return []


def load_dataset_stream(source_id: str, config_name: str | None):
    kwargs = {
        "path": source_id,
        "streaming": True,
    }
    if config_name:
        kwargs["name"] = config_name

    try:
        return load_dataset(**kwargs, trust_remote_code=True)
    except TypeError:
        try:
            return load_dataset(**kwargs)
        except Exception:
            raise


def ordered_splits(dataset_obj: Any) -> list[tuple[str, Any]]:
    if hasattr(dataset_obj, "items"):
        items = list(dataset_obj.items())
    else:
        items = [("train", dataset_obj)]

    priority = {name: index for index, name in enumerate(SPLIT_PRIORITY)}
    return sorted(items, key=lambda item: priority.get(item[0], len(priority)))


def iter_source_records(source_id: str, max_records: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_records: list[dict[str, Any]] = []
    stats = {
        "configs_loaded": 0,
        "records_seen": 0,
        "valid_text": 0,
    }

    for config_name in dataset_configs(source_id):
        if len(source_records) >= max_records:
            break
        try:
            dataset_obj = load_dataset_stream(source_id, config_name)
            stats["configs_loaded"] += 1
            LOGGER.info("Loaded %s config=%s", source_id, config_name or "default")
        except Exception as exc:
            LOGGER.warning("Skipping %s config=%s: %s", source_id, config_name or "default", exc)
            continue

        for split_name, split_dataset in ordered_splits(dataset_obj):
            if len(source_records) >= max_records:
                break
            try:
                iterator = iter(split_dataset)
                for row_index, row in enumerate(iterator):
                    if len(source_records) >= max_records:
                        break
                    if not isinstance(row, dict):
                        continue

                    stats["records_seen"] += 1
                    text = extract_text(row)
                    if text.strip():
                        stats["valid_text"] += 1

                    original_id = (
                        row.get("id")
                        or row.get("_id")
                        or row.get("example_id")
                        or row.get("idx")
                        or f"{config_name or 'default'}:{split_name}:{row_index}"
                    )
                    metadata = {
                        "dataset_config": config_name,
                        "row_keys": sorted(str(key) for key in row.keys()),
                        "raw_record": to_jsonable(row),
                    }
                    source_records.append(
                        {
                            "source": source_id,
                            "split": split_name,
                            "original_id": str(original_id),
                            "text": text,
                            "metadata": metadata,
                        }
                    )
            except Exception as exc:
                LOGGER.warning(
                    "Could not iterate %s config=%s split=%s: %s",
                    source_id,
                    config_name or "default",
                    split_name,
                    exc,
                )
                continue

    return source_records, stats


def main() -> int:
    args = parse_args()
    configure_logging()

    output_path = ensure_parent(Path(args.output_dir) / "raw_records.jsonl")
    all_records: list[dict[str, Any]] = []
    successful_sources = 0
    total_valid_text = 0

    for source_id in DATASET_SOURCES:
        LOGGER.info("Downloading source=%s", source_id)
        records, stats = iter_source_records(source_id, args.max_records_per_source)
        if records:
            successful_sources += 1
            all_records.extend(records)
        total_valid_text += stats["valid_text"]
        LOGGER.info(
            "Source %s: configs=%s records=%s valid_text=%s emitted=%s",
            source_id,
            stats["configs_loaded"],
            stats["records_seen"],
            stats["valid_text"],
            len(records),
        )

    if successful_sources == 0:
        LOGGER.error("No dataset source could be downloaded.")
        return 1

    write_jsonl(all_records, output_path)
    LOGGER.info("Wrote %s", output_path)
    LOGGER.info("Successful sources: %s", successful_sources)
    LOGGER.info("Total records: %s", len(all_records))
    LOGGER.info("Records with valid text: %s", total_valid_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

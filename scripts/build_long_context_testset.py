from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

try:
    from scripts.utils_text import ensure_parent, has_replacement_char, read_jsonl
except ModuleNotFoundError:
    from utils_text import ensure_parent, has_replacement_char, read_jsonl


LOGGER = logging.getLogger("build_long_context_testset")

SEPARATOR = "\n\n---\n\n"
TOKENIZER_FALLBACK = "bert-base-multilingual-cased"
GROUPS = {
    "4k": {"target": 4096, "min": 3500, "max": 5000, "count": 4},
    "8k": {"target": 8192, "min": 7000, "max": 9500, "count": 4},
    "16k": {"target": 16384, "min": 14000, "max": 18500, "count": 4},
}
SMOKE_GROUPS = {
    "4k": {"target": 1024, "min": 400, "max": 5000, "count": 1},
    "8k": {"target": 2048, "min": 600, "max": 9500, "count": 1},
    "16k": {"target": 4096, "min": 800, "max": 18500, "count": 1},
}


@dataclass
class CleanRecord:
    id: str
    source: str
    original_id: str | None
    text: str
    metadata: dict[str, Any]
    token_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small Vietnamese long-context test set.")
    parser.add_argument("--input", default="data/processed/cleaned.jsonl", help="Cleaned JSONL input.")
    parser.add_argument("--output", default="datasets/test_set_small.json", help="Output JSON path.")
    parser.add_argument(
        "--tokenizer-name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Preferred tokenizer name or path.",
    )
    parser.add_argument("--allow-smoke-test", action="store_true", help="Allow fewer/shorter samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_tokenizer(name: str):
    for candidate in [name, TOKENIZER_FALLBACK]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True)
            LOGGER.info("Using tokenizer=%s", candidate)
            return tokenizer, candidate
        except Exception as exc:
            LOGGER.warning("Could not load tokenizer %s: %s", candidate, exc)
    raise RuntimeError("Could not load preferred tokenizer or fallback tokenizer.")


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_max(tokenizer, text: str, max_tokens: int) -> tuple[str, int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, len(token_ids)
    truncated_ids = token_ids[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text, token_len(tokenizer, truncated_text)


def load_records(path: str | Path, tokenizer) -> list[CleanRecord]:
    records: list[CleanRecord] = []
    for row in read_jsonl(path):
        text = str(row.get("text") or "").strip()
        if not text or has_replacement_char(text):
            continue
        try:
            count = token_len(tokenizer, text)
        except Exception:
            continue
        records.append(
            CleanRecord(
                id=str(row.get("id")),
                source=str(row.get("source") or "unknown"),
                original_id=row.get("original_id"),
                text=text,
                metadata=row.get("metadata") or {},
                token_count=count,
            )
        )
    return records


def detokenize_ok(tokenizer, text: str) -> bool:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        return bool(decoded.strip())
    except Exception:
        return False


def select_records(records: list[CleanRecord], used_counts: dict[str, int], rng: random.Random) -> list[CleanRecord]:
    shuffled = records[:]
    rng.shuffle(shuffled)
    return sorted(shuffled, key=lambda record: (used_counts.get(record.id, 0), -record.token_count))


def build_context(
    tokenizer,
    records: list[CleanRecord],
    group_spec: dict[str, int],
    used_counts: dict[str, int],
    rng: random.Random,
    allow_short: bool,
) -> tuple[str, int, list[CleanRecord]]:
    selected: list[CleanRecord] = []
    parts: list[str] = []
    current_tokens = 0
    seen_in_sample: set[str] = set()

    for record in select_records(records, used_counts, rng):
        if record.id in seen_in_sample:
            continue
        candidate = SEPARATOR.join(parts + [record.text]) if parts else record.text
        candidate_tokens = token_len(tokenizer, candidate)
        if candidate_tokens > group_spec["max"] and current_tokens >= group_spec["min"]:
            break
        selected.append(record)
        parts.append(record.text)
        seen_in_sample.add(record.id)
        current_tokens = candidate_tokens
        if current_tokens >= group_spec["min"]:
            break

    if not parts:
        raise ValueError("No records available for context assembly.")

    text = SEPARATOR.join(parts)
    text, actual_tokens = truncate_to_max(tokenizer, text, group_spec["max"])

    if actual_tokens < group_spec["min"] and not allow_short:
        raise ValueError(
            f"Not enough tokens for group target. actual={actual_tokens} min={group_spec['min']}"
        )

    for record in selected:
        used_counts[record.id] = used_counts.get(record.id, 0) + 1

    return text, actual_tokens, selected


def sample_metadata(selected: list[CleanRecord], detok_ok: bool) -> dict[str, Any]:
    cleaning_pipeline: list[str] = []
    for record in selected:
        for step in record.metadata.get("cleaning_pipeline") or []:
            if step not in cleaning_pipeline:
                cleaning_pipeline.append(step)

    return {
        "title": None,
        "domain": None,
        "original_id": selected[0].original_id if selected else None,
        "source_record_ids": [record.id for record in selected],
        "cleaning_pipeline": cleaning_pipeline,
        "quality_flags": {
            "unicode_valid": True,
            "vietnamese_ratio_ok": True,
            "detokenize_ok": detok_ok,
        },
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    tokenizer, tokenizer_name = load_tokenizer(args.tokenizer_name)
    records = load_records(args.input, tokenizer)
    if not records:
        raise RuntimeError("No cleaned records available for test-set construction.")

    rng.shuffle(records)
    specs = SMOKE_GROUPS if args.allow_smoke_test else GROUPS
    mode = "smoke_test" if args.allow_smoke_test else "full"
    used_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []

    for group_name, spec in specs.items():
        for sample_index in range(1, spec["count"] + 1):
            try:
                text, actual_tokens, selected = build_context(
                    tokenizer=tokenizer,
                    records=records,
                    group_spec=spec,
                    used_counts=used_counts,
                    rng=rng,
                    allow_short=args.allow_smoke_test,
                )
            except ValueError as exc:
                if args.allow_smoke_test:
                    LOGGER.warning("Skipping smoke sample group=%s: %s", group_name, exc)
                    continue
                raise

            detok_ok = detokenize_ok(tokenizer, text)
            samples.append(
                {
                    "id": f"vi_lc_{group_name}_{sample_index:03d}",
                    "source": "mixed" if len({record.source for record in selected}) > 1 else selected[0].source,
                    "context_group": group_name,
                    "target_tokens": spec["target"],
                    "actual_tokens": actual_tokens,
                    "text": text,
                    "metadata": sample_metadata(selected, detok_ok),
                }
            )

    if not samples:
        raise RuntimeError("No test-set samples were created.")

    dataset = {
        "dataset_name": "vietnamese_long_context_test_set_small",
        "version": "0.1.0",
        "language": "vi",
        "created_by": "data_pipeline",
        "description": "Small cleaned Vietnamese long-context test set for LLM benchmark.",
        "mode": mode,
        "tokenizer": {
            "name_or_path": tokenizer_name,
            "token_count_method": "transformers AutoTokenizer encode length",
        },
        "samples": samples,
    }
    return dataset


def main() -> int:
    args = parse_args()
    configure_logging()

    try:
        dataset = build_dataset(args)
    except Exception as exc:
        LOGGER.error("Could not build test set: %s", exc)
        return 1

    output_path = ensure_parent(args.output)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    LOGGER.info("Wrote %s samples to %s", len(dataset["samples"]), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

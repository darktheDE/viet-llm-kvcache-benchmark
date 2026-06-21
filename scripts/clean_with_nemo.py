from __future__ import annotations

import argparse
from collections import Counter
import logging
from pathlib import Path
from typing import Any

from ftfy import fix_text

try:
    import nemo_curator  # noqa: F401

    NEMO_CURATOR_AVAILABLE = True
except Exception:
    NEMO_CURATOR_AVAILABLE = False

try:
    from scripts.nemo_backend import NemoBackendError, NemoTextBackend
    from scripts.utils_text import (
        char_ngrams,
        hamming_distance,
        jaccard,
        normalize_nfc,
        normalize_whitespace,
        read_jsonl,
        remove_control_chars,
        simhash_from_features,
        stable_text_hash,
        text_quality_flags,
        write_jsonl,
    )
except ModuleNotFoundError:
    from nemo_backend import NemoBackendError, NemoTextBackend
    from utils_text import (
        char_ngrams,
        hamming_distance,
        jaccard,
        normalize_nfc,
        normalize_whitespace,
        read_jsonl,
        remove_control_chars,
        simhash_from_features,
        stable_text_hash,
        text_quality_flags,
        write_jsonl,
    )


LOGGER = logging.getLogger("clean_with_nemo")

CLEANING_PIPELINE = [
    "nemo_document_batch",
    "nemo_unicode_reformatter",
    "nemo_newline_normalizer",
    "nemo_heuristic_document_filters",
    "unicode_normalization_nfc",
    "ftfy_fix_text",
    "control_char_removal",
    "whitespace_normalization",
    "length_filter",
    "vietnamese_quality_filter",
    "exact_dedup",
    "near_dedup",
]
PYTHON_FALLBACK_STEPS = [
    "unicode_normalization_nfc",
    "ftfy_fix_text",
    "control_char_removal",
    "whitespace_normalization",
    "length_filter",
    "vietnamese_quality_filter",
    "exact_dedup",
    "near_dedup",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw Vietnamese JSONL records.")
    parser.add_argument("--input", default="data/raw/raw_records.jsonl", help="Input JSONL path.")
    parser.add_argument("--input-dir", default=None, help="Directory containing JSONL files.")
    parser.add_argument("--output", default="data/processed/cleaned.jsonl", help="Output JSONL path.")
    parser.add_argument(
        "--backend",
        choices=["auto", "nemo", "python"],
        default="auto",
        help="Cleaning backend: auto prefers NeMo, nemo requires NeMo, python disables NeMo.",
    )
    parser.add_argument("--min-chars", type=int, default=200, help="Minimum cleaned text length.")
    parser.add_argument(
        "--near-duplicate-jaccard",
        type=float,
        default=0.92,
        help="Character n-gram Jaccard threshold for near-duplicate removal.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_dir:
        return sorted(Path(args.input_dir).glob("*.jsonl"))
    return [Path(args.input)]


def clean_text(text: str) -> str:
    fixed = fix_text(text or "")
    normalized = normalize_nfc(fixed)
    no_control = remove_control_chars(normalized)
    return normalize_whitespace(no_control)


def python_postprocess_text(text: str) -> str:
    normalized = normalize_nfc(text or "")
    no_control = remove_control_chars(normalized)
    return normalize_whitespace(no_control)


def resolve_nemo_backend(backend_arg: str) -> NemoTextBackend | None:
    if backend_arg == "python":
        LOGGER.info("Using python_fallback backend by request.")
        return None

    if not NEMO_CURATOR_AVAILABLE:
        message = "nemo_curator import failed; NeMo backend is unavailable."
        if backend_arg == "nemo":
            raise NemoBackendError(message)
        LOGGER.warning("%s Falling back to Python backend.", message)
        return None

    try:
        backend = NemoTextBackend()
        LOGGER.info("Using hybrid_nemo_python backend.")
        LOGGER.info("NeMo Curator steps: %s", ", ".join(backend.steps))
        return backend
    except Exception as exc:
        if backend_arg == "nemo":
            raise NemoBackendError(f"Could not initialize required NeMo backend: {exc}") from exc
        LOGGER.warning("Could not initialize NeMo backend: %s. Falling back to Python backend.", exc)
        return None


def is_near_duplicate(
    grams: set[str],
    fingerprint: int,
    seen_fingerprints: list[tuple[int, set[str]]],
    jaccard_threshold: float,
) -> bool:
    for previous_fingerprint, previous_grams in seen_fingerprints:
        if hamming_distance(fingerprint, previous_fingerprint) <= 4:
            if jaccard(grams, previous_grams) >= jaccard_threshold:
                return True
    return False


def raw_records_from_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = input_paths(args)
    if not paths:
        raise FileNotFoundError("No JSONL input files found.")

    records: list[dict[str, Any]] = []
    for path in paths:
        LOGGER.info("Reading %s", path)
        records.extend(read_jsonl(path))
    return records


def iter_clean_records(args: argparse.Namespace):
    stats = {
        "input_records": 0,
        "removed_nemo_filter": 0,
        "removed_too_short": 0,
        "removed_unicode_replacement": 0,
        "removed_low_quality": 0,
        "removed_exact_duplicate": 0,
        "removed_near_duplicate": 0,
        "final_records": 0,
    }
    backend_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    seen_fingerprints: list[tuple[int, set[str]]] = []

    raw_records = raw_records_from_inputs(args)
    stats["input_records"] = len(raw_records)
    nemo_backend = resolve_nemo_backend(args.backend)
    active_backend = "hybrid_nemo_python" if nemo_backend is not None else "python_fallback"

    nemo_results = None
    if nemo_backend is not None:
        raw_texts = [str(record.get("text") or "") for record in raw_records]
        try:
            nemo_results = nemo_backend.process_texts(raw_texts)
        except Exception as exc:
            if args.backend == "nemo":
                raise NemoBackendError(f"Required NeMo processing failed: {exc}") from exc
            LOGGER.warning("NeMo processing failed: %s. Falling back to Python backend.", exc)
            nemo_backend = None
            active_backend = "python_fallback"
            nemo_results = None

    for index, raw_record in enumerate(raw_records):
        nemo_steps: list[str] = []
        nemo_scores: dict[str, Any] = {}
        python_steps = PYTHON_FALLBACK_STEPS[:]

        if nemo_results is not None:
            nemo_result = nemo_results[index]
            nemo_steps = nemo_result.steps
            nemo_scores = nemo_result.scores
            if not nemo_result.keep:
                stats["removed_nemo_filter"] += 1
                continue
            text = python_postprocess_text(nemo_result.text)
        else:
            text = clean_text(str(raw_record.get("text") or ""))

        if len(text) < args.min_chars:
            stats["removed_too_short"] += 1
            continue

        flags = text_quality_flags(text)
        if not flags["replacement_char_free"] or not flags["unicode_valid"]:
            stats["removed_unicode_replacement"] += 1
            continue

        if not flags["vietnamese_ratio_ok"]:
            stats["removed_low_quality"] += 1
            continue

        text_hash = stable_text_hash(text)
        if text_hash in seen_hashes:
            stats["removed_exact_duplicate"] += 1
            continue
        seen_hashes.add(text_hash)

        grams = char_ngrams(text)
        fingerprint = simhash_from_features(grams)
        if is_near_duplicate(grams, fingerprint, seen_fingerprints, args.near_duplicate_jaccard):
            stats["removed_near_duplicate"] += 1
            continue
        seen_fingerprints.append((fingerprint, grams))

        stats["final_records"] += 1
        backend_counts[active_backend] += 1
        raw_metadata: dict[str, Any] = raw_record.get("metadata") or {}
        metadata = {
            **raw_metadata,
            "split": raw_record.get("split"),
            "nemo_curator_available": NEMO_CURATOR_AVAILABLE,
            "cleaning_backend": active_backend,
            "cleaning_implementation": active_backend,
            "nemo_curator_steps": nemo_steps,
            "python_fallback_steps": python_steps,
            "nemo_curator_scores": nemo_scores,
            "cleaning_pipeline": CLEANING_PIPELINE if nemo_steps else PYTHON_FALLBACK_STEPS,
            "quality_flags": {
                "unicode_valid": bool(flags["unicode_valid"]),
                "vietnamese_ratio_ok": bool(flags["vietnamese_ratio_ok"]),
                "replacement_char_free": bool(flags["replacement_char_free"]),
                "letter_ratio": flags["letter_ratio"],
                "symbol_ratio": flags["symbol_ratio"],
                "has_vietnamese_signal": flags["has_vietnamese_signal"],
            },
        }
        yield {
            "id": f"clean_{stats['final_records']:06d}",
            "source": raw_record.get("source"),
            "original_id": raw_record.get("original_id"),
            "text": text,
            "metadata": metadata,
        }

    LOGGER.info("nemo_curator_available=%s", NEMO_CURATOR_AVAILABLE)
    LOGGER.info("requested backend: %s", args.backend)
    LOGGER.info("active backend: %s", active_backend)
    LOGGER.info("backend counts: %s", dict(backend_counts))
    LOGGER.info("input records: %s", stats["input_records"])
    LOGGER.info("removed by NeMo filter: %s", stats["removed_nemo_filter"])
    LOGGER.info("removed too short: %s", stats["removed_too_short"])
    LOGGER.info("removed unicode/replacement: %s", stats["removed_unicode_replacement"])
    LOGGER.info("removed low quality: %s", stats["removed_low_quality"])
    LOGGER.info("removed exact duplicate: %s", stats["removed_exact_duplicate"])
    LOGGER.info("removed near duplicate: %s", stats["removed_near_duplicate"])
    LOGGER.info("final records: %s", stats["final_records"])


def main() -> int:
    args = parse_args()
    configure_logging()

    try:
        count = write_jsonl(iter_clean_records(args), args.output)
    except Exception as exc:
        LOGGER.error("Cleaning failed: %s", exc)
        return 1

    LOGGER.info("Wrote %s records to %s", count, args.output)
    if count == 0:
        LOGGER.error("No cleaned records were produced.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

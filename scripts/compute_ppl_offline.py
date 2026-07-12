from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils_generation_quality import (
    QUALITY_FIELDS,
    aggregate_quality_metrics,
    analyze_generated_text,
)
from scripts.utils_ppl import compute_perplexity, load_reference_model
from scripts.utils_text import read_jsonl


PPL_FIELDS = ["perplexity", "ppl_loss", "ppl_tokens", "ppl_status", "ppl_error"]
EXTRA_FIELDS = PPL_FIELDS + QUALITY_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill PPL and quality flags from generated-output JSONL."
    )
    parser.add_argument("--input_csv", required=True, help="Raw benchmark CSV to backfill.")
    parser.add_argument("--input_jsonl", required=True, help="JSONL containing prompt/generated text.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--reference_model", required=True, help="BF16 reference model path or HF id.")
    parser.add_argument("--reference_tokenizer", default=None, help="Optional tokenizer path or HF id.")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument(
        "--ppl_mode",
        default="conditional",
        choices=["conditional", "generated_only"],
        help="conditional scores generated tokens with prompt context; generated_only scores output alone.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output_csv.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output_csv exists, reuse it and skip rows with ppl_status=OK.",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace access token cho model gated (hoac dat env HF_TOKEN)")
    parser.add_argument("--progress_every", type=int, default=10)
    return parser.parse_args()


def read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_output_fields(fieldnames: list[str]) -> list[str]:
    output_fields = list(fieldnames)
    for field in EXTRA_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    return output_fields


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    return [record for record in read_jsonl(path)]


def _same_value(left: Any, right: Any) -> bool:
    return str(left or "") == str(right or "")


def _metadata_matches(row: dict[str, Any], record: dict[str, Any]) -> bool:
    for key in ("model", "kv_cache_type", "context_length"):
        if row.get(key) not in (None, "") and record.get(key) not in (None, ""):
            if not _same_value(row.get(key), record.get(key)):
                return False
    return True


def _same_path(left: str, right: str | Path) -> bool:
    if not left:
        return False
    return Path(left).name == Path(right).name


def select_records_for_row(
    row: dict[str, Any],
    records: list[dict[str, Any]],
    input_jsonl: str | Path,
) -> list[dict[str, Any]]:
    """Select exact or aggregate JSONL records for one CSV row."""

    sample_id = str(row.get("sample_id") or "")
    exact = [
        record
        for record in records
        if sample_id and str(record.get("sample_id") or "") == sample_id
    ]
    if exact:
        return exact

    if sample_id:
        prefix = f"{sample_id}__s"
        prefixed = [
            record
            for record in records
            if str(record.get("sample_id") or "").startswith(prefix)
            and _metadata_matches(row, record)
        ]
        if prefixed:
            return prefixed

    output_path = str(row.get("output_path") or "")
    if _same_path(output_path, input_jsonl):
        return [record for record in records if _metadata_matches(row, record)]

    return []


def aggregate_ppl_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [
        result
        for result in results
        if result.get("ppl_status") == "OK"
        and result.get("ppl_loss") not in (None, "")
        and int(result.get("ppl_tokens") or 0) > 0
    ]
    if ok_results:
        total_tokens = sum(int(result["ppl_tokens"]) for result in ok_results)
        total_nll = sum(float(result["ppl_loss"]) * int(result["ppl_tokens"]) for result in ok_results)
        mean_loss = total_nll / total_tokens
        return {
            "perplexity": round(math.exp(mean_loss), 4),
            "ppl_loss": round(mean_loss, 6),
            "ppl_tokens": total_tokens,
            "ppl_status": "OK",
            "ppl_error": "",
        }

    if not results:
        return {
            "perplexity": None,
            "ppl_loss": None,
            "ppl_tokens": 0,
            "ppl_status": "MISSING_TEXT",
            "ppl_error": "no matching JSONL record",
        }

    statuses = [str(result.get("ppl_status") or "ERROR") for result in results]
    errors = [str(result.get("ppl_error") or "") for result in results if result.get("ppl_error")]
    return {
        "perplexity": None,
        "ppl_loss": None,
        "ppl_tokens": 0,
        "ppl_status": ";".join(dict.fromkeys(statuses)),
        "ppl_error": "; ".join(errors),
    }


def _serialize_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def backfill_rows(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    input_jsonl: str | Path,
    ppl_fn: Callable[[dict[str, Any]], dict[str, Any]],
    resume: bool = False,
    progress_every: int = 10,
) -> list[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        if resume and str(row.get("ppl_status") or "") == "OK" and row.get("perplexity"):
            continue

        matched_records = select_records_for_row(row, records, input_jsonl)
        ppl_results = [ppl_fn(record) for record in matched_records]
        aggregate_ppl = aggregate_ppl_results(ppl_results)

        quality_metrics = [
            analyze_generated_text(str(record.get("generated_text") or ""))
            for record in matched_records
        ]
        aggregate_quality = aggregate_quality_metrics(quality_metrics)

        for key in PPL_FIELDS:
            row[key] = _serialize_metric(aggregate_ppl.get(key))
        for key in QUALITY_FIELDS:
            row[key] = _serialize_metric(aggregate_quality.get(key))

        if progress_every > 0 and index % progress_every == 0:
            print(f"Processed {index}/{len(rows)} CSV rows", flush=True)

    return rows


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    input_jsonl = Path(args.input_jsonl)
    output_csv = Path(args.output_csv)

    if output_csv.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"{output_csv} already exists; use --overwrite or --resume")

    source_csv = output_csv if args.resume and output_csv.exists() else input_csv
    fieldnames, rows = read_csv_rows(source_csv)
    fieldnames = ensure_output_fields(fieldnames)
    records = load_jsonl_records(input_jsonl)

    reference_model, tokenizer, device = load_reference_model(
        args.reference_model,
        tokenizer_name=args.reference_tokenizer,
        device=args.device,
        dtype=args.dtype,
        hf_token=args.hf_token,
    )

    def score_record(record: dict[str, Any]) -> dict[str, Any]:
        return compute_perplexity(
            reference_model=reference_model,
            tokenizer=tokenizer,
            generated_text=str(record.get("generated_text") or ""),
            prompt_text=str(record.get("prompt_text") or ""),
            device=device,
            mode=args.ppl_mode,
            max_length=args.max_length,
            stride=args.stride,
        )

    rows = backfill_rows(
        rows=rows,
        records=records,
        input_jsonl=input_jsonl,
        ppl_fn=score_record,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    write_csv_rows(output_csv, fieldnames, rows)
    print(f"Wrote backfilled CSV: {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

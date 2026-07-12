"""
Wrapper script to calculate and backfill Perplexity (PPL) for all models in a benchmark CSV.
Automatically splits the input CSV by model, runs scripts/compute_ppl_offline.py as a subprocess
for each model using its corresponding uncompressed reference model, and merges the results back.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Mapping from Ollama/CSV model names to HuggingFace reference models
MODEL_MAP = {
    "qwen3:8b-fp16": "Qwen/Qwen3-8B",
    "llama3.1:8b-instruct-fp16": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral:7b-instruct-v0.3-fp16": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5:7b-instruct-fp16": "Qwen/Qwen2.5-7B-Instruct",
    "phi4:mini-reasoning": "microsoft/Phi-4-mini-reasoning",
    "gemma3:4b-it": "google/gemma-3-4b-it",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and backfill PPL for all models in a benchmark CSV."
    )
    parser.add_argument("--input_csv", required=True, help="Raw benchmark CSV to backfill.")
    parser.add_argument("--input_jsonl", required=True, help="JSONL containing prompt/generated text.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--ppl_mode",
        default="conditional",
        choices=["conditional", "generated_only"],
        help="conditional or generated_only mode for perplexity",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HF access token")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it exists.")
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


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    input_jsonl = Path(args.input_jsonl)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        print(f"Error: Input CSV {input_csv} does not exist.")
        sys.exit(1)
    if not input_jsonl.exists():
        print(f"Error: Input JSONL {input_jsonl} does not exist.")
        sys.exit(1)

    if output_csv.exists() and not args.overwrite:
        print(f"Error: Output CSV {output_csv} already exists. Use --overwrite to replace.")
        sys.exit(1)

    # Read original CSV rows
    fieldnames, original_rows = read_csv_rows(input_csv)

    # Group rows by model
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in original_rows:
        model = row.get("model", "")
        grouped_rows.setdefault(model, []).append(row)

    print(f"Found unique models in CSV: {list(grouped_rows.keys())}")

    # Process each model group
    backfilled_by_model: dict[str, list[dict[str, Any]]] = {}
    temp_files = []

    try:
        for model, rows in grouped_rows.items():
            if not model:
                print("Skipping empty model row group.")
                continue

            ref_model = MODEL_MAP.get(model)
            if not ref_model:
                print(f"Warning: No uncompressed reference model mapped for model '{model}'. Skipping.")
                continue

            print(f"\nProcessing model '{model}' with reference model '{ref_model}'...")

            # Write temporary input CSV containing only this model's rows
            temp_in = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8", newline="")
            temp_out = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8", newline="")
            temp_files.extend([temp_in.name, temp_out.name])

            # Write group rows to temp_in
            writer = csv.DictWriter(temp_in, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            temp_in.close()
            temp_out.close()

            # Prepare subprocess command
            cmd = [
                sys.executable,
                str(REPO_ROOT / "compute_ppl_offline.py"),
                "--input_csv", temp_in.name,
                "--input_jsonl", str(input_jsonl),
                "--output_csv", temp_out.name,
                "--reference_model", ref_model,
                "--device", args.device,
                "--dtype", args.dtype,
                "--stride", str(args.stride),
                "--ppl_mode", args.ppl_mode,
                "--overwrite"
            ]

            hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                cmd.extend(["--hf_token", hf_token])

            # Run compute_ppl_offline for this model
            print(f"Running compute_ppl_offline.py for {model}...")
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            res = subprocess.run(cmd, env=env)

            if res.returncode != 0:
                print(f"Error: compute_ppl_offline failed for model {model} (exit code {res.returncode}).")
                continue

            # Read backfilled results for this model
            bf_fieldnames, bf_rows = read_csv_rows(temp_out.name)
            backfilled_by_model[model] = bf_rows
            # Update overall fieldnames with any new metrics fields added
            for fn in bf_fieldnames:
                if fn not in fieldnames:
                    fieldnames.append(fn)

        # Merge results back into original rows
        merged_rows = []
        for row in original_rows:
            model = row.get("model", "")
            # Find the matching backfilled row
            matched_bf_row = None
            if model in backfilled_by_model:
                for bf_row in backfilled_by_model[model]:
                    # Match by sample_id, kv_cache_type, context_length
                    if (
                        row.get("sample_id") == bf_row.get("sample_id")
                        and row.get("kv_cache_type") == bf_row.get("kv_cache_type")
                        and row.get("context_length") == bf_row.get("context_length")
                    ):
                        matched_bf_row = bf_row
                        break

            if matched_bf_row:
                # Merge backfilled perplexity/quality metrics into original row
                merged_row = {**row, **matched_bf_row}
            else:
                merged_row = row

            merged_rows.append(merged_row)

        # Save final merged CSV
        write_csv_rows(output_csv, fieldnames, merged_rows)
        print(f"\nSuccessfully wrote overall backfilled CSV to: {output_csv}")

    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for tf in temp_files:
            try:
                os.remove(tf)
            except Exception:
                pass


if __name__ == "__main__":
    main()

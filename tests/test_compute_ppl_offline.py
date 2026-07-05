import tempfile
import unittest
from pathlib import Path

from scripts.compute_ppl_offline import (
    aggregate_ppl_results,
    backfill_rows,
    ensure_output_fields,
    read_csv_rows,
    select_records_for_row,
    write_csv_rows,
)


class ComputePplOfflineTests(unittest.TestCase):
    def test_select_records_supports_aggregate_sample_id(self):
        row = {
            "model": "model-a",
            "kv_cache_type": "FP16",
            "context_length": "8000",
            "sample_id": "model-a__FP16__8000",
            "output_path": "results/run_generated.jsonl",
        }
        records = [
            {
                "sample_id": "model-a__FP16__8000__s0",
                "model": "model-a",
                "kv_cache_type": "FP16",
                "context_length": 8000,
            },
            {
                "sample_id": "model-a__FP16__8000__s1",
                "model": "model-a",
                "kv_cache_type": "FP16",
                "context_length": 8000,
            },
        ]

        selected = select_records_for_row(row, records, "results/run_generated.jsonl")

        self.assertEqual(len(selected), 2)

    def test_aggregate_ppl_is_token_weighted(self):
        result = aggregate_ppl_results(
            [
                {"ppl_status": "OK", "ppl_loss": 1.0, "ppl_tokens": 10},
                {"ppl_status": "OK", "ppl_loss": 2.0, "ppl_tokens": 30},
            ]
        )

        self.assertEqual(result["ppl_status"], "OK")
        self.assertEqual(result["ppl_loss"], 1.75)
        self.assertEqual(result["ppl_tokens"], 40)

    def test_backfill_rows_populates_ppl_and_quality_fields(self):
        rows = [
            {
                "model": "model-a",
                "kv_cache_type": "FP16",
                "context_length": "8000",
                "sample_id": "model-a__FP16__8000",
                "output_path": "run_generated.jsonl",
            }
        ]
        records = [
            {
                "sample_id": "model-a__FP16__8000__s0",
                "model": "model-a",
                "kv_cache_type": "FP16",
                "context_length": 8000,
                "prompt_text": "Xin chào",
                "generated_text": "Đây là một câu trả lời bình thường.",
            }
        ]

        def fake_ppl(record):
            return {
                "perplexity": 2.7183,
                "ppl_loss": 1.0,
                "ppl_tokens": 5,
                "ppl_status": "OK",
                "ppl_error": "",
            }

        output_rows = backfill_rows(rows, records, "run_generated.jsonl", fake_ppl)

        self.assertEqual(output_rows[0]["ppl_status"], "OK")
        self.assertEqual(output_rows[0]["perplexity"], "2.7183")
        self.assertEqual(output_rows[0]["repetition_flag"], "false")
        self.assertIn("quality_warning", output_rows[0])

    def test_csv_io_adds_backfill_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.csv"
            output = Path(tmpdir) / "output.csv"
            write_csv_rows(
                path,
                ["model", "sample_id"],
                [{"model": "model-a", "sample_id": "s0"}],
            )
            fieldnames, rows = read_csv_rows(path)
            fieldnames = ensure_output_fields(fieldnames)
            write_csv_rows(output, fieldnames, rows)

            header = output.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("ppl_status", header)
            self.assertIn("quality_warning", header)


if __name__ == "__main__":
    unittest.main()

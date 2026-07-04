import unittest

from scripts.utils_csv import CANONICAL_LOG_FIELDS, normalize_log_row


class TestCsvSchema(unittest.TestCase):
    def test_canonical_header_order(self):
        expected = [
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
        self.assertEqual(CANONICAL_LOG_FIELDS, expected)

    def test_normalize_log_row_removes_sentinels_from_numeric_fields(self):
        row = normalize_log_row(
            {
                "model": "x",
                "dataset": "y",
                "sample_id": 1,
                "kv_cache_type": "FP16",
                "kv_cache_dtype": "auto",
                "context_length": 8000,
                "peak_memory_mb": "OOM",
                "latency_ms_per_token": "ERROR",
                "throughput_tokens_per_s": "N/A",
                "generated_tokens": "10",
                "perplexity": "inf",
                "ppl_loss": "nan",
                "ppl_tokens": "5",
                "ppl_status": "OK",
                "ppl_error": "",
                "repetition_flag": True,
                "gibberish_flag": False,
                "repeated_ngram_ratio": 0.5,
                "special_char_ratio": 0.2,
                "output_length": 12,
                "quality_warning": "",
                "status": "OK",
                "error_message": "",
            }
        )
        self.assertEqual(row["peak_memory_mb"], "")
        self.assertEqual(row["latency_ms_per_token"], "")
        self.assertEqual(row["throughput_tokens_per_s"], "")
        self.assertEqual(row["perplexity"], "")
        self.assertEqual(row["ppl_loss"], "")
        self.assertEqual(row["ppl_tokens"], "5")
        self.assertEqual(row["repetition_flag"], "true")
        self.assertEqual(row["gibberish_flag"], "false")


if __name__ == "__main__":
    unittest.main()

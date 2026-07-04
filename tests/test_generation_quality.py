import unittest

from scripts.utils_generation_quality import analyze_generated_text


class GenerationQualityTests(unittest.TestCase):
    def test_normal_text_has_no_warning(self):
        result = analyze_generated_text("Đây là một câu trả lời tiếng Việt bình thường.")

        self.assertFalse(result["repetition_flag"])
        self.assertFalse(result["gibberish_flag"])
        self.assertEqual(result["quality_warning"], "")

    def test_empty_text_is_flagged(self):
        result = analyze_generated_text("   ")

        self.assertFalse(result["repetition_flag"])
        self.assertTrue(result["gibberish_flag"])
        self.assertEqual(result["quality_warning"], "empty_output;gibberish")

    def test_repeated_text_is_flagged(self):
        result = analyze_generated_text("xin chào xin chào xin chào xin chào xin chào")

        self.assertTrue(result["repetition_flag"])
        self.assertIn("repetition", result["quality_warning"])

    def test_symbol_heavy_text_is_flagged(self):
        result = analyze_generated_text("!!!! #### $$$$ %%%% ^^^^")

        self.assertTrue(result["gibberish_flag"])
        self.assertIn("gibberish", result["quality_warning"])


if __name__ == "__main__":
    unittest.main()

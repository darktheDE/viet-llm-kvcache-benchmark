import unittest

from scripts.utils_generation_quality import analyze_generated_text


class TestGenerationQuality(unittest.TestCase):
    def test_normal_case(self):
        result = analyze_generated_text("Xin chao viet nam day la mot cau binh thuong.")
        self.assertFalse(result["repetition_flag"])
        self.assertFalse(result["gibberish_flag"])
        self.assertGreater(result["output_length"], 0)

    def test_repetition_case(self):
        result = analyze_generated_text("hello hello hello hello hello hello")
        self.assertTrue(result["repetition_flag"])
        self.assertIn("repetition", result["quality_warning"])

    def test_gibberish_case(self):
        result = analyze_generated_text("!!!! #### $$$$ %%%% ^^^^")
        self.assertTrue(result["gibberish_flag"])
        self.assertIn("gibberish", result["quality_warning"])

    def test_empty_case(self):
        result = analyze_generated_text("   ")
        self.assertTrue(result["gibberish_flag"])
        self.assertEqual(result["quality_warning"], "empty_output;gibberish")


if __name__ == "__main__":
    unittest.main()

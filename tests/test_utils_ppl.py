import unittest

from scripts.utils_ppl import compute_perplexity


class DummyTokenizer:
    model_max_length = 16

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        token_count = max(1, len(text.split()))
        return {"input_ids": FakeTensor(token_count)}


class FakeTensor:
    def __init__(self, token_count):
        self._token_count = token_count

    def to(self, device):
        return self

    def size(self, dim=None):
        if dim == 1:
            return self._token_count
        if dim is None:
            return (1, self._token_count)
        return 1

    def clone(self):
        return self

    def __getitem__(self, item):
        return self

    def __setitem__(self, key, value):
        return None


class DummyModel:
    class _Config:
        max_position_embeddings = 16

    def __init__(self, loss_value=1.5):
        self.config = self._Config()
        self.loss_value = loss_value

    def eval(self):
        return self

    def __call__(self, input_ids=None, labels=None, attention_mask=None):
        class Output:
            def __init__(self, loss):
                self.loss = loss

        class FakeLoss:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def float(self):
                return self

            def item(self):
                return self.value

        return Output(FakeLoss(self.loss_value))


class TestUtilsPpl(unittest.TestCase):
    def test_empty_text(self):
        result = compute_perplexity(None, None, "", device="cpu")
        self.assertEqual(result["ppl_status"], "EMPTY")
        self.assertIsNone(result["perplexity"])

    def test_smoke(self):
        model = DummyModel(loss_value=1.5)
        tokenizer = DummyTokenizer()
        result = compute_perplexity(model, tokenizer, "xin chao ban", device="cpu", max_length=8, stride=4)
        self.assertEqual(result["ppl_status"], "OK")
        self.assertIsNotNone(result["perplexity"])


if __name__ == "__main__":
    unittest.main()

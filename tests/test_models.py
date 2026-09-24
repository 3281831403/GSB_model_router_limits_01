import dataclasses
import unittest

from model_router import Limits, ModelInfo

RAW = {
    "id": "anthropic/claude-opus-5.5",
    "context_length": 1_000_000,
    "pricing": {"prompt": "0.000004", "completion": "0.00002"},
    "top_provider": {"max_completion_tokens": 128_000},
    "description": "Flagship model.\nGreat at code.",
    "created": 1_790_000_000,
    "expiration_date": None,
}


class ModelInfoTest(unittest.TestCase):
    def test_parses_openrouter_entry(self):
        m = ModelInfo.from_openrouter(RAW)
        self.assertEqual(m.id, "anthropic/claude-opus-5.5")
        self.assertEqual(m.context_length, 1_000_000)
        self.assertEqual(m.max_output_tokens, 128_000)
        self.assertAlmostEqual(m.prompt_price, 4e-6)
        self.assertAlmostEqual(m.completion_price, 2e-5)
        self.assertEqual(m.created, 1_790_000_000)
        self.assertIsNone(m.expires)

    def test_description_newlines_are_flattened(self):
        self.assertEqual(ModelInfo.from_openrouter(RAW).description, "Flagship model. Great at code.")

    def test_negative_variable_price_is_treated_as_zero(self):
        m = ModelInfo.from_openrouter({**RAW, "pricing": {"prompt": "-1", "completion": "-1"}})
        self.assertEqual((m.prompt_price, m.completion_price), (0.0, 0.0))

    def test_missing_fields_get_safe_defaults(self):
        m = ModelInfo.from_openrouter({"id": "x/y"})
        self.assertEqual(m.context_length, 0)
        self.assertIsNone(m.max_output_tokens)
        self.assertEqual((m.prompt_price, m.completion_price), (0.0, 0.0))
        self.assertEqual(m.description, "")

    def test_expiration_date_is_kept(self):
        m = ModelInfo.from_openrouter({**RAW, "expiration_date": "2026-12-01"})
        self.assertEqual(m.expires, "2026-12-01")

    def test_cost_uses_input_and_output_prices(self):
        m = ModelInfo.from_openrouter(RAW)
        self.assertAlmostEqual(m.cost(1_000, 500), 1_000 * 4e-6 + 500 * 2e-5)

    def test_is_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ModelInfo.from_openrouter(RAW).context_length = 1


class LimitsTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(Limits(), Limits(output_tokens=1024, max_cost_usd=None))

    def test_is_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Limits().output_tokens = 5


if __name__ == "__main__":
    unittest.main()

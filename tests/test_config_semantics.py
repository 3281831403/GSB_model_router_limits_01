import unittest
from unittest.mock import patch

from model_router import Limits, ModelInfo, NoModelFitsError, Router

CATALOG = {
    "cheap/small": ModelInfo("cheap/small", 8_000, 2_000, 1e-7, 4e-7, "small fast model"),
    "big/smart": ModelInfo("big/smart", 1_000_000, 64_000, 5e-6, 2.5e-5, "frontier model"),
    "acme/new": ModelInfo("acme/new", 128_000, None, 1e-6, 1e-6, "", created=300),
    "acme/old": ModelInfo("acme/old", 128_000, None, 1e-6, 1e-6, "", created=100),
    "mystery/unknown": ModelInfo(
        "mystery/unknown", 0, None, 1e-6, 1e-6, "", context_length_known=False
    ),
}


def make_router(**kw):
    kw.setdefault("openrouter_api_key", "or-key")
    with patch("model_router.router.fetch_catalog", return_value=CATALOG):
        return Router(**kw)


class LimitsValidationTest(unittest.TestCase):
    def test_negative_output_tokens_rejected_with_field_and_value(self):
        with self.assertRaises(ValueError) as ctx:
            Limits(output_tokens=-5)
        self.assertIn("output_tokens", str(ctx.exception))
        self.assertIn("-5", str(ctx.exception))

    def test_zero_output_tokens_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            Limits(output_tokens=0)
        self.assertIn("output_tokens", str(ctx.exception))
        self.assertIn("0", str(ctx.exception))

    def test_positive_output_tokens_accepted(self):
        self.assertEqual(Limits(output_tokens=7).output_tokens, 7)

    def test_negative_max_cost_rejected_with_field_and_value(self):
        with self.assertRaises(ValueError) as ctx:
            Limits(max_cost_usd=-1.0)
        self.assertIn("max_cost_usd", str(ctx.exception))
        self.assertIn("-1.0", str(ctx.exception))

    def test_none_zero_and_positive_max_cost_accepted(self):
        self.assertIsNone(Limits().max_cost_usd)
        self.assertEqual(Limits(max_cost_usd=0.0).max_cost_usd, 0.0)
        self.assertEqual(Limits(max_cost_usd=0.5).max_cost_usd, 0.5)


class UnknownContextTest(unittest.TestCase):
    def test_missing_context_length_is_flagged_unknown(self):
        m = ModelInfo.from_openrouter({"id": "x/y"})
        self.assertFalse(m.context_length_known)

    def test_present_context_length_is_known(self):
        m = ModelInfo.from_openrouter({"id": "x/y", "context_length": 4096})
        self.assertTrue(m.context_length_known)
        self.assertEqual(m.context_length, 4096)

    def test_explicit_zero_context_length_is_known(self):
        m = ModelInfo.from_openrouter({"id": "x/y", "context_length": 0})
        self.assertTrue(m.context_length_known)
        self.assertEqual(m.context_length, 0)

    def test_unknown_context_rejection_says_unknown_not_zero(self):
        r = make_router(models=["mystery/unknown"])
        with self.assertRaises(NoModelFitsError) as ctx:
            r.route("hi")
        msg = str(ctx.exception)
        self.assertIn("unknown", msg.lower())
        self.assertNotIn("context 0", msg)

    def test_unknown_context_is_not_silently_passed(self):
        r = make_router(models=["mystery/unknown"])
        self.assertEqual(r.fitting("hi"), [])

    def test_known_context_judgement_unchanged(self):
        r = make_router(models=["cheap/small", "big/smart"])
        self.assertEqual([m.id for m in r.fitting("x" * 40_000)], ["big/smart"])
        with self.assertRaises(NoModelFitsError) as ctx:
            make_router(models=["cheap/small"]).route("x" * 40_000)
        self.assertIn("context 8000", str(ctx.exception))


class DedupTest(unittest.TestCase):
    def test_duplicate_model_ids_deduped_first_seen_order(self):
        r = make_router(models=["cheap/small", "big/smart", "cheap/small"])
        self.assertEqual([m.id for m in r.models], ["cheap/small", "big/smart"])

    def test_duplicate_provider_names_expand_once(self):
        r = make_router(models=None, providers=["acme", "acme"])
        self.assertEqual([m.id for m in r.models], ["acme/new", "acme/old"])

    def test_overlapping_providers_dedupe_expanded_models(self):
        r = make_router(models=None, providers=["acme", "big", "acme"])
        self.assertEqual([m.id for m in r.models], ["acme/new", "acme/old", "big/smart"])

    def test_models_only_config(self):
        r = make_router(models=["big/smart"])
        self.assertEqual([m.id for m in r.models], ["big/smart"])

    def test_providers_only_config(self):
        r = make_router(models=None, providers=["acme"])
        self.assertEqual([m.id for m in r.models], ["acme/new", "acme/old"])

    def test_models_and_providers_together_models_win(self):
        r = make_router(models=["big/smart", "big/smart"], providers=["acme"])
        self.assertEqual([m.id for m in r.models], ["big/smart"])


if __name__ == "__main__":
    unittest.main()

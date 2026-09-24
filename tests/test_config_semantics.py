import unittest
from unittest.mock import patch

from model_router import Limits, ModelInfo, NoModelFitsError, Router


def info(id, *, context=128_000, context_known=True, output=64_000):
    return ModelInfo(
        id,
        context,
        output,
        0.0,
        0.0,
        "",
        context_length_known=context_known,
    )


CATALOG = {
    "a/one": info("a/one"),
    "a/two": info("a/two"),
    "big/smart": info("big/smart", context=1_000_000),
}

OVERLAP = {
    "shared/m": info("shared/m"),
    "a/n": info("a/n"),
    "b/o": info("b/o"),
}


def make_router(*, models=None, providers=None, limits=None):
    with patch("model_router.router.fetch_catalog", return_value=CATALOG):
        return Router(
            openrouter_api_key="k",
            models=models,
            providers=providers,
            limits=limits or Limits(),
        )


class LimitsValidationTest(unittest.TestCase):
    def test_negative_output_tokens_rejected(self):
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
        self.assertEqual(Limits(output_tokens=128).output_tokens, 128)

    def test_defaults_unchanged(self):
        self.assertEqual(Limits(), Limits(output_tokens=1024, max_cost_usd=None))

    def test_negative_cost_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            Limits(max_cost_usd=-1.0)
        self.assertIn("max_cost_usd", str(ctx.exception))
        self.assertIn("-1.0", str(ctx.exception))

    def test_zero_cost_accepted(self):
        self.assertEqual(Limits(max_cost_usd=0.0).max_cost_usd, 0.0)

    def test_none_cost_accepted(self):
        self.assertIsNone(Limits(max_cost_usd=None).max_cost_usd)

    def test_positive_cost_accepted(self):
        limits = Limits(output_tokens=256, max_cost_usd=1.5)
        self.assertEqual((limits.output_tokens, limits.max_cost_usd), (256, 1.5))


class UnknownContextTest(unittest.TestCase):
    def test_missing_context_field_is_unknown_not_zero_metadata(self):
        m = ModelInfo.from_openrouter({"id": "u/m"})
        self.assertEqual(m.context_length, 0)
        self.assertFalse(m.context_length_known)

    def test_explicit_zero_context_is_known(self):
        m = ModelInfo.from_openrouter({"id": "z/m", "context_length": 0})
        self.assertEqual(m.context_length, 0)
        self.assertTrue(m.context_length_known)

    def test_explicit_context_is_known(self):
        m = ModelInfo.from_openrouter({"id": "k/m", "context_length": 200_000})
        self.assertEqual(m.context_length, 200_000)
        self.assertTrue(m.context_length_known)

    def test_unknown_context_rejection_reason_says_unknown(self):
        unknown = info("u/m", context_known=False)
        catalog = {**CATALOG, "u/m": unknown}
        with patch("model_router.router.fetch_catalog", return_value=catalog):
            r = Router(openrouter_api_key="k", models=["u/m"])
        with self.assertRaises(NoModelFitsError) as ctx:
            r.route("hi")
        msg = str(ctx.exception)
        self.assertIn("context length unknown", msg)
        self.assertNotIn("context 0", msg)

    def test_unknown_context_model_is_excluded_from_fitting(self):
        unknown = info("u/m", context_known=False)
        catalog = {**CATALOG, "u/m": unknown}
        with patch("model_router.router.fetch_catalog", return_value=catalog):
            r = Router(openrouter_api_key="k", models=["u/m", "big/smart"])
        self.assertEqual([m.id for m in r.fitting("hi")], ["big/smart"])

    def test_explicit_zero_context_rejected_like_before(self):
        zero = info("z/m", context=0, context_known=True)
        catalog = {**CATALOG, "z/m": zero}
        with patch("model_router.router.fetch_catalog", return_value=catalog):
            r = Router(openrouter_api_key="k", models=["z/m"])
        with self.assertRaises(NoModelFitsError) as ctx:
            r.route("hi")
        self.assertIn("context 0 <", str(ctx.exception))

    def test_explicit_known_context_passes_like_before(self):
        r = make_router(models=["big/smart"])
        self.assertEqual([m.id for m in r.fitting("hi")], ["big/smart"])


def overlap_provider_models(catalog, provider, limit=None):
    return {
        "a": [OVERLAP["shared/m"], OVERLAP["a/n"]],
        "b": [OVERLAP["shared/m"], OVERLAP["b/o"]],
    }[provider]


class DedupTest(unittest.TestCase):
    def test_duplicate_model_ids_collapse_keeping_first_order(self):
        r = make_router(models=["a/one", "a/two", "a/one"])
        self.assertEqual([m.id for m in r.models], ["a/one", "a/two"])

    def test_dedup_keeps_first_seen_order(self):
        r = make_router(models=["a/two", "a/one", "a/two"])
        self.assertEqual([m.id for m in r.models], ["a/two", "a/one"])

    def test_unknown_models_still_detected_after_dedup(self):
        from model_router import UnknownModelError

        with self.assertRaises(UnknownModelError):
            make_router(models=["nope/m", "nope/m"])

    @patch("model_router.router.provider_models", side_effect=overlap_provider_models)
    @patch("model_router.router.fetch_catalog", return_value=OVERLAP)
    def test_duplicate_provider_names_expand_once(self, fetch, providers):
        r = Router(openrouter_api_key="k", providers=["a", "a"])
        self.assertEqual([m.id for m in r.models], ["shared/m", "a/n"])

    @patch("model_router.router.provider_models", side_effect=overlap_provider_models)
    @patch("model_router.router.fetch_catalog", return_value=OVERLAP)
    def test_models_expanded_from_providers_are_deduped(self, fetch, providers):
        r = Router(openrouter_api_key="k", providers=["a", "b"])
        self.assertEqual([m.id for m in r.models], ["shared/m", "a/n", "b/o"])

    @patch("model_router.jev.request_json", return_value={"answers": {"model": {"choice": "m0"}}})
    def test_deduped_candidates_sent_to_ranking_service(self, req):
        r = make_router(models=["a/one", "a/two", "a/one"])
        r.route("hi")
        criteria = req.call_args.args[2]["questions"]["model"]["criteria"]
        self.assertEqual(set(criteria), {"m0", "m1"})


class ConfigShapesTest(unittest.TestCase):
    def test_models_only(self):
        r = make_router(models=["a/one"])
        self.assertEqual([m.id for m in r.models], ["a/one"])

    @patch("model_router.router.provider_models", side_effect=overlap_provider_models)
    @patch("model_router.router.fetch_catalog", return_value=OVERLAP)
    def test_providers_only(self, fetch, providers):
        r = Router(openrouter_api_key="k", providers=["b"])
        self.assertEqual([m.id for m in r.models], ["shared/m", "b/o"])

    @patch("model_router.router.provider_models", side_effect=overlap_provider_models)
    @patch("model_router.router.fetch_catalog", return_value={**CATALOG, **OVERLAP})
    def test_models_and_providers_models_win(self, fetch, providers):
        r = Router(openrouter_api_key="k", models=["a/one"], providers=["a"])
        self.assertEqual([m.id for m in r.models], ["a/one"])


if __name__ == "__main__":
    unittest.main()

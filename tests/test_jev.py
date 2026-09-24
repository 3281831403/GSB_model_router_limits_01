import json
import unittest
from unittest.mock import patch

from model_router import Limits, ModelInfo, RouterError, jev

CHEAP = ModelInfo("cheap/small", 8_000, 2_000, 1e-7, 4e-7, "small fast model")
BIG = ModelInfo("big/smart", 1_000_000, None, 5e-6, 2.5e-5, "frontier model " * 50)


class DescribeTest(unittest.TestCase):
    def test_includes_live_pricing_context_and_task_cost(self):
        line = jev._describe(CHEAP, 1_000, Limits(output_tokens=500), 200)
        self.assertIn("cheap/small", line)
        self.assertIn("$0.10/M in, $0.40/M out", line)
        self.assertIn("context 8000", line)
        self.assertIn("max output 2000", line)
        self.assertIn(f"est. ${CHEAP.cost(1_000, 500):.6f}", line)
        self.assertIn("small fast model", line)

    def test_unknown_max_output(self):
        self.assertIn("max output n/a", jev._describe(BIG, 1, Limits(), 0))

    def test_description_truncated_or_dropped(self):
        self.assertLessEqual(jev._describe(BIG, 1, Limits(), 40).count("frontier"), 4)
        self.assertNotIn("frontier", jev._describe(BIG, 1, Limits(), 0))


class FitTest(unittest.TestCase):
    def test_uses_longest_descriptions_that_fit(self):
        body = jev._fit(lambda chars: {"x": "a" * chars}, 1)
        self.assertEqual(body, {"x": "a" * jev.DESCRIPTION_CHARS[0]})

    def test_shrinks_when_too_big(self):
        def build(chars):
            return {"x": "a" * (jev.MAX_BODY_BYTES - 50 + chars)}

        body = jev._fit(build, 1)
        self.assertLessEqual(len(json.dumps(body).encode()), jev.MAX_BODY_BYTES)

    def test_raises_when_nothing_fits(self):
        with self.assertRaises(RouterError):
            jev._fit(lambda chars: {"x": "a" * (jev.MAX_BODY_BYTES + 1)}, 999)


class ChooseViaJevTest(unittest.TestCase):
    @patch("model_router.jev.request_json", return_value={"code": 0, "data": {"decision": "big/smart"}})
    def test_request_shape(self, req):
        self.assertEqual(jev.choose_via_jev("k", "do it", 3, [CHEAP, BIG], Limits()), "big/smart")
        url, key, body = req.call_args.args
        self.assertEqual(url, jev.JEV_ROUTE_URL)
        self.assertEqual(key, "k")
        self.assertIn("do it", body["task"])
        self.assertIn("~3 input tokens", body["task"])
        self.assertEqual(len(body["priorities"]), 2)
        self.assertEqual({c["id"] for c in body["candidates"]}, {"cheap/small", "big/smart"})

    @patch("model_router.jev.request_json", return_value={"code": 0, "data": {"decision": "someone/else"}})
    def test_decision_outside_candidates_is_rejected(self, req):
        with self.assertRaises(RouterError):
            jev.choose_via_jev("k", "t", 1, [CHEAP, BIG], Limits())

    @patch("model_router.jev.request_json", return_value={"code": 0, "data": None})
    def test_missing_data_is_rejected(self, req):
        with self.assertRaises(RouterError):
            jev.choose_via_jev("k", "t", 1, [CHEAP, BIG], Limits())


class ChooseViaOpenRouterTest(unittest.TestCase):
    @patch("model_router.jev.request_json", return_value={"answers": {"model": {"choice": "m0"}}})
    def test_request_shape(self, req):
        self.assertEqual(jev.choose_via_openrouter("k", "do it", 3, [CHEAP, BIG], Limits()), "cheap/small")
        url, _key, body = req.call_args.args
        self.assertEqual(url, jev.OPENROUTER_DECISIONS_URL)
        self.assertEqual(body["model"], jev.OPENROUTER_JEV_MODEL)
        q = body["questions"]["model"]
        self.assertEqual(q["type"], "choice")
        self.assertIn("cheap/small", q["criteria"]["m0"])
        self.assertIn("big/smart", q["criteria"]["m1"])

    @patch("model_router.jev.request_json", return_value={"answers": {"model": {"choice": "m9"}}})
    def test_unknown_alias_is_rejected(self, req):
        with self.assertRaises(RouterError):
            jev.choose_via_openrouter("k", "t", 1, [CHEAP, BIG], Limits())

    @patch("model_router.jev.request_json", return_value={"error": "boom"})
    def test_missing_answers_is_rejected(self, req):
        with self.assertRaises(RouterError):
            jev.choose_via_openrouter("k", "t", 1, [CHEAP, BIG], Limits())

    @patch("model_router.jev.request_json", return_value={"answers": {"model": {"choice": "m0"}}})
    def test_task_is_truncated(self, req):
        jev.choose_via_openrouter("k", "z" * 50_000, 1, [CHEAP, BIG], Limits())
        self.assertEqual(req.call_args.args[2]["state"].count("z"), jev.ROUTING_CHARS)


if __name__ == "__main__":
    unittest.main()

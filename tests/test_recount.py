"""Offline tests: no network, no API key, no spend. Run: python -m unittest -v  (or pytest)."""
import json
import os
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from recount import otari
from recount.otari import BudgetExceeded, OtariClient, OtariError, OtariResult, RateLimited, RoutingRejected, classify_error
from recount.pipeline import Recount, code_runs, majority, parse_json, parse_vote
from recount.replay import SAMPLES_DIR, Gateway, slugify

PROBES = SAMPLES_DIR / "probes"


def load_probe(name):
    return json.loads((PROBES / f"{name}.json").read_text())


class FakeGateway(Gateway):
    """Answers each step from a dict of OtariResult or exceptions."""

    def __init__(self, answers):
        super().__init__("replay", Path("/nonexistent"))
        self.mode = "live"
        self.answers = answers
        self.requests = {}

    def call(self, step, kind, request):
        self.requests[step] = request
        a = self.answers[step]
        if isinstance(a, Exception):
            raise a
        self.spent_usd += a.cost_usd
        return a


def chat(content, cost="0.001", n=1):
    return OtariResult({"choices": [{"message": {"content": c}} for c in ([content] * n if isinstance(content, str) else content)],
                        "usage": {"cost_usd": cost}}, {"Otari-Model": "test-model"})


class TestParsing(unittest.TestCase):
    def test_parse_json_with_fence_and_chatter(self):
        self.assertEqual(parse_json('Sure!\n```json\n{"a": {"b": "}"}}\n```'), {"a": {"b": "}"}})

    def test_parse_json_rejects_prose(self):
        with self.assertRaises(ValueError):
            parse_json("no json here")

    def test_parse_vote(self):
        self.assertEqual(parse_vote("VERDICT: FALSE - wrong"), "FALSE")
        self.assertEqual(parse_vote("**Verdict:** misleading"), "MISLEADING")
        self.assertEqual(parse_vote("I am not sure"), "UNVERIFIABLE")

    def test_majority(self):
        self.assertEqual(majority(["FALSE", "FALSE", "ACCURATE"]), ("FALSE", 2))
        self.assertEqual(majority(["FALSE", "ACCURATE"]), ("UNVERIFIABLE", 1))
        self.assertEqual(majority([]), ("UNVERIFIABLE", 0))

    def test_slugify(self):
        self.assertEqual(slugify("India's GDP: 7%!"), "india-s-gdp-7")


class TestRecordedOtariResponses(unittest.TestCase):
    def test_code_interpreter_items_are_the_executed_code(self):
        runs = code_runs(load_probe("code_interpreter_sandbox")["response"])
        self.assertEqual(len(runs), 1)
        self.assertIn("cagr", runs[0]["code"])
        self.assertIn("0.08706202972134314", runs[0]["output"])

    def test_chat_answer_with_code_in_text_is_not_a_sandbox_run(self):
        # deepseek wrote "code" and a (wrong) result without running anything.
        probe = load_probe("fake_code_run_deepseek")["response"]
        self.assertEqual(code_runs(probe), [])
        self.assertIn("0.08711682932518143", probe["choices"][0]["message"]["content"])

    def test_routing_policy_fallback_headers(self):
        p = load_probe("routing_policy_n3_fallback")
        r = OtariResult.from_json(p)
        self.assertTrue(r.fell_back)
        self.assertEqual(len(r.body["choices"]), 3)


class TestErrors(unittest.TestCase):
    def test_budget_402(self):
        e = classify_error(402, b'{"detail": "Budget exceeded for member", "code": "budget_exceeded"}')
        self.assertIsInstance(e, BudgetExceeded)
        self.assertIn("00:00 UTC", str(e))
        self.assertIn("--replay", str(e))

    def test_insufficient_funds(self):
        self.assertIsInstance(classify_error(402, b'{"detail": "no funds", "code": "insufficient_funds"}'), BudgetExceeded)

    def test_budget_in_403_detail(self):
        self.assertIsInstance(classify_error(403, b'{"detail": "Daily budget limit reached"}'), BudgetExceeded)

    def test_rate_limit(self):
        self.assertIsInstance(classify_error(429, b"slow down"), RateLimited)

    def test_routing_rejected_from_recording(self):
        body = json.dumps(load_probe("routing_policy_all_rejected")["response"])
        self.assertIsInstance(classify_error(400, body), RoutingRejected)

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {"OTARI_API_KEY": ""}):
            with self.assertRaises(OtariError) as cm:
                OtariClient(api_key="").chat("m", [{"role": "user", "content": "hi"}])
        self.assertEqual(cm.exception.code, "no_api_key")

    def test_base_url_fixed(self):
        self.assertEqual(OtariClient(api_key="x", base_url="https://api.otari.ai/v1").base_url, "https://api.otari.ai/api/v1")
        self.assertEqual(OtariClient(api_key="x").base_url, otari.DEFAULT_BASE_URL)

    def test_http_error_is_classified(self):
        import io
        import urllib.error
        err = urllib.error.HTTPError("u", 402, "Payment Required", {}, io.BytesIO(b'{"detail":"Budget exceeded"}'))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(BudgetExceeded):
                OtariClient(api_key="x").chat("m", [])


class TestPipeline(unittest.TestCase):
    def test_replay_full_audit(self):
        audit = Recount(Gateway.replay("india-population")).run(
            (SAMPLES_DIR / "india-population" / "claim.txt").read_text().strip())
        self.assertEqual([s.name for s in audit.steps], ["extract", "source", "recompute", "verdict"])
        self.assertFalse(audit.stopped)
        self.assertEqual(audit.verdict, "FALSE")
        rec = audit.steps[2].data
        self.assertTrue(rec["sandbox_verified"])
        self.assertIn("8.33", rec["runs"][0]["output"])
        self.assertGreater(audit.spent_usd, 0)

    def test_budget_error_stops_gracefully(self):
        gw = FakeGateway({
            "step1_extract": chat('{"figures": [{"label": "x", "value": 1}]}'),
            "step2_source": classify_error(402, b'{"detail":"Budget exceeded","code":"budget_exceeded"}'),
        })
        audit = Recount(gw).run("x grew 1 to 2")
        self.assertEqual(audit.verdict, "UNVERIFIABLE")
        self.assertIn("budget", audit.stopped.lower())
        self.assertFalse(audit.steps[-1].ok)

    def test_local_spend_cap(self):
        gw = FakeGateway({"step1_extract": chat('{"figures": []}', cost="0.5")})
        audit = Recount(gw, max_spend_usd=0.10).run("claim")
        self.assertEqual(len(audit.steps), 1)
        self.assertIn("max-spend", audit.stopped)

    def test_unverified_sandbox_and_requests(self):
        no_run = OtariResult({"output": [{"type": "message", "content": [{"type": "output_text", "text": "it is 8%"}]}],
                              "usage": {"cost_usd": "0.0001"}}, {})
        gw = FakeGateway({
            "step1_extract": chat('{"figures": [{"label": "pop", "value": 1.32e9, "year": 2015}]}'),
            "step2_source": chat('{"official": [{"label": "pop", "value": 1.31e9, "year": 2015, "url": "https://x"}]}'),
            "step3_recompute": no_run,
            "step4_verdict": chat(["VERDICT: FALSE - a", "VERDICT: FALSE - b", "VERDICT: ACCURATE - c"]),
        })
        audit = Recount(gw).run("claim")
        self.assertEqual(audit.verdict, "FALSE")
        self.assertEqual(audit.agreement, "2/3")
        self.assertIn("not trusted", audit.steps[2].data["warning"])
        # Requests carry the Otari features we rely on.
        self.assertEqual(gw.requests["step4_verdict"]["model"], "recount-auditor")
        self.assertEqual(gw.requests["step4_verdict"]["n"], 3)
        self.assertEqual(len(gw.requests["step2_source"]["mcp_server_ids"]), 2)
        self.assertEqual(gw.requests["step2_source"]["tools"][0]["type"], "otari_web_search")
        self.assertEqual(gw.requests["step3_recompute"]["tools"][0]["type"], "code_interpreter")
        self.assertFalse(audit.steps[1].data["official"][0]["url_seen_in_search"])

    def test_search_fallback_when_mcp_finds_nothing(self):
        ws = load_probe("web_search_worldbank")
        fb = OtariResult({"choices": [{"message": {"content": '{"official": [{"label": "pop", "value": 1428627663, "year": 2023, '
                                                              '"url": "https://data.worldbank.org/indicator/SP.POP.GROW?locations=IN"}]}'}}],
                          "usage": {"cost_usd": "0.02"}, "otari_tool_calls": ws["response"]["otari_tool_calls"]}, {})
        gw = FakeGateway({
            "step2_source": chat('{"official": [{"label": "pop", "value": null, "year": 2023}], "notes": "tool failed"}'),
            "step2b_search": fb,
        })
        step = Recount(gw).source("claim", {"figures": []})
        self.assertEqual(step.data["official"][0]["value"], 1428627663)
        self.assertTrue(step.data["official"][0]["url_seen_in_search"])
        self.assertTrue(step.data["web_searches"])
        self.assertAlmostEqual(step.cost_usd, 0.021, places=6)


class TestServer(unittest.TestCase):
    def test_replay_endpoint(self):
        from recount.server import make_handler
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(True, "india-population"))
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # ignore any HTTP(S)_PROXY
            self.assertIn(b"Recount", opener.open(base + "/").read())
            req = urllib.request.Request(base + "/api/audit", data=b'{"claim": ""}', method="POST",
                                         headers={"Content-Type": "application/json"})
            data = json.loads(opener.open(req).read())
            self.assertEqual(data["verdict"], "FALSE")
            self.assertEqual(data["mode"], "replay")
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main()

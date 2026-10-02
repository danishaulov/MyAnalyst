"""Regression tests for the live HTTP boundary, pandas queries, and provider failures."""
import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

import _groq
import index
from _ask import answer_question, _schema, _validate_plan
from _conclude import generate_conclusions, _result_from_raw
from _grounding import check_grounding
from _validation import dataframe, InputError
from _engine import profile


def conclusion(number):
    return json.dumps({"bottomLine": f"Revenue is ${number}.", "summary": f"Revenue is ${number}.",
                       "conclusions": [], "chartInsights": [], "actions": []})


class GroundingTests(unittest.TestCase):
    def test_scale_units_small_numbers_and_years(self):
        sources = [{"text": "Revenue $51.4M, margin 0.2%, 3 orders, in 2024."}]
        self.assertTrue(check_grounding("Revenue $51,400,000, margin 0.2%, 3 orders in 2024.", sources)["grounded"])
        for answer in ("Revenue $51.4B.", "Margin 0.8%.", "9 orders.", "In 2028.", "Revenue €51.4M.", "Margin 51.4%."):
            with self.subTest(answer=answer):
                self.assertFalse(check_grounding(answer, sources)["grounded"])

    def test_failed_repair_never_escapes(self):
        with patch("_groq.chat", side_effect=[conclusion(999), conclusion(888)]):
            result = generate_conclusions([{"text": "Revenue $100."}])
        self.assertEqual(result["provider"], "none")
        self.assertNotIn("888", json.dumps(result))
        self.assertTrue(result["grounding"]["grounded"])

    def test_successful_repair_and_schema_rejection(self):
        with patch("_groq.chat", side_effect=[conclusion(999), conclusion(100)]):
            result = generate_conclusions([{"text": "Revenue $100."}])
        self.assertEqual(result["bottomLine"], "Revenue is $100.")
        for malformed in ("[]", "null", '{"bottomLine": 123}', '{"bottomLine":"ok","summary":"ok","actions":[null]}',
                          '{"bottomLine":"ok","summary":"ok","conclusions":[123]}'):
            self.assertIsNone(_result_from_raw(malformed, []))


class AskTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"Revenue": [10, 30, 20, 80], "Cost": [3, 4, 5, 6],
                                "Region": ["East", "West", "East", "West"],
                                "Email": ["private1@example.com", "private2@example.com", "private3@example.com", "private4@example.com"]})

    def test_exact_grouped_sum_and_mean_without_ai(self):
        with patch("_groq.chat", return_value=None):
            total = answer_question(self.df, "total Revenue by Region?")
            mean = answer_question(self.df, "average Revenue by Region?")
        self.assertEqual(total["result"], [{"group": "West", "value": 110}, {"group": "East", "value": 30}])
        self.assertEqual(mean["result"], [{"group": "West", "value": 55}, {"group": "East", "value": 15}])
        self.assertIn("Grouped by Region", mean["method"])

    def test_unknown_metric_or_filter_is_not_a_guessed_total(self):
        with patch("_groq.chat", return_value=None):
            for q in ("total Profit?", "total Revenue in East", "how many customers?", "why did Revenue fall?", "total Revenue last year"):
                self.assertTrue(answer_question(self.df, q)["needsClarification"], q)
            count = answer_question(self.df, "how many rows?")
            self.assertEqual(count["result"], [{"value": 4}])

    def test_schema_only_filtered_plan_and_aggregate_only_narration(self):
        plan = {"op": "sum", "metric": "Revenue", "filters": [{"column": "Region", "op": "eq", "value": "East"}]}
        with patch("_groq.chat", side_effect=[json.dumps(plan), '{"answer":"The total Revenue is 30."}']) as chat:
            result = answer_question(self.df, "total Revenue in East?", [{"text": "Invented profit 999"}])
        self.assertEqual(result["result"], [{"value": 30}])
        self.assertEqual(result["provider"], "groq")
        sent = json.dumps(chat.call_args_list)
        self.assertNotIn("private1@example.com", sent)
        self.assertNotIn("Invented profit", sent)
        self.assertNotIn('"rows"', sent)

    def test_unique_counts_nulls_and_date_filter(self):
        df = pd.DataFrame({"Customer": ["A", "A", "B", None], "Revenue": [10, None, 30, 40],
                           "Date": ["2024-01-01", "2024-01-02", "2025-01-01", "2025-01-02"]})
        plan = {"op": "distinct", "metric": "Customer"}
        with patch("_groq.chat", side_effect=[json.dumps(plan), None]):
            self.assertEqual(answer_question(df, "how many unique Customer values?")["result"], [{"value": 2}])
        plan = {"op": "sum", "metric": "Revenue", "filters": [{"column": "Date", "op": "gte", "value": "2025-01-01"}]}
        with patch("_groq.chat", side_effect=[json.dumps(plan), None]):
            self.assertEqual(answer_question(df, "total Revenue from 2025?")["result"], [{"value": 70}])

    def test_sparse_metric_keeps_its_numeric_type(self):
        profiles = profile(pd.DataFrame({"Revenue":[10, None, None, 20]}))
        self.assertEqual(profiles[0]["role"], "metric")
        self.assertEqual(profiles[0]["fill"], 0.5)
        self.assertEqual(profiles[0]["numeric"]["sum"], 30)

    def test_bad_plans_are_rejected_without_execution(self):
        schema = _schema(self.df)
        for plan in ({"op": "exec", "code": "delete()"}, {"op": ["sum"]}, {"op": "sum", "metric": "Missing"},
                     {"op": "sum", "metric": "Revenue", "groupBy": "Email"},
                     {"op": "sum", "metric": "Revenue", "limit": 999},
                     {"op": "sum", "metric": "Revenue", "order": []},
                     {"op": "sum", "metric": "Revenue", "filters": [{"column":"Revenue","op": [],"value": 1}]}):
            self.assertIsNone(_validate_plan(plan, schema))

    def test_hallucinated_answer_falls_back_to_computed_result(self):
        with patch("_groq.chat", return_value='{"answer":"The total Revenue is 999."}'):
            result = answer_question(self.df, "total Revenue?")
        self.assertEqual(result["provider"], "none")
        self.assertIn("140.00", result["answer"])
        self.assertNotIn("999", result["answer"])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), index.handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, body, path="/api/analyze", headers=None, method="POST"):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
            return response.status, dict(response.getheaders()), json.loads(data) if data else None
        finally:
            connection.close()

    def test_invalid_json_and_dataset_shapes_are_client_errors(self):
        for body in ("[]", "null", "not json", '{"rows": [[1]],"columns":["a","b"]}',
                     '{"rows": [[1,2]],"columns":["a","a"]}', '{"rows": [[{}]],"columns":["a"]}',
                     '{"rows": [[NaN]],"columns":["a"]}', '{"rows": [[1]],"columns":["a"],"sourceRowCount":0}'):
            status, headers, data = self.request(body)
            self.assertEqual(status, 400, body)
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertEqual(data["requestId"], headers["X-Request-ID"])

    def test_limits_routing_content_type_and_cors(self):
        self.assertEqual(self.request("{}", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("{}", path="/api/typo")[0], 404)
        self.assertEqual(self.request("{}", headers={"Content-Type":"application/json", "Content-Length":"4000001"})[0], 413)
        self.assertEqual(self.request("{}", headers={"Content-Type":"application/json", "Origin":"https://untrusted.example"})[0], 403)
        status, headers, _ = self.request(None, method="OPTIONS", headers={"Origin":"https://myanalyst.net"})
        self.assertEqual(status, 204)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://myanalyst.net")

    def test_safe_error_and_finite_json_and_scope(self):
        payload = json.dumps({"columns": ["Revenue"], "rows": [[1], [2]], "sourceRowCount": 10})
        with patch("index.analyze", side_effect=RuntimeError("SECRET_RAW_DATA")):
            status, _, data = self.request(payload)
            self.assertEqual(status, 500)
            self.assertNotIn("SECRET_RAW_DATA", json.dumps(data))
        with patch("index.analyze", return_value={"values":[float("nan"), np.float64("inf"), np.int64(3)]}):
            status, _, data = self.request(payload)
        self.assertEqual(status, 200)
        self.assertEqual(data["values"], [None, None, 3])
        self.assertEqual(data["scope"], {"sourceRows":10,"analyzedRows":2,"sampled":True})

    def test_busy_instance_rejects_instead_of_queuing(self):
        index._COMPUTE.acquire()
        index._COMPUTE.acquire()
        try:
            status, headers, _ = self.request('{"columns":["Revenue"],"rows":[[1]]}')
            self.assertEqual(status, 429)
            self.assertEqual(headers["Retry-After"], "2")
        finally:
            index._COMPUTE.release()
            index._COMPUTE.release()

    def test_csv_contract_and_numeric_kpi(self):
        for csv in ("a,a\n1,2", "a,b\n1,2,3"):
            with self.assertRaises(InputError):
                dataframe({"csv":csv})
        with patch("_groq.chat", return_value=None):
            status, _, data = self.request(json.dumps({"facts":[{"text":"Revenue 123."}],"kpis":[{"name":"Revenue","value":123}]}), path="/api/conclude")
        self.assertEqual(status, 200)
        self.assertTrue(data["grounding"]["grounded"])


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LLM_API_KEY":"test-key", "LLM_PROVIDER":"groq", "LLM_MODEL":"llama-3.3-70b-versatile", "LLM_BASE_URL":""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def response(self, content="{}", finish="stop"):
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.read.return_value = json.dumps({"choices":[{"message":{"content":content},"finish_reason":finish}]}).encode()
        return resp

    def test_token_budget_for_non_reasoning_models_and_hidden_reasoning(self):
        with patch("urllib.request.urlopen", return_value=self.response()) as send:
            self.assertEqual(_groq.chat([], max_tokens=456), "{}")
            self.assertEqual(json.loads(send.call_args.args[0].data)["max_tokens"], 456)
        with patch.dict(os.environ, {"LLM_MODEL":"openai/gpt-oss-120b"}), patch("urllib.request.urlopen", return_value=self.response()) as send:
            _groq.chat([])
            self.assertEqual(json.loads(send.call_args.args[0].data)["reasoning_format"], "hidden")

    def test_short_retry_long_quota_and_auth_fail_fast(self):
        short = urllib.error.HTTPError("https://example.com", 429, "rate", {"Retry-After":"0"}, None)
        with patch("urllib.request.urlopen", side_effect=[short, self.response()]) as send:
            self.assertEqual(_groq.chat([]), "{}")
            self.assertEqual(send.call_count, 2)
        for status, delay in ((401,"0"),(429,"100")):
            error = urllib.error.HTTPError("https://example.com", status, "rate", {"Retry-After":delay}, None)
            with patch("urllib.request.urlopen", side_effect=error) as send:
                self.assertIsNone(_groq.chat([]))
                self.assertEqual(send.call_count, 1)

    def test_expired_budget_and_truncation(self):
        with _groq.budget(0), patch("urllib.request.urlopen") as send:
            self.assertIsNone(_groq.chat([]))
            send.assert_not_called()
        with patch("urllib.request.urlopen", return_value=self.response(finish="length")):
            self.assertIsNone(_groq.chat([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

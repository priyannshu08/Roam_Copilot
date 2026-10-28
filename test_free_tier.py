import unittest
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.api_core.exceptions import InvalidArgument, ResourceExhausted
from google.rpc.error_details_pb2 import RetryInfo

from free_tier import FreeTierClient, QuotaUnavailable


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class FakeClient:
    def __init__(self, clock, outcomes=()):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append((self.clock(), prompt))
        outcome = self.outcomes.pop(0) if self.outcomes else SimpleNamespace(content="OK")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FreeTierTests(unittest.TestCase):
    def make_client(self, outcomes=(), **kwargs):
        clock = FakeClock()
        raw = FakeClient(clock, outcomes)
        wrapper = FreeTierClient(raw, clock=clock, sleep=clock.sleep, **kwargs)
        return clock, raw, wrapper

    def test_parallel_requests_share_spacing(self):
        clock, raw, client = self.make_client()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(client.invoke, [str(index) for index in range(8)]))
        self.assertEqual([stamp for stamp, prompt in raw.calls], [15 * index for index in range(8)])

    def test_success_cache_expires_and_returns_copies(self):
        clock, raw, client = self.make_client(ttl=60)
        client.invoke("trip").content = "modified"
        self.assertEqual(client.invoke("trip").content, "OK")
        self.assertEqual(len(raw.calls), 1)
        clock.now += 61
        client.invoke("trip")
        self.assertEqual(len(raw.calls), 2)

    def test_retry_honors_server_delay(self):
        retry = RetryInfo()
        retry.retry_delay.seconds = 42
        clock, raw, client = self.make_client([ResourceExhausted("rate limit", details=[retry])])
        self.assertEqual(client.invoke("trip").content, "OK")
        self.assertEqual(clock.waits, [42])
        self.assertEqual(client.get_status(), "")

    def test_retries_are_bounded_and_failure_not_cached(self):
        clock, raw, client = self.make_client([ResourceExhausted("rate limit")] * 3)
        with self.assertRaises(QuotaUnavailable):
            client.invoke("trip")
        self.assertEqual(len(raw.calls), 3)
        self.assertEqual(client.invoke("trip").content, "OK")
        self.assertEqual(len(raw.calls), 4)

    def test_daily_quota_does_not_retry(self):
        clock, raw, client = self.make_client([ResourceExhausted("GenerateRequestsPerDay")])
        with self.assertRaisesRegex(QuotaUnavailable, "daily"):
            client.invoke("trip")
        self.assertEqual(len(raw.calls), 1)
        self.assertEqual(clock.waits, [])

    def test_authentication_errors_do_not_retry(self):
        clock, raw, client = self.make_client([InvalidArgument("API_KEY_INVALID")])
        with self.assertRaises(InvalidArgument):
            client.invoke("trip")
        self.assertEqual(len(raw.calls), 1)

    def test_long_retry_delay_does_not_block_or_retry_early(self):
        retry = RetryInfo()
        retry.retry_delay.seconds = 300
        clock, raw, client = self.make_client([ResourceExhausted("rate", details=[retry])])
        with self.assertRaises(QuotaUnavailable):
            client.invoke("trip")
        self.assertEqual(clock.waits, [])
        self.assertEqual(client.next_request, 300)
        with self.assertRaisesRegex(QuotaUnavailable, "cooling down"):
            client.invoke("trip")
        self.assertEqual(len(raw.calls), 1)
        self.assertEqual(clock.waits, [])


class PlannerTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        st.cache_data.clear()
        st.cache_resource.clear()
        self.clock = FakeClock()
        self.raw = FakeClient(self.clock)
        self.search = Mock()
        self.search.search.return_value = {
            "results": [{"title": "Travel source", "content": "Source summary", "url": "https://example.com/travel"}],
            "images": []
        }
        replacements = [
            patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=self.raw),
            patch("tavily.TavilyClient", return_value=self.search),
            patch("langchain_community.utilities.WikipediaAPIWrapper", return_value=Mock()),
            patch("free_tier.FreeTierClient", side_effect=lambda client: FreeTierClient(
                client, clock=self.clock, sleep=self.clock.sleep
            ))
        ]
        for replacement in replacements:
            replacement.start()
            self.addCleanup(replacement.stop)
        self.addCleanup(st.cache_data.clear)
        self.addCleanup(st.cache_resource.clear)
        self.app = AppTest.from_file(str(Path(__file__).with_name("agents.py")), default_timeout=20)
        self.app.secrets["GOOGLE_API_KEY"] = "test-only"
        self.app.secrets["TAVILY_API_KEY"] = "test-only"
        self.app.run()
        self.assertFalse(self.app.exception)

    def submit(self):
        next(button for button in self.app.button if "Plan My Amazing Trip" in button.label).click().run()
        self.assertFalse(self.app.exception, [error.message for error in self.app.exception])

    def fill_form(self):
        self.app.text_input[0].set_value("London")
        self.app.text_input[1].set_value("Paris")

    def saved_states(self):
        return [value for key, value in self.app.session_state.items()
                if key.startswith("trip_result_")]

    def test_form_uses_one_generation_and_caches_complete_trip(self):
        self.fill_form()
        self.submit()
        self.assertEqual(len(self.raw.calls), 1)
        searches = self.search.search.call_count
        self.assertGreater(searches, 0)
        self.submit()
        self.assertEqual(len(self.raw.calls), 1)
        self.assertEqual(self.search.search.call_count, searches)
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Your Complete Trip Itinerary" in element.value for element in self.app.markdown))

    def test_cache_key_includes_origin_and_interests(self):
        self.fill_form()
        self.submit()
        self.app.text_input[0].set_value("Berlin")
        self.submit()
        self.app.multiselect[0].set_value(["Museums"])
        self.submit()
        self.assertEqual(len(self.saved_states()), 3)
        self.assertEqual(len(self.raw.calls), 3)

    def test_expired_trip_regenerates_its_pdf(self):
        self.fill_form()
        self.submit()
        key = next(key for key in self.app.session_state if key.startswith("trip_result_"))
        old_pdf = self.app.session_state[f"pdf_buffer_{key}"]
        state = self.app.session_state[key]
        state["_saved_at"] = 0
        self.app.session_state[key] = state
        self.clock.now += 3601
        self.raw.outcomes = [SimpleNamespace(content="Updated itinerary")]
        self.submit()
        self.assertEqual(len(self.raw.calls), 2)
        self.assertIsNot(self.app.session_state[f"pdf_buffer_{key}"], old_pdf)
        self.assertEqual(self.app.session_state[key]["final_itinerary"], "Updated itinerary")

    def test_quota_failure_preserves_research_and_retries_only_itinerary(self):
        self.raw.outcomes = [ResourceExhausted("per minute quota")] * 3
        self.fill_form()
        self.submit()
        state = self.saved_states()[0]
        self.assertEqual(state["final_itinerary"], "")
        self.assertIn("limit", state["itinerary_error"])
        self.assertTrue(state["flight_options"])
        self.assertFalse(any("Your Complete Trip Itinerary" in element.value for element in self.app.markdown))
        searches = self.search.search.call_count
        self.submit()
        self.assertEqual(self.search.search.call_count, searches)
        self.assertEqual(len(self.raw.calls), 4)
        self.assertEqual(self.saved_states()[0]["final_itinerary"], "OK")
        self.assertEqual(self.saved_states()[0]["itinerary_error"], "")

    def test_daily_quota_is_reported_without_retries(self):
        self.raw.outcomes = [ResourceExhausted("GenerateRequestsPerDay")]
        self.fill_form()
        self.submit()
        self.assertEqual(len(self.raw.calls), 1)
        self.assertTrue(any("daily allowance" in element.value for element in self.app.warning))

    def test_search_credit_errors_are_visible(self):
        self.search.search.side_effect = RuntimeError("credits exhausted")
        self.fill_form()
        self.submit()
        self.assertTrue(any("credit allowance" in element.value for element in self.app.warning))

    def test_chat_retry_uses_saved_research_without_reextracting_details(self):
        self.app.radio[0].set_value(self.app.radio[0].options[1]).run()
        details = dict(origin="London", destination="Paris", start_date="2026-11-01",
                       end_date="2026-11-04", budget=3000, num_travelers=2,
                       interests=["Food"], is_complete=True)
        extraction = SimpleNamespace(content=json.dumps({
            "updated_details": details, "response_message": "Planning", "needs_revision": True
        }))
        self.raw.outcomes = [extraction] + [ResourceExhausted("per minute quota")] * 3
        self.app.chat_input[0].set_value("Plan London to Paris").run()
        self.assertFalse(self.app.exception, [error.message for error in self.app.exception])
        self.assertEqual(len(self.raw.calls), 4)
        searches = self.search.search.call_count
        self.app.run()
        next(button for button in self.app.button if button.label == "Retry itinerary").click().run()
        self.assertFalse(self.app.exception, [error.message for error in self.app.exception])
        self.assertEqual(len(self.raw.calls), 5)
        self.assertEqual(self.search.search.call_count, searches)
        self.assertEqual(self.saved_states()[0]["final_itinerary"], "OK")


if __name__ == "__main__":
    unittest.main()
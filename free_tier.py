import copy
import threading
import time
from collections import OrderedDict

from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from google.rpc.error_details_pb2 import QuotaFailure, RetryInfo


class QuotaUnavailable(RuntimeError):
    pass


def quota_details(error):
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ResourceExhausted, TooManyRequests)):
            break
        current = current.__cause__ or current.__context__
    else:
        return None

    delay = None
    exhausted = False
    details = getattr(current, "details", ())
    if not isinstance(details, (list, tuple)):
        details = ()
    for detail in details:
        if isinstance(detail, RetryInfo):
            delay = detail.retry_delay.seconds + detail.retry_delay.nanos / 1e9
        elif isinstance(detail, QuotaFailure):
            for violation in detail.violations:
                identifier = getattr(violation, "quota_id", "").lower()
                description = violation.description.lower()
                exhausted |= "perday" in identifier or "per day" in description
        elif isinstance(detail, dict):
            kind = detail.get("@type", "")
            if kind.endswith("RetryInfo"):
                try:
                    delay = float(detail.get("retryDelay", "0s").removesuffix("s"))
                except (ValueError, TypeError):
                    pass
            elif kind.endswith("QuotaFailure"):
                exhausted |= any(
                    "perday" in violation.get("quotaId", "").lower()
                    or "per day" in violation.get("description", "").lower()
                    for violation in detail.get("violations", [])
                )
    message = str(current).lower()
    exhausted |= any(term in message for term in ("perday", "per day", "daily", "limit: 0"))
    response = getattr(current, "response", None)
    headers = getattr(response, "headers", {})
    if delay is None and headers:
        try:
            delay = float(headers.get("Retry-After", ""))
        except (ValueError, TypeError):
            pass
    return delay, exhausted


class FreeTierClient:
    def __init__(self, client, interval=15.0, attempts=3, ttl=3600, cache_size=64,
                 max_wait=120.0, clock=time.monotonic, sleep=time.sleep):
        self.client = client
        self.interval = interval
        self.attempts = attempts
        self.ttl = ttl
        self.cache_size = cache_size
        self.max_wait = max_wait
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.next_request = 0.0
        self.cache = OrderedDict()
        self.status = ""

    def get_status(self):
        with self.status_lock:
            return self.status

    def set_status(self, message):
        with self.status_lock:
            self.status = message

    def clear_cache(self):
        with self.lock:
            self.cache.clear()

    def invoke(self, prompt):
        with self.lock:
            cached = self.cache.get(prompt)
            if cached is not None and self.clock() - cached[0] < self.ttl:
                self.cache.move_to_end(prompt)
                return copy.deepcopy(cached[1])
            try:
                for attempt in range(self.attempts):
                    delay = max(0.0, self.next_request - self.clock())
                    if delay > self.max_wait:
                        raise QuotaUnavailable(
                            f"Gemini is cooling down. Retry in at least {delay:.0f} seconds. Your research is saved."
                        )
                    if delay:
                        self.set_status(f"Waiting for API quota ({delay:.0f}s)")
                        self.sleep(delay)
                    self.set_status("Generating with Gemini")
                    self.next_request = self.clock() + self.interval
                    try:
                        response = self.client.invoke(prompt)
                    except Exception as error:
                        quota = quota_details(error)
                        if quota is None:
                            raise
                        retry_delay, exhausted = quota
                        if exhausted:
                            raise QuotaUnavailable(
                                "Gemini's daily allowance is exhausted or this model has no free quota. "
                                "Your research is saved; retry after the quota resets."
                            ) from error
                        delay = max(self.interval, retry_delay if retry_delay is not None else 30 * (2 ** attempt))
                        self.next_request = max(self.next_request, self.clock() + delay)
                        if attempt + 1 == self.attempts or delay > self.max_wait:
                            raise QuotaUnavailable(
                                "Gemini's request or token limit is still active. "
                                f"Retry in at least {delay:.0f} seconds. Your research is saved."
                            ) from error
                        continue
                    if not getattr(response, "content", None):
                        raise RuntimeError("Gemini returned an empty response. Your research is saved; retry the itinerary.")
                    self.cache[prompt] = (self.clock(), copy.deepcopy(response))
                    self.cache.move_to_end(prompt)
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
                    return response
            finally:
                self.set_status("")
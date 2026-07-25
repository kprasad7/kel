"""Token-bucket rate limiter for requests-per-minute / tokens-per-minute.

Distinct from `kel.budget`: a Budget is a total cap over a run's lifetime
(raises once exceeded); a RateLimiter throttles the *pace* of calls
(blocks/sleeps until capacity is available) so a burst of calls doesn't
exceed a provider's RPM/TPM limits. The two compose — use both.

Known simplification: token cost is reserved using an *estimate*
(`kel.context.estimate_total_tokens` on the input + `max_tokens` for the
worst-case output) taken up front, not refunded down to the real usage
after the call completes. This is deliberately conservative (throttles
slightly more than strictly necessary) rather than under-throttling.
"""

from __future__ import annotations

import asyncio
import threading
import time


class RateLimiter:
    def __init__(self, *, requests_per_minute: float | None = None, tokens_per_minute: float | None = None):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._request_capacity = requests_per_minute or float("inf")
        self._token_capacity = tokens_per_minute or float("inf")
        self._request_tokens = self._request_capacity
        self._token_tokens = self._token_capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        if self.rpm:
            self._request_tokens = min(self._request_capacity, self._request_tokens + elapsed * (self.rpm / 60))
        if self.tpm:
            self._token_tokens = min(self._token_capacity, self._token_tokens + elapsed * (self.tpm / 60))

    def _try_acquire_locked(self, tokens_needed: int) -> float | None:
        """Returns None if acquired, else the number of seconds to wait before retrying."""
        self._refill_locked()
        request_ready = self._request_tokens >= 1
        token_ready = self._token_tokens >= tokens_needed
        if request_ready and token_ready:
            self._request_tokens -= 1
            self._token_tokens -= tokens_needed
            return None
        wait_for_request = (1 - self._request_tokens) / (self.rpm / 60) if self.rpm and not request_ready else 0.0
        wait_for_tokens = (
            (tokens_needed - self._token_tokens) / (self.tpm / 60) if self.tpm and not token_ready else 0.0
        )
        return max(wait_for_request, wait_for_tokens, 0.01)

    def acquire(self, tokens_needed: int = 0) -> None:
        while True:
            with self._lock:
                wait = self._try_acquire_locked(tokens_needed)
                if wait is None:
                    return
            time.sleep(wait)

    async def aacquire(self, tokens_needed: int = 0) -> None:
        while True:
            with self._lock:
                wait = self._try_acquire_locked(tokens_needed)
                if wait is None:
                    return
            await asyncio.sleep(wait)

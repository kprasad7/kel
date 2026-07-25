import time

from kel.models.base import ChatModel
from kel.models.types import Message, ModelResponse, TextPart, Usage
from kel.ratelimit import RateLimitedChatModel, RateLimiter


def test_rate_limiter_allows_calls_within_capacity_without_delay():
    limiter = RateLimiter(requests_per_minute=60)
    start = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05


def test_rate_limiter_throttles_requests_beyond_capacity():
    # 2 requests/minute = 1 every 30s; starting bucket is full (2 tokens),
    # so the 3rd immediate call must wait.
    limiter = RateLimiter(requests_per_minute=120)  # 2/sec, easier to test quickly
    limiter._request_tokens = 1  # start with only 1 token available
    start = time.monotonic()
    limiter.acquire()  # consumes the 1 available token, instant
    limiter.acquire()  # must wait for refill (~0.5s at 2/sec)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3


def test_rate_limiter_tpm_throttles_on_token_budget():
    limiter = RateLimiter(tokens_per_minute=6000)  # 100 tokens/sec
    limiter._token_tokens = 50
    start = time.monotonic()
    limiter.acquire(tokens_needed=50)  # exactly enough, instant
    limiter.acquire(tokens_needed=50)  # needs to wait ~0.5s for refill
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3


async def test_rate_limiter_async_acquire_throttles_too():
    limiter = RateLimiter(requests_per_minute=120)
    limiter._request_tokens = 1
    start = time.monotonic()
    await limiter.aacquire()
    await limiter.aacquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3


class _FakeModel(ChatModel):
    provider = "fake"
    model_id = "fake-1"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return ModelResponse(
            id="r1", model="fake-1", content=[TextPart(text="hi")], stop_reason="end_turn", usage=Usage()
        )

    def stream(self, messages, **kwargs):
        raise NotImplementedError

    async def agenerate(self, messages, **kwargs):
        return self.generate(messages, **kwargs)


def test_rate_limited_chat_model_delegates_generate():
    limiter = RateLimiter(requests_per_minute=6000)
    wrapped = _FakeModel()
    model = RateLimitedChatModel(wrapped, limiter)

    resp = model.generate([Message.user("hi")])

    assert resp.text == "hi"
    assert wrapped.calls == 1


async def test_rate_limited_chat_model_agenerate_uses_async_acquire():
    limiter = RateLimiter(requests_per_minute=6000)
    wrapped = _FakeModel()
    model = RateLimitedChatModel(wrapped, limiter)

    resp = await model.agenerate([Message.user("hi")])

    assert resp.text == "hi"

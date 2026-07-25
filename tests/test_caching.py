import tempfile
from pathlib import Path

from kel.budget import Budget, BudgetTracker
from kel.caching import CachedChatModel, InMemoryCache, SQLiteCache, make_cache_key
from kel.models.base import ChatModel
from kel.models.types import Message, ModelResponse, TextPart, Usage
from kel.observability import ListSink, Tracer


class _CountingModel(ChatModel):
    provider = "fake"
    model_id = "fake-1"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return ModelResponse(
            id=f"r{self.calls}",
            model=self.model_id,
            content=[TextPart(text=f"response {self.calls}")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

    def stream(self, messages, **kwargs):
        raise NotImplementedError

    async def agenerate(self, messages, **kwargs):
        return self.generate(messages, **kwargs)


def test_in_memory_cache_get_set_roundtrip():
    cache = InMemoryCache()
    response = ModelResponse(
        id="r1", model="m", content=[TextPart(text="hi")], stop_reason="end_turn", usage=Usage()
    )
    assert cache.get("key1") is None
    cache.set("key1", response)
    assert cache.get("key1").text == "hi"


def test_sqlite_cache_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.sqlite"
        response = ModelResponse(
            id="r1", model="m", content=[TextPart(text="persisted")], stop_reason="end_turn", usage=Usage()
        )
        cache1 = SQLiteCache(path)
        cache1.set("key1", response)
        cache1.close()

        cache2 = SQLiteCache(path)
        loaded = cache2.get("key1")
        cache2.close()
        assert loaded is not None
        assert loaded.text == "persisted"


def test_make_cache_key_is_stable_and_sensitive_to_inputs():
    key_a = make_cache_key(
        provider="anthropic", model_id="m", messages=[Message.user("hi")], system=None, tools=None,
        max_tokens=100, temperature=None,
    )
    key_a2 = make_cache_key(
        provider="anthropic", model_id="m", messages=[Message.user("hi")], system=None, tools=None,
        max_tokens=100, temperature=None,
    )
    key_b = make_cache_key(
        provider="anthropic", model_id="m", messages=[Message.user("different")], system=None, tools=None,
        max_tokens=100, temperature=None,
    )
    assert key_a == key_a2
    assert key_a != key_b


def test_cached_chat_model_only_calls_wrapped_once_for_identical_requests():
    wrapped = _CountingModel()
    cache = InMemoryCache()
    model = CachedChatModel(wrapped, cache)

    r1 = model.generate([Message.user("hi")])
    r2 = model.generate([Message.user("hi")])

    assert wrapped.calls == 1
    assert r1.text == r2.text == "response 1"


def test_cached_chat_model_different_requests_both_call_wrapped():
    wrapped = _CountingModel()
    model = CachedChatModel(wrapped, InMemoryCache())

    model.generate([Message.user("hi")])
    model.generate([Message.user("bye")])

    assert wrapped.calls == 2


def test_cached_chat_model_emits_hit_and_miss_spans():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])
    model = CachedChatModel(_CountingModel(), InMemoryCache(), tracer=tracer)

    model.generate([Message.user("hi")])
    model.generate([Message.user("hi")])

    cache_spans = [s for s in sink.spans if s.name == "kel.model.cache"]
    assert [s.attributes["hit"] for s in cache_spans] == [False, True]


async def test_cached_chat_model_agenerate_caches_too():
    wrapped = _CountingModel()
    model = CachedChatModel(wrapped, InMemoryCache())

    await model.agenerate([Message.user("hi")])
    await model.agenerate([Message.user("hi")])

    assert wrapped.calls == 1


def test_cache_hit_does_not_double_charge_budget():
    # cache wraps a budgeted model the way get_model() composes them —
    # a cache hit must not call into the inner budgeted model at all
    tracker = BudgetTracker(Budget(max_cost_usd=100.0))

    class _BudgetCharging(ChatModel):
        provider = "fake"
        model_id = "fake-1"
        calls = 0

        def generate(self, messages, **kwargs):
            _BudgetCharging.calls += 1
            tracker.record_usage(Usage(input_tokens=10, output_tokens=5), cost_usd=1.0)
            return ModelResponse(
                id="r1", model="fake-1", content=[TextPart(text="hi")], stop_reason="end_turn",
                usage=Usage(input_tokens=10, output_tokens=5),
            )

        def stream(self, messages, **kwargs):
            raise NotImplementedError

    model = CachedChatModel(_BudgetCharging(), InMemoryCache())
    model.generate([Message.user("hi")])
    model.generate([Message.user("hi")])

    assert _BudgetCharging.calls == 1
    assert tracker.cost_usd_used == 1.0

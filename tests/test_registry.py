import pytest

from kel.models.base import ChatModel, EmbeddingModel
from kel.models.errors import ModelNotFoundError
from kel.models.registry import get_embedding_model, get_model, register_embedding_provider, register_provider


class _DummyModel(ChatModel):
    provider = "dummy"

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs

    def generate(self, messages, **kwargs):
        raise NotImplementedError

    def stream(self, messages, **kwargs):
        raise NotImplementedError


def test_register_and_resolve_provider():
    register_provider("dummy", _DummyModel)
    model = get_model("dummy:model-x", extra="y", instrument=False)
    assert isinstance(model, _DummyModel)
    assert model.model_id == "model-x"
    assert model.kwargs == {"extra": "y"}


def test_get_model_instruments_by_default():
    from kel.observability import InstrumentedChatModel

    register_provider("dummy", _DummyModel)
    model = get_model("dummy:model-x")
    assert isinstance(model, InstrumentedChatModel)
    assert model.provider == "dummy"
    assert model.model_id == "model-x"


def test_get_model_wraps_with_budget_when_given():
    from kel.budget import Budget, BudgetedChatModel

    register_provider("dummy", _DummyModel)
    model = get_model("dummy:model-x", budget=Budget(max_tokens=1000))
    assert isinstance(model, BudgetedChatModel)
    assert model.tracker.budget.max_tokens == 1000
    # budget wraps the already-instrumented model, so tracing still happens underneath
    from kel.observability import InstrumentedChatModel

    assert isinstance(model.wrapped, InstrumentedChatModel)


def test_get_model_wraps_with_cache_as_outermost_layer():
    from kel.budget import Budget, BudgetedChatModel
    from kel.caching import CachedChatModel, InMemoryCache

    register_provider("dummy", _DummyModel)
    model = get_model("dummy:model-x", budget=Budget(max_tokens=1000), cache=InMemoryCache())

    assert isinstance(model, CachedChatModel)
    assert isinstance(model.wrapped, BudgetedChatModel)


def test_get_model_wraps_with_rate_limit_as_innermost_layer():
    from kel.ratelimit import RateLimitedChatModel

    register_provider("dummy", _DummyModel)
    model = get_model("dummy:model-x", rate_limit={"requests_per_minute": 60})

    # instrument (default True) wraps rate-limit, which wraps the raw dummy model
    assert model.wrapped.__class__.__name__ == "RateLimitedChatModel"
    assert isinstance(model.wrapped, RateLimitedChatModel)
    assert isinstance(model.wrapped.wrapped, _DummyModel)


def test_missing_colon_raises():
    with pytest.raises(ModelNotFoundError):
        get_model("no-colon-here")


def test_unknown_provider_raises():
    with pytest.raises(ModelNotFoundError):
        get_model("nope:some-model")


def test_builtin_providers_are_registered():
    from kel.models.registry import _PROVIDERS

    assert "anthropic" in _PROVIDERS
    assert "openai" in _PROVIDERS
    assert "cohere" in _PROVIDERS


class _DummyEmbeddingModel(EmbeddingModel):
    provider = "dummy"

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_register_and_resolve_embedding_provider():
    register_embedding_provider("dummy", _DummyEmbeddingModel)
    model = get_embedding_model("dummy:embed-x", extra="y")
    assert isinstance(model, _DummyEmbeddingModel)
    assert model.model_id == "embed-x"
    assert model.kwargs == {"extra": "y"}
    assert model.embed(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]


def test_get_embedding_model_missing_colon_raises():
    with pytest.raises(ModelNotFoundError):
        get_embedding_model("no-colon-here")


def test_get_embedding_model_unknown_provider_raises():
    with pytest.raises(ModelNotFoundError):
        get_embedding_model("nope:some-model")


def test_builtin_embedding_providers_are_registered():
    from kel.models.registry import _EMBEDDING_PROVIDERS

    assert "openai" in _EMBEDDING_PROVIDERS
    assert "cohere" in _EMBEDDING_PROVIDERS
    assert "anthropic" not in _EMBEDDING_PROVIDERS  # Anthropic has no embeddings API

"""Model spec -> ChatModel factory.

Deliberately small core-maintained provider set (see DESIGN.md 3.1): more
providers register themselves the same way instead of the registry growing
an ever-longer if/elif chain the core team has to maintain forever.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kel.models.base import ChatModel, EmbeddingModel
from kel.models.errors import ModelNotFoundError

_ProviderFactory = Callable[..., ChatModel]
_EmbeddingProviderFactory = Callable[..., EmbeddingModel]

_PROVIDERS: dict[str, _ProviderFactory] = {}
_EMBEDDING_PROVIDERS: dict[str, _EmbeddingProviderFactory] = {}


def register_provider(prefix: str, factory: _ProviderFactory) -> None:
    """Register a `ChatModel` factory under a `provider:model_id` prefix.

    `factory` is called as `factory(model_id, **kwargs)`.
    """
    _PROVIDERS[prefix] = factory


def _register_builtin_anthropic(model_id: str, **kwargs: Any) -> ChatModel:
    from kel.models.providers.anthropic import AnthropicChatModel

    return AnthropicChatModel(model_id, **kwargs)


def _register_builtin_openai(model_id: str, **kwargs: Any) -> ChatModel:
    from kel.models.providers.openai import OpenAIChatModel

    return OpenAIChatModel(model_id, **kwargs)


def _register_builtin_cohere(model_id: str, **kwargs: Any) -> ChatModel:
    from kel.models.providers.cohere import CohereChatModel

    return CohereChatModel(model_id, **kwargs)


def _register_builtin_gemini(model_id: str, **kwargs: Any) -> ChatModel:
    from kel.models.providers.gemini import GeminiChatModel

    return GeminiChatModel(model_id, **kwargs)


def _register_builtin_mistral(model_id: str, **kwargs: Any) -> ChatModel:
    from kel.models.providers.mistral import MistralChatModel

    return MistralChatModel(model_id, **kwargs)


register_provider("anthropic", _register_builtin_anthropic)
register_provider("openai", _register_builtin_openai)
register_provider("cohere", _register_builtin_cohere)
register_provider("gemini", _register_builtin_gemini)
register_provider("mistral", _register_builtin_mistral)


def register_embedding_provider(prefix: str, factory: _EmbeddingProviderFactory) -> None:
    """Register an `EmbeddingModel` factory under a `provider:model_id` prefix."""
    _EMBEDDING_PROVIDERS[prefix] = factory


def _register_builtin_openai_embedding(model_id: str, **kwargs: Any) -> EmbeddingModel:
    from kel.models.providers.openai import OpenAIEmbeddingModel

    return OpenAIEmbeddingModel(model_id, **kwargs)


def _register_builtin_cohere_embedding(model_id: str, **kwargs: Any) -> EmbeddingModel:
    from kel.models.providers.cohere import CohereEmbeddingModel

    return CohereEmbeddingModel(model_id, **kwargs)


def _register_builtin_gemini_embedding(model_id: str, **kwargs: Any) -> EmbeddingModel:
    from kel.models.providers.gemini import GeminiEmbeddingModel

    return GeminiEmbeddingModel(model_id, **kwargs)


register_embedding_provider("openai", _register_builtin_openai_embedding)
register_embedding_provider("cohere", _register_builtin_cohere_embedding)
register_embedding_provider("gemini", _register_builtin_gemini_embedding)


def get_embedding_model(spec: str, **kwargs: Any) -> EmbeddingModel:
    """Resolve a `"provider:model_id"` spec to an `EmbeddingModel` instance.

    Example: `get_embedding_model("openai:text-embedding-3-small")`.
    Anthropic doesn't offer an embeddings API (they point users at Voyage
    AI) so there's no "anthropic" embedding provider — that's accurate to
    the actual provider landscape, not a kel gap.
    """
    if ":" not in spec:
        raise ModelNotFoundError(
            f"Embedding model spec must be 'provider:model_id', got {spec!r}. "
            f"Known providers: {sorted(_EMBEDDING_PROVIDERS)}"
        )
    provider, model_id = spec.split(":", 1)
    factory = _EMBEDDING_PROVIDERS.get(provider)
    if factory is None:
        raise ModelNotFoundError(
            f"Unknown embedding provider {provider!r}. Known providers: {sorted(_EMBEDDING_PROVIDERS)}"
        )
    return factory(model_id, **kwargs)


def get_model(
    spec: str,
    *,
    instrument: bool = True,
    budget: Any = None,
    cache: Any = None,
    rate_limit: Any = None,
    **kwargs: Any,
) -> ChatModel:
    """Resolve a `"provider:model_id"` spec to a `ChatModel` instance.

    Example: `get_model("anthropic:claude-sonnet-5", api_key=...)`.

    Wrapped with tracing by default (`instrument=True`) — observability is
    not optional per DESIGN.md's design principles; pass `instrument=False`
    to get the raw adapter back (e.g. for tight benchmark loops).

    Pass `budget=` a `kel.budget.Budget` or `BudgetTracker` to meter every
    call through this model against it; a Budget is wrapped in a fresh
    tracker, a BudgetTracker is used as-is (share one across multiple
    models/agents to enforce a single combined budget).

    Pass `cache=` a `kel.caching.ResponseCache` (e.g. `InMemoryCache()` or
    `SQLiteCache(path)`) to cache identical `generate()`/`agenerate()`
    calls. The cache is the outermost layer: a cache hit short-circuits
    before budget charging or the inner instrumentation span, so cached
    responses cost nothing and don't double-count against a budget — the
    cache itself still emits a `kel.model.cache` span with a `hit` attribute.

    Pass `rate_limit=` a `kel.ratelimit.RateLimiter` or a dict of its kwargs
    (`requests_per_minute=`/`tokens_per_minute=`) to throttle the pace of
    real calls to the provider. Applied innermost — closest to the actual
    network call — so a cache hit never gets throttled (it never reaches
    this layer at all).
    """
    if ":" not in spec:
        raise ModelNotFoundError(
            f"Model spec must be 'provider:model_id', got {spec!r}. Known providers: {sorted(_PROVIDERS)}"
        )
    provider, model_id = spec.split(":", 1)
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ModelNotFoundError(f"Unknown provider {provider!r}. Known providers: {sorted(_PROVIDERS)}")
    model: ChatModel = factory(model_id, **kwargs)

    if rate_limit is not None:
        from kel.ratelimit.limiter import RateLimiter
        from kel.ratelimit.rate_limited import RateLimitedChatModel

        limiter = rate_limit if isinstance(rate_limit, RateLimiter) else RateLimiter(**rate_limit)
        model = RateLimitedChatModel(model, limiter)

    if instrument:
        from kel.observability.instrumented import InstrumentedChatModel

        model = InstrumentedChatModel(model)

    if budget is not None:
        from kel.budget.budgeted import BudgetedChatModel
        from kel.budget.tracker import BudgetTracker
        from kel.budget.types import Budget

        if isinstance(budget, BudgetTracker):
            tracker = budget
        else:
            tracker = BudgetTracker(budget if isinstance(budget, Budget) else Budget(**budget))
        model = BudgetedChatModel(model, tracker)

    if cache is not None:
        from kel.caching.cached import CachedChatModel

        model = CachedChatModel(model, cache)

    return model

"""Wraps a ChatModel with a response cache. Only `generate()`/`agenerate()`
are cached — `stream()`/`astream()` aren't, because replaying a cached
complete response chunk-by-chunk isn't the same thing as a real stream and
would be misleading to claim. A cache hit is a real trace event of its own
(`kel.model.cache`, `hit=True/False`) — visible even though the wrapped
model's own generate() is never called on a hit, so a hit doesn't also
trigger whatever's inside (budget charges, inner instrumentation spans)."""

from __future__ import annotations

from typing import Any

from kel.caching.cache import ResponseCache
from kel.caching.key import make_cache_key
from kel.models.base import ChatModel
from kel.models.types import Message, ModelResponse, ToolSpec
from kel.observability.tracer import Tracer, get_tracer


class CachedChatModel(ChatModel):
    def __init__(self, wrapped: ChatModel, cache: ResponseCache, *, tracer: Tracer | None = None):
        self._wrapped = wrapped
        self.provider = wrapped.provider
        self.model_id = wrapped.model_id
        self.cache = cache
        self._tracer = tracer or get_tracer()

    @property
    def wrapped(self) -> ChatModel:
        return self._wrapped

    def _key(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSpec] | None,
        max_tokens: int,
        temperature: float | None,
    ) -> str:
        return make_cache_key(
            provider=self.provider,
            model_id=self.model_id,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        key = self._key(messages, system, tools, max_tokens, temperature)
        with self._tracer.span("kel.model.cache", provider=self.provider, model=self.model_id) as span:
            cached = self.cache.get(key)
            span.set_attribute("hit", cached is not None)
            if cached is not None:
                return cached

        response = self._wrapped.generate(
            messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        self.cache.set(key, response)
        return response

    def stream(self, messages: list[Message], **kwargs: Any):
        yield from self._wrapped.stream(messages, **kwargs)

    async def agenerate(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        key = self._key(messages, system, tools, max_tokens, temperature)
        with self._tracer.span("kel.model.cache", provider=self.provider, model=self.model_id) as span:
            cached = self.cache.get(key)
            span.set_attribute("hit", cached is not None)
            if cached is not None:
                return cached

        response = await self._wrapped.agenerate(
            messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        self.cache.set(key, response)
        return response

    async def astream(self, messages: list[Message], **kwargs: Any):
        async for event in self._wrapped.astream(messages, **kwargs):
            yield event

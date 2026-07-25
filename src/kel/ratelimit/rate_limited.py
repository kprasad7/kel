"""Wraps a ChatModel to throttle calls through a RateLimiter — same
decorator pattern as Instrumented/Budgeted/Cached."""

from __future__ import annotations

from typing import Any

from kel.context.tokens import estimate_total_tokens
from kel.models.base import ChatModel
from kel.models.types import Message
from kel.ratelimit.limiter import RateLimiter


class RateLimitedChatModel(ChatModel):
    def __init__(self, wrapped: ChatModel, limiter: RateLimiter):
        self._wrapped = wrapped
        self.provider = wrapped.provider
        self.model_id = wrapped.model_id
        self.limiter = limiter

    @property
    def wrapped(self) -> ChatModel:
        return self._wrapped

    def _estimate(self, messages: list[Message], max_tokens: int) -> int:
        return estimate_total_tokens(messages) + max_tokens

    def generate(self, messages: list[Message], *, max_tokens: int = 1024, **kwargs: Any):
        self.limiter.acquire(self._estimate(messages, max_tokens))
        return self._wrapped.generate(messages, max_tokens=max_tokens, **kwargs)

    def stream(self, messages: list[Message], *, max_tokens: int = 1024, **kwargs: Any):
        self.limiter.acquire(self._estimate(messages, max_tokens))
        yield from self._wrapped.stream(messages, max_tokens=max_tokens, **kwargs)

    async def agenerate(self, messages: list[Message], *, max_tokens: int = 1024, **kwargs: Any):
        await self.limiter.aacquire(self._estimate(messages, max_tokens))
        return await self._wrapped.agenerate(messages, max_tokens=max_tokens, **kwargs)

    async def astream(self, messages: list[Message], *, max_tokens: int = 1024, **kwargs: Any):
        await self.limiter.aacquire(self._estimate(messages, max_tokens))
        async for event in self._wrapped.astream(messages, max_tokens=max_tokens, **kwargs):
            yield event

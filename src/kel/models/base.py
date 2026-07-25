"""The single interface every provider adapter implements.

Sync (`generate`/`stream`) is required of every adapter. Async
(`agenerate`/`astream`) is implemented for the built-in providers
(Anthropic, OpenAI, Cohere) using each SDK's real async client — not a
thread-pool wrapper around the sync path. Third-party adapters that don't
implement async get the NotImplementedError default below rather than
being forced to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from kel.models.types import Message, ModelResponse, StreamEvent, ToolSpec


class ChatModel(ABC):
    provider: str
    model_id: str

    @abstractmethod
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
        """Send messages, return the complete response."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Send messages, yield StreamEvents as the response is generated."""

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
        raise NotImplementedError(f"{self.provider} adapter does not support async generation yet")

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError(f"{self.provider} adapter does not support async streaming yet")
        yield  # pragma: no cover — makes this an async generator per its declared return type

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_id={self.model_id!r})"


class EmbeddingModel(ABC):
    provider: str
    model_id: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_id={self.model_id!r})"

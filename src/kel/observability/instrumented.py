"""Wraps any ChatModel to emit a trace span around every call. This is how
observability stays "not optional" (DESIGN.md design principle 2) without
every provider adapter needing to know about tracing itself — the registry
applies this wrapper by default (see kel.models.registry.get_model)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from kel.models.base import ChatModel
from kel.models.types import Message, MessageStop, ModelResponse, StreamEvent
from kel.observability.tracer import Tracer, get_tracer


class InstrumentedChatModel(ChatModel):
    def __init__(self, wrapped: ChatModel, *, tracer: Tracer | None = None):
        self._wrapped = wrapped
        self.provider = wrapped.provider
        self.model_id = wrapped.model_id
        self._tracer = tracer or get_tracer()

    @property
    def wrapped(self) -> ChatModel:
        return self._wrapped

    def generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        with self._tracer.span("kel.model.generate", provider=self.provider, model=self.model_id) as span:
            response = self._wrapped.generate(messages, **kwargs)
            span.set_attribute("input_tokens", response.usage.input_tokens)
            span.set_attribute("output_tokens", response.usage.output_tokens)
            span.set_attribute("stop_reason", response.stop_reason)
            return response

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[StreamEvent]:
        with self._tracer.span("kel.model.stream", provider=self.provider, model=self.model_id) as span:
            for event in self._wrapped.stream(messages, **kwargs):
                if isinstance(event, MessageStop):
                    span.set_attribute("input_tokens", event.response.usage.input_tokens)
                    span.set_attribute("output_tokens", event.response.usage.output_tokens)
                    span.set_attribute("stop_reason", event.response.stop_reason)
                yield event

    async def agenerate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        with self._tracer.span("kel.model.agenerate", provider=self.provider, model=self.model_id) as span:
            response = await self._wrapped.agenerate(messages, **kwargs)
            span.set_attribute("input_tokens", response.usage.input_tokens)
            span.set_attribute("output_tokens", response.usage.output_tokens)
            span.set_attribute("stop_reason", response.stop_reason)
            return response

    async def astream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamEvent]:
        with self._tracer.span("kel.model.astream", provider=self.provider, model=self.model_id) as span:
            async for event in self._wrapped.astream(messages, **kwargs):
                if isinstance(event, MessageStop):
                    span.set_attribute("input_tokens", event.response.usage.input_tokens)
                    span.set_attribute("output_tokens", event.response.usage.output_tokens)
                    span.set_attribute("stop_reason", event.response.stop_reason)
                yield event
